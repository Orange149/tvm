# E1 第二级与 E2 静态部分：代表编译及切点内存差异

日期：2026-09-08。本轮为 compile-only，没有新增 AutoTVM 搜索、板端运行或 FPS。23 段代表全部完成，包含对旧 3 个 anchor 的带身份映射重放；旧结果不覆盖、不删除。

后续状态：本文之后已完成全部 23 段的 E2 上板计数/输出资格化，见 [E2 报告](E2_ON_BOARD_QUALIFICATION.md)。本文“仍待运行时验证”等段落保留编译阶段的历史证据边界，最新计划以 E2 报告为准；静态 census 原始文件不回写 qualification 标记，以独立上板 sidecar 关联。

## 1. 做了什么，覆盖到哪里

代表集合沿用上一轮冻结的 [23 段清单](precodegen87_run1/coverage_and_representatives.json)，没有根据性能挑选。复用正式 `build_vta_stage`、独立量化、boundary bridge 和 TopHub；执行前核对第一级冻结的全部源码哈希，保存实际 libtvm SHA、VTA 配置、主机线程环境和各段 TopHub 审计。

证据：[编译汇总](representatives23_run1/summary.json)、[环境](representatives23_run1/provenance.json)、[上下文/TIR/CPU 检查](representatives23_run1/context_verification_v2.json)。

| 检查 | 实际覆盖 |
|---|---|
| 第一级（上一轮） | 87/87 段；693 次卷积出现；19 个语义卷积 |
| 本轮正式 build | 23/23 代表；72/72 类 primitive（16 类卷积、56 类非卷积） |
| 原图到编译图清单 | 23/23 段的 primitive 签名多重集合相同，保留调用次数 |
| Graph JSON → TIR | 195 次图函数调用，182 个段内唯一函数，全部关联 |
| 同类 primitive 的跨代表 TIR 比较 | 110 次非自比较，全部相同；72 类中 37 类只有一个代表实例，不伪称这些类做了跨实例比较 |
| 同一语义卷积的跨代表 TIR 比较 | 84 次出现覆盖 19 个语义卷积，65 次非自比较全部相同；19 个卷积均有至少两个代表 placement |
| 完整图静态 LOAD/STORE 对账 | 23/23；输入、权重、STORE 与裸卷积聚合一致，LOAD 剩余差异全部为 ACC LOAD |
| CPU helper 逻辑读写计数 | 23/23 段覆盖；静态尾部条件已处理，当前无 unsupported helper |

关联方式不是“相似函数名/相同 shape”：在 `AnnotateMemoryScope` 后保留 primitive 的规范化签名、编译器 hash、packed kernel 内容哈希；Graph JSON 的 `hash` 将其关联到 `func_name`，再读对应 PrimFunc。校验 hash 无签名歧义、hash→函数在段内唯一、调用多重集合相等和全部函数覆盖。源码路径见 `src/relay/backend/graph_executor_codegen.cc:434`、`src/relay/backend/te_compiler.cc:759`。跨段卷积身份沿用第一级的完整 packed weight 校验。

TIR 比较只统一 `global_symbol` 并使用 `structural_equal(map_free_vars=True)`；不删除循环、量化、DMA 或存储参数。因此结论是：**冻结 incumbent 下，已测代表中的同一卷积生成代码没有随切图变化。** 这不是 87 次完整 TIR 构建，也不是完整 tile 空间最优解不变的证明。

## 2. 裸卷积聚合漏掉的部分现在能解释了

[完整 LOAD/STORE census](representatives23_run1/logical_dma_census_v2.json)按静态循环与 GraphExecutor 调用次数计数，包含 ACC。与旧 10 类 workload 的静态聚合逐项比较，23/23 段的额外 LOAD 次数、字节数、小请求和 stride 差异均完全由 ACC 描述符解释。输入、权重、STORE 以及其请求形态没有额外差异。见 [对账 JSON](representatives23_run1/dma_aggregation_comparison.json) / [CSV](representatives23_run1/dma_aggregation_comparison.csv)。对账使用 v1 census；v2 的 LOAD/STORE 数据不变，只增加 push-scope 计数。

例如 `15..15 → 15..16`：新增 LOAD `66` 次 / `219,648 B`，其中裸 projection 为 `64` 次 / `217,600 B`，差额恰为 ACC `2` 次 / `2,048 B`。这个结果支持增加明确的 fused ACC 修正项，而不是用自由参数把这部分误归因于 DDR 竞争或随机 DMA 碎片化。

另已统计 `coproc_uop_scope` 的静态执行次数：三个层级 `main+projection → +tail` 的 ALU push-scope 分别保持 `112/56/28` 次不变。LLVM 将此 scope 交给 `CreateStaticInit`，运行时 `VTAPushALUOp` 对应 host push 入口（`src/target/llvm/codegen_cpu.cc:1512`、`vta/runtime/runtime.cc:2617`）。这里不把 scope 次数当作硬件 ALU 操作数、uop 首次填充次数或 compute stall cycle；还须与运行时 push counter 对齐。

## 3. 与共享内存更直接相关的相邻切点规律

三个残差块都观察到：把 `add_relu_tail` 放在 main+projection 所在 stage 中，卷积集合、卷积 TIR、VTA DMA 和 ALU push-scope 均不变，但边界从两个 float32 张量变成一个，CPU helper 的执行与物化路径改变。

| 切点移动 | VTA LOAD B（前后相同） | stage 内 CPU 逻辑读写 B：前 → 后 | 减少 | float32 输出 payload B：前 → 后 |
|---|---:|---:|---:|---:|
| 05..06 → 05..07 | 1,846,784 | 2,007,040 → 1,806,336 | 200,704（196 KiB） | 802,816 → 401,408 |
| 10..11 → 10..12 | 1,736,192 | 1,003,520 → 903,168 | 100,352（98 KiB） | 401,408 → 200,704 |
| 15..16 → 15..17 | 3,913,216 | 501,760 → 451,584 | 50,176（49 KiB） | 200,704 → 100,352 |

在这三个例子里，令一个 packed int8 输出大小为 `B`：

- tail 在外：两个 packed 输出分别转换成 float32，CPU 逻辑读 `2B`、写 `8B`。
- tail 在内：CPU 先在 packed buffer 上做 add/ReLU，再转换一个输出，读 `3B`、写 `5B`。
- 因而该局部路径的逻辑访问净减少 `2B`，输出 contract 的 payload 减少 `4B`。输入路径在每个配对内相同。

这里的“VTA stage”包含 CPU helper，并非所有操作都由 FPGA 执行；CPU 通过 `VTABufferCPUPtr` 访问 buffer。该指针 API 本身不能证明实际 cache/coherence 或物理路由。

重要边界：这些是当前 pass 的逻辑 BufferLoad/Store 计数，包含向量 lane 与静态尾部条件，**不是实际 DDR bytes，也不是性能收益**。表格比较的是选中 stage 内的访问；旧方案在 stage 外执行的 CPU tail 尚未计入这一表，不能把表中差值直接当作完整流水总访存差值。边界 payload 也不等于框架 API 复制量，更不能直接乘 K2 后声称整条流水峰值内存下降。

这条规律只对上述三类相邻端点成立，不扩写成“stage 越长越好”；其他切点还会引入额外输入、不同融合、CPU 线程和单 VTA owner 串行成本。

## 4. 对创新点与下一步的约束

E1 的两级编译审计已经完成。它支持复用冻结 incumbent 的卷积编译成本，但不支持以“切图普遍改变卷积最优 tile”作为当前证据结论。也没有证据要求立即扩大 AutoTVM 搜索。

目前更具体的 C1 路线是：`可复用的卷积/DMA 基础项 + fused ACC 修正 + 切点相关 CPU adapter/残差路径 + shared-edge contract`。其中后两项与共享缓冲的张量数、dtype、layout 和物化直接相关。当前只完成机制及静态特征，尚未证明新特征改善 Top-20 排名或 FPS，不能宣布新的模型增益。

下一步按顺序收口：

1. E2 运行时资格化：对冻结的层级端点与不变负对照，对齐完整 LOAD/STORE/ACC、host ALU/GEMM push counter 和数值正确性。只测少量代表，不扩大 tile 搜索。
2. E3/E6-S：为这三对切点列出完整前后 topology 的 CPU tail 归属、线程、producer/consumer、输出 contract、adapter 与 shared-slot，避免局部计数或重叠成本被重复相加；量化一致性沿用原 E3 gate，不因卷积 TIR 一致而自动跳过。
3. 将通过资格化的修正项纳入 frozen search，并做独立消融。若排序仍不变，保留为可审计机制/复用边界，不包装成 FPS 加速。

H6 的 tile 排名验证本轮没有触发卷积 context 差异；E3 因果拆分和 E2 运行时验证尚未闭合，因此不在本轮把所有 H5/H6/整篇论文门槛一起勾完。

## 5. 自检、失败与复现

CPU 计数初版对静态 if 尾部拒绝计数，保留 `context_verification.json`，其中总量是已覆盖函数的小计，不用于上表。v2 支持仅依赖静态循环变量的 then/else 条件枚举，动态条件仍拒绝，并显式记录 totals_complete。测试中一次 `IfThenElse` 构造漏传本版本所需的 `else_case`，修正后重跑；不把这个测试构造错误算作编译失败。

最终新增与既有 tuning 回归共 `31 passed`；包括 CPU 标量/向量字节数、静态 then/else、动态条件拒绝、uop scope 只乘外部 host 循环而不误乘内部 kernel 循环等检查。`py_compile` 与 `git diff --check` 通过；另逐段确认 census v1/v2 的全部 LOAD/STORE 总量一致。

脚本在 `vta/tutorials/frontend/`：`capture_vta_segment_tir.py --selection .../coverage_and_representatives.json --scan .../precodegen87_run1 --output <new>`，随后运行 `verify_vta_representative_contexts.py`、`summarize_vta_archived_dma.py` 和 `compare_vta_segment_dma_aggregation.py`。全部输出拒绝覆盖；本轮编译日志保存在 `/tmp/vta_representatives23_run1.log`（临时，持久证据以报告目录内 JSON/TIR/object 为准）。
