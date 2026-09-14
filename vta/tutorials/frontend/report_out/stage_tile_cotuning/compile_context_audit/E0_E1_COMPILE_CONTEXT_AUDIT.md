# E0/E1：切图、融合与 AutoTVM 编译上下文审计

日期：2026-09-08。本文保存第一轮 E0、87 段 E1 第一级与三个 anchor 的证据。**后续 23 段第二级及 E2 上板资格化已全部完成**，分别见 [代表编译报告](E1_REPRESENTATIVES_AND_MEMORY.md)、[E2 上板报告](E2_ON_BOARD_QUALIFICATION.md)。本文“下一批/剩余”叙述为当时执行记录，以新报告及计划最新状态为准。本报告对应的编译轮次没有板端测量或 AutoTVM 搜索，不与后续 E2 上板混同。

## E0：各层分别决定什么

| 层 | 本仓库执行位置 | 决定的内容 | 不应混同的内容 |
|---|---|---|---|
| stage 构造与导入 | `split_resnet18_stages.py:1728,1792` | 本次 Relay module 包含的 unit、输入/输出依赖和参数 | 不是选择矩阵 tile |
| 独立量化与 pack/bridge | `profile_split_resnet18_stages.py:348`；`quantize.py:333`；`graphpack.py:863` | `global_scale=8.0`、量化节点、边界与 packed layout | 不是 AutoTVM 搜索融合 |
| Relay 优化和 FuseOps | `src/relay/backend/build_module.cc:328,350` | 各 primitive 的融合分组、输入类型和表达式结构 | primitive 相同不代表不同完整 stage/流水耗时相同 |
| VTA template 与调度 | `vta/python/vta/top/vta_conv2d.py:170,197,227,254` | template 决定 cache、ewise ACC、DMA pragma 与 tensorize 结构；本版本暴露 `tile_b/h/w/ci/co` 和 `oc_nthread/h_nthread` | 参数空间由 template 决定，不代表 AutoTVM 的普遍能力边界 |
| 后端 lowering/codegen | `src/relay/backend/build_module.cc:419`；`vta/python/vta/build_module.py:72` 起 | 基于前述图和配置生成 TIR、VTA API 调用与目标模块 | 逻辑 DMA 不等于物理 DDR burst、DMA 时间或 compute stall |

实际链路为：`切图限定 module → quantize/graph_pack → FuseOps primitive → AutoTVM dispatch / template schedule → lowered TIR / 逻辑 DMA`。因此本项目的 AutoTVM 不自行搜索 Relay FuseOps 分组，但融合上下文可能改变其 template 实际收到的表达式/额外输入；是否有影响要比较实际编译产物，不能仅凭 workload key 回答。

源码 SHA-256、dirty-worktree 基础 commit、TopHub 来源与哈希、target、量化配置、禁用 pass 及 87 段来源 manifest 均保存在 [扫描 provenance](precodegen87_run1/provenance.json)。声明针对该冻结源码与配置，不对其他 TVM/VTA 版本作普遍结论。

## 审计入口为什么可比

`relay.optimize` 和完整 `relay.build` 共用 C++ `OptimizeImpl`；后者在其返回后才进入完整目标 GraphExecutor codegen。审计入口显式带上与 build 相同的 graph Executor/cpp Runtime 属性，复用正式 `prepare_vta_stage`、TopHub context 和 `vta.build_config`。这里“不做完整后端构建”不排除常量折叠内部使用主机计算/编译辅助代码。

为正式入口增加默认关闭的只读 observer/PassInstrument 参数，原有调用方式不变。三个预先选择的 layer4 endpoint（`vta:15:15/15:16/15:17`）分别运行 optimize 和正式 `build_vta_stage`：FuseOps 之前、之后的 IRModule 均通过 `tvm.ir.structural_equal(map_free_vars=True)`；由优化图恢复并显式查询的 workload/config 与正式 lowering 实际 dispatch 记录也全部一致。见 [3 段完整资格化](qualification2/summary.json)。

单独 optimize 不触发实际 VTA schedule dispatch；第一级的记录必须称为“从优化后的 packed convolution 恢复 workload，再显式查询 TopHub”，不能伪装成已完成全部段的真实 lowering。3 段资格化只验证这 3 段的路径对应，不把它写成 87 段全量实际 build。

## E1 第一级的字段与身份映射

每段保留量化前、量化后、packed、优化后 Relay 文本、原参数内容哈希、primitive 规范化表达式 DAG、卷积 workload/config 和完整输入输出 contract。文本省略大型 constant 元数据，不将原始文本的符号名/地址差异当成结构差异。

primitive 结构签名忽略变量命名、自动生成 `hash/global_symbol` 和单独冻结的 virtual-device 属性；保留 op、依赖 DAG、参数及输出 shape/dtype、算子属性和函数内部量化标量常量。kernel/bias 作为 primitive 参数时，其 shape/dtype 被保留，参数内容另记录，不把网络权重不同本身当成不同 schedule 类。

语义 occurrence 使用 `unit index + unit 内 conv ordinal`：从冻结域最小 main_preadd 段建立两个有依赖顺序的卷积参考；projection 使用 main_preadd+skip_proj 段中唯一 1×1 卷积。跨段通过完整 packed int8 kernel 的 shape/dtype/content SHA-256 精确匹配，并检查 unit 所属、同段唯一性与期望卷积数量。**不是仅按 shape 或跨段遍历位置匹配。** 若量化使 kernel 内容变化、碰撞或无法匹配，必须标 unresolved，不自动套用旧 occurrence；这将限制复用结论。

扫描对每个 occurrence 同时报告 placement 数、唯一 workload/config 数和唯一卷积 primitive context 数；完整 stage 的 adapter、残差/尾部 primitive 另保留，不能以卷积 primitive 不变证明整个图的融合不变。当前保存的是编译上下文，不是 slot/coherence 或物理 DDR 性能模型。

## 当前结果与剩余门槛

三个 layer4 endpoint 的共同卷积均命中原 TopHub 配置：main 两个卷积为 `203/243`，projection 为 `203`。在这些端点间，各自卷积 primitive 的 typed DAG 签名一致；完整 stage 的输入输出、残差/转换 primitive 仍随范围改变。这个局部结果不能代替 87 段矩阵，更不能证明 tile 全空间最优解不变。

### 87 段第一级最终结果

见 [完整扫描](precodegen87_run1/summary.json)、[身份与上下文矩阵](precodegen87_run1/occurrence_contexts.json)、[覆盖与代表选择](precodegen87_run1/coverage_and_representatives.json)。

| 指标 | 结果及分母 |
|---|---|
| 第一级成功段 | 87/87；失败 0 |
| 卷积身份映射 | 693/693 次出现；19 个语义卷积；unresolved 0 |
| 同一语义卷积跨 placement 的 workload/config | 19/19 均各只有 1 种；全域共 10 类 workload；fallback 0/693 |
| 同一语义卷积跨 placement 的 typed primitive context | 19/19 均只有 1 种，覆盖 693 次出现 |
| 全图 primitive | 1,267 次调用，72 类签名，其中卷积类 16、非卷积类 56 |
| 已捕获完整 TIR 的代表段 | 3/87；覆盖 3/19 个语义卷积、9/72 类 pre-codegen primitive |

这里“19 个卷积、16 类卷积 primitive、10 类 workload”使用不同等价关系，不互相混充：同形状卷积可能因是否融合 ReLU 等而属于不同 primitive 类，不同语义卷积也可能属于同一类。本次证明的是**同一个语义卷积换 stage 后的第一级上下文未变**，不是所有命中同一 workload 的不同卷积都具有相同完整融合表达式。

这比旧 H1 的“同 workload 命中相同 config”强，但尚不是所有 lowered TIR 都相同，更不是完整 AutoTVM 空间中的最优 tile 不变。不能据此把完整 stage 建模为裸卷积时间简单相加：非卷积 adapter、CPU tail、调用次数和 shared-edge 仍需计入。

### 第二级已冻结的下一批代表

代表选择覆盖全部 72 类 primitive，固定 layer2/3/4 的 9 个相邻端点，另取 `01..02、03..04、18..19` 三个分层负对照，再按新增类覆盖数贪心补齐，共 23 段。因 19 个卷积均未出现跨 placement context 差异，卷积差异段集合为空；这不表示非卷积 stage 图不变。选择只依据第一级签名，未依据板端性能挑例。

已有三个 layer4 anchor 留作已执行项，剩余 20 段为 `01..02/03/04/05/08、03..04、05..05/06/07、08..09/10/18、10..10/11/12、13..14/15/16、18..18/19`。下一轮执行这些完整 build/TIR、建立 primitive 到 Graph JSON 函数的身份映射，再做 E2 完整 descriptor 与 occurrence 聚合/运行时交叉验证。不能把 23 段代表编译外推为 87 次精确完整构建。

第二级覆盖和 E2 验证结束前，不宣称完整 H5/E1 完成，也不直接跳过 H6。已有 Graph JSON/object 与 TIR 快照不称板端正确性或加速结果。

### 三个 layer4 anchor 的 TIR 与逻辑 DMA

[完整 TIR 捕获](anchor_tir3/summary.json)逐一核对 Graph JSON 中的函数及调用次数，覆盖率为 `6/6、5/5、6/6` 个唯一生成函数，对应各 `6` 次图函数调用。`15:16` 的输出转换函数被调用两次，不能只按唯一生成函数各加一次。原始 JSON 有逐文件 SHA，TIR 文本仅供阅读；本地重新加载时必须先注册 VTA intrinsics。

[结构比较](anchor_tir3/structural_comparison.json)只统一 `global_symbol`，随后使用 TVM `structural_equal(map_free_vars=True)`；不删除 DMA、量化或循环参数。两个 main 卷积在三段中的 TIR 均相同，projection 在后两段中相同。因此共同卷积的循环、计算和逻辑 DMA 调用结构在这三个 anchor 的冻结 incumbent 下不变。这里比较的是完整 PrimFunc，而不是文本哈希或函数名相同。

[静态 LOAD/STORE 描述符普查](anchor_tir3/logical_dma_census.json)对静态循环展开并按 GraphExecutor 调用次数加权；包含 ACC LOAD。含 DMA 函数若出现不支持的条件/动态循环则拒绝计数，不默认为精确。当前结果如下：

| VTA segment | LOAD 调用 | LOAD B | 其中 ACC LOAD B | STORE 调用 | STORE B |
|---|---:|---:|---:|---:|---:|
| 15..15 | 196 | 3,693,568 | 4,096 | 4 | 50,176 |
| 15..16 | 262 | 3,913,216 | 6,144 | 6 | 75,264 |
| 15..17 | 262 | 3,913,216 | 6,144 | 6 | 75,264 |

`15..15 → 15..16` 增加 `66` 次 LOAD、`219,648 B` LOAD、`25,088 B` STORE，数值与旧实验不同起点 `03..15 → 03..16` 的运行时 delta 一致；这是跨起点描述性交叉核对，不代替这三个新产物自身的板端验证。ACC 的 `2` 次 / `2,048 B` 增量也被完整 fused TIR 捕获，说明 bare-convolution 聚合漏项可以有明确来源，而不是直接归咎 DDR 拥塞。

`15..16 → 15..17` 的 VTA LOAD/STORE 描述符相同，但完整 stage 不相同：后者新增 residual add/ReLU 的主机并行循环，通过 `VTABufferCPUPtr` 访问 packed buffer，且输出转换调用数从两次变为一次。见 [tail TIR](anchor_tir3/vta_15_17/lowered_004.tir)。**名为 VTA stage 不意味着其中每个操作都在 FPGA 上执行；VTA DMA 不变也不意味着 CPU 共享内存访问或交接成本不变。** 此处不据 CPU 指针 API 单独推断 cache/coherence 属性或物理 DDR bytes。

本普查尚不包括运行时 uop/instruction traffic、ALU 执行计数、CPU adapter 的访存量、物理 burst 或 stall，也未完成 occurrence 聚合逐项校验和代表集合覆盖，因此 E2 仍未完成。

## 审计工具自检与失败保留

- 7 个本地测试已通过，覆盖 alpha-renaming、量化标量、shape、自动 hash 属性、参数依赖关系，以及 TIR 变量/导出符号归一化和循环 extent 保留；真实 build 路径比较另有上述 3 段。
- `qualification1` 因当前 TVM `DictAttrs` 不提供 `.get()`，在分析阶段失败；修正后 `qualification2` 的三个正式 build 全部通过，失败目录保留，不计作编译非法段。
- 编译主机单线程 probe 对 `vta:01:14` 的全部卷积 context/workload/config 与默认线程运行完全相同；该单段 probe 不构成全域身份映射，故其单独报告里 14 个 occurrence unresolved 是缺少最小参考段导致，不能解读成 kernel 变化。未凭此宣称编译速度提升或改变板端线程。
- TIR 捕获器早期 `anchor_tir1/2` 因 Environment 属性 API 使用错误，在实际 build 之前退出；修正为本版本 `cfg_dict`，另起目录保存后续结果，不将这些失败计为 stage 编译失败。
- TIR 比较器首次独立加载缺少 VTA intrinsic 注册，随后发现本版本 PrimFunc 不支持 `without_attr`；修正为导入 VTA 和 `with_attr("global_symbol", "normalized")` 后，最终比较及新增测试通过。没有用失败运行生成结论。
- 本轮成功的完整 target build 共 6 次、覆盖 3 个不同 segment：资格化 3 次，TIR 捕获 3 次；其余 87 段只运行第一级审计。新测试与已有 tuning 回归合计 `26 passed`，`py_compile` 与 `git diff --check` 通过。没有重新调优、刷板或测流水 FPS。

## 复现入口

使用 `/home/orange/miniconda3/envs/vta-resnet/bin/python`，工作目录为仓库根目录，数据缓存环境 `TEST_DATA_ROOT_PATH=/tmp/tvm_test_data`、`MPLCONFIGDIR=/tmp/mpl`。每次选择新的输出目录，工具拒绝覆盖既有结果。

- `audit_vta_compile_context.py --output <new_scan>`：87 段第一级；添加 `--segments vta:15:15 vta:15:16 vta:15:17 --qualify-builds` 则复现三段正式路径资格化。
- `capture_vta_segment_tir.py --output <new_archive>`：三个冻结 anchor 完整 build/TIR；本次设置 `TVM_THREAD_POOL_SPIN_COUNT=0`，其他主机环境见 archive provenance。
- `compare_vta_segment_tir.py --archive <archive> --output <new_json>`：完整图函数覆盖、调用次数、TIR 结构比较。
- `summarize_vta_archived_dma.py --archive <archive> --output <new_json>`：逻辑 LOAD/STORE census；不自动宣布 E2 完成。
- `summarize_vta_compile_context.py --scan <completed_scan> --output <new_json>`：全域统计与第二级待执行代表选择。覆盖定义包括非卷积 adapter；代表选择不是已经编译的证据。

上述脚本均位于 `vta/tutorials/frontend/`。
