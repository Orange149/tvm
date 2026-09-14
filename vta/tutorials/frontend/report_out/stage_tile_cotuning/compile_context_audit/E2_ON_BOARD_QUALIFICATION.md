# E2 上板：静态 DMA 与完整 segment 的运行时对齐

日期：2026-09-08。**23/23 冻结代表全部通过，E2 已完成。** 第一批 11 段加剩余 12 段共 69 组输入、207 次 profiler 采样、138 次 warmup；3,726 项计数字段比较与 27,998,208 个输出元素比较（包含重复样本）均无差异。见 [全量合并与独立复核](board_e2_all23_summary.json)。本实验不做新 AutoTVM，不报告 FPS 或切点性能提升。

## 实验协议与证据边界

- 开发板 `192.168.1.247:9091`，boot `a5220e22-f562-40be-bbe9-058b6fbeed50`；192 MiB u-dma-buf 与 FPGA `operating`。本轮未重刷 bitstream 或替换 runtime。
- 每段测量前后重新核对 bitstream、libvta、libtvm_runtime 的冻结哈希、boot、CPU 频率及 IIO PS 温度。RPC 独占顺序运行，设置一个 TVM 线程、affinity mode 1；板上没有 taskset，未声称额外显式固定 host CPU 核。没有多 boot 性能统计。
- 第一批事先固定 layer2/3/4 的三个端点各 3 段，以及 `01..02、18..19` 两个不变对照。通过后，为覆盖全部冻结 primitive 类，对原 23 段清单中所有剩余 12 段做相同检查，不按性能挑选。
- 每段使用 3 组确定输入：均匀 `[-1,1]`、正态 `(0,2)`、全零；seed 为 `260908/260909/260910`。每组 2 次 warmup，再清零 profiler 并分别采样 3 次。输出获取在 profiler 快照之后，不把 RPC 取回输出的复制计入 graph-run DMA。
- 每段独立量化到 LLVM 生成参考，与 VTA 的全部输出逐元素精确比较；这检查该段自己的实现正确性，**不证明不同切图的整网输出数值等价**，后者仍由 E3 控制量化后检查。
- 上板前重新构建 graph/library/params；Graph JSON 哈希必须与静态 census 相同，每个 lowered PrimFunc 须与原捕获 TIR 结构相等。保存实际上传 `.so` 和 params 的 SHA，不把不同二进制的计数混用。

预注册：[第一批](board_e2_run1/preregistered.json)、[剩余集合](board_e2_remaining_run1/preregistered.json)。各目录包含实际 `.so`、`graph.params`、输入 npz、TIR、TopHub 审计、逐次 profile 与输出比较记录。TopHub 之外的 LLVM reference 调度可能打印“未调优”提示；VTA 构建启用 `require_tuned=True`，不忽略 VTA fallback。

两批为不相交的完整 23 段集合，复核工具检查覆盖与预注册一致、每段 input/repeat 笛卡尔积恰为 3×3，并从原始 runtime profile 重新逐字段比较静态 census，而不是仅相信脚本的 passed 标记。CPU governor 始终观察为 userspace、频率 1,066,666 kHz，逐段前后 PS 温度采样为 31.80–34.51 °C；这是离散快照范围，不是全程峰值保证。69 组输入中每段的 3 个分布是功能覆盖，207 次采样不是 207 个独立 boot。

本轮同时闭合了 23 段的静态与运行时逻辑 DMA、ACC、host ALU/GEMM push 证据。因此这些实际捕获并运行过的段可以称“segment 级精确静态逻辑 DMA（运行时资格化）”；其他 64 段仍只有第一级上下文或代表复用依据，不能声称逐段都上板验证了。CPU helper 逻辑 BufferLoad/Store 的静态计数没有物理 DDR counter 对应验证，仍保留原限定。

## 已确认的相邻切点结果

下表为第一批已经实测通过的每次 graph-run 计数；同段的 9 次采样一致。

| segment | LOAD 次数 | LOAD B | ACC LOAD B | STORE B | host ALU push 次数 |
|---|---:|---:|---:|---:|---:|
| 05..05 | 208 | 1,445,888 | 3,072 | 200,704 | 80 |
| 05..06 | 280 | 1,846,784 | 5,120 | 301,056 | 112 |
| 05..07 | 280 | 1,846,784 | 5,120 | 301,056 | 112 |
| 10..10 | 200 | 1,488,896 | 2,048 | 100,352 | 40 |
| 10..11 | 268 | 1,736,192 | 4,096 | 150,528 | 56 |
| 10..12 | 268 | 1,736,192 | 4,096 | 150,528 | 56 |
| 15..15 | 196 | 3,693,568 | 4,096 | 50,176 | 20 |
| 15..16 | 262 | 3,913,216 | 6,144 | 75,264 | 28 |
| 15..17 | 262 | 3,913,216 | 6,144 | 75,264 | 28 |

检查不只比较总 bytes，还包括输入/权重/ACC/out 分类、LOAD/STORE 调用、小请求与 stride 计数，以及 host ALU/GEMM push 次数。缺失字段即失败，不将 missing 当 0。这里 ALU 指 `VTAPushALUOp` 调用次数，不是硬件 ALU 操作数或等待周期。

现在能够支持的结论：**在固定合法 schedule 和已资格化编译上下文中，可以从静态 TIR 预测完整 segment 的逻辑 DMA 请求次数与 payload，并与真实运行时精确对齐。** 裸卷积聚合需要 fused ACC 修正。不能进一步把逻辑描述符自动换算成物理 DDR burst、传输时间或 compute 等待 LOAD/STORE 的周期。

## 完整路径边界对账：尾部不能“凭空消失”

新增 [六条完整路径的 contract ledger](endpoint_full_path_contracts.json)，从冻结 manifest 取合法的三段路径，保持 CPU prefix 不变、名义 CPU 线程固定为 `1/1`：

| 层级 | tail 在外 | tail 在内 | 两条 inter-stage 边合计 payload 变化 | 理论 K2 对齐 slot 变化 |
|---|---|---|---:|---:|
| layer2 | CPU00..04 → VTA05..06 → CPU07..20 | CPU00..04 → VTA05..07 → CPU08..20 | −401,408 B | −802,816 B |
| layer3 | CPU00..09 → VTA10..11 → CPU12..20 | CPU00..09 → VTA10..12 → CPU13..20 | −200,704 B | −401,408 B |
| layer4 | CPU00..14 → VTA15..16 → CPU17..20 | CPU00..14 → VTA15..17 → CPU18..20 | −100,352 B | −200,704 B |

CPU→VTA 边的 contract 不变，VTA→CPU 边从两个 float32 张量变成一个；tail 从原 CPU suffix 移到 VTA stage 内的 CPU helper。上轮 stage 内逻辑 CPU 访问减少 196/98/49 KiB，不能代替这两条完整方案的总访存变化；还需实际编译 CPU suffix、核对量化和对应执行时间。

ledger 的职责划分：VTA DMA 归 VTA stage；内部 CPU helper 已包含在该 stage 内，不能再收一次 edge copy；外部 tail 只在 tail-outside 方案归 CPU suffix；边界只记录一次 contract payload，实际 get/set 复制次数单独核对；理论 K2 容量按每张量 align256，不等于 bump allocator 的真实 high-water。形状/dtype 匹配只是零拷贝必要条件，不声明这六条流水已完成物理绑定或整网正确性验证。

## 后续出口

E2 已结束，下一项是 E3 的量化/边界因果对照及 E6-S 全 4623 topology 的 shared-edge 审计；本 ledger 只覆盖 6 条路径，不能把 E6-S 勾完。没有观察到共同卷积 context 变化，因此不立即扩大 tile 搜索，也不把“能预测 DMA”写成“已改善切图排名/FPS”。

本轮新增工具与已有编译/tuning 回归共 `36 passed`，Python 语法与 diff 检查通过。RPC 服务由本轮启动，测量结束后关闭；[收尾日志](board_e2_after.txt)确认 9091 不再监听、FPGA 和 u-dma-buf 保持可用。RPC child 的 SIGTERM/status 15 为显式 CloseRPCConnection 后服务端的正常回收，不能误读成算子超时；主实验脚本两批均正常退出。
