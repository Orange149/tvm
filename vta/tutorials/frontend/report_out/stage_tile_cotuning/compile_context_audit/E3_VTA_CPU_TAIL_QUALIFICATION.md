# E3：冻结 VTA producer 与共同参考 CPU tail 的板端资格化

日期：2026-09-08。**本次单帧三臂数值、逻辑 DMA 与共享缓冲范围资格化完成；完整 E3 未完成。** 不含新 AutoTVM、流水 FPS、多 boot 性能或双 slot 并发安全实验。保留已有 int8 回绕，不是量化精度修复。

## 实验做了什么

复用 E2 的六个冻结 VTA 二进制和 params，不重新编译或调优 VTA：

- layer2：producer `05..06`，joined `05..07`；
- layer3：producer `10..11`，joined `10..12`；
- layer4：producer `15..16`，joined `15..17`。

仅新增三个 AArch64 CPU tail，直接编译上轮已资格化的 once-quantized `float32_consumer/relay.json`，不重新量化。它将 producer 的 float32 量化格点精确恢复为 int8，再执行共同参考中的 tail；不是原来的无窄化 float32 add/ReLU。

每层比较三臂：

| 臂 | 路径 | 边界复制 |
|---|---|---|
| mono | 冻结 joined VTA stage | 无跨 Executor 交接 |
| shared | 冻结 producer → CPU tail | 两个 ext_dev owner 的 CPU view 直接绑定 consumer |
| copy | 同一 producer → 同一 CPU tail | 从上述 CPU view 各调用一次 consumer set_input |

shared/copy 共用相同 producer 输出绑定方式，避免把 producer 内部输出存储差异混入交接对照。两张量同帧顺序完成后才运行 consumer；single-inflight、每张量一个交接缓冲，不是双 slot 多帧流水。

三组原 E2 输入全部复用，包含上一轮 43 个回绕位置；每个 case/arm 采样 3 次，按 case/repeat 轮换三臂顺序。每层正式采样前各臂 2 次 warmup。诊断 profiler、输出下载、RPC 与日志均存在，**不采用本轮时间字段推断性能**。

## 数值与计数结果

| 项目 | 实际完成 |
|---|---:|
| 正式 VTA invocation | 81（每层 27） |
| VTA warmup invocation | 18 |
| 正式 CPU tail invocation | 54 |
| 最终输出元素比较 | 4,741,632，全部相同 |
| producer 中间输出元素比较 | 6,322,176，全部相同 |
| copy 臂边界 set_input 调用 | 54（27 次路径运行，每次 2 个 tensor） |
| copy 臂累计交接 payload | 12,644,352 B |

最终输出以冻结 once-quantized 主机 A 为共同参考；producer 中间输出也与原独立量化 producer reference 比较。shared/copy/mono 均保留相同的回绕结果，因此不能称为“修复后精度验证”。

每次 producer/joined 的软件计数与各自冻结完整 TIR census 一致；producer 到 CPU tail 执行后，VTA 逻辑 DMA 计数无新增。独立审计还逐 case/repeat 对齐三臂的 LOAD/STORE 请求数和各内存类型 payload，均相同。enqueue 时间不是计数，未要求相等。

| 层 | LOAD 请求数 / payload B | STORE 请求数 / payload B | copy 每次边界 payload B | shared 每次边界 copy payload B |
|---|---:|---:|---:|---:|
| layer2 | 280 / 1,846,784 | 24 / 301,056 | 802,816 | 0 |
| layer3 | 268 / 1,736,192 | 12 / 150,528 | 401,408 | 0 |
| layer4 | 262 / 3,913,216 | 6 / 75,264 | 200,704 | 0 |

两列 copy payload 来自显式 API 账本：copy 执行两次 CPU→CPU set_input，shared 不执行边界复制。CPU→CPU memcpy 不由 VTA 软件 profiler 完整计数，不能把其 profiler 的零差值伪称为拷贝流量实测。输出下载在 profiler 快照之后单列排除，shared 的“0”不包含这些诊断读回，也不表示物理 DDR traffic 为零。

## 共享内存实现资格化

开发板为 `192.168.1.247:9091`，boot `a5220e22-f562-40be-bbe9-058b6fbeed50`；u-dma-buf 为 192 MiB，基址 `0x68500000`。bitstream、libvta 和 libtvm_runtime 与 E2 冻结哈希一致，未替换。CPU 为 1 个 TVM 线程；各层前后 PS 温度采样范围 32.39–34.69°C，不等于连续温度控制或多 boot 统计。

独立诊断模块 `vta_e3_buffer_probe.cc` 查询 CPU view 的虚拟地址及驱动物理地址；检查连续、零 offset、256 B 对齐、完整范围位于 u-dma-buf 内、两张量不重叠，并在该层全部采样后再次核对范围不变。

| 层 | tensor 0 物理范围（左闭右开） | tensor 1 物理范围（左闭右开） |
|---|---|---|
| layer2 | `[0x68880c00, 0x688e2c00)` | `[0x688e2c00, 0x68944c00)` |
| layer3 | `[0x68849800, 0x6887a800)` | `[0x6887a800, 0x688ab800)` |
| layer4 | `[0x68cc7000, 0x68cdf800)` | `[0x68cdf800, 0x68cf8000)` |

各层在独立 RPC child 中执行，层间不同时存活；不要求不同 child 的地址范围彼此不重叠。保留 ext_dev owner 到 producer/consumer 完成，既有 shared CPU view 保留视图容器引用，producer 后显式调用 host-readable 同步。此为当前调用序列的功能/范围证据，**不是 cache/coherence 端口结构归因、任意重叠一致性或 E6-R 全域覆盖**。

RPC 的 NDArray 参数会以 DLTensor 传到服务端，现有 typed NDArray 视图函数无法直接接收。诊断模块仅做参数包装并调用既有 runtime 函数，同时提供只读范围查询；没有替换 runtime/驱动，也没有更改正式 pipeline。原 ext_dev allocation owner 的存活由本实验显式保持，不将此适配模块当成通用 buffer 所有权实现。

## 对第一创新点的结论

1. 这三对端点的 **VTA DMA 不因 tail 的归属改变**，该结论现已在共同数值参考下上板复核，不再仅依靠各 stage 自参考。
2. 改变的是边界 tensor 数、表示、转换及跨 Executor 交接，而非必然改变卷积 tile 或 VTA DMA。共享内存感知成本必须将这些项与 VTA LOAD/STORE 分开，避免重复收费。
3. 旧 float32 producer 可复用，但新的 CPU tail 不是旧无窄化 tail；其 service 成本必须重新测量，不能直接回填旧 CPU profile、M1 排名或宣称吞吐改善。
4. 共同数值参考提供了公平对照的基础，不代表整个切图域已数值等价或精度合格；真实流水激活、block-aligned 负对照、旧 Experiment C 归因及正式四臂性能仍待完成。

下一步优先补真实激活与负对照，并冻结最终数值政策；然后对限定、合格路径做无 profiler 的 shared/copy 配对 service 测量，再决定是否影响 C1 排名。主机去屏障 6→5 primitive 的结果不能外推到 VTA，不自动启动额外 tile 搜索。

## 归档与失败透明性

- 正式结果：[run4 原始样本](e3_board_tail_run4/summary.json)、[独立审计](e3_board_tail_run4/audited_summary.json)、[事前协议与哈希](e3_board_tail_run4/preregistered.json)。每次输出 NPZ、CPU binary/graph/params、范围及 profiler 均保留。
- run1：诊断模块缺少 RPC device 常量头文件，交叉编译失败，未采样。
- run2：RPC DLTensor→typed NDArray 参数不兼容，未采样。
- run3：范围查询返回 runtime.String，旧 RPC 不支持该对象返回类型；改为 std::string 后重跑，未采样。
- 三次接口失败日志分别保留在各 run 的 `host.log`，不计入成功样本；正式 run4 完整执行，退出码 0。
- [板端 RPC 日志](e3_board_tail_run4/board_rpc.log)和[收尾状态](e3_board_tail_run4/board_after.txt)已归档。只停止本轮启动的 PID 3928；9091 不再监听，FPGA 和 u-dma-buf 保持可用。

运行脚本为 `qualify_vta_cpu_tail_handoff.py`，需先按现有脚本启动独占 HPC RPC，再 source 现有 AArch64 SDK。输出目录必须不存在；用 `summarize_vta_cpu_tail_handoff.py --run <目录>` 独立审计。新增物理范围测试覆盖相邻、重叠、越界及虚拟/物理未对齐，6 项通过。
