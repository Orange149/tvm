# G0 第二阶段起步：真实 stage 配对 harness 资格化

2026-09-08；boot `a5220e22-f562-40be-bbe9-058b6fbeed50`。这是一个真实 pair 的双方向工具资格化，**不是完整 9-pair discovery、五 boot confirmation 或流水策略实验**。

## 先冻结什么，再测什么

选取冻结 A/rank01 的 CPU stage0（unit00–02，线程 4，CPU affinity 0/1/2/3）与 VTA stage1（unit03–17，VTA host 固定 CPU3）。选择首个 upstream CPU 与首个 VTA 是为了验证工具，不根据并发 slowdown 选择。所有 graph/library/params/runtime/bitstream 哈希与前轮冻结产物一致，没有 AutoTVM 或重编译 stage。

两个方向的预定偏移取自前轮 A 的自然 encounter active-age 中位数，分别四舍五入为 `24.78 ms` 与 `0.38 ms`。参数、顺序、旧 trace 哈希与工具哈希在测量前写入 [preregistered.json](pair_a5220e22_qualification1/preregistered.json)，SHA-256 为 `4424fdf881db7db5321aa5ac050b1f9ba2eb6e6c85b2e6ec4d043b1e8c78da06`。

每个方向先各策略 5 次 warmup，再做 10 个 `allow/wait/wait/allow` ABBA block，即 50 个 pair 样本。两方向共 100 个 pair 样本、200 次被测 stage invocation，另外各执行一次完整串行 reference 来准备冻结输入并核对最终输出。所有样本均保留，无按性能挑选或删点。

## 同步机制：为什么没有沿用旧进程入口 barrier

新增独立 `vta_stage_pair_runner.cc`，复用原有 LoadStage、input routing、RunStage，不修改生产流水线调度。`RunStage` 新增默认关闭的 optional hooks：binding 后 `before_run`、graph-run 开始 `started`、结束 `finished`。原有调用不传 hooks，维持原语义。

计划中的 ready/start/done token 在本次资格化里用**同进程两个常驻 worker 的 condition-variable 状态**实现：Python 只在样本开始前冻结方向、offset、ABBA 日程与命令；native coordinator 发出 ready/release/start/done 事件。这样不把逐样本 SSH、文件轮询或跨进程时钟换算误差混入 makespan。下一阶段扩展 pair 时沿用这一同步语义，不要求退回文件 token。

1. 用同一固定输入串行执行完整 A，保留各 stage 正确输入及参考输出。
2. CPU/VTA worker 在两侧都空闲时分别执行 set_input，然后在 binding 后等待。
3. 两边 binding 完成后释放 active；以实际 run-start 时间为基准等待冻结 offset，再将 pending 标记为 eligible。
4. `allow` 立即释放 pending；`wait` 等 active 的 graph-run done 后才释放 pending。这是不抢占的 wait，已经 active 的一侧不暂停。
5. 两侧 graph-run 都结束之后，才允许 get_output/CPU materialization 和输出校验，防止这些拷贝污染对方的 graph-run。全部 JSON 样本先缓存在内存，整组结束后落盘。

此 harness 在样本内恰好只有一个 VTA，因此不存在第二个 VTA owner；hook 位于实际 VTA invocation 之前，但这里没有生产流水线的多 island mutex 竞争。没有 managed slot，不能虚构 slot-ready 或据此资格化 K2 handoff。双方提前绑定后延迟 eligible 的方式，是受控的 gate-level ready-state，不是自然流水全过程重放；自然流水里的 adapter/cache/binding 与 slot 压力尚未复现。

记录 binding、set 时间、release、run start/end、eligible、实际 affinity 和是否仍有 active。主终点为 `max(active_end,pending_end)-active_start`，即两项 graph-run 全部完成时间；不包括 gate 前 binding 或两项结束后的 get_output，**不是单帧流水时延，也不能取倒数称流水 FPS**。

## 结果

| 方向 | 预定 offset / ms | 实际 active-age 中位 / ms | allow makespan 中位 / ms | wait makespan 中位 / ms | ABBA block 配对 wait 退化中位 |
|---|---:|---:|---:|---:|---:|
| CPU0 active → VTA1 ready | 24.78 | 24.863 | 93.943 | 120.757 | +28.52% |
| VTA1 active → CPU0 ready | 0.38 | 0.442 | 70.887 | 120.570 | +66.18% |

最后一列先在每个 ABBA block 内分别求 allow/wait 两次测量均值，再计算 `wait/allow-1`，最后跨 10 个 block 取中位；它不是前两列中位数直接相除。两个方向分别有 10/10 block 为 wait 更慢。帧/重复样本并非独立 boot，不能把这些 block 当成跨 boot 显著性证据。

两个方向的计分样本均有 40/40 在 eligible 时 active 仍未结束；release→start P95 分别约 `0.03255/0.02607 ms`。这只资格化 token 唤醒/发令延迟，不是新旧 runner policy-off 开销对照。时间戳顺序、wait 不早于 active done、ABBA 顺序和 makespan 公式校验全部通过。

每次被测 stage 的完整 RawOutputsJSON 都与串行 reference 一致，两次完整 reference 的最终 FNV 均为冻结 A 的 `00b1f37535ac3647`。重复输入不能替代唯一 frame/generation 异常压力测试。

补充机制指标：CPU0 在两个方向的 allow 下 run 中位约 `61.54/69.15 ms`，wait 下约 `52.17/52.03 ms`；VTA run 对应 allow 约 `69.05/69.85 ms`，wait 约 `68.53/68.49 ms`。**并发时某个 stage 变慢，但允许重叠的 pair 总完成时间仍更短。** 因此不能依据单 stage slowdown 就生成 protect 标签。这里的 wait 下 run 也不是独立 isolated-service 测量；尚无 compute-only 负对照和 memory-intensity dose response，不能把 CPU slowdown 直接归因于物理 DDR。CPU 的 4 个 worker 与固定 CPU3 的 VTA host 仍可能发生 CPU 调度争用。

## 对第三创新点的影响

- 工具已经能执行 binding 后的 allow/wait 因果对照，不再只看自然 overlap。
- 该真实 pair 的两个方向暂时都支持保留重叠，未发现 protect 候选。但动作仍记 `unclassified`，尚未满足五 boot 的正式 allow 非劣判定。
- 不能据一个 pair 否定全部 C3，也不能为了寻找正例改变本次 offset 后不报告原结果。下一步按 static/isolated 特征冻结完整 CPU/VTA pair 矩阵与自然 active-age 分层，完整报告可达方向，再决定是否存在 protect/allow 异质性。
- 同帧数 ABBA instrumentation-overhead、isolated/sequential 对照、compute-only 负对照、内存强度归因、重复噪声门槛、opportunity ceiling 与五 boot confirmation 均未完成。**G0 尚未通过，G2 控制器和第三创新点都未成立；C1 的融合/编译审计也仍待做。**

## 环境与审计产物

- FPGA/u-dma-buf/CPU 频率在运行前通过复核，结束后 boot、FPGA `operating`、192 MiB 缓冲不变，无残留 runner/RPC。
- 每个方向前后记录 IIO PS/PL raw 温度，见产物目录的 `*_temp_before/after.txt`。仍不是逐 block 温度 gate，不称 confirmation 环境控制已完成。
- [完整汇总与 ABBA block 数据](pair_a5220e22_qualification1/summary.json)、[CPU active 原始样本](pair_a5220e22_qualification1/active0_ready1.jsonl)、[VTA active 原始样本](pair_a5220e22_qualification1/active1_ready0.jsonl)。目录同时保存实际命令、stdout、前置环境与 stage/runtime 哈希。
- 新 runner SHA-256：`e0e36ad3bbf7a9f0cc9680673aec9752e73331bfba060ab68e23d53663055f01`；编译时 pair 源码：`1c4b0bcc85d11e5704a312850f959c8ba872740324b868d5b41699ae5c00648f`；复用 pipeline 源码：`84528e647c2533287a4015b3d5ca6ff0bf50a61100edc5ba3237ad827922d27d`。
- 构建沿用前阶段 SDK/编译参数，将源文件换成 `vta/apps/native_deploy/vta_stage_pair_runner.cc`、输出换成 `/tmp/vta_stage_pair_runner_g0`。独立部署于 `/media/sd-mmcblk1p2/vta_stage_pair_runner_g0`，旧 runner 未覆盖。
- 主机侧 G0 回归测试扩至 9 项，覆盖配对 makespan、wait 时序、错误输出/ABBA 顺序与前阶段 interval/slot 审计；交叉编译与 Python 语法检查通过。
