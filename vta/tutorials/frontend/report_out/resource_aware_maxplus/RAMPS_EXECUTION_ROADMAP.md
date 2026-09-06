# CPU-VTA Pipeline V1 Execution Roadmap

更新日期：2026-09-06

主设计见 [HARDWARE_LAYER_DESIGN_REVIEW.md](HARDWARE_LAYER_DESIGN_REVIEW.md)。本文只记录执行顺序、
产物和 gate。理论全集 [RAMPS_SYSTEM_MODEL.md](RAMPS_SYSTEM_MODEL.md) 仅作为 V2 参考，不是 V1
实现清单。

## 1. V1 范围

```text
自由变量：CPU/VTA 切图、stage cuts、每个 CPU stage 的 TVM threads（1..4）
固定变量：VTA schedule/tile、量化、fusion、FIFO、runtime policy、bitstream
成本项：segment CPU/VTA service、单物理 VTA 串行约束、四核 core-work 下界、复合 boundary、共享 DDR 下界
求解器：PartitionTuner grouped cost + Tarnawski/DNNPipe Pipeline objective 的 k-best DP
验证：预测 Top-K 原生编译、correctness 和 Pipeline 上板
```

已完成的 V1 baseline 不实现 tile 联合搜索、pairwise DDR contention surface、FIFO/Max-Plus 或
overlap 拟合；它只保留共享 DDR transaction ownership 和聚合 resource-demand 下界。P7 在继续
固定 tile 的前提下补做物理组件 profile，不回到 212/80-case 全硬件画像路线。

### 1.1 论文主线

V1 只保留两项主贡献，不把标准 DP、TVM 已有 zero-copy API 或双缓冲本身写成创新：

1. **共享内存感知的静态切图评估与预算化 Top-K 筛选。** 对 compiler-valid CPU/VTA segment、
   每个 CPU stage 的 TVM threads、单 VTA 串行负载、CPU core demand、边界 tensor 的复制、
   layout/量化转换以及共享 DDR demand 统一计费，再由 k-best DP 从大空间中输出少量候选。
   共享物理 DDR 不等于通信免费；它只省去独立内存之间的必然迁移，框架 materialization、表示转换、
   VTA LOAD/STORE 和并发带宽竞争仍可能产生代价。
2. **面向固定 CPU-FPGA 流水线的共享 slot zero-copy runtime。** RIMMS 处理运行时才确定的动态
   PE/location；本项目利用运行前已知的设备映射和共享 DDR，由切图 manifest 自动生成有界 slot，
   为同一地址建立 CPU/VTA tensor view，queue 只传 slot token，并以 generation、owner 和
   completion 管理双缓冲生命周期。TVM Pipeline Executor 当前源码可确认采用 owning/deep-copy
   forwarding；本文不推断其作者为何选择该语义，也不把 GraphExecutor 已有的 zero-copy API
   重新包装成贡献。

两项贡献分别验收。第一项必须在不读取候选吞吐标签拟合参数的前提下报告冻结集的 Top-K/regret，
并明确区分“筛出高性能区域”和“准确细排”；当前 P7D 未通过后者的 gate，不再写成绝对性能模型。
第二项必须相对最快正确的普通复制，通过逐帧 correctness、边界实际复制字节、单帧 latency、
Pipeline II/FPS 和多 boot 稳定性验证。若只降低复制字节和单帧 latency、但没有改善 II/FPS，论文
只能声明内存物化与时延优化，不能声明稳态吞吐提升。第二网络迁移性是增强证据，不在 V1 收口前
强制扩大实现范围。

两条主线通过同一份切图 manifest 连接：静态规划器同时给出 stage、边界 tensor contract 和
zero-copy eligibility，运行时据此分配 slot 并执行；组件 profile 得到的复制、转换、DDR demand 和
slot-wait 再反馈给成本模型。论文应将其表述为 planner/runtime 协同，而不是两个互不相关的技巧。

板端 4 个 CPU 核按同构核心处理，不引入 big.LITTLE 变量。`threads` 是每个 CPU stage 独立传给
TVM thread-pool 的并行度参数，不是从 4 个物理核心中切出的互斥配额。因此只要求每个 stage 的
`threads in {1,2,3,4}`，不约束所有 CPU stage 的 threads 总和。为保持测量可复现，每个 stage
独立使用 `[0,threads)` mask；不同 stage 的 mask 可以重叠。搜索另外累加各 stage 的实测
`process CPU core-ms`。对每个前缀宽度 `k=1..4`，所有 `threads<=k` 的 CPU stage 都只能在前
`k` 个核心上运行，因此容量下界取：

```math
D_{core}=\max\left(\frac{W_{host,total}}{4},
\max_{k=1..4}\frac{\sum_{j:t_j\le k}W_j}{k}\right).
```

当前 runtime affinity mode 允许 worker 在各自 `[0,threads)` mask 内迁移，不能假设每个 stage
把 `core-ms` 永久均分到具体核心。cache、调度和并发效率损失仍属于后续修正，也不能用
`sum(threads)<=4` 代替。

当前板端运行端点为：

```text
SSH: root@192.168.1.247
TVM RPC: 192.168.1.247:9090
```

新命令必须显式传入 host/port 并把端点写入 run manifest。IP 是运行连接信息，不替代由板卡、
bitstream、时钟、compiler 和 runtime 构成的稳定硬件指纹；历史产物保留采集时的原始 IP。

板端部署不得重新探索，统一使用现有 runbook：

```text
主说明：apps/vta_rpc/README.md
目录约定：apps/vta_rpc/axu5evb_board_layout.md
V1 bitstream/runtime：HPC/coherent
启动入口：/mnt/sd/tvm_deploy/start_axu5evb_hpc_rpc.sh 9090
SSH 兼容参数：HostKeyAlgorithms=+ssh-rsa, PubkeyAcceptedAlgorithms=+ssh-rsa
```

只有 runbook 与当前板端出现明确不一致时才记录差异并修复部署；不得从通用目录扫描重新推导 native
部署方式。已有 HP/HPC 对照已经选择 HPC/coherent 为主线，V1 不再次比较 bitstream。

## 2. 阶段控制规则

本路线图严格一次只执行一个阶段：

1. 任何时刻只能有一个阶段标记为 `current`，其余阶段保持 `pending` 或 `conditional`。
2. 当前阶段内可以完成该阶段所需的代码、静态检查和明确列出的实验，但不得提前执行下一阶段任务。
3. 当前阶段结束后先提交产物、测试结果、失败项、实际耗时和下一阶段建议，然后停止。
4. 只有用户审阅并明确要求继续，才把当前阶段标记为 `completed`，并把下一阶段改为 `current`。
5. 需要上板的阶段必须再次报告命令、case 数和预计耗时；用户确认前不启动性能实验。
6. Gate 未通过时留在当前阶段修正，不通过扩大 profile、增加模型复杂度或跳到后续阶段规避问题。

当前 `V1-P1`--`V1-P5B` 已完成单 session 验证。Iteration 2 在不读取候选吞吐的条件下，从
972528 个合法配置产生自然 Top-20；2026-09-04 已在新端点 `.247` 编译 4 种唯一拓扑并运行全部
20 个线程配置，20/20 通过完整输出正确性检查。静态筛选进入了高性能区域，但 Top-20 内部细排
相关性较弱，历史 `+34 ms` 也不能直接迁移为物理常量。P7A、P7B-1 和 P7B-2 已完成；P7C
单 boot shared-memory qualification 已通过并停在 review。尚未把任何新量写入正式公式。

## 3. 预算与指标

```text
B_profile: 去重局部 signature 的构建、上板数量和 wall-clock
B_build:   DP shortlist 的 native compile/reference 数量和 wall-clock
B_search:  完整 Pipeline candidate 的上板数量和 wall-clock
```

通用指标：

```text
throughput regret@1/3/5/10/20
evaluations to 95% measured-pool oracle
Top-K recall
profile + build + search 总 wall-clock
build/correctness failure rate
```

这里必须区分两种 `oracle`：P3 的 enumeration oracle 是同一静态成本函数在 972528 个配置上的精确
最优，只验证 DP 实现；P4/P5 的 measured-pool oracle 是冻结候选池里实测 FPS 最高的候选，只评价
该池内的排序质量，不是完整空间的全局最优。`regret@K = 1 - FPS_best_in_topK/FPS_pool_oracle`。
`evaluations to 95%` 是实验结束后的回顾性指标，真实运行时 oracle 未知，不能作为提前停止条件。

P4R 前瞻 pilot 的单一主指标固定为 `throughput regret@3`，不再从五个 K 中事后选择最有利结果；
`regret@1`、thread 配对结果和完整成本为次要诊断。所有通过 correctness gate 的冻结候选都要按固定
顺序测量，失败或超时照常消耗预算。

## 4. 执行阶段

### V1-P0：冻结接口

状态：`completed`

任务：

1. 冻结 ResNet18 compute-unit 顺序和静态 tensor-contract cut 规则；native lowering 合法性留给 P1。
2. 冻结 VTA lowering 配置、compiler/runtime/bitstream 指纹；实际 TIR/module hash 留给 P1。
3. 对 `root@192.168.1.234` 和 RPC `192.168.1.234:9090` 执行 SSH/RPC preflight，并记录板端身份。
4. 定义每个 CPU stage 独立的 TVM `threads=1..4`；明确 threads 总和不是约束。
5. 定义单 VTA mutex、boundary 和共享 DDR transaction 的唯一计费位置。
6. 修正指标名称，区分 cycle regret 与 throughput regret。

产物：

```text
v1_protocol.json
v1_hardware_fingerprint.json
v1_unit_and_boundary_schema.json
v1_preflight_evidence.json
v1_execution_state.json
```

Gate：自动采集的 SSH/RPC evidence 指向预期板端且组件 hash 与当前本地构建一致；同一候选必须
产生确定的 stage、threads 和允许重叠的 CPU mask。P0 只冻结 lowering 配置，segment TIR/module hash 和
compiler-valid cuts 是 P1 Gate。通过后提交 P0 产物并等待用户确认，不自动进入 P1。

2026-09-02 执行快照：

```text
静态接口：完成
ResNet18 原子 compute units：21
静态相邻 cut contracts：20/20 compatible；native lowering 0/20 checked（P1 执行）
CPU：4 个同构核；每个 CPU stage 独立选择 threads in {1,2,3,4}；不对 threads 求和
Pipeline affinity：每个 CPU stage 独立使用 [0,threads)，mask 可重叠；TVM affinity mode=-3
VTA island 上限：3；单物理 VTA mutex
固定 runtime：queue_depth=2，poll_sleep_ns=1000
P0 静态测试：7 passed；native runner 交叉编译通过

SSH：connected；板端 vta_linux，Linux 5.4.0，aarch64，4 CPU cores
RPC：connected；runtime.NumThreads=4；config_threadpool=true
VTA profiler：clear/status/events 均存在
bitstream：vta_hpc.bit；SHA256 7bf1ac95b1182c670cd25241111f33725b4ea37ed6d66f2df37b0b2e7df528d6
runtime：自动采集发现旧 libtvm_runtime.so 漂移，按 HPC runbook 同步并重启后，
         tvm_rpc/libtvm_runtime.so/libvta.so 与仓库 build_axu_aarch64 逐项同 hash
CPU policy：4 cores，userspace governor，1066666 kHz
FPGA manager：operating
u-dma-buf：udmabuf0=201326592 bytes；/dev/udmabuf0 可用
P0 Gate：通过；用户已确认并进入 P1
```

已生成并显式纳入版本跟踪的 P0 产物：

```text
v1_preflight_evidence.json
  artifact SHA256 cdf31e884728b6a262883ee918be0a40dcf988ba41b40d7b208462961564d427
v1_unit_and_boundary_schema.json
  artifact SHA256 4ef28d59d36ad33d32f4b0b5fce35dd4cb8dd63be7de7cdacdcd9f2e303fa237
v1_hardware_fingerprint.json
  artifact SHA256 957d2e836a69097d431c5b47ffb6247e117598c173bec61a943c69f07507606a
v1_protocol.json
  artifact SHA256 654c038a18d4ffd03c9d0a3c4b4d70ce05ca676b1e4b982b01cccc89cc25d408
```

硬件指纹引用独立 preflight evidence，包含板端实例代理、HPC bitstream、RPC/runtime 三组件、CPU
frequency/governor、FPGA 状态和本地 compiler/source hash。不可变实验定义位于 `v1_protocol.json`；
可变阶段状态位于 `v1_execution_state.json`，避免阶段推进改变 protocol hash。

### V1-P1：生成局部成本 manifest

状态：`completed`

任务：

1. 从真实 lowering 导出 CPU/VTA segment、boundary 和 DDR access signatures。
2. 按完整 signature 去重，不按 DNN 层名重复 profile。
3. 审计历史 archive 的 fingerprint、correctness 和计时边界。
4. 标记 `reusable`、`missing`、`stale`，只补 missing。
5. 预先冻结完整 stage grouped holdout。

产物：

```text
v1_profile_manifest.json
v1_archive_reuse_audit.json
v1_grouped_holdout.json
```

Gate：每个 DP 可达 transition 都有 segment-level lowering manifest，能够映射到成本表或被明确
标记为 unsupported；每笔 DDR transaction 只有一个 owner。通过后等待用户确认，不自动进入 P2。

2026-09-02 执行结果：

```text
合法 canonical partitions：4623（相邻同设备 stage 已合并/拒绝）
去重 segments：172 = 85 CPU + 87 VTA
CPU thread execution signatures：340
异构 boundaries：27；只含 CPU->VTA 和 VTA->CPU
代表编译：8 个 unique segment cache key 全部通过；其余 164 个要求 shortlist 后再编译
历史复用：无性能数据可直接复用；旧计时均因指纹、计时边界或 correctness 过期
成本形式：additive_atomic_logical_ops_v1；不拟合原先不可辨识的 30 个原子参数
```

产物 artifact SHA256：

```text
v1_profile_manifest.json     6abf9de3b35475e993dbdd1edc1ca73c19743b1b3ac32de124ddab6198b955e5
v1_archive_reuse_audit.json  6fcaa968979381e74030a9bf511391c75e1e3f5ed140c8a44086dee3bea3b758
v1_grouped_holdout.json      cbe6f5a9e677cf3a6e2334e87b6ba7c6898c0e93c0da852616bb399577922cfb
```

### V1-P2：最小局部 Profile

状态：`completed`

任务：

1. CPU signature 使用 `threads=1..4`，记录 affinity、wall 和 process CPU time。
2. VTA signature 只使用冻结 schedule/tile，记录互斥 device-run service。
3. Boundary 使用真实 adapter/set/get 路径并检查 accounting id；submit/sync 归属 VTA service。
4. 测量少量持续 DDR bandwidth，并运行 CPU-only、VTA-only、concurrent matched control。
5. 第一轮只运行一个 session；先验证 correctness、segment 可加性和资源 ownership。
6. grouped holdout 未通过时修正成本所有权，不立即增加 tile 或 contention surface。

产物：

```text
v1_local_cost_table.json
v1_profile_cost_ledger.json
v1_holdout_report.md
```

Gate：correctness 全部通过；每 stage TVM threads 配置真实生效；holdout 误差足以支持候选排序；并发 matched
control 的残差被记录为 V2 判据，不在 V1 中临时拟合。通过后等待用户确认，不自动进入 P3。

2026-09-02 执行结果：

```text
冻结性能 case：19 = 7 native stage modes + 12 DDR cases
采样：84 native stage frames；96 DDR samples；前 2 个 stage warmup 不计分
成本：262.90 s；35 条成功命令、2 次 pre-sample 失败尝试、1 条旧计时异常已显式标记；
      candidate throughput evaluations=0
correctness：独立 MXNet top1=282；仅 1/6 package 与其 top1 精确相同，其余 split 输出为 285；
            所有输出 hash 稳定且落在预注册 cat 等价类。这是语义/确定性 gate，不是数值等价证明
runtime：VTA run/sync/load/store profiler 非零，driver timeout=0
CPU：内存 worker 的 threads=1..4 affinity 与预期一致，TVM CPU stage 的 t4 均快于 t1；
     尚未逐线程观测每个 TVM worker 的实际驻留 CPU
```

冻结的最小模型为无截距、非负的 Tarnawski 式加性服务模型：

```text
CPU ms/logical-GOP：t1=225.166，t2=114.763，t3=95.856，t4=67.244
VTA ms/logical-GOP：27.551
CPU->VTA boundary：3.790 ms/MiB
VTA->CPU boundary：3.536 ms/MiB
DDR read GB/s：t1=2.163，t2=4.167，t3=5.879，t4=7.336
DDR write GB/s：t1=4.100，t2=7.749，t3=7.943，t4=7.757
DDR copy GB/s：t1=4.275，t2=6.293，t3=6.867，t4=7.529
```

grouped holdout `three_stage_b` 未参与拟合：预测 cycle `227.916 ms`，实测 Pipeline
`222.580 ms`，残差 `-2.34%`；预测与实测瓶颈均为 CPU stage0。但它使用旧的
`serial_reuse_or_pipeline_contiguous_disjoint_v1` affinity，只能作为历史策略诊断，不能验证当前
overlapping-mask 下的 CPU contention。串行 fit 的实际 masks 与当前策略一致，9 个 P2 gate 通过。

P2 还从串行 `process_cpu_ms` 拟合了共享四核所需的工作量，而不是把 threads 当核心配额：

```text
CPU core-ms/logical-GOP：t1=221.207，t2=218.671，t3=226.789，t4=234.191
VTA host core-ms/logical-GOP：6.435
boundary host core-ms/MiB：CPU->VTA=3.785，VTA->CPU=3.564
```

当前限制：CPU 局部 fit MAPE 为 `20.66%--37.78%`，明显弱于 VTA/boundary；P2 不继续扩大
profile，先由 P4 的 Top-K 排序结果判断这种粗模型是否够用。P3 shortlist 仍必须逐段 native compile，
并通过独立数值 reference；不能用 cat 类别等价替代 tensor correctness。

产物 artifact SHA256：

```text
v1_p2_frozen_plan.json       43243a0e0756ebd1ffa4adde0a84d8eb01cdb81806a601f31623e4bc524c51e1
v1_p2_reference.json         db3392fd71b90e976c899209266f5fcc00251938a9256bd68da41bb3565ea198
v1_local_cost_table.json     f33f81add7b47605ea988dbf14f1eda0b5617a398ac5360a6e7db5dd191d00d6
v1_profile_cost_ledger.json  98229cab26bfec58b988508bacf2898575dbead73dd01d429fca32adccec597c
```

### V1-P3：实现 DP 求解器

状态：`completed`

任务：

1. 状态使用 `prefix + VTA island count + last device`；threads 是转移变量，不是静态核心配额。
2. 标签保存 `max CPU stage`、`VTA + boundary sum`、`total host core-ms/4` 和共享 `DDR demand`。
3. 实现精确 Top-K 所需的 k-best predecessor 保留和单调下界停止规则。
4. 输出每项成本的 provenance，不允许候选实测吞吐进入 score。
5. 对 ResNet18 全部执行配置做流式枚举并逐项验证 Top-K。

产物：

```text
v1_dp_ranked_candidates.json
v1_dp_vs_enumeration_report.json
```

Gate：DP 与完整枚举在相同成本表上的最优值和 Top-K 集合一致；重复候选为零。通过后等待用户
确认，不自动进入 P4。

2026-09-02 执行结果：

```text
静态切图拓扑：4623
加入每 stage 独立 TVM threads 后的执行配置：972528
CPU thread 约束：每个 CPU stage 独立取 1..4；threads 总和不构成约束；mask 可重叠
Top-20：基于单调 partial objective 的 exact label-setting；Top-K 模式不做不安全 Pareto 删除
验证 oracle：主机内流式遍历全部 972528 个执行配置并保留精确 Top-20；无编译、无上板、无候选实测吞吐
```

DP 的 Top-20 candidate ID 和预测周期与完整枚举逐项完全一致，最大差值 `0 ms`，无重复候选。
单最优预测为：

```text
CPU[unit 00..02] t3 -> VTA[03..17] -> CPU[18..20] t2

max CPU stage with owned boundary：67.535 ms
single VTA service：67.977 ms
CPU-owned boundary total：0.407 ms
VTA-mutex-owned boundary total：2.830 ms
direct CPU-VTA boundary total：3.238 ms
single VTA + mutex-owned boundaries：70.807 ms
host core work / 4：279.377 / 4 = 69.844 ms
shared DDR physical-demand bound：9.532 ms
原始资源下界 II：70.807 ms；下界倒数：14.123（仅作排序分数，不是最终预测 FPS）
```

Top-20 中 11 个包含 1 个 VTA island，9 个包含 2 个；不再因遗漏 CPU 总工作量而偏向 3 islands。
预测 II 为 `70.807--72.045 ms`，单 VTA 路径与四核容量下界共同决定排名。直接 boundary service
为 `3.238--4.639 ms`，共享 DDR 下界为 `9.510--9.628 ms`，尚未成为瓶颈。

共享 DDR demand 的构成为：CPU stage 输入/输出的方向化 read/write 下界，以及由 P2 VTA profiler
拟合的 physical LOAD/STORE bytes。direct boundary bytes 已体现在相邻 CPU/VTA stage 流量中，
不能再把 boundary wall time作为 DDR service 加一次。P2 两个去重 VTA segment
拟合得到 LOAD `4.612 MB/logical-GOP`、STORE `0.486 MB/logical-GOP`；冻结 grouped holdout 的
total-traffic APE 为 `17.42%`。device DMA 速率暂用既有 Stage 4a-Q 的 LOAD `1.414 GB/s`、STORE
`0.954 GB/s`，该实验通过 `R^2=0.9973` 辨识 gate，但其 busy-poll policy 与 V1 runtime 不同，
因此只作为 provisional V1 component rate，P4/P5 shortlist 必须在当前 runtime 下复核。

VTA LOAD/STORE 已包含在 VTA `run_ms` 中，只作为共享 DDR 的平行 resource constraint，不再加到
VTA 路径。boundary 的 host adapter/set-get service 与 lowered stage DMA accounting id 也已拆开，
stage DMA additional service 固定为 0，防止重复计费。

boundary 进一步按 native runner 锁范围拆分：CPU producer `get`/CPU consumer `set` 加入对应 CPU
stage；VTA `set/get` 才加入全局 VTA mutex。不同资源之间通过 Pipeline `max(resource load)` overlap，
不再把全部 direct boundary 串行到 VTA。P2 grouped holdout 的 CPU->VTA 与 VTA->CPU boundary
total APE 分别为 `0.84%` 和 `1.66%`。

这里“CPU、VTA、通信处于同一层级”只表示它们都是候选启动间隔的瓶颈约束，不表示三项串行相加。
CPU 侧通信只与所属 CPU stage 相加，VTA 侧通信只与单 VTA mutex 相加，共享 DDR demand 独立作为
资源下界；最终对 CPU 单 stage、单 VTA 路径、四核总 core-work 和共享 DDR 四类负载取 `max`。
VTA `run_ms` 已经包含 LOAD/compute/STORE 的硬件内部 overlap，不能再把同一 DMA 时间加到 VTA
路径。

直接 boundary 消融后，Top-20 与完整模型重合 `0/20`，Top-1 也发生变化。
这说明通信已经显著改变 shortlist；历史 200-candidate 实验还显示加入直接 boundary 后 MAE 从
`17.3617 ms` 降至 `12.8220 ms`（改善 `26.15%`）。历史候选实测吞吐未进入本次 score。

取消错误总线程预算后，4623 个切图拓扑对应 `972528` 个执行配置。加入 core-work 下界后，Top-20
label-setting 扩展 `83061` 个部分路径、生成 `280526` 个 label，在确认 27 个完整候选后停止；
流式遍历全部 972528 个
配置后，Top-20 id 和 score 逐项一致。生产排名不使用 Pareto shortcut，因为等目标路径的候选 id
tie-break 不能由该 dominance 安全保留。Top-20 全部仍需 shortlist native compile/reference，
当前排名不等于可上板清单。

全部 14 个 P3 gate 通过。产物 artifact SHA256：

```text
v1_dp_ranked_candidates.json      360aa854ddbe7f5546e0e688e577bbf3fc3504506405c173b20b0b84197d11a2
v1_dp_vs_enumeration_report.json  c7a52205c3e9c661fcaf4e35b4b2fce467f27ca51bd77509d22046fa167c86ab
```

### V1-P4：回顾性排序验证

状态：`completed`

在已有 measured candidate pool 上比较，分别隔离已有工作的能力与 V1 的新增约束：

```text
B0 random / stratified random
B1 type-fixed Synergy/NEURAghe-like mapping，固定 CPU thread policy
B2 PartitionTuner-like grouped segment cost，优化单帧 latency
B3 Tarnawski/DNNPipe-like Pipeline max load，忽略 CPU-stage contention 和 DDR coupling
B4 V1 single-VTA + per-stage TVM threads + composite boundary + shared-DDR bound
```

Gate：B4 至少在一个预注册低 K 上同时优于 B0-B3；否则不启动新 candidate 上板，也不
通过增加 tile/FIFO/Max-Plus 掩盖成本表问题。通过后等待用户确认，不自动进入 P5。

2026-09-02 执行结果：

```text
历史 measured rows：200
成功映射到当前 21-unit 表示：200
去重执行配置/拓扑：199/199
仍属于当前搜索语法的历史 segment/thread 配置：69
历史 CPU stage threads 总和：7 有 78 条；8 有 109 条；9 有 13 条，仅作诊断
当前 P3 执行配置：972528
segment/thread 语义重合：70 条记录、69 个唯一执行配置
当前 overlapping-affinity/source fingerprint 精确重合：0
```

此前把 threads 总和当作物理核心预算，因而错误拒绝了这 69 个语义配置；该结论已撤销。但反向把
语义相同写成当前 runtime 精确相同也不成立：历史包没有当前 affinity/source fingerprint。
69 个配置只允许做历史 runtime 上的探索性 B0-B4 regret；P5 仍保持禁止启动。

产物：

```text
v1_p4_measured_pool_compatibility.json  5667840c7ddefd4fa5aa6ce10eccda24a45301fccadc5ec3291e954ff31d533f
v1_p4_historical_labels.json            09ed0378c8d2fe07ce10f57449f97a4ccdc360bb76300be457e30195d3312bb4
v1_p4_retrospective_report.json         ebf145a22a332d872c1740ade9af73385fc667e3d0fceeeb8e9bf7e2d4cbb8dd
```

### V1-P4B：冻结基线并做同池回顾性比较

状态：`completed_exploratory_gate_inconclusive`

冻结 B0-B4 的 score、tie-break、K 和去重规则，再只对 69 个语义兼容配置计算历史-runtime
探索性排序；全部 score 冻结后才读取已有 measured throughput 标签计算 regret。该结果不能替代
当前 runtime 的 P5。该阶段不上板，不拟合 DDR contention，不扩展 tile/FIFO/Max-Plus。

实现采用物理隔离的两阶段：P4 candidate-pool JSON 不含任何 measured 字段，历史 throughput
单独封存在 labels JSON；`score` 进程只加载前者并先封存哈希，`evaluate` 才加载后者并按 execution
candidate id 连接。重复配置的两条标签按预注册 median 合并。结果如下：

| 模型 | regret@1 | regret@3 | regret@5 | 到 95% oracle 次数 | Spearman |
|---|---:|---:|---:|---:|---:|
| B2 single-frame | 24.34% | 23.22% | 20.53% | 15 | 0.081 |
| B3 Pipeline max-load | 30.58% | 25.66% | 25.66% | 27 | 0.271 |
| B4 V1 | 9.60% | 7.62% | 7.62% | 12 | 0.482 |

B0 uniform random 的 regret 中位数在 K=1/3/5 为 `21.85%/15.16%/7.90%`，分层随机为
`22.99%/17.17%/10.52%`。所以 B4 在同一个预注册 K=1/3/5 上同时优于 B0、B2 和 B3。
但 B4 达到 95% measured-pool oracle 需要 12 次评价，uniform random 中位数只需 9 次
（P10--P90 为 2--26），因此“用更少上板次数达到近优”这一主张尚未成立；不能只选择性报告
低 K regret。2000 次 uniform random 中有 `62.4%` 不晚于 B4 达到95% oracle；在 K=1/3/5，
随机 regret 不高于 B4 的概率分别为 `15.35%/31.60%/48.30%`，说明 B4 的优势主要集中在非常低的 K。

Top-K recall 给出同样的限制：B4 在 K=1/3/5 的 recall 都是 0，K=10 为 `0.30`，K=20 为
`0.60`。精确历史池 oracle 位于 B4 第 12 名；此前“到 K=20 才命中”的写法只是把离散报告点
`K=10/20` 误当成实际首次命中位置，现已撤销。B4 的低 K regret 较小，是因为它找到接近最优但
不属于实测 Top-K 的候选，不代表已经恢复了真实排序前部。

历史69配置也不是97万合法配置的随机样本，而且只覆盖三种 CPU thread 向量：`(3,4)` 13个、
`(3,1,4)` 44个、`(3,1,1,4)` 13个。首尾 threads 固定为3/4，中间固定为1；所以本结果主要验证
切图排序，不能验证独立 per-stage thread 联合优化。

B1 固定策略的依据是 stem/head 只能放 CPU，其余 VTA-supported units 全部放进单 island，CPU 使用
固定最大 thread 参数，即 `CPU[00] t4 -> VTA[01..19] -> CPU[20] t4`。它不在历史69配置中；代码明确禁止用
“最近候选”替代。因此完整 B0--B4 gate 仍为 false，且历史 affinity/source fingerprint 与当前
runtime 不同，`p5_may_start=false`。B1 只有一个候选，未来只能和 B4 regret@1 做等预算比较。

产物：

```text
v1_p4b_scoring_protocol.json      bfe4f474b2bcef2911caa4ff23e2d2807636ff41c1a61ccbec99fb2cea624ab4
v1_p4b_frozen_scores.json         361599911cd61b25052e41808c60900f57f470e8bbac6fc9974dab93f22f7dc3
v1_p4b_retrospective_report.json  d34b455a0db45c1e6888c5bc36f4dba2d1fff177ad71969161c59a5f6c095514
v1_execution_state.json           efe12064498a95439b2345700d8d3e4e7974852b892c35e06fd503ea6bca63a2
```

### V1-P4R：证据缺口修复方案

状态：`completed`

P4R 不修改 B0--B4 score，也不读取历史 throughput。它重新遍历当前全部 972528 个合法配置，冻结
一个 20-candidate prospective pilot：B4/B2/B3/uniform-random 各取 3 个，加入精确 B1、B4 Top-1
同 topology 的 6 个单变量 thread 对照，再用 3 个不同 topology 的高排名 B4 候选填满预算。B1 与
B2 rank-1 是同一候选，去重后仍正好 20 个、覆盖 10 种 topology；13 个候选需要新 native compile。

主比较固定为四组的 `regret@3`；B1 只与 B4 做单次等预算比较。20 是整个 pilot 的唯一候选上板
上限，不是每个模型各测 20 个。冻结顺序由 seed `20260902` 生成；不能根据中途吞吐改变顺序，
也不能使用未知 oracle 提前停止。该设计只能产生 20-candidate pool 内的 pilot 证据，不能声明完整
97 万空间的 global oracle。

产物：

```text
v1_p4r_prospective_plan.json   f74f949ce50934c566fa04cc188af1d7df03a4cd67ed1ef3a76d21d1d85f4f03
v1_p4r_candidate_manifest.json cea6029c11f51c4ccc345dd6a34462f23258ff956ae889ebd1c20c30c17ceeff
v1_execution_state.json        7eac953342aa4ddea641d4409fc74025851310fc6de12fe643ba64a9fba15869
```

P4R 静态 gate 为 `3 passed`。本阶段没有 compile shortlist、没有连接开发板，也没有产生任何新
候选吞吐。

#### P4R 后整体审查结论

当前 V1 的合理部分是：切图与每 stage TVM threads 被正确视为独立变量；单物理 VTA、CPU
core-work、复合 boundary 和 DDR transaction 没有重复计费；DP Top-K 与相同成本函数的完整枚举
一致；P4B/P4R score 阶段均与 throughput 标签隔离。这些足以进入小规模 prospective pilot。

仍不能提前声称有效的部分是：CPU 局部成本 MAPE 仍为 `20.66%--37.78%`；VTA physical traffic
只由两个 segment 拟合；DDR 只有聚合 demand 下界，没有 concurrent slowdown；历史69配置既不是
随机样本也不覆盖独立 threads；20-candidate pilot 的 measured-pool oracle 也不是全局 oracle。
因此当前最合理的优化不是继续增加 profile 或引入 tile/FIFO/Max-Plus，而是先用 P5A/P5B 判断这些
误差是否真的破坏低预算排序。只有冻结残差明确指向 DDR contention 或 tile 时才启动对应扩展。

P4R 还修复了三项实验设计问题：将多 K 探索改为单一 primary `K=3`；用同 topology 单变量对照
隔离 thread 效果；将 compile/reference readiness 与板端 correctness/performance 分成 P5A/P5B。
旧 P4 审计器也允许在 P4R/P5 后只读重放，避免阶段推进破坏历史证据的可复现性。

### V1-P5A：shortlist compile/reference

状态：`completed; awaiting_user_confirmation; local only`

任务：

1. 为冻结的 20 个候选生成 `scheme_cfg` package，复用 7 个当前指纹 compile，补编译其余 13 个。
2. 审计实际 stage/module hash、VTA lowering、threads、affinity、accounting ids 和 native-only 标记。
3. 生成独立期望张量和 raw-output comparator；稳定 hash 或 ImageNet cat 等价不能替代数值 reference。
4. 生成 compile/reference failure ledger；本阶段不连接开发板、不产生性能结果。

Gate：20 个候选全部具有可审计 package，并具备独立 expected-tensor comparator。允许
unsupported/failure 保留并消耗后续预算；真实 native-vs-reference 必须在 P5B 板端运行且先于性能
计时。完成后停止并等待用户确认。

实际结果：

1. 20/20 个冻结候选均成功生成 AArch64 native package；共引用 88 个 stage、去重后为 33 个
   compile key（15 CPU、18 VTA）。manifest 记录 59 次 cache hit 和 29 次 cache miss。
2. 每个 VTA stage 的二进制均引用原生 `VTAPushGEMMOp` 或 `VTAPushALUOp`；stage module hash、
   thread 参数、overlapping-prefix affinity、输入输出 ABI 和 stage-level accounting id 全部通过审计。
3. runner 现在可分别导出 serial/pipeline 的完整 `[1,1000] float32` logits；独立 MXNet reference
   top1 为 282，comparator 在任何 P5B 输出出现前已冻结。稳定 hash 和 cat 类别等价不再作为充分条件。
4. 首次 20-candidate 完整构建与初审耗时 `626.303 s`；最终缓存复审耗时
   `0.895 s`，二者在成本账中分开。硬链接将 package 逻辑 `3.165 GB` 压到约 `540 MB` 唯一 inode
   数据，并在去重后重新验证 artifact hash。
5. 初审的 5 个失败均来自 tuple producer 的描述名 `out0/out1` 与 consumer 的 `main/residual`
   不同，不是 compile failure。真实 runner ABI 按 ordinal/shape/dtype 传递，因此审计已按 ABI 修正；
   7 个 tuple 边界的语义顺序仍必须由 P5B 完整 tensor correctness 证明。

产物：

```text
v1_p5a_reference.json     8f384ca84076b544037acd021160e69ba2f25060e9944735f3e9501d1c472f8f
v1_p5a_compile_audit.json 9b52bb532eee8a7f65715098f6e66e895240953dd5e68090d0f3319d66512791
v1_p5a_review.json        c8e79a264da8291321bc9463d106fce1eac5b0ef8092906be80df69e91ea0947
v1_execution_state.json   15919c026eedc0505b4822f7e464c4ae8cb670c4b2a5c37f50f1e6566a9b5d26
```

证据边界：P5A 只证明 buildability、native lowering 和接口一致。MXNet float gate 是对量化输出的
语义数值检查，不是 bit-exact reference；P5B 必须同时要求 serial/pipeline tensor 等价和独立
reference gate 通过后才允许计时。P5A 没有连接开发板，也没有生成任何 throughput。

### V1-P5B：前瞻 Pipeline 上板

状态：`completed; iteration2 natural Top-20 single-session board validation complete`

按 P4R 冻结顺序运行候选；每个候选先做 native-vs-reference correctness，通过后才计时，失败和
超时照常计入 20-candidate 总预算。报告主指标
`regret@3`、次要 regret/thread 对照、measured-pool oracle 边界以及 profile/build/search 总成本。
只有 B4 在同预算下优于 B0/B2/B3，且结果不依赖事后修改，才能声称该 pilot 支持资源感知筛选。

Qualification smoke（2026-09-02）：

1. 只运行冻结执行顺序第 1 的线程对照候选（不是模型 rank-1）
   `cpu-00-02_t1__vta-03-17_t1__cpu-18-20_t2`，serial/pipeline 各 1 帧；没有执行其余候选，
   也没有将单帧延迟换算为流水线吞吐。
2. serial 与 pipeline 的 `[1,1000] float32` logits 逐字节一致，SHA256 均为
   `12060d163e13649461b73449380e0ba63a723b0ff04153b65ec6ebb6a7da5922`，top1 均为 285。
3. 相对冻结 MXNet reference（top1 282），cosine=`0.995778`、normalized RMSE=`0.093204`、
   Top-10 overlap=`9/10`，预注册数值 gate 通过；top1 不同，因此不能表述为独立 reference bit-exact。
4. 单帧观察值为 serial `302.282 ms`、pipeline `317.069 ms`。单帧尚未填满流水线，只用于检查
   native 执行和 stage 交接，不能用于 FPS、speedup 或 regret。
5. FPGA 运行前后均为 `operating`，RPC 保持运行。增量部署后 SD 卡只剩 `21 MiB`；完整 P5B 前
   必须先审计并清理已确认无引用的旧部署数据，不能把空间不足造成的失败计作算法结果。

证据：`v1_p5b_quick_smoke.json`（artifact SHA256
`d7027bc352f5d4522a734ffc648fb0aa60d24257b4f4368632bc3565119fed6c`）及
`v1_p5b_quick_smoke/` 下的 manifest、JSONL 和完整 logits。

执行顺序第 1 的线程对照初步 FPS（2026-09-02）：复用同一个 `input.bin` 连续运行 22 帧，
丢弃完成顺序最前面的 2 帧，
按后 20 帧 `stage2_end_ms` 的首尾 span 得到周期 `184.757 ms`、吞吐 `5.413 FPS`；完成间隔中位数
`186.078 ms`，P95 `187.057 ms`。22 帧的输出摘要 hash 均一致且 top1 均为 285。冻结 B4 对该
候选预测周期为 `158.137 ms`，实际周期高 `16.8%`；这说明绝对成本仍有低估，但单候选不能判断
排序质量。证据保存在 `v1_p5b_quick_fps.json`（artifact SHA256
`3e43bb84ec34fc5d298eb6fe36adabea19ebb5400a32b646edfb591b8a687820`）和
`v1_p5b_quick_fps/`。

#### P5B Iteration 1：线程曲线修正与未见候选验证

1. 实测同一 `cpu:00:02 -> vta:03:17 -> cpu:18:20` topology，固定尾段 `t2`，只改变首段
   TVM threads。22 帧中丢弃 2 warmup，得到：`t1=5.413`、`t2=6.237`、`t3=9.312`、
   `t4=10.449 FPS`。这证明 threads 是有效变量，但速度提升明显非线性。
2. `t4` 的首轮正确参数 session 出现一次 `2.44 s` 暂停，整轮拒绝后重跑；不从首轮手工删点。
   另一次 `t4` 运行因复用包被部署工具原地刷新，实际变成 `4,1,4`，按错误配置拒绝。工具已改为
   在临时副本中刷新 reuse package；P5A 审计也新增实际 shell thread 参数检查，被污染的第 8 个包
   已按 `4,1,2` 重建并通过复审。
3. 原 B4 Top-1 是 `t3` 三阶段候选，原始资源分数倒数为 `14.123`，实测 `9.312 FPS`，主要误差来自首段
   CPU：预测约 `67.16 ms`，流水线中位 run 为 `105.15 ms`；VTA 中位 run `69.66 ms` 接近原估计。
4. 只修正一个 exact segment 会让搜索逃逸到相邻未修正 segment，因此该方案被否决。P2 的 CPU
   segment 本来都由同一 `ms/GOP × threads` 斜率生成，Iteration 1 改为用代表段得到每线程倍率，
   一致修正全部 85 个 CPU segment；VTA、boundary 和 DDR 公式不变。修正后 DP Top-20 与
   972528 配置完整枚举仍一致。
5. 修正后的未见 Top-1 为五阶段、两个 VTA island：
   `cpu-00-00_t1__vta-01-09_t1__cpu-10-12_t2__vta-13-17_t1__cpu-18-20_t2`。
   该候选新编译 5 个 stage（3 个新 cache key），native/reference gate 通过；serial/pipeline logits
   逐字节一致，独立 reference top1 也精确匹配 282。
6. 该候选原始资源下界为 `75.526 ms`、倒数为 `13.240`，板端实测 `112.747 ms / 8.869 FPS`，周期低估
   `49.29%`。当前 5-candidate measured-pool oracle 是三阶段 `t4` 的 `10.449 FPS`，因此修正后
   Top-1 的 throughput regret 为 `15.11%`。本轮不能声称“用更少上板次数找到近优方案”。
7. 误差仍集中在 CPU segment：五阶段候选三个 CPU stage 的预测 run 约
   `62.95/50.68/65.29 ms`，实测中位数约 `117.12/115.23/88.82 ms`。这直接否定单一 logical-GOP
   斜率跨 stem、residual transition 和 head 的充分性；当前没有证据把该误差归因于 DDR/FIFO。

Iteration 1 证据：

```text
v1_p5b_iteration1_measurements.json
v1_p5b_iteration1_ranked_candidates.json
v1_p5b_iteration1_dp_report.json
v1_p5b_iteration1_validation_scheme.json
v1_p5b_iteration1_review.json
v1_p5b_iteration1/
```

#### P5B Iteration 2：atomic CPU 成本与前缀核心容量修正

1. 将 ResNet18 的 CPU 路径拆成 21 个相邻 atomic unit，分别测量 `threads=1..4` 的 wall time 和
   process CPU core-ms。额外保留 stem、transition、terminal/head 和旧 fused segment 作为冻结
   grouped holdout，不使用完整候选吞吐拟合成本。
2. 23 个 grouped checks 的 median/P95/max APE 分别为 `0.89%/4.17%/17.96%`，通过预注册的
   `5%/15%/25%` gate。atomic t1 reference 与冻结 MXNet 输出近似一致，top1=`282`、Top-10
   overlap=`10/10`。
3. review 发现旧求解器把所有 host core work 只除以固定 4，忽略所有 stage 都使用
   `[0,threads)` 前缀 mask。现改为：

```math
D_{core}(p)=\max\left(
\frac{W_{host,total}(p)}{4},
\max_{k=1..4}\frac{\sum_{j:t_j\le k}W_j(p)}{k}
\right).
```

   这同时检查四核总容量以及每个嵌套前缀 mask 的容量，且不假设 worker 固定在某一核心。
   `sum(t_j)<=4` 仍不是合法性约束。该下界可随 stage 单调累加；DP Top-20 与 972528 配置完整
   枚举逐项一致。
4. 在不拟合 5 个候选吞吐的前提下，修正模型复现了它们的完整实测顺序和 Top-1。绝对 cycle
   MAPE 仍为 `14.66%`，因此它改善的是当前池的排序解释，不代表已经精确预测板端 FPS。
5. 新的未见验证候选为
   `cpu-00-02_t4__vta-03-17_t1__cpu-18-20_t1`，原始资源下界为 `70.807 ms`，其倒数
   `14.123` 只作为排序分数。
   该数值尚未上板验证。它与当前 measured-pool 最优候选只差尾段 `t1/t2`；模型判断尾段都低于
   VTA 瓶颈，选择 t1 是为了减少 CPU work，而不是宣称 t1 一定带来更高实测吞吐。
6. 部署该候选时内核报告 `/dev/mmcblk1p2` 的 bad block bitmap、bad `extra_isize` 和 delayed
   allocation failure。上传已中止，候选没有运行。部署工具现会在任何远端写入前检查目标设备的
   当前 boot ext4 错误并拒绝继续，避免把存储故障误记为算法失败或损坏更多证据。

Iteration 2 证据：

```text
v1_p5b_iteration2_cpu_measurements.json
v1_p5b_iteration2_ranked_candidates.json
v1_p5b_iteration2_validation_scheme.json
v1_p5b_iteration2_review.json
v1_p5b_iteration2/
```

#### P5B Iteration 2：自然 Top-20 上板结果（2026-09-04）

本轮没有使用 `+34 ms` 重新排序，因为对所有候选加同一个常量不会改变名次。执行对象就是
Iteration 2 在 972528 个合法配置上自然产生的 Top-20。20 个候选只有 4 种唯一计算图拓扑，故每种
拓扑只 native compile 一次，再把各候选的 `threads` 作为 TVM runtime 参数运行。每项执行 22 帧，
丢弃前 2 帧，以后 20 帧最后 stage completion span 计算稳态 FPS。

正确性 gate 为：22/22 serial 与 pipeline 最终 `[1,1000] float32` tensor 逐字节一致，并且每个候选
相对冻结 MXNet reference 均通过 cosine、normalized RMSE 和 Top-10 overlap 门槛。结果为 20/20
通过；失败、超时和事后删点均为 0。

| 实测池内名次 | 原静态名次 | 切图与关键线程配置 | 实测周期 | 实测 FPS |
|---:|---:|---|---:|---:|
| 1 | 9 | CPU00--02 t4 / VTA03--16 / CPU17--20 t3 | 86.872 ms | 11.511 |
| 2 | 12 | CPU00--02 t4 / VTA03--15 / CPU16--20 t1 | 90.280 ms | 11.077 |
| 3 | 14 | 双 VTA island，CPU13--14 t1，CPU20 t2 | 90.565 ms | 11.042 |
| 4 | 7 | CPU00--02 t4 / VTA03--16 / CPU17--20 t1 | 91.525 ms | 10.926 |
| 5 | 1 | CPU00--02 t4 / VTA03--17 / CPU18--20 t1 | 91.850 ms | 10.887 |

这里的 `11.511 FPS` 只是这 20 个已测候选中的 measured-pool oracle，不是 972528 个配置的全局
最优。按冻结静态顺序，`regret@1/3/5=5.419%`，到第 9 次才首次达到该池 oracle 的 95%；
`regret@10/20=0`。静态名次与 Top-20 内部实测名次的 Spearman 仅 `0.155`。因此合理结论是：

1. 筛选有效：20 个候选的实测范围为 `10.220--11.511 FPS`，其中 16 个高于此前另一 session 的
   `10.449 FPS` 结果；同一 rank-2 候选本轮为 `10.324 FPS`，说明跨 session 比较存在约 1.2% 漂移。
2. 细排不足：原始静态周期只跨 `70.807--72.045 ms`，无法表达实测 `86.872--97.852 ms` 的线程与
   contention 差异；同 topology 的 `t1..t4` 实测顺序也不是单调关系。所有候选使用相同首段
   `CPU00--02 t4`，但其并发 stage 中位时间仍为 `91.6--105.3 ms`，说明其他 CPU stage 的线程数
   会通过共享核心/cache/DDR 反向影响该段，不能只查孤立 stage 时间。
3. `+34 ms` 不是已验证硬件常量：原始周期 MAPE 为 `23.523%`；加 34 ms 后降为 `12.859%`，但
   统一偏向高估周期。本轮实际减原始预测的残差均值/中位数只有 `22.054/22.353 ms`。
4. 多 island 不是自然更优：双 VTA island 的最好结果为 `11.042 FPS`，低于单 island 的
   `11.511 FPS`；本轮支持保留 island/boundary 代价，但不足以单独辨识 DDR contention。

这批结果不得反向拟合并改写本轮排名。它们用于确认 P7 的重点应从“继续加统一常量”改为：测量
CPU stage 并发效率、共享 DDR contention，以及按 stage/island/boundary 次数变化的 runtime 固定
成本。由于目前只有一个 boot/session，前三名的稳定顺序还需独立 session 复测后才能作为论文最终
数值。

证据：

```text
v1_p7_top20_board_20260904/top20_board_summary.json
v1_p7_top20_board_20260904/results/rank01_valid/ ... rank20_valid/
run_cpu_vta_pipeline_v1_p7_top20.py
```

#### P5B 离线历史 200-case 公式回放

开发板不可用期间，使用已封存的 200 条 Pipeline 实测记录检查修改后的静态公式。score 阶段只读取
无吞吐标签的 compatibility/profile 产物并先封存；evaluation 阶段才连接历史 cycle 标签。200 条
记录对应 199 个唯一执行方案，其中 70 条记录、69 个唯一方案符合当前 DP 的异设备交替 stage 语义。

| 范围与模型 | cycle MAPE | 平均有符号误差 | Spearman | regret@5 |
|---|---:|---:|---:|---:|
| 199 unique，旧前缀并集平均 | 28.36% | -35.50 ms | 0.803 | 10.28% |
| 199 unique，嵌套前缀容量（正式） | 28.02% | -35.14 ms | 0.775 | 10.28% |
| 199 unique，固定均分到每核（仅诊断） | 23.71% | -17.48 ms | 0.480 | 10.28% |
| 69 canonical，嵌套前缀容量（正式） | 26.70% | -34.27 ms | 0.786 | 7.30% |

上表是把静态公式回放到历史池本身的误差诊断，不是当前算法自然产生的 Top-20。若在 199 个历史
方案内部重新排序，正式公式的 predicted Top-1 实测排名为 50，`regret@1/3/5/10/20` 为
`18.69%/10.28%/10.28%/7.72%/7.72%`。在更可比的 69 个 canonical 方案中，predicted Top-1
实测排名为 18，`regret@1/3/5/10/20` 为 `16.43%/7.30%/7.30%/7.30%/0.40%`。因此当前公式能
捕获总体趋势，但这个回顾性排序不能冒充自然 shortlist 的性能。

正确的自然 shortlist 检查是：先让 Iteration 2 DP 在全部 `972528` 个合法配置上冻结 Top-20，再
连接历史标签求交集，历史数据不参与选择。结果为：**精确 execution candidate 交集为 `0/20`**。
当前 Top-20 的首段 CPU 都选择 `t4`，而历史200中相同拓扑只测过首段 `t3`，因此不能把旧 FPS
直接当成新候选实测值。

当前 Top-20 只有4种唯一切图拓扑。历史200覆盖其中3种，共对应 Top-20 中12个 thread 变体：

| 当前 Top-20 rank | 当前拓扑 | 原始资源分数倒数 | 历史同拓扑执行配置 | 历史实测 FPS | 全199实测 rank |
|---:|---|---:|---|---:|---:|
| 1--4 | CPU00--02 / VTA03--17 / CPU18--20 | 14.123 | t3 / t1 / t4 | 9.258 | 18 |
| 5,6,11,12 | CPU00--02 / VTA03--15 / CPU16--20 | 13.916--14.073 | t3 / t1 / t4 | 8.769 | 37 |
| 7--10 | CPU00--02 / VTA03--16 / CPU17--20 | 14.063 | t3 / t1 / t4 | 9.461 | 13 |
| 13--20 | CPU00--02 / VTA03--12 / CPU13--14 / VTA15--19 / CPU20 | 13.880 | 无 | 无 | 无 |

在 2026-09-04 上板前，这只能提供 topology-level 初步证据：算法集中选择的三阶段大 VTA island
在旧线程配置下达到 `8.769--9.461 FPS`，历史200无法给出当前 `t4` 首段配置的精确性能。当时
P5B 在历史200之外只精确测过当前 rank2，为 `10.449 FPS`。该证据缺口现已由上面的自然 Top-20
单 session 实验补齐；本段保留用于说明离线检查当时能够支持的边界。

跨批次得到的比例 `1.423--1.432` 读取了完整候选的历史 cycle 标签，只保留为 post-hoc
遗漏开销诊断。它不得写入正式 score，也不得用于声称当前 Top-20 的绝对 FPS；当前 Top-20
只有资源下界排序和已明确标注的个别板端实测，不能区分同一瓶颈下的 thread 变体。
自然 Top-20 只有4种 topology、包含大量同分 thread 变体，也暴露了 shortlist 预算利用率问题；后续
上板清单应在不修改 score 的前提下按 topology 分组，保留少量预注册 thread control 和更多 topology
代表，不能把20个近重复配置当作20次独立搜索证据。

绝对周期存在稳定的全局少估。为检查该现象是否跨批次存在，使用 warm100 学一个正比例系数、只在
theory100 验证，再反向执行：least-squares 系数为 `1.402/1.349`，测试 MAPE 为
`7.65%/8.19%`；robust median-ratio 系数为 `1.432/1.423`，测试 MAPE 为 `7.78%/6.94%`。
这说明当前物理公式遗漏了系统效率损失，但不能证明该损失是一个可迁移常数。正比例校准既不改变
候选顺序，也不能改善 regret；由于它来自候选结果，正式模型明确禁用该系数。

进一步以相同的跨批次规则比较 `T_raw+beta`、`alpha*T_raw` 和 `alpha*T_raw+beta`。纯加法模型
从两个训练批次得到的 mean offset 为 `36.459/33.764 ms`，median offset 为
`36.541/36.784 ms`，在对侧测试批次的 MAPE 为 `5.535%/6.046%`，优于纯乘法的
`7.651%/8.188%`。仿射模型测试 MAPE 为 `5.808%/5.516%`，但两个方向的 scale/intercept 分别为
`1.072 + 30.081 ms` 和 `0.780 + 54.069 ms`，参数不稳定。当前最有依据的结论是存在约
`34--37 ms` 的可迁移加法残差，而不是已经识别出一个 `1.42` 硬件常数。P7 首要目标是用
empty/short/long matched control 查明这笔残差属于 runtime 调度、submit/sync、cache maintenance
还是并发降速。

##### 固定加法残差模型及当前效果

令 `p` 表示一个完整 Pipeline 切图和线程配置，`T_static(p)` 是 CPU、VTA、boundary 与 DDR
资源公式给出的原始周期下界，`T_measured(p)` 是板端稳态周期。当前离线诊断使用：

```math
T_{corrected}(p)=T_{static}(p)+b,
\qquad FPS_{corrected}(p)=\frac{1000}{T_{corrected}(p)}.
```

最小二乘常量由训练集残差均值获得，稳健版本使用残差中位数：

```math
\hat b_{LS}=\frac{1}{N}\sum_{i=1}^{N}
\left(T_{measured}(p_i)-T_{static}(p_i)\right),
\qquad
\hat b_{robust}=\operatorname{median}_i
\left(T_{measured}(p_i)-T_{static}(p_i)\right).
```

在全部 200 条历史记录上仅用于描述性回放时，统一取 `b=34 ms` 后，cycle MAE 为
`7.373 ms`、cycle MAPE 为 `5.896%`、cycle median/P95 APE 为 `4.710%/14.636%`；换算到 FPS 后，
MAPE 为 `5.858%`、median/P95 APE 为 `4.865%/14.321%`。这不是“所有方案误差都只有
6%--8%”：约 6% 是全池平均相对误差，尾部方案仍可达到约 14% 以上误差。

为避免只报告同池拟合，另将历史数据按原始两个 100-case 批次交叉验证。用 warm100 拟合得到
`b=36.459 ms`，在 theory100 上 cycle MAPE 为 `5.535%`、FPS MAPE 为 `5.142%`；反向拟合得到
`b=33.764 ms`，在 warm100 上 cycle MAPE 为 `6.046%`、FPS MAPE 为 `6.176%`。两个方向得到相近
常量，说明遗漏项更像稳定的加法服务时间，而不是把所有计算时间统一放大的比例误差。

该常量对所有候选相同，因此：

```math
T_{static}(p_a)<T_{static}(p_b)
\Longleftrightarrow
T_{static}(p_a)+b<T_{static}(p_b)+b.
```

所以 `+34 ms` 只能改善绝对周期/FPS 的数值校准，**不会改变候选排序、Top-20、Top-K recall 或
regret**，不能作为搜索算法变好的证据。2026-09-04 自然 Top-20 全部上板后，原始周期 MAPE 为
`23.523%`，加 34 ms 后为 `12.859%`；本轮实际残差中位数为 `22.353 ms`。它仍改善了绝对值，
但明显过度修正，进一步证明历史候选残差不能直接作为固定硬件参数。

`b` 当前仍是读取完整候选标签后得到的 post-hoc 残差，不是可直接写入正式硬件模型的已辨识参数。
P7 应通过组件级 matched control 将其替换为可测项，例如：

```math
b(p)=\alpha_{frame}
    +N_{cpu\_stage}(p)\alpha_{cpu\_stage}
    +N_{vta\_island}(p)\alpha_{vta\_island}
    +N_{boundary}(p)\alpha_{boundary}
    +N_{dma\_call}(p)\alpha_{dma}.
```

候选间不相同的 DDR contention、cache miss、layout 按字节成本和 CPU 并发效率不得塞入这个固定
常量。只有上述组件参数由独立 profile 得到并通过 holdout，才允许进入 P7D 正式 score。

统一报告口径如下：封存 JSON 中保留原始资源下界及其倒数，以复现 DP 排序，但字段
`predicted_fps` 按 legacy schema 读取时必须解释为 `raw_resource_bound_fps`。自然 Top-20 的实测
FPS 只从 `v1_p7_top20_board_20260904` 读取；rank-1 为 `10.887 FPS`，池内最好为静态 rank-9 的
`11.511 FPS`。这些是单 session 结果，不替代 P7 受控物理 profile。

固定均分到每核虽然降低绝对 MAPE，却使 Spearman 降到 `0.480`，而且不符合当前可迁移 worker
语义，故不得进入正式 score。全 199 方案中还有 130 条记录含相邻 VTA stage；当前回放保留其
串行 VTA service，但没有重建未知的 VTA-to-VTA materialization，完整池结果只能作为历史诊断。
历史 manifest 也未记录当前 source fingerprint 和明确 affinity mode，不能宣称与当前 runtime
完全同分布。

产物：

```text
v1_historical200_affinity_scores.json      3c1033e44ff72ac653813b262de0470e08fc28e7ad7c2d98647d9146d1f33898
v1_historical200_affinity_evaluation.json  c0cfcc2131a3e3cb273a2fb42df990c9be93256283d4fa9c6755154f321a7566
```

### V1-P6：跨模型检查

状态：`host-only retrospective audit complete; prospective board validation conditional`

冻结 ResNet18 得到的算法、成本字段和实验规则，在 YOLOv3-tiny 或另一个网络上重新生成 workload
和局部 signature。允许补测新 signature，不允许读取候选吞吐后修改 score 结构。

#### P6-A：YOLOv3-tiny 无板静态筛选审计

1. 按 YOLO 双检测头的合法 branch-aware split point 生成 `684` 个切图 topology。固定 tile，最多
   3 个 VTA island；每个 CPU stage 独立选择 `threads=1..4`，共 `105696` 个执行配置。本次规模可在
   主机完整枚举，因此先用完整枚举得到精确排序，作为后续 branch-aware k-best DP 的 oracle；当前
   不能把该实现声称为已经解决含 tile 的大搜索空间。
2. 目标函数与 ResNet V1 相同：

```math
II(p)=\max\{D_{CPU-stage},D_{single-VTA},D_{CPU-core},D_{DDR}\}.
```

   多个 VTA island 的 service 与 VTA 侧 boundary mutex 时间顺序相加；CPU core 使用嵌套
   `[0,threads)` 容量下界；DDR 仍只是总服务需求下界。搜索不读取 YOLO pipeline FPS。
3. 直接迁移 ResNet CPU/VTA 每 GOP 价格失败：自然 topology-diverse Top-20 只有 `2/20` 个被旧
   YOLO 实测池覆盖，最好仅为 169 个实测候选中的第 81 名，regret 为 `31.60%`。根因是 ResNet
   4-thread 原子 profile 约为 `18.14 GOP/s`，严重高估 YOLO 高分辨率 CPU 前缀的 backend 效率，
   导致错误选择 `pool4_route` 才进入 VTA。这个消融证明“硬件相同”不等于算子 shape 价格可直接
   跨模型复用。
4. 使用旧数据中固定的 `pool2 -> dual_pre_logits` 六组 **serial component profile** 做最小目标域
   校准，只读取 stage `run_ms` 和 thread 参数，不读取对应 pipeline FPS。它区分普通卷积与
   255-channel logits 两类 CPU slope，并校准 VTA 主干 slope；缺失 thread 点只继承 ResNet 的相对
   thread scaling。该步骤是 profile，不是候选吞吐拟合。旧 serial 文件没有 process CPU clock，
   因而 YOLO core demand 暂由目标 wall slope 乘 ResNet 实测的 core/wall 比例得到，属于待验证推断。
5. 校准后 topology-diverse Top-20 中有 `13/20` 个 topology 在旧 169-case 池中有测量；历史
   Top-20 的 14 个唯一 topology 命中 `7/14`。静态第 1 名的同 topology 历史最好为第 22 名，说明
   Top-1 仍不可靠；但静态第 4 名覆盖历史第 4 名，静态第 6 名
   `pool2 -> logits` 覆盖历史全局第 1 名 `5.440 FPS`。按已覆盖 topology 回放，
   `regret@1/3/5/6/20 = 16.22%/16.22%/6.01%/0%/0%`。
6. 为避免校准自证，holdout 指标排除整个 `pool2 -> dual_pre_logits` topology。剩余 162 个实测
   candidate、80 个 topology 的 oracle 仍是 `pool2 -> logits`；Top-20 覆盖其中 12 个 topology，
   最佳覆盖仍为 oracle，holdout regret 为 0。
7. 证据仍是 topology-level：当前建议的 CPU thread 配置与旧记录没有精确 execution match，不能
   把 5.440 FPS 当成新配置的实测 FPS。Top-20 中 13 个 topology 过去 compile 成功、4 个过去
   compile 失败、3 个从未 compile；恢复开发板后必须先做 compile/reference screen，再按冻结顺序
   测量，不能把历史 buildability 或 throughput 重新写进 score。

P6-A 产物：

```text
evaluate_yolov3_tiny_static_shortlist.py
test_evaluate_yolov3_tiny_static_shortlist.py
yolov3_tiny_static_v1_evaluation.json
yolov3_tiny_static_v1_shortlist.csv
```

当前封存 JSON 的 canonical SHA256 为
`3d9bb9e633fa952be5b94464b9dfe4a86e64a76c4cdddedd6c4aa7b38fc9c201`；脚本每次重跑都会重新计算，
若 profile、候选规则或历史连接逻辑变化，必须同步更新该值。

P6-A 支持的结论仅是：6 组目标域 component profile 后，同一资源约束搜索能把历史 YOLO oracle
topology 放入前 6。它不证明绝对 FPS 准确，也不是前瞻跨模型验证。真正的 P6 gate 仍需在未参与
校准的新 topology/thread 配置上完成 native compile、reference 和上板吞吐测量。

## 5. 下一轮物理 Profile

固定 tile 继续保持不变。下一轮不拟合候选级全局比例，而是只用组件级受控实验补齐当前公式中
已确认缺失的物理量。参数必须能对应到一个明确资源，且能由 matched control 单独辨识。

### P7A：本地协议与 instrumentation（completed）

1. 从 ResNet18 冻结 Top-20 与 lowering manifest 提取真实 segment、CPU threads 和 boundary bytes，
   不读取候选吞吐标签，也不做 stage pair 的规则笛卡尔大网格。
2. CPU runner 记录 wall time、process CPU time、cycles、instructions、cache references/misses；若
   板端 PMU 不支持某个事件，manifest 必须显式标记 unavailable，不能填零。
3. VTA runner 分开记录 LOAD、STORE、compute、submit、wait，以及 cache flush/invalidate 的 bytes、
   calls 和 wall/CPU time；同一 transaction 只允许一个 owner。
4. 统一 orchestrator 生成实验计划、同步两个 barrier-aware 进程、采集环境、检查 correctness/
   determinism 并生成 review。本阶段不产生任何性能参数。

2026-09-04 本地结果：native runner 已支持独立 warmup、ready/start barrier、单 stage 重复运行、
process CPU time 和进程级 PMU；PMU 不可用时输出 `null` 及原因。CPU memory benchmark 已支持相同
barrier 和 `cache_resident/transition/streaming` 压力类。AArch64 runner 与 memory benchmark 均
交叉编译成功；相关 Python 测试 `24 passed`，逐样本同步和多 tensor stage 输入审计通过。计划固定为
36 个 memory baseline、16 个 CPU stage pair、6 个 device runtime、6 个 boundary 和 12 个
CPU-VTA DDR matched run；计划 SHA256 为
`598e63a93fc77737be68f32f5d574031e4e1ba74778837a15a6648a5085f61ec`。memory copy 的计划与
输出已进一步区分 operand bytes 和双数组实际活跃 working set，避免低估 cache/DDR 压力。

产物：`v1_p7_measurement_plan.json`、`v1_p7_local_instrumentation_audit.json` 和
`v1_p7_review.md`。后两阶段需要的 session/profile/formula validation 仍为 pending，不能从本地
smoke 伪造。

### P7B-1：CPU 并发与 Cache/DDR（completed）

先执行单 boot qualification，每项 5 次 warmup、20 次计分：

1. 36 组 memory baseline：256 KiB、4 MiB、64 MiB × read/write/copy × threads 1..4。
2. 16 组 Top-20 来源 CPU pair，每组执行 A-only、B-only、A+B concurrent。
3. 每个 stage 使用独立进程，barrier 后同时起跑，分别保留 wall、process CPU 和 PMU 计数。

只有 slowdown 超过 5%、95% CI 不包含 1 且三个 boot 可重复，才进入模型。完成本类后停止并 review。

2026-09-04 qualification 使用实际端点 `.247` 和 boot
`08189a44-9662-422e-9db7-e1e040043691`。重启后已按 runbook 重新部署 native runtime、HPC
bitstream 与 RPC；FPGA 为 `operating`，`u-dma-buf=192 MiB`，测量前后 preflight 均通过。

36/36 组 memory case 通过。64 MiB streaming copy 的 t1/t4 中位带宽为
`4.502/7.886 GB/s`；streaming write 的 t3/t4 为 `7.992/7.785 GB/s`，说明 3--4 线程已进入
共享内存带宽饱和区。16/16 组真实 CPU pair 通过；A/B slowdown 范围分别为
`1.013--1.700x` 和 `1.018--2.221x`，各有 14 组在单 boot 下满足 slowdown >5% 且 95% CI
不含 1。大 `cpu:00:02_t4` 与后段 2--4 线程并发时影响明显，短 `cpu:20:20` 的影响较小，因而
不能用一个统一 CPU 并发常量替代 shape/thread-aware slowdown。

并发/孤立 PMU 指令数中位比只在 `0.999--1.010`，cache-miss 比值中位数约为 `1.012`；stage
执行的工作量基本不变，而 elapsed time 显著增加。这把问题定位为 CPU 调度、共享 cache/内存等
资源竞争，但尚不能单独归因于 DDR；CPU-VTA DDR 归因仍由 P7C matched control 完成。

qualification review 后，统计 gate 冻结为：孤立 stage CV 不超过 10%，slowdown 中位数
bootstrap 95% CI 相对半宽不超过 25%。并发 wall-time CV 反映 Linux/TVM 调度竞争，保留为
不确定性但不单独作为失败条件。该修正后统一重跑全部 16 组。runner 也增加多输入文件支持，保证
`cpu:16:20` 和 `cpu:17:20` 同时消费 main/residual 两个真实输入。

首次 boot 完成时未生成 `v1_p7_physical_profile.json`，slowdown 也未进入正式 score。
该组合 session 为 `v1_p7_session_boot1.json`，canonical SHA256 为
`23d2c77ee38eb44f101b77b672392297b88c8ee480b7c550fb2c7828ca091d80`。

2026-09-04 第二个独立 boot `c918d6e5-c843-49f9-888a-9574dbe47b40` 已按相同冻结计划完成。
重启后重新建立 `/mnt/sd -> /media/sd-mmcblk1p2`，由同一 runbook 加载 HPC bitstream、192 MiB
`u-dma-buf` 和 RPC。memory 首轮因 `cpu_mem_streaming_read_t2` 的 wall-time CV 为 `10.91%`
未通过；失败产物完整保留，随后不改 case、流量或 gate，统一重跑 36 组并全部通过。CPU pair
16/16 通过，A/B 分别有 14/13 个单 boot 显著信号。

两个 boot 的 memory 带宽 Pearson 相关性为 `0.998`，中位相对差 `0.86%`；A/B slowdown 的
相关性分别为 `0.981/0.982`，中位相对差分别为 `1.37%/1.10%`。A 的 14 个显著 case 完全一致；
B 有 13 个一致，只有边界较弱的 `cpu_pair_16` 从 boot 1 的显著变为 boot 2 不显著。第二个组合
session 为 `v1_p7_session_boot2.json`，canonical SHA256 为
`25a91e171072897cc26a9aaefcc907691ea8d9e784676bb74c38b3eb4454c4d3`。完成该轮时仍少一个独立
boot，故当时没有生成正式 slowdown profile。

2026-09-05 第三个独立 boot `a5f1e917-0021-4674-91d9-679107024d21` 也已完成。memory 首轮的
`cpu_mem_streaming_read_t3` 因两个长尾样本导致 CV=`17.47%` 而失败；失败 session 保留后，
按相同计划统一重跑 36 组并通过。CPU pair 16/16 通过，A/B 单 boot 显著信号为 14/14。boot 3
组合 session SHA256 为
`02e4af42f9b896c4119c81d41e9ac2bec4f3b940f33d822d0a865050eaf58d09`。

三 boot 汇总 `v1_p7b1_three_boot_reproducibility.json` 已通过，SHA256 为
`b4fc0f24319af9890cc81fcd9f5134076acbe68bcb2cc999e1c7c639ce2a4056`。memory 三轮两两 Pearson
最小值为 `0.994`，跨 boot CV 中位数/最大值为 `0.45%/6.01%`；A/B slowdown 三轮两两 Pearson
最小值为 `0.974/0.965`，跨 boot CV 中位数为 `1.17%/2.17%`。按“三轮均 slowdown >5% 且单 boot
95% CI 不含 1”的冻结条件，准入 A/B 精确观测 14/13 个。

该 gate 只证明这些已测 stage pair 的局部 slowdown 可重复。当前 16 组没有覆盖全部 DP stage-pair
特征域，因此 `generalized_slowdown_surface_ready=false`，不得把精确观测静默外推到其他候选；
特征化拟合和 grouped holdout 留到 P7D。P7B-1 完成不等于已经生成最终 physical profile。

### P7B-2：固定运行时与 Boundary 开销（completed）

执行 host empty、VTA short/long × poll sleep 0/1000 ns，以及双方向 boundary 的实际
low/median/high 字节数。Review 时发现原计划拟合
`alpha_frame + N_cpu*alpha_cpu + N_vta*alpha_vta` 不可辨识，并且 P2/P3 的 CPU/VTA `run_ms`
已经包含 graph invocation，VTA `run_ms` 还包含 LOAD/STORE、submit、poll 和 sync；再次叠加
per-stage 截距会重复计费。因此保留原 P7 总计划 SHA256 不变，新增冻结子协议
`v1_p7b2_measurement_plan.json`，只输出 host 队列下界、poll matched delta、boundary set/get
观测和 inclusive VTA runtime 分解。

2026-09-05 在 boot `a5f1e917-0021-4674-91d9-679107024d21` 完成单 boot qualification，9/9 个
实际运行 case 通过 correctness、determinism、CV 与 profiler gate。1/3 个空载 actor 队列链为
`0.028/0.056 ms/帧`；poll sleep 从 0 改为 1000 ns 后，short/long VTA `run_ms` 只增加
`0.222/0.531 ms`。CPU->VTA 与 VTA->CPU boundary 的单 boot through-origin slope 分别为
`3.222/3.215 ms/MiB`，fit MAPE 为 `0.71%/1.05%`。这些量都不足以解释 22--34 ms 候选残差；
旧 P2 boundary slope 还略高于本轮结果，强行替换反而会扩大周期低估。

P2 单一 VTA GOP slope 在 `vta:03:17` 上 APE 约 1%，但在本轮短 segment 上达到 11%--23%，
说明应继续使用 segment/shape-aware service，而不是增加全局 runtime 常量。所有 profiler 分量
只作 inclusive `run_ms` 的解释证据，不重复相加。因当前只有一个 boot，boundary slope 不进入
正式 profile；host queue floor 也未经过候选 holdout，不进入公式。本类实验完成后按阶段规则停止。

### P7C：共享 CPU/DDR 并发 Profile（completed，单 boot 证据未准入）

使用 CPU-only、VTA-only 和 CPU+VTA concurrent matched control；CPU 选择 256 KiB cache-resident
与 64 MiB streaming，VTA 选择 compute-heavy 与 DMA-heavy，共 4 个组合、12 种运行模式。优先拟合
聚合共享带宽；只有明显的非对称仲裁无法被聚合模型解释时才增加方向化 slowdown。不能用完整候选
残差反推参数，也不展开所有 stage pair 的笛卡尔积。完成本类后停止并 review。

本地审计发现原总计划不能直接执行：compute-heavy 与 DMA-heavy 都指向同一 `vta:03:17`
placeholder，256 KiB cache read 实际只分配了 128 KiB，且 4 个 CPU worker 会占满全部核心，
把 VTA host 的用户态调度竞争混进 DDR 归因。总计划及既有 session hash 不改，P7C 子协议改用
真实 `vta:01:03` 与 `vta:15:19`；P7B-2 profiler 显示二者 DMA MiB/run-ms 相差约 `3.92x`。
cache read 的 operand/working set 均修正为 256 KiB；CPU 使用核 0--2，VTA host 使用核 3。

首轮 runner 在 `runtime.config_threadpool` 后把 VTA host affinity 从核 3 覆盖为核 0，因此 0/4
通过 affinity gate；该失败 session 已保留。第二轮在每次 threadpool 配置后重新绑定当前线程，
但 review 发现不能证明已创建的 TVM 辅助线程也在核 3，因此只保留为过程证据。最终 runner 枚举
`/proc/self/task`，把全部现有线程绑定核 3，并让后续线程继承；二进制 SHA256 为
`dbc0caa3885e6d8b5722901a7b62d66feb139099c27bed2d021a900ef9083aec`。四组 workload 不变，最终
4/4 通过 correctness、determinism、PMU、CV、置信区间、affinity 和 VTA 指令工作量 matched gate。

最终单 boot 中，CPU slowdown 为 `0.995--1.065x`，只有 `cache_resident + compute_heavy` 的
`1.065x` 超过 gate；由于它来自 cache control，不能归因为 DDR。VTA 在 cache control 下为
`1.010/1.014x`，在 streaming 下为 `1.052/1.066x`，两组 streaming 均超过单组 gate；相对各自
cache control 的额外 slowdown 比率为 `1.042x/1.052x`。两组 VTA stage 增量分别为
`2.110/1.359 ms`，driver run 仅增加 `0.023/0.266 ms`，大部分增量位于 host runtime 的
run-minus-driver 部分。因此当前只证明轻度、非对称的 CPU-VTA shared-memory/host-runtime
interference，不能拟合纯 DDR 带宽常数，也远不足以解释 22--34 ms 残差。正式准入仍需两个
额外独立 boot。产物中的 GB/s 只是声明流量/最长 wall 的诊断比率，不是实测 DDR bandwidth。

P7D 准入审计决定不继续用两个 boot 扩展该信号：当前观测既不是纯 DDR 参数，也没有证据表明它能
改善冻结 Top-20。P7C 因此以“完成受控单 boot、保留负结果、不进入正式公式”收口；若后续新的
前瞻候选把 shared-memory contention 暴露为排序主误差，再按同一冻结协议补 boot，而不是现在
为了得到参数而重复测量。

### P7D：公式重建与冻结验证（completed）

正式模型只组合 P7B/P7C 的组件参数：

```text
CPU stage service = isolated service × admitted slowdown + owned boundary
VTA service       = isolated service + owned boundary
boundary service  = layout/quantization + actually executed cache maintenance (HPC/coherent V1 = 0)
pipeline II       = max(CPU stage、单 VTA 总和、CPU core pool、共享 DDR) + frame intercept
```

其中 `frame intercept` 不是默认项。只有独立 empty/matched 实验能够辨识、所有权不与 stage
`run_ms` 重叠，并且在冻结 holdout 上改善误差时才加入；否则取 0。P7B-2 当前测得的 host queue
floor 小于 `0.1 ms`，尚不满足进入正式公式的条件。

先在组件 grouped holdout 上冻结公式，再连接历史 200 条候选标签做纯评价。历史候选不得参与参数
拟合或选择公式。恢复开发板后还需要测冻结的新候选，才能建立前瞻绝对 FPS 证据。

P7D 验收门槛固定为：冻结历史 200 条 cycle MAPE 低于其 `+34 ms` 诊断基线 `5.896%`，自然
Top-20 cycle MAPE 低于其 `+34 ms` 基线 `12.859%`，Top-20 固定顺序 Spearman 高于 `0.155`，
`regret@5` 低于 `5.419%`；每个新增参数必须来自组件实验。某项不能改善冻结验证集时不纳入
正式公式。

2026-09-06 已完成 P7D。三 boot 的 64 MiB streaming CPU read/write 带宽通过组件 gate，替换原
CPU 逻辑 DDR read/write rate；36 个完整 memory case 仍保留，只有 8 个 streaming read/write
参数进入公式。16 组 CPU pair slowdown 因 `generalized_slowdown_surface_ready=false` 不向全部
DP transition 外推；P7B-2 boundary/runtime 与 P7C interference 只有单 boot 或所有权不可唯一
辨识，也不进入。`+34 ms` 和候选比例拟合明确禁止进入排名。

使用冻结参数重新遍历 `4623` 个 topology、精确核对 `972528` 个执行配置，DP Top-20 与完整枚举
一致。新旧 Top-20 的候选、顺序和预测周期完全相同，Top-1 仍为
`cpu-00-02_t4__vta-03-17_t1__cpu-18-20_t1`。这说明更新后的 DDR 下界在该高性能区域不是关键
约束，不应为了改变排名而加入未经验证的 contention 系数。

冻结评价如下：历史 200 条 raw cycle MAPE=`28.022%`，其 `+34 ms` 诊断 MAPE=`5.896%`；自然
Top-20 raw cycle MAPE=`23.523%`，`+34 ms` MAPE=`12.859%`；固定顺序 Spearman=`0.154887`，
并列分数感知 Spearman=`0.053274`，`regret@5=5.419%`。两种 Spearman 的差异来自大量完全同分
线程配置的名次处理，不是候选集合变化。质量 gate 未通过，因此 P7D 决定为
`freeze_as_ranking_only_v1_and_do_not_claim_absolute_calibration`：保留其筛选高性能区域的能力，
不宣称绝对 FPS 已完成物理校准。

P7D 产物为 `v1_p7_physical_profile.json`、`v1_p7d_ranked_candidates.json`、
`v1_p7d_historical_scores.json`、`v1_p7d_historical_evaluation.json`、
`v1_p7_formula_validation.json` 和 `v1_p7d_review.md`。其中 profile/ranked/validation SHA256
分别为 `b4d36712ef7baac6c33179de7d2767fc4405015fadc68f20774521f598b42557`、
`bfa0e9ecba2eaafbc188e33752961ec54edebc3feaa019faa2a93b97a78edd35`、
`983afc8d7e0f6f919aae7c0d40c2c8a041203f670c9366f01d84c7ad60664aa6`。

以下证据至少满足一项，才考虑 tile 联合搜索：

```text
同一 VTA workload 的固定 tile 在候选关键路径上稳定低效；
排序残差与 tile occupancy/physical DMA 显著相关；
替换少量 tile 后 Top-K regret 明显下降；
V1 其余成本项已通过 grouped holdout。
```

V2 采用有限合法 tile 集合和分层/Pareto DP，不直接展开 AutoTVM 全笛卡尔积。

共享 DDR contention 已由原“条件 D1”提升为 P7C 的明确实验，但仍与 tile、FIFO 和 Max-Plus
解耦。若 matched control 显示并发影响不显著，则保留聚合 DDR 下界，不强行增加参数。

### P8：跨 Stage 共享内存与边界零拷贝（current，P8B 已完成待 review）

#### P8.1 问题边界与设计依据

当前 AXU5EVB driver 已将一整块 `u-dma-buf` 映射到 CPU 虚拟地址，并以 256-byte
对齐的 bump allocator 生成 VTA 可使用的物理地址；同一块内存可作为 VTA LOAD/STORE
的源或目标。因此 BiMapTab 类工作中“统一 MM2S/S2MM 物理 chunk”的底层问题已基本解决，
**P8 不重新实现 arena 或物理内存分配器**。当前 driver 的 `Free` 是 no-op，但每次 native
runner 只加载一个候选且进程结束后整体释放，第一版不需要动态 compact/free-list。

真正缺失的是 `u-dma-buf` 之上的跨 Executor 管理：native runner 仍调用 GraphExecutor
普通 `set_input/get_output`，并对每个 stage 输出执行 `CopyTo(kDLCPU)`。Host/VTA copy
在 AXU5EVB driver 中使用 `memcpy` 或默认的逐字节 volatile safe-copy，取决于
`AXU5EVB_DRIVER_SAFE_COPY`。当前产物没有记录该环境变量，所以已观测的 `3.2--3.6 ms`
只能称为当前 rank-1 路径的 stage 间 materialization 墙钟开销，不是固定通信常量，
也不等于零拷贝后 Pipeline II 必然下降同样数值。publication baseline 必须先在 correctness
通过后使用最快的正确普通 copy 路径；若逐字节 safe-copy 只是调试规避措施，不能把它的开销
算成零拷贝方案的收益。

TVM 官方 Pipeline Executor 解决的是“多个 GraphExecutor 如何按依赖关系并行运行”，不是
“跨 Executor 的同一块物理内存如何复用”。当前源码中的 `QueueData::CreateCopyFrom` 会调用
`TVMArrayCopyFromTo`；SPSC queue 的 push/poll 都通过赋值复制 payload，consumer 最后又通过普通
`SetInput` 复制到自己的输入。也就是说，官方实现提供 worker、queue、notification 和 DAG
连接，但其数据通路是 owning tensor/deep-copy 语义，不能自动利用本板共享 DDR 的同址 buffer。
P8 的目标不是重新发明 Pipeline Executor，而是把其“控制流水线”与共享物理 slot 的“数据通路”
分开。

RIMMS 面向 task-to-PE 映射在运行时才确定的 CPU/GPU/FPGA 系统，因此需要跟踪数据最新位于哪个
memory/PE，并在需要时迁移。本项目的切图和设备映射在运行前已经固定，CPU 与 VTA 又共享同一
DDR，不需要复制 RIMMS 的通用位置迁移层。可迁移的思想只有“不要隐式猜测数据状态”：这里把它
收窄为 edge/slot 的 owner、frame generation、consumer count 和 completion 状态。

P8 只借鉴下列可迁移机制，不直接套用其它系统的性能数字：

| 依据 | 借鉴内容 | 不照搬的部分 |
|---|---|---|
| BiMapTab/PSoC memory organization | 虚拟/物理地址统一的双向 chunk | 底层 chunk 已由 `UdmabufPool` 提供 |
| RIMMS | 将数据状态和必要迁移显式化 | 本项目映射静态且共享 DDR，不需要动态位置迁移或多副本一致性 |
| TVM Pipeline Executor | worker、queue、notification 和 stage DAG | 当前 `QueueData/SetInput` 是深拷贝数据通路，不直接作为共享 slot 管理器 |
| TVM GraphExecutor zero-copy API | 通过外部 `DLTensor` 重绑定输入/输出指针 | API 不分配共享内存，也不管理跨 Executor 生命周期与帧间覆盖 |
| NEURAghe/batched pipeline | 预分配 slot、异步队列与跨帧重叠 | 不修改 VTA 硬件调度器 |
| SoC-FPGA coherency analysis | 区分硬件一致性、软件 cache maintenance 和执行完成同步 | V1 已固定 HPC/coherent，不重新搜索 HP/HPC/ACP |

因此可声明的系统贡献不是 `set_*_zero_copy` API 或双缓冲本身，而是：从切图 manifest 判定哪些
CPU-VTA edge 能共享；为可共享 edge 分配有界物理 slot；跨 CPU/VTA Executor 绑定同址双 view；
用 generation/owner/completion 保证多帧复用；把不可消除的 adapter 直接写入目标 slot；最后把
真实 materialization/slot-wait 反馈给静态切图成本模型。上述机制必须通过消融实验后才能称为贡献。

#### P8.2 最小可用设计

复用现有 `UdmabufPool`，只新增轻量的 `BoundarySlotManager`。每个跨 stage 的 live tensor
拥有两个预分配 slot，每个 slot 保存：

```text
u-dma-buf virtual address + physical address + allocated bytes
shape + dtype + layout + alignment + producer/consumer ids
CPU DLTensor view + VTA DLTensor view
state + frame generation + remaining-consumer count
```

CPU view 和 VTA view 的 `data` 指针相同，但 `device_type` 分别是 `kDLCPU` 与
`kDLExtDev`，以满足 GraphExecutor 对 device、shape 和 alignment 的检查。管理器拥有底层内存和
view 生命周期，Executor 只借用指针。绑定前额外审计 dtype、layout 和实际分配范围；
TVM 当前检查不充分的字段不能默认为合法。

单生产者/单消费者的基本状态机为：

```text
FREE -> PRODUCER_WRITING -> READY -> CONSUMER_USING -> FREE
```

slot 转移使用 mutex/condition-variable 和 generation id，不忙等，并防止旧帧 token 重用后的
ABA/覆盖。Producer 在 `run()` 前获取空闲输出 slot 并调用 `set_output_zero_copy`；
consumer 在 `run()` 前对同一 slot 调用 `set_input_zero_copy`，在同步 `run()` 返回后释放输入。
队列只传递 `{edge_id, slot_id, generation, tensor_contract}`，不再传独立中间 NDArray。

第一版固定每条边两个 slot，把“可等待帧数”与原 `queue_depth=2` 语义分开。两个 slot
可容纳一个 consumer 正在使用和一个 producer 正在准备/READY 的帧；额外队列深度不改变
稳态瓶颈，不在第一版增加更多 slot。当前 rank-1 两个边界的双 slot 合计约
`2 * (802816 + 100352) = 1.72 MiB`，远小于 192 MiB pool。

当前 VTA `run()` 是同步返回，因此返回后可认为该 slot 的 device 访问已完成；若未来改成
异步提交，必须以 completion event 代替返回点，不得延用当前假设。单 VTA 互斥约束仍保留。

#### P8.3 一致性与转换规则

HPC 硬件一致性、线程同步和 buffer 生命周期是三个不同问题：

1. V1 的正式路径固定为已经选定的 HPC/coherent bitstream 与
   `VTA_COHERENT_ACCESSES=1` runtime。该组合下 VTA runtime 本来就跳过
   `VTAFlushCache/VTAInvalidateCache`；P8 **不得重新插入** `SyncForDevice/SyncForCpu`。
2. 上板 preflight 只做一次配置不变量审计：bitstream/runtime hash 必须属于同一 HPC bundle，
   编译宏必须为 coherent。`u-dma-buf/dma_coherent` 只描述该 u-dma-buf device 的分配/绑定元数据；
   当前通过 module parameter 建立、未绑定到 PL master device 的 pool 报告 `0`，它不能单独证明或
   否定 PL 实际使用的 HPC/AxCACHE 路径。正式证据必须分开记录为“allocator 元数据不背书”与
   “冻结 HPC bitstream/runtime hash + coherent 编译宏 + 既有 HP/HPC 受控对照”。这不是新的性能
   实验，也不靠偶然正确的一次输出“证明”HPC。
3. A/B alternating-input、generation 和 reference 测试验证的是 zero-copy 绑定、帧顺序、完成点和
   slot 重用是否正确，不是重新验证硬件 cache coherence。mutex/condition-variable 的
   release/acquire 负责线程可见性，VTA `run()` 返回负责 device completion；二者仍然必须保留。
4. 若上述 HPC 配置下出现旧数据，按 bitstream/runtime/AXI 属性或生命周期实现错误处理并失败，
   不在同一个正式配置中自动回退软件 flush/invalidate。HP/non-coherent 只能作为使用另一套
   bitstream/runtime 的独立可选 baseline。
5. HP/ACP 接口选择需要硬件连接与 bitstream 支持，P8 不做“运行时动态切换”的虚假抽象。
6. layout/量化/padding/slice 不匹配时禁止冒充 zero-copy。后续 adapter 可直接写入
   consumer slot，但 adapter compute 与其读写字节仍独立计费。
7. 多输出/多消费者边界为每个 live tensor 独立管理 slot，`remaining_consumers=0` 后才能
   重用。ResNet 残差仍保持在现有合法 unit 内，不为测试 slot 而改变切图语义。

#### P8.4 分阶段实施

P8 仍遵守一次只执行一个子阶段：

1. **P8A：ABI 与内存属性可行性，本地+单 boot。**
   增加 `set_input_zero_copy/set_output_zero_copy` 句柄、CPU/VTA 双 view 和字段检查；记录
   `AXU5EVB_DRIVER_SAFE_COPY`、u-dma-buf module/capability、mapping 属性、物理地址、alignment
   与实际 copy bytes。先对逐字节 safe-copy 和 `memcpy` 做 correctness-matched control；正式 B0
   使用最快的正确普通 copy。再用普通 CPU buffer 与 u-dma-buf 分别作为相同 CPU kernel 的
   输出/输入，检查直写/直读共享 mapping 是否使 CPU stage 变慢。若 CPU penalty 已大于待消除
   copy，停止直接绑定路线。

   2026-09-06 已完成。GraphExecutor 的 zero-copy API 只负责重绑定外部 `DLTensor`，不会为本
   native 进程自动取得 u-dma-buf 物理内存；runner 因此通过现有 VTA driver 显式分配一个
   256-byte 对齐的 cached slot，再构造 CPU/VTA 两个同址 view。冻结边界为
   `[1,64,56,56] float32`、`802816 bytes`，物理地址检查与 A/B 交替输入逐帧 ordinary-copy
   等价检查均通过。初版按“全部 ordinary 后全部 zero-copy”执行时，CPU 温漂会把 direct-mapping
   penalty 的符号反转，因此已改为逐帧匹配并平衡 `AB/BA` 顺序，不采用任何一次较好旧结果。
   修正后逐字节 safe-copy 的普通边界 materialization 中位数为 `2.807218 ms`，`memcpy` 为
   `0.714217 ms`，故 B0 固定选最快且正确的 `memcpy`，不得把调试用 safe-copy 的额外耗时记为收益。
   同址路径把该 stage 边界的 framework materialization 从 `1605632 bytes` 降到 `0`，相对 B0 的
   单帧串行 latency 配对差值中位数为 `+1.157847 ms`；CPU producer 直写共享 mapping 的配对
   penalty 中位数为 `-0.251582 ms`，没有显示抵消 copy 收益。该结果是单 boot、单边界、单 slot
   串行可行性证据，不是
   双缓冲 Pipeline FPS 结论，也不是独立 CPU 数值 reference。

   本 boot 的 u-dma-buf sysfs 报告 `dma_coherent=0`。这被记录为 allocator 元数据未背书，而不是
   改写成“非一致路径”；VTA 路径仍由匹配的 HPC bitstream/runtime hash、
   `VTA_COHERENT_ACCESSES=true` 和既有 HP/HPC 受控对照限定。P8A gate 通过并停在 review。
2. **P8B：单边界、单 slot、串行 matched control。**
   先做约 802816-byte CPU->VTA 边界，再做约 100352-byte VTA->CPU 边界。串行模式排除
   流水调度干扰，使用 A/B 交替的两种真实输入，不使用单图重复哈希作为覆盖证据。
   普通 set/get 与 TVM 官方 zero-copy API 手工绑定作为成对 baseline，仅证明可消除的
   materialization 和数值正确性，不声明 Pipeline 收益。

   2026-09-06 已完成单 boot qualification。runner 的单 slot 绑定已从 CPU->VTA 推广为双向
   heterogeneous edge，并按方向分别检查 CPU producer 直写或 CPU consumer 直读 penalty；初版
   两边都检查 producer 的 owner 错误已在冻结前修复。两个方向均执行 safe-copy/`memcpy` control，
   每个 control 为 20 个逐帧 matched pair，AB/BA 各 10 个。两种输入的 ordinary-copy/zero-copy
   输出逐帧等价，并单独引用冻结 Top-20 的独立 reference 证据。

   最快正确 B0 均为 `memcpy`：CPU->VTA 的 `802816-byte` tensor 普通边界 copy 中位数为
   `0.726637 ms`，framework materialization 从 `1605632` 降到 `0 bytes`，CPU 共享 mapping penalty
   为 `-0.253247 ms`；VTA->CPU 的 `100352-byte` tensor copy 为 `0.116311 ms`，materialization 从
   `200704` 降到 `0 bytes`，CPU 直读 penalty 为 `+0.005370 ms`。对应整帧
   `baseline - zero-copy` 配对差值中位数为 `+1.204572/+0.846264 ms`，但它们大于边界 copy 本身，
   包含 stage/runtime 波动，不得全部归因于 zero-copy。P8B gate 只准入双向 ABI、字节消除和 mapping
   未抵消 copy 三项结论；仍不声明 Pipeline FPS 提升。
3. **P8C：两边界、双 slot Pipeline。**
   实现 `BoundarySlotManager`、状态机、generation、超时/错误传播，保留单 VTA executor/mutex。
   中间输出默认不复制作 debug dump；只在计时区间外按预注册帧抽样检查。比较完整
   latency、Pipeline II 和 FPS，不把四段边界计时之和直接当成 II 收益。

   2026-09-06 已完成最终 binary 的单 boot qualification。`BoundarySlotManager` 对每条边执行
   `FREE -> PRODUCER_WRITING -> READY -> CONSUMER_USING -> FREE`，token 绑定 edge/slot/generation/
   frame owner；condition-variable 等待具有超时，任一 worker 失败会中止全部 manager，流水排空后
   强制检查所有 slot 已释放。四个 slot 的虚拟/物理地址均 256-byte 对齐且物理区间不重叠，两条边
   均实际轮换 slot 0/1。A/B 两种输入在 B0/B1/B2 的逐帧输出一致，独立 reference 证据另行保留；
   前后 preflight 和存储检查均通过。

   同一 boot 采用 `B0/B1/B2/B2/B1/B0` 对称 block 顺序，每 block 5 次预热和 10 帧计分。后续审计
   发现旧分析把相邻完成间隔的中位数误作稳态 II；在 completion 呈突发成批分布时，该统计会漏掉
   block 内的长空洞并产生虚假的正收益。正式口径已改为每个 block 使用
   `(末次完成时间-首次完成时间)/(N-1)`，再对同模式 block 等权取中位数；逐帧间隔中位数只作
   调度抖动诊断。旧 `+4.42%` 单 boot 结果已废止，不再作为证据。

   归因计数显示 B0 的两条边每帧共有 `1806336 bytes` framework materialization；VTA profiler
   直接观测其中 host->VTA `802816 bytes` 和 VTA->host `100352 bytes`，B2 两项均为 0。VTA 内部
   SRAM 搬运仍为 `LOAD 12025856 + STORE 1229312 bytes/frame`，B0/B2 完全相同，故本轮没有把
   VTA 内部 LOAD/STORE 伪装成 zero-copy 收益。B2 的主要 backpressure 是 VTA->CPU 边 producer
   等待 CPU 尾 stage 释放 slot，中位数 `18.541 ms`；它是流水线瓶颈反馈且可与其他 stage 重叠，
   不得作为 additive latency 再计一次。
4. **P8C-L：通信密集切图的串行单帧时延验证。**
   P8C 的双 slot 吞吐实验已经证明边界物化不在当前一/双 island Top-20 的稳态关键路径。为回答
   “zero-copy 是否真实降低单张图片经过多次 CPU-VTA 交接的时延”，另冻结一个三 VTA-island
   stress case，比较完全相同的逐 stage 串行调度：L0 使用普通 `get_output/set_input`，L1 使用
   单 slot zero-copy。该实验不使用双 slot，也不把流水线重叠记作 zero-copy 收益。

   2026-09-06 已完成第一个 boot qualification，并补做了与历史 runner 默认复制实现一致的
   qualification。候选为
   `CPU(0)-VTA(1..3)-CPU(4)-VTA(5..8)-CPU(9)-VTA(10..17)-CPU(18..20)`，含 7 个 stage、
   6 条异构边和残差产生的多张量边界。编译图中的 shape/dtype/device view、slot 物理地址对齐与
   不重叠、A/B 输入输出等价均通过。普通路径每帧物化 `9031680 bytes`，L1 为 `0 bytes`。

   两种普通复制口径必须分开报告。显式 `AXU5EVB_DRIVER_SAFE_COPY=0` 的最快正确 `memcpy` 基线中，
   边界 API 服务为 `4.483829/0.111196 ms`，L0/L1 单帧中位时延为
   `167.398779/162.661537 ms`，减少 `4.737242 ms`（`2.83%`）。撤销该覆盖并跟随 runner 默认
   安全复制后，边界 API 服务为 `16.324229/0.111167 ms`，单帧中位时延为
   `178.855128/163.092956 ms`，减少 `15.762172 ms`（`8.81%`）。后者与该切法历史 profile 的
   `17.238750 ms` direct copy 接近，解释了历史三-island分组中位数 `11.366 ms` 与前次
   `4.484 ms` 的差异。两组 VTA 内部 LOAD/STORE 字节均完全相同。

   `8.81%` 只能回答“相对历史 runner 默认路径”的收益；评价 zero-copy 相对优化后普通复制的
   独立价值时，仍应使用更保守的 `memcpy` 基线和 `2.83%`。两项都只有一个 boot，均不得写成
   正式稳定提升。

   本阶段最终保留的系统规律是“latency 与 throughput 的收益条件不同”。逐 stage 串行执行时，
   每条边界都位于单帧关键路径，因而消除边界物化会近似累加到单帧时延收益：

   ```math
   L_{serial}=\sum_i T_{stage,i}+\sum_e C_{boundary,e}+T_{runtime},
   \qquad
   \Delta L\approx\sum_e(C_{copy,e}-C_{zc,e}).
   ```

   稳态流水线的周期则由最慢工位或共享资源需求决定：

   ```math
   II=\max(T_{stage/resource}),
   \qquad
   \Delta II\approx\max(0,\Delta C_{boundary}-slack_{boundary}).
   ```

   当 boundary copy 小于该工位相对瓶颈的余量、或能够与瓶颈 stage 重叠时，即使复制字节减少
   100%，`II/FPS` 也可能不变。只有 boundary/DDR 已在关键周期上，zero-copy 才会转化为吞吐提升。
   因此三-island串行实验用于验证累计 latency 收益；P8C 三 boot 用于证明当前自然 Top-20 中这项
   工作不在稳态吞吐关键路径。两者并不矛盾。

   这是人为增加交接次数的 communication-intensive stress case，用于说明收益随可消除边界工作
   累积，不代表静态搜索应选择三个 island，也不声称吞吐提升。结果仍只来自一个 boot；正式
   单帧时延结论要求至少三个独立 boot 的配对区间排除 0。实验实现和产物位于
   `v1_p8_latency_island_scaling/`。
5. **P8D：adapter 直写与调度最小化，条件执行。**
   HPC 主线不优化不存在的软件 cache maintenance。只有 P8C profiler 显示 adapter 或 slot wait
   仍处于关键路径时，才实现 adapter-direct-write 或调整 slot 调度；每个机制单独消融，不一次
   合并后归因。HP/non-coherent 的 dirty-range sync 只能在独立 baseline 中研究。
6. **P8E：manifest 驱动的通用化，条件执行。**
   从 stage DAG 和 tensor contract 生成 edge/slot，按 live interval 和 consumer 计数管理复用；超过
   `u-dma-buf` 容量、contract 不兼容或出现动态 shape 时显式拒绝。只在 ResNet18 两方向通过后
   再用 YOLOv3-tiny 的 tuple/route 边界验证通用性。

#### P8.5 Baseline、指标与验收

固定消融顺序，避免把 TVM 已有 API 当作创新：

```text
F0: TVM 官方 Pipeline Executor 的 QueueData + 普通 SetInput（仅在 VTA/ext_dev 正确运行时测性能）
B0: 当前 native pipeline + 最快的正确普通 set/get + CopyTo(CPU)
B1: TVM 官方 GraphExecutor zero-copy API + 手工单 slot + 串行
B2: B1 + BoundarySlotManager + 双 slot Pipeline
B3: B2 + 有证据的 adapter-direct-write/slot scheduling 条件优化
```

F0 主要确认官方控制流水线的数据 forwarding 语义；若现有版本不能正确执行 VTA/ext_dev，记录
兼容性结果，不把它伪装成性能 baseline。`B1-B0` 回答显式复制能否消除；`B2-B1` 回答本项目的
生命周期/双缓冲管理能否把零拷贝安全带入流水线；`B3-B2` 回答转换直写或 slot 调度优化。

P8C-L 额外固定两套不得混合的 B0 口径：

```text
B0-historical: 不设置 AXU5EVB_DRIVER_SAFE_COPY，跟随当前 runner 默认安全复制
B0-optimized:  设置 AXU5EVB_DRIVER_SAFE_COPY=0，使用已通过 correctness 的 memcpy
B1:            相同串行 schedule 的 single-slot zero-copy
```

`B0-historical` 只用于复现并解释历史 200 条 profile 的通信时间；一个通过全部功能 gate 的 boot
已经足以完成该诊断，不再为它追加正式性能 session。`B0-optimized` 是评价 zero-copy 独立贡献的
严格基线。每个 session 必须记录 copy policy、runner source/binary hash、bitstream、runtime 和
boot id。两种 B0 不能混合样本，也不能只报告收益较大的 `B0-historical`。当前同一 boot 的观察值
分别是 `8.81%` 和 `2.83%`；只有 `B0-optimized` 的 `2.83%` 进入三 boot 配对 95% 区间验收。

每个上板子阶段使用同一 stage、输入、threads、poll、bitstream 和 affinity，每 boot `5 warmup +
20 scored`，采用 B0/B1/B2 交错顺序抑制温度漂移，正式结论至少三个独立 boot。报告：

```text
per-boundary materialization bytes/time
CPU shared-mapping direct-read/direct-write penalty
HPC configuration fingerprint, software cache-maintenance calls and slot wait time
CPU core-ms and VTA LOAD/STORE/submit/sync
single-frame latency, Pipeline II and FPS
shared-slot bytes, peak slots and rejected contracts
```

以下 gate 全部满足才称为端到端 stage-boundary zero-copy：

1. A/B 交替输入的独立 reference、逐帧 determinism 和 generation 对应全部通过，无旧帧、覆盖、
   悬空 view、死锁和隐式 fallback。
2. 目标 stage 边界的 Host/VTA materialization bytes 降为 0，或降到预注册的不可消除
   adapter bytes。VTA 从 DDR 到片上 SRAM 的 LOAD/STORE 仍存在，不计入“已消除 copy”。
3. HPC/coherent 主线的 `SyncForDevice/SyncForCpu` 调用必须为 0；线程交接与 VTA completion wait
   不得被误计为 cache maintenance。
4. 只有 `Delta II = II_B0 - II_B2` 或 FPS 改善的跨 boot 95% CI 排除 0，才声明吞吐提升。
   若只降低了 copy bytes/单帧 latency，则只报告对应收益。
5. 零拷贝不得使 producer/consumer 的 CPU `run_ms` 因共享 mapping 内存属性而退化到抵消
   materialization 收益；否则停止 B2，保留普通拷贝为该边界的更优策略。

#### P8.6 共享内存分配后端选择

TVM 没有一个适用于所有 target 的“官方 ION 分配器”。本仓库 VTA PYNQ driver 使用的是
PYNQ-specific CMA API；ION/rpcmem 主要出现在 Android/Hexagon 路径，与本板 VTA 不是同一
baseline。ION 已是 Android 的 legacy 接口，V1 不为使用旧接口而改成 ION。

| 后端 | 优点 | 当前限制 | V1 决策 |
|---|---|---|---|
| 当前 `u-dma-buf` | mmap 简单；提供 VTA 所需物理地址；已在本板 HPC 路径运行 | 固定大块；当前 `UdmabufPool::Free` 为 no-op；版本/能力未进 manifest；直接物理地址方式不利于 IOMMU/多进程隔离 | **保留为主线** |
| VTA PYNQ `/dev/cma` | 动态物理连续分配；已有 VTA driver 接口 | 板级定制 CMA module/API，不是统一上游 userspace ABI；需要重新移植和验证 cache 语义 | 只作历史实现对照 |
| ION | heap/dma-buf fd 曾广泛用于 Android | vendor flags/heap id 不统一，Android 已迁移到 DMA-BUF heaps | 不采用 |
| Linux DMA-BUF heaps (`cma` heap) | 上游稳定 UAPI；fd 便于跨驱动/框架共享和权限隔离 | 当前板 kernel 是否支持未知；VTA userspace driver 还需可靠 importer/DMA address，不能把 fd 直接当物理地址 | 条件替代实验 |
| XRT buffer object | allocation/import/sync 生命周期完整 | 需要 XRT shell/runtime，改变当前 VTA/PetaLinux 栈与论文变量 | V1 不采用 |

因此近期优化放在 `UdmabufPool` 上层，而不是先换 kernel allocator：

1. preflight/manifest 固定记录 module version、size、physical range、`dma_coherent`（若可用）、
   mmap/sync mode 和 `AXU5EVB_DRIVER_SAFE_COPY`。
2. 使用离线 slot plan 一次预分配并跨帧复用；只在所有 Executor 静止时允许 pool reset。当前
   单候选单进程不需要通用 free-list，多候选常驻同一进程时再实现回收。
3. typed allocation 返回 `{vaddr,paddr,size,alignment,edge_id,slot_id}`，所有绑定先做范围、重叠、
   dtype、shape、layout 和 generation 检查。
4. HPC 主线不执行每帧 sysfs sync。逐字节 safe-copy、sysfs sync 或 `msync` fallback 都只能作为
   调试/非一致性路径，不能进入最快正确 baseline。
5. 只有出现以下任一实测问题，才启动 DMA-BUF heap 对照：pool 容量浪费阻塞模型、长期进程
   因 no-op free 耗尽、需要跨进程共享 fd、需要 IOMMU/访问隔离，或 u-dma-buf CPU mapping 性能
   显著劣于等价 CMA heap。allocator 对照不与 zero-copy/双 slot 同轮混改。

P8 产物固定为 `v1_p8_protocol.json`、`v1_p8_local_abi_audit.json`、
`v1_p8_session_<boot>.json`、P8A/P8B 每个 edge/control 的 `v1_p8*_raw_*_<boot>.jsonl`、
`v1_p8_ablation.json` 和 `v1_p8_review.md`。P8 已在 P7D review 后进入 `current`；不改变切图、
tile、CPU threads 或原 Top-20，先只在固定候选上归因内存管理收益。

P8C-L 的产物按普通复制策略隔离保存：`three_island_sessions/` 保留显式 `memcpy` 结果，
`three_island_sessions_runner_default/` 保存历史 runner 一致性结果。每个目录独立生成 protocol、
raw JSONL、runtime profile、session、cross-boot summary 和 review；cross-boot loader 按
`ordinary_copy_policy` 过滤，禁止把两种基线当成独立 boot 合并。

## 6. 当前下一步

P7A--P7D、P8A 和 P8B 已收口。P8C 已完成双 slot 状态机、AArch64 编译、本地测试和三个独立 boot
的 B0/B1/B2 qualification，当前停在 `P8C completed/throughput claim rejected`。三个 boot 的
功能 gate 均通过。按完整 block 窗口重算后，B2 相对 B0 的 II 差值分别为
`-0.679958/+0.270163/-1.261018 ms`，相对 FPS 差值为 `-0.74%/+0.29%/-1.37%`；II 差值均值为
`-0.556938 ms`，配对 Student-t 95% 区间为 `[-2.477096, 1.363220] ms`。区间包含 0，P8C 正式
吞吐 gate 不通过，旧的 `+3.90%` 结论由错误的逐帧间隔中位数产生，现已废止。

组件归因同时通过：B0两条异构边界的框架物化/API服务时间平均为`1.446899 ms/frame`，B2同址
绑定/发布平均为`0.041830 ms/frame`，减少`1.405069 ms`（`97.11%`），其95%区间为
`[1.368642, 1.441496] ms`；framework materialization由`1806336`降至`0 bytes/frame`（100%）。
这不是全部DDR通信时间：VTA内部`LOAD 12025856 + STORE 1229312 bytes/frame`保持不变。

跨 boot 统计以每个 boot 的 `B0-B2 session-median block-window II` 为配对效应，帧和同 boot block
不作为独立重复；达到至少三个 boot 后使用 `paired_boot_student_t_95_v1` 计算均值的 95% 区间。
原始 session 不修改，修正结果由其原始 JSONL 和 SHA256 重建到派生汇总中。
P8C 吞吐线在此冻结并先提交 review，不自动进入 P8D。VTA->CPU producer slot wait 的跨 boot 中位数为
`18.541/16.229/0.001 ms`，说明背压存在但强度并不稳定，不能据此加入固定penalty。只有后续证据
能把剩余关键路径明确归因于adapter或slot调度时，才进入P8D的adapter-direct-write或最小slot
调度优化；当前不增加第三个 slot，因为这只能吸收瞬态队列，未证明能改善由最慢 stage 或共享资源
决定的稳态吞吐。P7D 未通过的绝对 FPS gate 作为公开限制保留，不由 P8 结果反向改写。

作为与稳态吞吐正交的补充，P8C-L 已完成三 VTA-island stress case 的第一个单帧串行 boot：
在同一 7-stage schedule 下，单 slot zero-copy 将 6 条边界的框架物化从 `9.031680 MB/frame`
降为 0。最快正确 `memcpy` 基线下，边界 API 服务 `4.484 -> 0.111 ms`，单帧中位时延
`167.399 -> 162.662 ms`（`2.83%`）；与历史 runner 默认安全复制一致时，对应结果为
`16.324 -> 0.111 ms` 和 `178.855 -> 163.093 ms`（`8.81%`）。当前只允许声明本 boot 的
qualification 观察值和组件字节/时间减少；完成至少两个新 boot 前，两项百分比均不是正式稳定提升。
用户确认后于 2026-09-06 追加了 **P8 Top-20 单 boot 外部有效性扫描**。该扫描不是重新拟合，也不
改动 Top-20；在同一 boot、同一输入和同一候选内，按奇偶 rank 反转的
`B0/B2/B2/B0` 或 `B2/B0/B0/B2` 顺序比较普通拷贝与双 slot zero-copy，每个 block 为
5 次 warmup 和 10 个计分帧。执行中发现 rank 5--12 的合法切点会跨越残差 tuple，原单张量 slot
无法运行；runner 已扩展为一个逻辑 slot 原子管理多个 u-dma-buf tensor view，并记录每个 tensor
的字节数和物理地址。三 stage、五 stage、单/双 VTA island 共 20 个候选均通过输出等价、双 slot
轮换、地址对齐/不重叠、frame/generation owner 和终态释放 gate。

组件结论具有一致性：B0 到 B2 的边界 API 服务时间减少均值为 `96.59%`、中位数为 `96.96%`；
根据切点不同，每帧消除的 Executor 间 framework materialization 为
`1806336/2007040/2207744/2609152 bytes`。按完整窗口重新分析后，
端到端 FPS 增量均值为 `-1.20%`、中位数为 `-1.09%`，范围为 `-8.51%--+3.27%`，仅 `6/20`
为正；`12/20` 的 B0/B2 block spread 均不超过 5%，`14/20` 的两次配对方向一致。rank 1 为
`-1.03%`，所以旧扫描中按逐帧间隔中位数得到的结果全部废止。

因此当前只声明：slot runtime 正确支持 Top-20 的单张量和残差多张量边界，并稳定消除了框架边界
物化工作；不声明 zero-copy 对任意切图都提高端到端 FPS。copy 若不在关键流水线工位上，其消除可
被其他 stage 隐藏。若论文需要 Top-20
吞吐泛化结论，下一步应先降低同模式 block 波动，再从四种唯一 topology 各冻结代表做至少三个
独立 boot，而不是把同 boot 的 20 个候选当作 20 个独立统计样本。完整产物位于
`v1_p8_top20_zero_copy/`。

2026-09-06 对 Top-20 中旧结果最稳定为负的 rank 20 进行了 VTA mutex 计时审计。审计发现 runner
原有口径不对称：B0 在 `stage_ms/run_ms` 开始前获取全局 VTA mutex，B2 则在 stage 计时开始后
获取，导致 B2 后一个 VTA stage 的锁等待被误报为 `set/run` 服务，而 B0 的同类等待被隐藏。
runner 现已分别输出 `vta_mutex_wait_ms`、`vta_mutex_held_ms`、`vta_run_call_ms` 和包含等待的
`scheduled_service_ms`；边界 API 统计只使用纯 `set/get` 时间。

修正观测后，同一 boot 对 rank 20 使用每模式两个 block、每 block 10 次 warmup 和 50 个计分帧
复测。完整窗口重分析得到 B0/B2 II 为 `95.023/95.150 ms`，相对 FPS 为 `-0.13%`；第二个独立
boot 为 `94.523/94.973 ms`，相对 FPS 为 `-0.47%`。旧 `99.247/97.008 ms` 与 `+2.31%` 来自
逐帧完成间隔中位数，已经废止。每帧消除 `2609152 bytes` framework materialization，边界 API 从
`2.176 ms` 降至 `0.115 ms`，减少 `94.72%`。两个 VTA stage 的真正 `run` 调用在 B0/B2 中分别
约为 `49.63/49.69 ms` 和 `20.11/20.15 ms`，VTA profiler 的 instruction、LOAD/STORE bytes 也
完全一致，因此没有证据表明 zero-copy 使设备计算或必要 DMA 变慢。

两个 boot 的完整窗口结果都接近 0，说明边界 copy 节省被更慢的流水线工位覆盖。当前可以撤销
“zero-copy 导致 VTA 执行变慢”的解释，但不能把组件服务减少解释为端到端吞吐提升。正式扩展
结论仍需按唯一 topology 选择代表完成独立 boot；原始诊断分别保存在
`v1_p8_mutex_diagnosis_rank20/` 和
`v1_p8_mutex_diagnosis_rank20_v2/`，前者的边界 API 汇总因旧计时口径无效，仅保留 FPS 和原始
时间线作为审计证据。

为降低短窗口误差，当前 boot `1617c571-1b81-41d9-9eff-b7224dd9d5c4` 又对 rank 1 和 rank 20
执行了每模式两个 block、每 block 10 次预热和 100 帧计分的长窗口复测。rank 1 的 B0/B2 II 为
`91.684/92.024 ms`，相对 FPS `-0.37%`；rank 20 为 `96.734/96.548 ms`，相对 FPS `+0.19%`。
两个候选的两次配对 block 均一正一负，不能形成方向一致的吞吐结论。边界 API 服务仍分别减少
`95.96%/93.24%`，VTA `run` 基本不变；因此当前最合理的解释是边界物化已被消除，但它未处于
这两个候选的稳态关键路径，约正负 1% 的端到端差异属于系统波动。产物位于
`v1_p8_representative_long_boot_1617c571/`。

2026-09-06 当前健康 boot 的 P0 preflight 已通过：SSH/RPC 使用 `.247`，`u-dma-buf=192 MiB`，FPGA
状态为 `operating`；`/mnt/sd` 实际解析到 `/media/sd-mmcblk1p2`，剩余约 `298.6 MiB`，当前 dmesg
未出现 EXT4/I/O 错误。P7 临时 package 和输出位于 `/var/volatile/ramps_p7`，不继续消耗 SD。
此前 ext4 损坏记录仍要求保留防护：部署器会在首次远端写入前解析目标文件系统并检查当前
boot 日志；错误再次出现时立即停止写入并离线 `e2fsck` 或更换 SD 卡。

已确认主机备份位于 Windows
`C:\Users\orange\Desktop\sd_backup_parts`（WSL：`/mnt/c/Users/orange/Desktop/sd_backup_parts`）：

```text
partition_table.sfdisk  208 B       SHA256 4410688ad5969551ce602664cc72dd19d89dced3f7cc79161847ad14e3f66911
boot_sde1.img           1123024896 B SHA256 21330c107ce058c90a8c5b28c84ff170f74d6b0f8daca4c23ecd5d841b42a3a6
data_sde2.img           2202009600 B SHA256 7e9bc764c4d95d7fdc597ba6e874e9cea04d455da77fb7933a878e364e2b0bad
```

两个镜像的字节数与备份分区表逐扇区一致，FAT32 boot 和 ext4 data 签名可识别；备份中的根文件树
及当前故障涉及的 inode 29 可读取。但 `data_sde2.img` 的只读 `e2fsck -fn` 报告保留 resize inode 7
异常，因此它是可恢复来源，不是已经证明完全健康的黄金镜像。恢复时必须保留原镜像不动，先制作
工作副本，在副本上完成 `e2fsck` 修复和文件抽查，再写入新卡并执行 P0 preflight。不得在唯一备份
上直接运行写修复，也不得把旧卡错误和备份镜像的检查结果混为一谈。

### 2026-09-06 计划复审结论

继续完整原计划的收益已经很低，现收缩为“证据闭环”，不再扩展机制：

1. **保留且停止重复 `runner_default`。** 它已经把当前切法的 `16.324 ms` 与历史 `17.239 ms`
   对齐，解释了三-island历史中位数 `11.366 ms` 的来源；该路径本身慢于正确 `memcpy`，不应作为
   zero-copy 主要性能 baseline，也不值得再占用两个 boot。
2. **`memcpy` vs single-slot zero-copy 的 boot 2 和 boot 3 降为可选复核。** 当前可以把
   `2.83%` 严格写成“一个 boot 的三-island stress-case观察值”，而把已有三 boot 的
   `97.11%` 边界 API 服务减少和 100% 物化字节消除作为正式组件证据。只有论文需要把
   `2.83%` 升格为跨 boot 端到端时延结论时，才补两个 boot。
3. **P8C 吞吐实验永久收口。** 三 boot 已证明边界 API 服务减少 `97.11%`，但 II/FPS 没有显著
   改善；不再重复 Top-20，不通过增加 island、分辨率或样本量制造吞吐收益。
4. **P8D 暂停，P8E 降为可选增强。** 当前没有 adapter/slot scheduling 位于关键路径的证据，继续
   P8D 缺少问题驱动。YOLOv3-tiny 只在论文评审明确要求第二网络 runtime 迁移性、且时间允许时做
   一个代表边界的正确性与物化字节验证，不重新展开大规模性能搜索。
5. **P8C-L 收口后转入论文证据冻结。** 静态切图方法只声明能在 972528 个配置中产生并精确求解
   预算化 shortlist，且自然 Top-20 落入 `10.220--11.511 FPS` 高性能区域；同时公开
   `Spearman=0.155`、`regret@5=5.419%` 和 P7D 细排 gate 未通过，不再追加无法改善排序的 profile。

因此 P8 当前冻结，不再要求 P8C-L boot 2。下一步转为论文证据整理：将“串行 latency 可累加受益、
流水线 throughput 仅在 boundary 位于关键周期时受益”写成问题、模型、实验和负结果闭环，并明确
区分三类结论：跨三 boot 的组件收益、单 boot 的三-island时延观察、跨三 boot 未发现吞吐提升。
若后续评审要求端到端时延置信区间，再恢复两个 `B0-optimized memcpy` boot；FIFO、tile、Max-Plus、
P8D 和 runner-default 重测均不启动。
