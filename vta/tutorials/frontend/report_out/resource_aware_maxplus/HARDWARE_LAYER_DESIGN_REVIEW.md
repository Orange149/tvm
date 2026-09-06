# CPU-VTA Pipeline V1 设计与实施计划

更新日期：2026-09-06

> 执行状态以 [RAMPS_EXECUTION_ROADMAP.md](RAMPS_EXECUTION_ROADMAP.md) 为准。V1 已明确放弃先完成
> 212-case/80-case 完整 HardwareProfile 的路线，固定 tile 后先使用加性 CPU/VTA 服务、复合边界和
> DDR 下界验证切图排序价值。2026-09-04 已在不读取本轮吞吐进行排序的前提下，把 Iteration 2
> 自然 Top-20 全部上板：20/20 correctness 通过，实测 `10.220--11.511 FPS`。算法成功筛到高性能
> 区域，但 Top-20 内静态/实测名次 Spearman 只有 `0.155`，说明细排仍受 CPU 线程并发效率、共享
> DDR contention 和 runtime 固定开销遗漏影响。历史 `+34 ms` 把本轮 cycle MAPE 从 `23.523%`
> 降到 `12.859%`，但本轮残差中位数只有 `22.353 ms`，故它仍是历史诊断而非硬件常量。下一阶段
> 固定 tile，使用受控组件实验辨识这些开销，不用本轮候选吞吐反推参数。P7B-1 三个独立 boot
> 已完成：每个合格 session 均为 36/36 组 memory case 和 16/16 组 CPU pair 通过；三轮 memory
> 最小相关性为 `0.994`，CPU slowdown A/B 最小相关性为 `0.974/0.965`。该结果准入 27 个已测
> side-specific 观测，但尚不能外推成覆盖全部 DP 候选的 slowdown surface。P7B-2 单 boot 的
> 9/9 个 runtime/boundary case 也已通过：它排除了普通 host 队列、1 us poll 和 boundary copy
> 是 22--34 ms 残差主因的假设，并发现短 VTA segment 需要 shape-aware service。P7C 单 boot
> 4/4 组 matched control 已通过：全线程核隔离后，CPU 有一个来自 cache control 的 `1.065x`
> 信号，不能归因为 DDR；VTA 在两组 streaming 压力下出现 `1.052x/1.066x` slowdown。它们尚未
> 通过三 boot 准入，也不能解释 22--34 ms 残差。P7D 随后只准入三 boot streaming CPU
> read/write 带宽，完整枚举核对后 Top-20 未变，正式模型仍定位为 ranking-only。P8A 已完成一个
> `802816-byte` CPU->VTA 边界的单 slot 串行验证：正确普通 `memcpy` 路径为 `0.714217 ms`，
> zero-copy 将 framework materialization 从 `1605632 bytes` 降到 `0`，配对 latency 差值中位数
> 为 `+1.157847 ms`。P8B 又完成两方向串行 matched control：两条边界的 framework
> materialization 均降到 0，且 CPU 直写/直读共享 mapping penalty 没有抵消普通 copy；该结果
> 尚不是双缓冲流水线 FPS 证据。P8C 的三个 boot 表明双 slot 对当前 Top-20 的稳态 FPS 无显著
> 改善，因此又补充了不含流水线重叠的 P8C-L 单帧实验：冻结三 VTA-island、6 条异构边的 stress
> case 后，首个 boot 的最快正确 `memcpy`/single-slot zero-copy 中位时延为
> `167.399/162.662 ms`，边界 API 服务为 `4.484/0.111 ms`，每帧消除 `9.032 MB` 框架物化。
> 为对齐历史 runner，又撤销 `SAFE_COPY=0` 覆盖并补测默认安全复制：对应结果为
> `178.855/163.093 ms` 与 `16.324/0.111 ms`。历史一致口径的 `8.81%` 和保守 memcpy 口径的
> `2.83%` 都只是单 boot qualification，需至少三个独立 boot 后才能作为正式时延提升。

当前绝对值诊断采用 `T_corrected(p)=T_static(p)+b` 和
`FPS_corrected(p)=1000/T_corrected(p)`。全部 200 条历史记录统一加 `34 ms` 后，cycle/FPS MAPE
分别为 `5.896%/5.858%`；双向 100/100 跨批次拟合得到 `b=36.459/33.764 ms`，对侧 cycle MAPE
为 `5.535%/6.046%`。这说明静态公式可能遗漏了一组近似稳定的调度、submit/sync、cache 维护或
调用启动时间，但尚未证明存在一个物理意义明确的 `34 ms` 常数。相同 `b` 不改变任何候选的相对
顺序，因此不能改善 Top-20、recall 或 regret；它只校准绝对周期。正式模型只能加入由独立组件
profile 可辨识、所有权唯一且通过 holdout 的开销，不能要求所有残差都必须被拆成正的固定项，
也不能继续由 200 个完整候选标签反推。P7B-2 已证明普通 per-frame/per-stage intercept 不是主因。

最新自然 Top-20 的 measured-pool oracle 是静态 rank-9 的 `11.511 FPS`；这里的 oracle 只表示这
20 个已测候选中的最好值。按冻结静态顺序，`regret@1/3/5=5.419%`，第 9 次测量首次达到该池
oracle 的 95%。这些数值来自单 boot/session，不能称为完整搜索空间的全局最优或最终统计结果。

## 1. 当前决策

第一版不联合搜索 TVM/VTA tile，也不建立完整 RAMPS HardwareProfile。先实现并验证下面这个
最小闭环：

```text
PartitionTuner 式 compiler-valid segment/profile
+ Tarnawski/DNNPipe 式 Pipeline 最大负载目标和 DP
+ Synergy/NEURAghe 式 SoC 跨帧执行
+ 单物理 VTA 串行约束、每个 CPU stage 独立的 TVM threads 和四核 core-work 容量下界
+ CPU-VTA 复合边界和共享 DDR 聚合资源下界
+ Top-K 原生编译与上板验证
```

V1 的目标不是精确复原所有硬件时序，而是回答：

> 能否通过少量、可复用的局部 profile 和一个离线划分算法，从大量合法切图中选出很小的
> Top-K，并在明显少于穷举的上板次数内找到接近最优吞吐的方案？

下列变量在 V1 中固定并写入实验指纹：

```text
VTA schedule 和 tile
量化与 graph packing
compiler pass 和 fusion policy
FIFO depth
runtime poll/sync policy
HPC/coherent bitstream、runtime 与时钟
```

板端部署直接遵循 `apps/vta_rpc/README.md` 和 `apps/vta_rpc/axu5evb_board_layout.md`，启动入口固定为
`/mnt/sd/tvm_deploy/start_axu5evb_hpc_rpc.sh 9090`。已有 HP/HPC 对照已经选择 HPC/coherent 为主线，
V1 不重新探索部署目录或重做 bitstream 选择。

tile 联合优化是 V2 扩展。只有 V1 已经跑通，而且固定 tile 被证明是主要排序误差来源时，才启动。

### 1.1 当前开发板端点

```text
SSH: root@192.168.1.247
TVM RPC: 192.168.1.247:9090
```

所有新生成的 profile、compile、reference 和 Pipeline 命令都必须显式记录该端点，不能依赖旧脚本中
可能残留的默认 IP。端点属于 run manifest 和连通性证据；稳定硬件身份仍由板卡序列/配置、bitstream、
时钟、compiler 和 runtime 指纹确定。仅 IP 变化不应使同一物理硬件的历史测量失效，历史产物中的旧
IP 也不得改写。

## 2. 相关工作定位与研究边界

不能只与 Tarnawski 做一对一比较。当前问题位于“DNN 图放置、SoC CPU-FPGA 协同、跨帧流水、
共享内存竞争、编译器合法性”五条研究线的交集。下表只使用与本项目方法边界直接相关的工作。

### 2.1 直接相关工作矩阵

| 工作 | 硬件和目标 | 已解决的部分 | 对当前问题仍缺少的部分 |
|---|---|---|---|
| Tarnawski et al. [R1] | 多设备 placement；延迟或离线 Pipeline 吞吐 | 节点/通信 cost，非连续 placement，IP/DP 最优算法 | 不描述 Zynq 共享 DDR、单 VTA 复用、每-stage TVM threads 和 TVM/VTA lowering |
| PartitionTuner [R2] | ZCU102，4 核 ARM + FPGA EVTA；单次推理延迟 | individual/group operator profile，fusion-aware DP，DAG 分支跨 CPU/NPU 并行 | 顺序分支仍顺序执行；不优化跨帧 II，不建模多 CPU stage 并发和共享 DDR |
| NEURAghe [R3] | Zynq ARM + FPGA CNN engine；端到端吞吐 | CPU 执行难加速层，FPGA 执行卷积，真实多帧 Pipeline 和 data marshaling | 设备分工由系统设计固定，没有对任意 compiler-valid 切图做预算化搜索 |
| Synergy [R4] | Zynq ARM/NEON + FPGA；高吞吐 CNN | 跨帧 HW/SW Pipeline、FIFO/mailbox、tile work stealing、固定 bitstream 复用 | 主要按算子类型固定角色并在卷积 tile 内平衡，不搜索任意 CPU/VTA segment 和 per-stage threads |
| CoDL [R5] | 移动 SoC 统一内存 CPU+GPU；单次推理延迟 | 显式考虑同步、mapping、dtype/layout 转换及并发非线性 | 是单算子协同分块，不是单 FPGA mutex 下的跨帧图划分 |
| HaX-CoNN [R6] | 共享内存 CPU/GPU/DSA SoC；并发多 DNN | layer/group mapping、跨加速器转换、共享内存 contention、SAT 最优调度 | 目标是多个独立 DNN 的并发延迟/总吞吐，不是同一 DNN 的 CPU-VTA 跨帧 stage Pipeline；无 FPGA lowering |
| DNNPipe [R7] | 多个异构 IoT 节点；跨设备 Pipeline | 最大 stage time 目标、DP 和保持最优性的 pruning | 公开 conference formulation 假设相邻通信小于最大 stage time；没有片上共享 DDR、单 VTA 和 host adapter |
| Rios-Navarro et al. [R8] / SECDA [R9] | Zynq/PYNQ CPU-FPGA 数据通路和 HW/SW co-design | 证明 packet、driver、DMA、数据准备和 unpack 可能主导开销 | 不提供 DNN 图 placement 和跨帧 stage 搜索 |

这张表带来两个结论：

1. “CPU-FPGA 切分”“Zynq 上的跨帧 Pipeline”“共享内存竞争”“layout/sync 通信代价”均已有工作，
   不能单独声称为创新。
2. 在本次审查的工作中，尚未发现一个方案同时覆盖：**TVM/VTA compiler-valid segment、单物理
   VTA 串行复用、多个 CPU stage 的 TVM threads、共享 DDR 约束、跨帧稳态吞吐和低预算 Top-K
   上板验证**。这是 V1 可以验证的研究交集，而不是无条件的“首次”声明。

### 2.2 从已有方法组合出 V1

V1 不重新发明上述基础方法，而是明确继承并修改：

```text
PartitionTuner：grouped operator/segment cost、fusion-aware compiler legality
Tarnawski：设备负载表示、通信边和 placement DP/IP 基础
DNNPipe：最大 stage time 的 Pipeline 目标和可证明 pruning 思路
Synergy/NEURAghe：跨帧执行语义、单板 CPU/FPGA 协同和真实 runtime
CoDL/Rios/SECDA：边界不能只用 tensor bytes，必须包含转换与同步
HaX-CoNN：共享内存竞争必须作为独立资源或 slowdown 进入验证
```

在此基础上，V1 只增加当前硬件必需的状态和约束：

1. 多个逻辑 VTA island 共用一个物理 VTA，不能当作多个独立加速器。
2. 每个 CPU stage 的 TVM threads 是独立候选变量；不同 stage 的 threads 不构成静态核心配额，
   但所有 host 工作的 `core-ms/4` 必须形成共享四核容量下界。
3. CPU、VTA LOAD/STORE、DMA 和 host adapter 竞争同一 DDR 系统。
4. CPU-VTA 边界包含实际发生的 layout/quantization、copy、cache、submit 和 sync。
5. 候选 segment 必须通过真实 TVM/VTA lowering、native build 和 correctness gate。
6. 最终价值由 Top-K 上板命中率和总实验成本证明，而不是由模型复杂度证明。

### 2.3 当前 SoC 的具体差异

当前 SoC 的关键差异不是“FPGA 型号不同”，而是同一颗 SoC 内的资源所有权和执行目标不同：

| 项目 | 通用 placement / 分布式 Pipeline | 当前 CPU-VTA SoC |
|---|---|---|
| 加速器 | 多个可独立占用的设备或节点 | 一个物理 VTA，全部 island 共用 mutex |
| 主存 | 设备内存或网络通信通常分别计价 | CPU、VTA 和 DMA 共享 DDR 与仲裁路径 |
| CPU | 节点 cost 或固定线程 policy | 每个 Pipeline stage 有独立 TVM threads，实际并发时仍共享 4 核 |
| 通信 | tensor transfer cost | adapter、set/get、cache、一到多次 DMA、VTA LOAD/STORE |
| 子图成本 | 节点代价或已 profile group | TVM fusion/materialization 使 segment 可能非可加 |
| 目标 | 单次延迟、多 DNN 并发或多节点吞吐 | 同一 DNN 的多帧稳态 II，单 VTA 被时分复用 |
| 可行性 | 图依赖与设备容量 | 还受量化、layout、VTA backend 和 native runtime 约束 |

### 2.4 SoC 问题一：共享 DDR 导致资源耦合

独立 profile 得到的 `T_cpu`、`T_vta` 和 `T_boundary` 不能直接证明并发时仍保持不变。在稳态
Pipeline 中，不同帧的 CPU stage、VTA LOAD/STORE 和 adapter 可能同时访问 DDR：

```math
T_{stage}\neq\max(T_{cpu}^{isolated},T_{vta}^{isolated},T_{comm}^{isolated})
```

一般情况依赖调度、每帧 DDR traffic、DMA overlap 和仲裁。V1 不拟合完整黑盒函数 `F`，但必须
保留每笔 DDR transaction 的唯一 ownership，并加入“每帧总 DDR demand / 可持续带宽”的资源
下界。若并发 matched control 证明该下界仍系统性误排，再进入 V2 contention 模型。

### 2.5 SoC 问题二：通信不是 tensor bytes 除以 DMA 带宽

同为 1 MiB 的边界，连续传输、64 次小传输、layout 转换、量化、cache 状态和共享 buffer 复用
可能产生不同时间。正式边界 signature 至少记录：

```text
CPU->VTA / VTA->CPU
host adapter 与 set/get copy
physical bytes、DMA calls、row/pitch/padding
layout、quantize/dequantize、slice
cache flush/invalidate 是否由当前端口和 runtime 实际执行
VTA lowered LOAD/STORE 的 transaction ids
```

cache maintenance 只有在当前 PS-PL 端口和 runtime 路径中被实际调用或测得时才计费；不能因为
理论上可能存在就手写 penalty。中间结果能否留在共享 buffer 或 VTA 片上，也必须由 lowered
TIR/runtime manifest 证明。

### 2.6 SoC 问题三：连续子图代价可能不具有可加性

V1 的搜索对象是连续 segment，而不是互相独立的单算子标签。对设备 `d`：

```math
T_d(S)\neq\sum_{u\in S}T_d(u)
```

可能的原因包括 fusion、内部 store/load 消除、padding、buffer 不足和固定 schedule 的重新分块。
V1 仍固定 tile，但 DP 的 transition API 必须接受 `Cost(S,d,config)`；单 unit 求和只是初始估计。
每个高排名 segment 必须有 compile-only lowering manifest，完整 stage holdout 用于判断是否需要
低维 segment correction。tile 作为自由变量继续延后到 V2。

### 2.7 可防守的贡献与判定条件

V1 不把“硬件不同”、标准 DP、TVM 已有 zero-copy API 或双缓冲本身写成创新。论文主线收敛为
两项系统贡献，每一项都绑定实验判据：

| 主贡献 | 与已有工作的区别 | 成立条件 |
|---|---|---|
| C1 共享内存感知的静态切图评估与预算化筛选 | 对 compiler-valid CPU/VTA segment 和 per-stage TVM threads，联合计算隔离 stage service、单 VTA 串行负载、CPU core demand、边界 layout/量化/复制以及共享 DDR demand，再用 k-best DP 输出少量 Top-K | 不读取候选吞吐拟合参数；相对 compute-only、忽略转换和 PartitionTuner-like 基线改善冻结集的 Top-K/regret；native runtime 执行同一 plan，并在第二个网络验证迁移性 |
| C2 固定 CPU-FPGA 流水线的共享 slot zero-copy runtime | RIMMS 面向动态 PE/location；TVM Pipeline Executor 当前源码采用 owning/deep-copy forwarding，但本文不推断其设计动机。本项目利用运行前已知的设备映射和共享 DDR，以 manifest 生成有界 slot，跨 Executor 绑定同址 CPU/VTA view，并用 generation/owner/completion 管理双缓冲 | 对比最快正确的普通 copy、手工单-slot zero-copy 和双-slot pipeline；逐帧 reference 通过，边界 materialization bytes 降为 0 或仅余必要 adapter，且多 boot Pipeline II/FPS 显著改善；在第二个网络验证 |

其中 DP、双缓冲和 `set_input_zero_copy/set_output_zero_copy` 都是实现手段，不单独作为创新。
若后续提出区别于标准 DP 的状态压缩、可证明下界或 pruning，并给出正确性/复杂度分析，才可追加
“搜索算法贡献”；否则只称为针对上述成本与约束的 k-best DP 实例化。CPU 并发、DDR contention
和 island 数量规律是 C1 的解释与消融结果，不再拆成多个主贡献。

C1 与 C2 通过同一份切图 manifest 形成闭环：规划器给出 stage、边界 tensor contract 和
zero-copy eligibility，runtime 据此生成 slot 与生命周期；组件实验测得的复制、转换、DDR demand
和 slot-wait 再进入成本模型。可防守的系统点是这种 planner/runtime 协同，而不是简单拼接两个技巧。

论文不得声称“首次 CPU-FPGA Pipeline”“首次考虑共享内存”或“首次考虑转换代价”。更稳妥的
主张是：

> 面向单 VTA、共享 DDR 的嵌入式 SoC，提出共享内存感知的 CPU-VTA 流水线切图评估与预算化
> Top-K 筛选方法，并针对静态设备映射设计基于固定 slot、双缓冲和跨 Executor 指针绑定的边界
> zero-copy runtime，在保证多帧生命周期正确的前提下减少框架复制。

如果 C1 的通信/共享内存项不能改善冻结排序，不能声称新的资源感知评估方法；如果 C2 只减少
copy bytes 而不改善 Pipeline II/FPS，只能报告内存流量优化，不能声称吞吐优化。任一项未完成时，
论文应相应缩小贡献范围，不用另一项结果替代其证据。

## 3. 决策变量与约束

将量化后的 ResNet18 转换为有序、不可再切的 compute units：

```text
U = (u_1, u_2, ..., u_n)
```

一个 V1 候选为：

```math
p=(z,cuts,t),
```

其中：

- `z`：每个 unit 放在 CPU 或 VTA。
- `cuts`：相邻 units 是否合并为同一 stage。
- `t_j in {1,2,3,4}`：第 `j` 个 CPU stage 传给 TVM runtime 的 threads 参数。

各 CPU stage 的 `t_j` 独立选择，不存在 `sum(t_j)<=4` 约束。还需满足：

```text
相邻同设备 stage 规范化合并
VTA island 数不超过预注册上限 R
所有 VTA unit 使用同一冻结 schedule/tile policy
tensor contract、量化、layout 和 native lowering 合法
每个 CPU stage 独立使用 [0,threads) affinity；不同 stage 的 mask 允许重叠
```

这里 affinity 只用于固定同构核心上的测量条件，不表示各 stage 独占这些核心。V1 累加串行测得的
process CPU time，并同时使用四核物理池下界和前缀 mask 下界；额外 cache、调度和效率损失需要
另行测量，不能通过相加 threads 近似。

## 4. V1 成本模型

### 4.1 CPU stage

对 CPU stage `S_j` 和 TVM threads `t_j`，DP 查询 segment cost：

```math
\widehat T_{cpu}(S_j,t_j)
=Cost_{cpu}(S_j,t_j).
```

第一版用 `sum(unit cost)+stage intercept` 初始化 `Cost_cpu`。若 compile manifest 或完整 stage
holdout 证明不可加，只对 DP 产生的少量高排名 segment 增加 correction，不建立全 shape surface。

threads 不是核心配额，但所有 CPU stage、VTA host 调用和 boundary adapter 最终共享四个核心。
因此还保留串行 profile 的 process CPU work：

```math
\widehat D_{core}(p)=
\max\left(
\frac{\sum_s \widehat W_{host}(s)+\sum_e \widehat W_{boundary}(e)}{4},
\max_{k=1..4}\frac{\sum_{s\in CPU:t_s\le k}\widehat W_{cpu-stage}(s)}{k}
\right).
```

第一项是四核总 work-conservation；第二项逐个检查嵌套前缀 mask：所有 `t_s<=k` 的 stage 都只能
在前 `k` 个核心内完成。当前 affinity mode 允许 worker 在 `[0,t_s)` 内迁移，所以不把 core-ms
固定均分到具体核心。这仍是容量下界，不是假设各 stage 的 threads 互斥，也不等价于
`sum(threads)<=4`。各前缀累计 work 只增不减，可直接用于 DP 的单调剪枝下界。

### 4.2 单物理 VTA

所有 VTA island 使用同一个物理 VTA mutex，因此每帧的 VTA demand 为：

```math
\widehat T_{vta-total}(p)=\sum_{k\in VTA\ islands}\widehat T_{vta}(S_k).
```

`T_vta(S_k)=Cost_vta(S_k,tau0)` 是 segment-level cost。它使用冻结 schedule 下互斥计时的
device-run service，包含 compute、lowered
LOAD/STORE、submit 和 sync，但不包含下面单独计费的 host adapter/set/get copy。V1 不再单独拟合
load/compute/store overlap。

### 4.3 CPU-VTA 边界

每条异构边界只计费一次：

```math
\widehat T_{boundary}(e)
=\widehat T_{adapter}(contract_e)
+\widehat T_{set/get-copy}(direction_e,bytes_e).
```

边界表至少区分方向、tensor bytes、dtype/layout、padding/slice 和是否量化转换。VTA lowered
LOAD/STORE、submit 和 sync 不得在 boundary 再次计费；正式实现必须通过 accounting id 检查。
如果现有 profiler 只能得到包含 set/run/get 的 VTA island 端到端时间，则该记录只能整体查表，
不能再叠加独立 boundary；正式局部成本表需要先增加互斥 instrumentation。

P3 根据 native runner 的实际执行位置进一步拆分：

```text
CPU -> VTA: CPU producer get 属于 producer CPU stage；VTA set 属于 VTA mutex
VTA -> CPU: VTA get 属于 VTA mutex；CPU consumer set 属于 consumer CPU stage
```

因此 direct boundary 总时间用于通信诊断，但不会整体串到 VTA 路径，也不会作为另一笔 DDR wall
time 重复计费。CPU-owned
分量加入对应 CPU stage，VTA-owned 分量才加入单 VTA mutex。P2 grouped holdout 的两个方向总时间
APE 分别为 `0.84%` 和 `1.66%`。这组 grouped Pipeline holdout 使用历史 disjoint-affinity policy，
只能验证历史运行策略下的局部成本和方向分解；当前 overlapping-prefix affinity 的串行 fit 可复用，
但并发 contention 必须由 P5 当前 runtime 候选重新验证。

### 4.4 共享 DDR 下界

对候选 `p`，按 accounting id 汇总每帧由 CPU 和 VTA 唯一拥有的物理 DDR traffic：

```math
\widehat D_{ddr}(p)=
\sum_{a\in DDR\ access\ classes}\frac{Q_a(p)}{B_a^{sustained}}.
```

V1 先按方向/访问类使用少量可持续带宽常数，不拟合任意 pairwise slowdown。该项是共享资源 busy
time 下界，与已经包含单 stage 内存 stall 的 isolated stage wall time取最大值，不再相加，因此
不是把同一时间重复计费；但 physical bytes 的 accounting id 仍必须唯一。boundary tensor 已由
相邻 CPU logical input/output 与 VTA physical LOAD/STORE 表示，不能再把 boundary wall time 加入
DDR demand。

### 4.5 Pipeline 目标

V1 使用按实际执行资源归属的最大负载目标：

```math
\widehat{II}(p)=\max\left(
\max_j\left(\widehat T_{cpu-run}(S_j,t_j)+\widehat T_{boundary,cpu}(S_j)\right),
\widehat T_{vta-total}(p)+\sum_{e\in hetero}\widehat T_{boundary,vta-mutex}(e),
\widehat D_{core}(p),
\widehat D_{ddr}(p)
\right),
```

```math
\widehat{FPS}(p)=1000/\widehat{II}(p).
```

该式已经允许 CPU stage、VTA 和 boundary 的异资源部分跨帧重叠；同一 CPU stage 或同一 VTA mutex
内的工作才相加。VTA `run_ms` 已包含硬件内部 LOAD/compute/STORE 的实际 overlap，不再人为拆分。
V1 仍未描述共享 DDR contention 导致的 slowdown，也不加入独立 PS-PL 仲裁、FIFO、Max-Plus 或
额外 overlap 参数。

因此，CPU、VTA 和通信在公式中“同层”是指它们都能限制稳态启动间隔，而不是三项必须串行。
例如 CPU stage 与 VTA stage 在处理不同帧时可以并行，取两者较慢者；只有落在同一 CPU stage
线程或同一全局 VTA mutex 内的 `set/get/run` 才顺序累计。共享 DDR 项表示即使执行时间可重叠，
所有并发访问最终仍不能超过同一个 DDR 的服务能力。

## 5. 多来源约束的 k-best DP

V1 不把每个完整候选逐一上板。搜索骨架继承 Tarnawski/DNNPipe 的连续划分和最大负载目标，
segment 合法性与 grouped cost 继承 PartitionTuner/HaX-CoNN 的做法，再加入单 VTA 和共享 DDR
负载。它是针对当前约束的 DP 实例化；在没有新的 pruning 证明前不称为新算法。

对线性化后的 units，DP 状态定义为：

```text
DP[i, r, d]

i: 已覆盖前 i 个 units
r: 已使用的 VTA island 数
d: 最后一个 stage 的设备类型
```

每个状态的标签为：

```text
(max_cpu_stage_run_only_ms, max_cpu_stage_with_owned_boundary_ms,
 last_cpu_stage_with_owned_boundary_ms,
 vta_service_sum_ms, vta_mutex_boundary_sum_ms, cpu_boundary_work_sum_ms,
 cpu_core_work_sum_ms, boundary_host_core_work_sum_ms,
 shared_ddr_physical_demand_ms, predecessor)
```

`last_cpu_stage_with_owned_boundary_ms` 用于处理中间 CPU stage：它先接收前一条 VTA->CPU 边界的
CPU `set`，随后切回 VTA 时再接收 CPU->VTA 边界的 CPU `get`。若只保存全局 CPU 最大值，会漏掉
这两个边界分量属于同一个 CPU stage 的事实。

转移选择下一个连续 segment `S=(i+1...j)`：

1. 放到 CPU：独立枚举 `t in {1,2,3,4}`，更新 CPU 最大 stage 时间。
2. 放到 VTA：增加一个 VTA island，累加固定 schedule segment service 和新产生的异构边界。
3. 相邻同设备 segment 直接合并，不生成等价重复状态。
4. 每次转移累加 segment 的唯一 DDR demand，并累加 segment/boundary 的 host core-ms；boundary
   bytes 不作为第三份 DDR traffic。
5. 精确 Top-K 使用单调 partial objective 的 label-setting；不使用可能破坏候选 id tie-break 的
   Pareto shortcut。

终点按 `max(max_cpu_stage_with_owned_boundary_ms,
vta_service_sum_ms + vta_mutex_boundary_sum_ms,
cpu_core_work_sum_ms / 4,
max_k(cpu_stage_core_work_with_threads_le_k / k),
shared_ddr_physical_demand_ms)` 排序并回溯方案。为产生精确 Top-K，使用
k-best label-setting：只有当队列中所有 partial lower bound 都大于第 K 个完整解时才停止。最终
再按候选 ID 去重并输出 Top-K。

DP 是当前默认求解器。对 ResNet18 流式遍历全部 972528 个执行配置作为正确性 oracle，只保留
精确 Top-K；该遍历只发生在主机上，不等于逐候选编译或上板。

## 6. 局部 Profile 计划

V1 不执行原 212-case HardwareProfile，也不预设 16 个点能够覆盖所有 CPU threads 配置。先从真实
lowering manifest 中按 signature 去重：

```text
CPU signature:
  segment lowering signature + physical shape + dtype/layout + threads

VTA signature:
  segment lowering/fusion signature + physical shape + frozen schedule id

Boundary signature:
  direction + src/dst dtype/layout + physical bytes/calls + padding/slice/quantization/cache policy

DDR access signature:
  owner + direction/access class + physical bytes/calls + transaction ids
```

测量原则：

1. 相同 signature 只测一次并复用，不按层名重复测。
2. CPU signature 对 `threads=1..4` 测量；同时记录 affinity、wall time 和 process CPU time。
3. VTA 只测冻结 schedule，不搜索 tile。
4. Boundary 使用真实 native adapter/runtime path，不使用单纯 memcpy 代替转换。
5. 测量少量 DDR sustained bandwidth，并做 CPU-only、VTA-only、concurrent matched control；V1
   只消费聚合下界，concurrent 结果用于判断是否需要 V2。
6. 每类至少留一个完整 stage 作为 grouped holdout，验证 unit 求和误差。
7. 优先复用 fingerprint 一致、correctness 通过的历史 archive，只补缺失 signature。
8. 第一轮一个 session；只有排序和 holdout gate 通过后才增加独立 boot。

profile manifest 生成后才确定 `B_profile`，不能先凑固定整数。

## 7. Top-K 上板验证

DP 输出预测排名后执行：

```text
1. 取预测 Top-K，并保留 type-fixed、PartitionTuner-like、Tarnawski/DNNPipe-like 和随机对照。
2. 对 shortlist 做 compile-only/native/reference gate。
3. 只对通过 gate 的方案执行 Pipeline 上板。
4. K 预注册为曲线，例如 1/3/5/10/20，而不是只报告成功的 K。
5. 编译失败、correctness 失败和超时都记录并计入对应预算。
```

必须比较：

```text
B0 uniform / stratified random
B1 type-fixed Synergy/NEURAghe-like mapping，固定 CPU thread policy
B2 PartitionTuner-like grouped segment cost，优化单帧 latency
B3 Tarnawski/DNNPipe-like Pipeline max load，忽略 CPU-stage contention 和 DDR coupling
B4 V1 single-VTA + per-stage TVM threads + composite boundary + shared-DDR bound
B5 historical/exhaustive measured-pool oracle，仅用于回顾性评价
```

主指标：

```text
throughput regret@K = 1 - FPS_best_in_topK / FPS_oracle
达到 95% measured-pool oracle 所需 board evaluations
Top-K recall
B_profile、build 和 B_search 的 case 数与 wall-clock
correctness/build failure rate
```

回顾性 ResNet 结果使用 `measured-pool oracle`；未测完整合法空间时不得简称 global oracle。

## 8. 实施阶段

### V1-P0：冻结实验定义

状态：`completed`

- 冻结 schedule/tile、量化、compiler、runtime 和 bitstream 指纹。
- 冻结当前运行端点 `root@192.168.1.234`、RPC `192.168.1.234:9090`，并完成 SSH/RPC preflight。
- 定义 compute units、静态 contract-compatible cuts、最大 VTA islands 和同构 CPU affinity 规则。
- 冻结评价指标和 profile/search 两类预算账本。
- 实际 compiler-valid cuts 与 segment TIR/module hash 属于 P1，不由 P0 提前声称。

### V1-P1：生成局部成本 manifest

状态：`completed`

- 从 ResNet18 lowering 生成去重 signature manifest。
- 审计并复用历史 CPU/VTA/boundary 数据。
- 标记 reusable、missing、stale，冻结 grouped holdout；本阶段不做性能测量。

### V1-P2：最小局部 Profile

状态：`completed`

- 只构建和测量 P1 识别出的缺失 signature。
- 检查 CPU affinity 真实生效，并用完整 stage holdout 检查局部成本可加性。

### V1-P3：DP 求解器

状态：`completed`

- 实现 per-stage TVM threads、单 VTA 累加、四核 core-work、复合边界和 DDR demand 转移。
- 实现单调下界 k-best label-setting 和精确 Top-K 输出。
- 在 ResNet18 上流式遍历全部静态配置，验证 Top-K 最优性。
- 当前 4623 个拓扑对应 972528 个执行配置；DP Top-20 与完整流式枚举逐项一致。
- P3 已把 VTA service、direct boundary service、四核总 work 和共享 DDR demand 分开。P2 profiler
  的实际 LOAD/STORE traffic 经两个 fit segment 拟合，grouped holdout total-traffic APE 为
  `17.42%`；Top-20 DDR demand 为 `9.510--9.628 ms`，没有重复加入 boundary wall time。
- 旧模型遗漏 CPU 总 work，错误偏向 3 个 VTA island，并给出下界倒数 `20.393`。加入 `core-ms/4` 后，
  Top-20 为 11 个单 island、9 个双 island；Top-1 是 CPU[00..02] t3 -> VTA[03..17] ->
  CPU[18..20] t2，原始资源下界为 `70.807 ms`，其倒数 `14.123` 只用于排序。该数字既不是
  校准后的 FPS，也不是板端实测吞吐。
- 直接 boundary 消融使 Top-20 只保留 `0/20`，与历史 200-candidate 通信实验一致，说明通信会改变
  shortlist；但 Top-20 仍无 DDR bottleneck，尚不能声称共享 DDR contention 模型改善了最终 regret。
- 当前 VTA traffic 仍是 logical-GOP 粗拟合，CPU internal weight/activation/cache traffic 和并发
  contention 未建模；方向化 DMA rate 来自已通过辨识 gate、但 runtime poll policy 不同的历史
  qualification，只能作为 provisional V1 rate，必须由当前 shortlist 复核。
- boundary 已按 runner 锁范围拆成 CPU-owned set/get 与 VTA-mutex-owned set/get。Top-1 的两部分
  分别为 `0.407 ms` 和 `2.830 ms`；CPU、VTA 与异资源通信可以 overlap，不再把完整 boundary
  错误串行到 VTA 路径。该修订仍不等于已经识别 DDR contention slowdown。

### V1-P4：回顾性验证

状态：`completed`

- 在已有 measured pool 上比较 B0-B4。
- 只有 B4 的低 K regret 明显优于 B0-B3，才启动新上板实验。
- 若失败，先修正局部成本和所有权，不增加 tile/FIFO/Max-Plus。

P4 先执行标签可比较性 gate。历史 200 条记录都能映射到 21-unit 表示，其中 70 条记录、69 个唯一
配置在 segment/thread 语义上重合。历史 CPU stage threads 总和为 7--9 不能作为物理核心预算 gate；
但历史 manifest 也没有当前 overlapping-affinity/source fingerprint，因此当前 runtime 精确重合为
0。69 个配置只用于探索性历史-runtime regret，P5 仍不得启动。

### V1-P4B：冻结基线并做同池回顾性比较

状态：`completed_exploratory_gate_inconclusive`

- candidate pool 与 historical labels 已拆成两个封存 JSON；score 进程只加载无标签 pool，标签仅在
  evaluate 阶段连接，不再只是“同一 JSON 中忽略 measured 字段”。
- B4 regret@1/3/5 为 `9.60%/7.62%/7.62%`，低于 B2、B3 和两种随机中位数；Spearman 为
  `0.482`，高于 B2 的 `0.081` 和 B3 的 `0.271`。
- B4 达到 95% measured-pool oracle 需要 12 次，uniform random 中位数为 9 次；当前还没有证明
  “更少上板次数达到近优”这一主目标；`62.4%` 的 uniform random 顺序不晚于 B4 达到该阈值。
- B4 Top-K recall 在 K=1/3/5 均为0，K=10/20 为 `0.30/0.60`；精确历史池 oracle 位于 B4
  第 12 名。低 K regret 改善表示找到接近最优候选，不表示恢复了真实 Top-K 排序。
- 历史池只覆盖 `(3,4)`、`(3,1,4)`、`(3,1,1,4)` 三种 thread 向量，因此只能支持切图排序诊断，
  不能验证 per-stage thread 联合搜索。
- 精确 B1 固定映射不在历史池，禁止用最近候选替代；B1 未来只与 B4 K=1 做等预算比较。因此完整
  gate 仍未通过，P5 不启动。
- 本阶段没有上板，也没有拟合 DDR contention；`D1` 仍由独立 matched-control 与排序残差 gate 触发。

### V1-P4R：证据缺口修复方案

状态：`completed; awaiting_user_confirmation`

- 保持现有 score 不变且不读取 throughput，重新遍历 972528 个合法配置，冻结 20 个候选：
  B4/B2/B3/uniform-random 各 3 个、精确 B1、6 个同 topology 单变量 thread 对照和 3 个 B4
  topology-diversity fill；去重后覆盖 10 种 topology，13 个候选尚需 native compile。
- primary K 固定为 3，20 是整个 prospective pilot 的总上板预算，不是每个模型各 20 个。
- `evaluations to 95%` 只在结果揭晓后报告，不允许作为运行时停止条件；P4R 本身不上板。

### V1-P5A/P5B：先 correctness，后前瞻上板

状态：`P5A completed; P5B iteration2 natural Top-20 single-session validation completed`

- P5A 已在本地构建并审计 20/20 个冻结 package：88 个 stage 引用对应 33 个唯一 compile key，
  VTA 二进制含原生 runtime 调用，threads/affinity/hash/ABI 均与冻结清单一致；本阶段 0 次上板。
- P5A 已生成独立 MXNet `[1,1000] float32` logits 和 raw-output comparator；稳定 hash 和 cat 等价
  都不足以替代数值 reference。首次完整构建与初审为 `626.303 s`，缓存复审 `0.895 s`，不能混用；
  P5B package 固定为 2 个 warmup 加 20 个 scored frame。
- 7 个 tuple 边界的运行 ABI 只能静态证明 ordinal/shape/dtype；`out0/out1` 与 `main/residual` 是描述名
  差异，最终语义顺序仍由 P5B 的完整 tensor 比较证明。float reference gate 也不是 bit-exact 量化证明。
- P5A 审阅通过后才进入 P5B；每个候选先在板端通过 native-vs-reference correctness，再按冻结顺序
  计时。报告 primary `regret@3`、thread 配对、完整成本和 failure ledger；结果只对应冻结
  measured pool，不冒充 972528 配置的 global oracle。
- P5B Iteration 1 已证明 thread 参数有效但非线性；原 B4 Top-1 `t3` 为 `9.312 FPS`，同 topology
  `t4` 为 `10.449 FPS`。基于四点修正全局 CPU slope 后，新的五阶段 Top-1 correctness 通过，
  但仅为 `8.869 FPS`，低于 measured-pool oracle `15.11%`。
- 三个 CPU stage 的实测中位 run 约 `117.12/115.23/88.82 ms`，而预测约
  `62.95/50.68/65.29 ms`。Iteration 2 已用 21 个 atomic CPU unit 和 23 个 grouped checks 替代
  单一 logical-GOP 斜率；holdout median/P95/max APE 为 `0.89%/4.17%/17.96%`。
- 求解器已修复 overlapping-prefix affinity 的容量遗漏。修正后不读取吞吐拟合即复现 5 个已测
  候选的完整顺序，cycle MAPE 为 `14.66%`；这支持排序诊断，但不支持精确 FPS 声明。
- 历史 200 条记录的标签隔离回放得到：199 个唯一方案上正式模型 cycle MAPE=`28.02%`、
  Spearman=`0.775`、regret@5=`10.28%`；当前 DP 可直接比较的 69 个唯一方案上 MAPE=`26.70%`、
  Spearman=`0.786`、regret@5=`7.30%`，到 K=20 才降至 `0.40%`。
- 上述历史数值是池内回顾性重排，不是当前算法的自然 Top-20。2026-09-04 已把 Iteration 2 自然
  Top-20 全部上板；4 种唯一拓扑只各编译一次，20 个 runtime thread 配置分别执行 22 帧并丢弃
  2 帧 warmup。20/20 的 serial/pipeline tensor 逐字节一致且独立 MXNet reference gate 通过。
- 自然 Top-20 的实测范围为 `10.220--11.511 FPS`，池内最好是静态 rank-9 的
  `cpu-00-02_t4__vta-03-16_t1__cpu-17-20_t3`。静态 rank-1 为 `10.887 FPS`；按冻结顺序
  `regret@1/3/5=5.419%`，第 9 次首次达到当前 measured pool oracle 的 95%。该 oracle 只覆盖这
  20 个候选，且当前只有单 session。
- 静态名次与 Top-20 内实测名次 Spearman 为 `0.155`。这说明当前模型能筛到高性能区域，但
  `70.807--72.045 ms` 的紧密静态分数不足以细排实测 `86.872--97.852 ms` 的近邻候选，尤其不能
  正确表达同 topology 下非单调的 CPU thread 并发效果。相同 `CPU00--02 t4` 首段的并发中位时间
  仍跨 `91.6--105.3 ms`，因此 P7B-1 测 CPU-CPU 重叠，P7C 再独立测 CPU-VTA 共享 DDR contention。
- 用 warm100 学一个全局效率比例并只在 theory100 验证，再反向执行，held-out MAPE 为约
  `6.94%--8.19%`。这只证明存在系统性遗漏，因参数来自完整候选吞吐而从正式模型排除；它不改变
  排序，也不能作为 Top-K 有效性证据。假设每段
  core-ms 固定均分到具体核心虽降低 MAPE，却把 Spearman 降至 `0.480` 且不符合 worker 迁移语义，
  只保留为诊断。
- 历史 `+34 ms` 在自然 Top-20 上把 cycle MAPE 从 `23.523%` 降到 `12.859%`，但本轮实际残差
  均值/中位数仅 `22.054/22.353 ms`。它改善绝对值却统一过度修正，仍不得进入正式 score。
- roadmap 的 P7A 本地 instrumentation 已实现并通过审计：native runner 支持独立 warmup、逐样本
  barrier、多 tensor stage 输入、process CPU time 和 PMU，CPU memory benchmark 使用持久 worker；
  AArch64 交叉编译和相关测试通过。P7B-1 已在 `.247` 的三个独立 boot 完成，测量前后 preflight
  均通过；每个合格 session 都是 36/36 memory case 和 16/16 CPU pair 通过。三轮 memory 最小
  相关性为 `0.994`，A/B slowdown 最小相关性为 `0.974/0.965`，跨 boot CV 中位数为
  `1.17%/2.17%`；冻结 gate 准入 A/B 精确观测 14/13 个。该结果说明资源竞争可重复，但精确
  stage-pair 表尚不能外推为通用 surface，slowdown 尚未进入 score；本轮 Top-20 留到新公式冻结后
  作外部验证。
- P7B-2 已在第三个 boot 完成单 boot qualification，9/9 个实际 case 通过。1/3 actor 的 host
  空队列下界仅为 `0.028/0.056 ms/帧`；poll 0/1000 ns matched delta 为 `0.222/0.531 ms`。
  CPU->VTA 与 VTA->CPU boundary slope 为 `3.222/3.215 ms/MiB`，且旧 P2 模型还略高估该项，
  因而这些分量均不能解释 22--34 ms 周期低估。P2 的 VTA service 对主 Top-20 segment
  `vta:03:17` 误差约 1%，但短 segment 误差达到 11%--23%，后续应使用 segment/shape-aware
  service。当前只有一个 boot，P7B-2 参数均未进入正式公式。
- P7C 已完成单 boot qualification。原计划把两种 VTA 压力写成同一个 placeholder、cache read
  字节口径不一致，并让 CPU 占满四核，无法归因 DDR；修订子协议改用 DMA 强度相差 `3.92x` 的
  真实 `vta:01:03/vta:15:19`，并把 CPU/VTA host 分别固定到核 0--2/3。首轮 affinity 被 TVM
  threadpool 覆盖的失败记录和第二轮仅验证当前线程的过程记录均已保留；最终 runner 对现有全部
  线程应用 affinity，原 case 4/4 通过。CPU 仅 `cache_resident + compute_heavy` 达到 `1.065x`
  gate，不能作为 DDR 证据；VTA 在两组 streaming control 下达到 `1.052x/1.066x`。对应 stage
  增量为 `2.110/1.359 ms`，driver run 只增加 `0.023/0.266 ms`，故当前称为
  shared-memory/host-runtime interference，不拟合纯 DDR bandwidth。P7D 准入审计未继续为该
  信号补 boot，因其既不是纯 DDR 参数，也没有改善冻结排名的证据；该负结果不进入正式公式。
- P7D 已完成：只准入三 boot streaming CPU read/write 带宽，精确枚举 `972528` 个执行配置后
  Top-20 未变，绝对 FPS gate 未通过，因此冻结为 ranking-only V1。
- P8A 已完成：GraphExecutor zero-copy API 之上由 VTA driver 显式分配单个 u-dma-buf slot，构造
  CPU/VTA 同址 view；20 个逐帧 matched pair 平衡 AB/BA 顺序，A/B 输入逐帧路径等价、地址稳定和
  256-byte 对齐均通过。最快正确普通路径 `memcpy` 的边界复制中位数为 `0.714217 ms`，zero-copy
  的 `baseline - zero-copy` 单帧 latency 配对差值中位数为 `+1.157847 ms`。当前只允许进入 P8B，
  不声明双 slot Pipeline 吞吐提升。
- P8B 已完成：同一绑定协议推广到 CPU->VTA 和 VTA->CPU，修复了反向边界必须检查 CPU consumer
  直读而非 VTA producer 的 owner 错误。最快正确 `memcpy` baseline 分别为
  `0.726637/0.116311 ms`，两边的 framework materialization 分别从 `1605632/200704 bytes` 降到
  0；CPU mapping penalty 为 `-0.253247/+0.005370 ms`。整帧配对差值包含 runtime 波动，不能替代
  P8C 的 Pipeline II/FPS 实验。
- P8C 已完成三个独立 boot qualification：新增双边界 `BoundarySlotManager`，以
  edge/slot/generation/frame owner 和四态状态机管理两个预分配 slot；等待超时、错误广播、物理
  地址对齐/不重叠及排空后全 FREE 检查均已实现。统计审计后，Pipeline II 固定使用完整 block
  completion 窗口，逐帧间隔中位数只作抖动诊断。三个 boot 的相对 FPS 差值为
  `-0.74%/+0.29%/-1.37%`，II 差值的 95% 区间为 `[-2.477, 1.363] ms`，所以不允许声明吞吐
  提升；旧 `+4.42%/+3.90%` 结果已废止。B2 把每帧
  `1806336 bytes` framework 边界物化降为 0，同时 VTA 内部
  `12025856-byte LOAD + 1229312-byte STORE` 保持不变，证明消除的是 Executor 边界副本而不是
  加速器必要 DMA。边界 API 服务平均从 `1.446899 ms` 降到 `0.041830 ms`，减少 `97.11%`；这是
  当前可发表的组件结果。VTA->CPU producer slot wait 跨 boot 波动明显，该等待不能再次加到
  服务时间中，也不能据此预设固定瓶颈。

### V2：条件扩展

只有固定 tile 被证明是主要误差来源时，才加入有限 tile 候选和分层 DP。DDR contention 已进入
P7 的受控 profile；FIFO 和完整 Max-Plus 继续保持条件扩展。

其中 `D1: shared-DDR contention` 单独登记为后续待优化项：V1 的 DDR 总服务需求只是容量下界，
没有表达 CPU 与 VTA 并发访存造成的双向 slowdown。只有 matched concurrent control 和冻结候选
排序残差同时支持该机制时才增加修正；不得仅因为 DDR 是共享资源就预设一个 contention penalty。

## 9. V1 完成标准

V1 完成必须同时满足：

```text
每个 CPU stage 的 TVM threads 真实生效，且未被误解释为互斥核心配额；
单 VTA 串行所有权与 native mutex 一致；
CPU/VTA boundary 无重复计费；
共享 DDR transaction ownership 完整且聚合下界可复现；
高排名 segment 通过 lowering manifest 与非可加性 holdout；
DP 与静态枚举在相同成本表上结果一致；
所有上板候选通过 reference correctness；
B4 在预注册低 K 上优于 type-fixed、single-frame placement 和无共享资源 Pipeline 基线；
profile + Top-K 总成本显著低于大规模候选上板。
```

若最后两项不成立，论文应诚实定位为 CPU-VTA Pipeline 系统实现和负面实证，不继续用模型复杂度
掩盖排序无效。

## References

- **[R1]** J. Tarnawski et al., “Efficient Algorithms for Device Placement of DNN Graph
  Operators,” NeurIPS 2020. [Paper](https://proceedings.neurips.cc/paper/2020/file/b14680dec683e744ada1f2fe08614086-Paper.pdf)
- **[R2]** M. Yu et al., “PartitionTuner: An Operator Scheduler for Deep-Learning Compilers
  Supporting Multiple Heterogeneous Processing Units,” ETRI Journal, 2023.
  [Paper](https://onlinelibrary.wiley.com/doi/pdf/10.4218/etrij.2021-0446)
- **[R3]** P. Meloni et al., “NEURAghe: Exploiting CPU-FPGA Synergies for Efficient and
  Flexible CNN Inference Acceleration on Zynq SoCs,” ACM TRETS, 2018.
  [Paper](https://arxiv.org/pdf/1712.00994)
- **[R4]** G. Zhong et al., “Synergy: A HW/SW Framework for High Throughput CNNs on
  Embedded Heterogeneous SoC,” ACM TECS, 2019. [Paper](https://arxiv.org/pdf/1804.00706)
- **[R5]** F. Jia et al., “CoDL: Efficient CPU-GPU Co-execution for Deep Learning Inference
  on Mobile Devices,” MobiSys, 2022. [Paper](https://chrisplus.me/assets/pdf/mobisys22-CoDL.pdf)
- **[R6]** I. Dagli and M. E. Belviranli, “Shared Memory-contention-aware Concurrent DNN
  Execution for Diversely Heterogeneous SoCs,” PPoPP, 2024.
  [Paper](https://mehmet.belviranli.com/papers/ppopp24.pdf)
- **[R7]** W. Seo et al., “DNNPipe: Dynamic Programming-Based Optimal DNN Partitioning
  for Pipelined Inference on IoT Networks,” JSA, 2025.
  [Paper](https://www.sciencedirect.com/science/article/pii/S1383762125001341)
- **[R8]** A. Rios-Navarro et al., “Performance Evaluation over HW/SW Co-design SoC
  Memory Transfers for a CNN Accelerator,” IEEE-NANO, 2018.
  [Paper](https://arxiv.org/pdf/1806.01106)
- **[R9]** J. Haris et al., “SECDA: Efficient Hardware/Software Co-Design of FPGA-based DNN
  Accelerators for Edge Inference,” ISPASS, 2022. [Paper](https://arxiv.org/pdf/2110.00478)
- **[R10]** T. Moreau et al., “A Hardware-Software Blueprint for Flexible Deep Learning
  Specialization,” IEEE Micro, 2019. [PDF](https://homes.cs.washington.edu/~arvind/papers/vta.pdf)
