# RAMPS 系统模型与理论基础

> **理论全集。** 本文件保存 RAMPS 可选的完整资源图设计，不表示当前论文实例必须实现
> 每一层。当前主路径和停止条件以
> [HARDWARE_LAYER_DESIGN_REVIEW.md](HARDWARE_LAYER_DESIGN_REVIEW.md) 为准：先验证
> M0/M1 的低预算候选选择价值，M2/M3 和完整 Max-Plus 仅在简单模型失败时启用。
> 日期目录中的 `SYSTEM_MODEL_AND_THEORY.md`、`MODEL_SPEC.md` 和旧 12-anchor
> 协议只用于历史复现。已成立的实验数字统一维护在
> [PAPER_EVIDENCE.md](PAPER_EVIDENCE.md)。
>
> 多阶段实现状态、阶段验收和下一步命令维护在
> [RAMPS_EXECUTION_ROADMAP.md](RAMPS_EXECUTION_ROADMAP.md)。publication-mode 字段见
> [RAMPS_MODEL_SCHEMA.md](RAMPS_MODEL_SCHEMA.md)。当前实现边界和硬件层缺口统一见
> [HARDWARE_LAYER_DESIGN_REVIEW.md](HARDWARE_LAYER_DESIGN_REVIEW.md)。

## 1. 目标与边界

RAMPS（Resource-Aware Max-Plus Stage Search）的目标不是针对每个新模型重新拟合一个
吞吐回归器，也不是先穷尽所有硬件细节。它先在固定预算内辨识对候选排序最有用的
CPU-FPGA 资源代价，再根据新 DNN 的
计算图、tensor 形状、layout、量化类型和切分方案计算资源需求。完整框架允许求解：

```text
算子设备分配 + stage 切点 + CPU 线程数 + FIFO 深度
+ VTA tile/layout + boundary adapter + 预测 pipeline 吞吐
```

当前论文实际实现和验证的是完整框架的**固定算子实现图级实例**。本轮不联合搜索
AutoTVM schedule、fusion、tile、CPU 线程和 FIFO 深度，而是固定编译与运行策略，只优化：

```text
compute-unit CPU/VTA assignment + stage cuts
```

固定变量不从资源模型中消失。它们作为已知条件决定 CPU/VTA 服务时间、DMA、SRAM
可行性和 FIFO token；只是本轮求解器不对它们枚举。论文不得把完整框架的扩展接口表述为
已经完成联合优化。

硬件模型只在以下硬件指纹变化时重新校准：bitstream、VTA 配置、CPU/DDR/PL 时钟、
runtime、编译 target、bridge 实现或操作系统调度配置发生变化。对于同一硬件指纹上的
ResNet18、YOLOv3-tiny、SqueezeNet 等新 DNN，只提取静态属性并求解，不重新训练模型。

### 1.1 完整扩展架构：两层模型

完整 RAMPS 的预测路径分为两层：

```text
Layer H: HardwareProfile(board, bitstream, compiler schedule, runtime policy)
  CPU primitive service + core demand + memory bandwidth
  VTA compute/load/store + overlap + submit/sync
  DMA access models + boundary adapter models + measured resource capacities

Layer D: PartitionWorkload(new DNN, candidate cuts)
  lowered primitive + logical/physical OP + bytes + tile/DMA calls
  tensor layout/padding/quantization + boundary adapter
  stage dependency + fixed threads/queue depth/schedule hash

HardwareProfile + PartitionWorkload -> StageService -> Max-Plus event graph
```

当前最低可用实例不要求先完成完整 HardwareProfile。它可以使用相同的
`PartitionWorkload -> StageService` 边界，依次计算 single-VTA compute balance、communication-
aware single-VTA 和 executor resource bound。只有这些轻量模型在低预算排序上不足时，才恢复
完整 service surface 和 Max-Plus event graph。候选实测预算与硬件 profile 预算分别记账。

CPU core 数、VTA 实例数、PS-PL channel 和 bridge capacity 属于 Layer H，不能在求解器中按
某一块开发板写死。Layer D 的 schedule ID、compiler hash 和 hardware fingerprint 必须与 Layer H
一致；否则拒绝预测，而不是插值到另一套 lowering/runtime。

Layer H 禁止包含模型名、层名、候选 ID 或候选实测吞吐；Layer D 的 identity metadata 仅用于
报告，在进入预测器前剥离。缺失 primitive 或硬件服务点时直接拒绝预测，不允许从其他模型
bucket、旧 static score 或默认 GOP/s 静默补值。代码契约分别定义在：

- `ramps_hardware_profile.py`
- `ramps_workload_schema.py`
- `ramps_hardware_calibration_protocol.py`

旧 Stage 4b 的 ResNet-shaped buckets 和 `resource_aware_dataset.py` 中按层名重建历史特征的逻辑
只用于解释已归档实验，不再是 publication prediction path；旧 runner/protocol 源码已由 v4
预算化协议替代。

本工作的直接思想起点是 DNNPipe：两者都把 DNN partition 看成以稳态吞吐为目标的 stage
划分问题，并认为理想切分应尽可能避免慢 stage [R14]。关键差异不是应用名称，而是系统
假设。DNNPipe 在独立 IoT 设备和通信可忽略的条件下最小化最大 compute-stage time；本项目
面对共享内存的嵌入式 CPU-FPGA SoC，切图会引入 layout/quantization adapter、PS-PL DMA、
同步和 FIFO，并且多个 stage 竞争 CPU、DDR、bridge 和唯一 VTA。因此本文研究的问题是：
当通信和共享资源不能忽略时，如何把 DNNPipe 式的 stage balance 推广为通信和资源感知的
稳态时序优化。

最接近的同平台工作是 PartitionTuner [R23]。它已经使用单算子/算子组 profile、动态规划和
CPU/EVTA backend mapping 优化 DNN 推理，并允许同一输入中的独立 branch 在不同设备上并行。
因此本文不能声称首次完成 CPU--FPGA 图划分、首次使用 profile 或首次考虑 transfer。两者的
边界在执行目标和代价抽象：PartitionTuner 对顺序 branch 仍采用串行 partition schedule，并把
候选子图的计算与传输聚合为执行时间；RAMPS 面向连续输入的 inter-frame pipeline，把可迁移的
CPU/VTA/DMA/boundary 服务映射为共享资源事件，再求稳态关键环。

Synergy [R24] 已证明 Zynq 上多线程 CPU/NEON/FPGA pipeline 和 work stealing 能提高 CNN
吞吐。因此 native multi-request executor 和“实现了 CPU--FPGA pipeline”是 RAMPS 的必要系统
基础，而不是独立理论创新。本文的新问题是：在固定后端和运行时上，如何自动选择合法 DNN
子图边界，使通信、共享资源和流水关键路径共同决定的稳态吞吐最优。

Tarnawski 等人的 device-placement 工作 [R25] 已经覆盖 CPU/加速器节点代价、跨设备边代价和
以最大设备负载表示的 Pipeline 吞吐，并给出 DP/IP 求解。因此 RAMPS 不能把
“profile + communication + Pipeline max-load”本身作为创新。当前需要验证的差异是单 VTA
被多个 island 重复占用、共享 DDR/PS-PL 和 native executor boundary 是否会改变低预算候选
排序；若不会，RAMPS 应退化为该类 aggregate device-load 模型的系统实现。

RAMPS 的主要目标是稳态吞吐。单帧 serial latency 作为通信分解、executor-matched 对照和
“serial 最优是否等于 pipeline 最优”的诊断目标单独报告，但不与吞吐共同组成一个未定义权重
的多目标函数。对候选方案 `p`，最终目标为：

```math
p^* = \arg\min_{p \in \mathcal{P}_{safe}}
      \left[\widehat T_{cycle}(p) + z_\alpha\widehat\sigma(p)\right],
\qquad
\widehat{FPS}(p)=\frac{1000}{\widehat T_{cycle}(p)}.
```

其中 `P_safe` 只包含 layout、padding、quantization、shape 和 correctness contract 均有效的
候选。`z_alpha sigma` 是风险项，不改变物理模型，仅避免优先选择不确定性过大的候选。

### 1.2 成功标准不是精确回归每个候选

RAMPS 的论文主目标是 candidate triage：用更少的 board evaluations 找到接近 oracle
的 partition。因此指标优先级是：

```text
regret@K
Top-K recall
board evaluations required to reach 95% of oracle throughput
calibration + build + board-search wall-clock cost
cycle MAE/MAPE and rank correlation as diagnostics
```

一个毫秒预测不完全准确、但能稳定把近优候选放入 Top-K 的模型仍然有搜索价值；
反之，只改善全局 MAE 却不降低 regret 或上板次数，不足以支撑论文主张。

硬件校准也必须进入成本账本。对复用同一 HardwareProfile 的第 `m` 个 DNN：

```math
C_{RAMPS}(m)=\frac{C_{profile}}{m}+C_{static}+K C_{board},
\qquad C_{exhaustive}=N C_{board}.
```

只有当 RAMPS 在预注册的未见候选上降低 regret/上板次数，且摊销后总成本小于穷举或
更简单的 profile-guided baseline，才能声称复杂建模有实际价值。

当前 V1 图级实例把目标写为：

```math
(z^*,cuts^*,t^*)=\arg\min_{z,cuts,t}
\max\left(
\widehat T_{cpu-max},
\widehat T_{single-vta+boundary},
\widehat D_{shared-ddr}
\right)
\quad\text{s.t. }q=2,\tau=\tau^0,fusion=fusion^0,
```

其中每个 CPU stage 的核心数 `t` 是 V1 决策变量；queue depth、VTA lowering/schedule/tile 和
compiler pass 在协议冻结后对所有候选保持一致。`adapter` 由 boundary contract 确定，不作为任意
可调 penalty。V1 使用 Tarnawski-style 最大负载加共享 DDR 聚合 resource-demand 下界，不以完整
Timed Event Graph 为前置条件。

## 2. 符号与系统抽象

将量化后的 DNN 表示为 DAG `G=(V,E)`：

- `u in V`：一个可编译算子或融合 compute unit。
- `e=(u,v) in E`：算子之间的 tensor 边界。
- `d(u) in {CPU,VTA}`：算子的目标设备。
- `S_k`：由一个或多个连续 compute unit 构成的 stage。
- `t_k in {1,2,3,4}`：CPU stage 的线程数。
- `q_e`：stage 边界 FIFO 可保存的 tensor token 数。
- `O_u`：逻辑或物理 OP 数。
- `Q_u`：执行产生的必要数据流量，单位 B。
- `C_b`：VTA 某类 SRAM buffer 的容量。

硬件资源集合为：

```text
4-token CPU core pool
1-token CPU memory-bandwidth domain
1-token VTA global mutex
1-token PS-PL load channel
1-token PS-PL store channel
1-token bridge pack/unpack resource
1-token worker per stage
bounded FIFO/buffer tokens
```

资源 token 不是软件线程。它表示某种硬件资源能同时服务多少项工作。例如 VTA global
mutex 只有一个 token，因此属于不同 stage 的 VTA 工作最终必须串行占用同一 VTA。

### 2.1 通用决策变量与当前冻结实例

对每个 compute unit `u`，定义二元设备变量：

```math
z_{u,d}\in\{0,1\},\qquad
\sum_{d\in\{CPU,VTA\}}z_{u,d}=1.
```

切点变量决定相邻 unit 是否属于同一 stage；每个 CPU stage 选择线程数 `t_k`，每条跨
stage 边选择 FIFO 深度 `q_e`，每个 VTA unit 选择合法 tile `tau_u`。完整方案写为：

```math
p=(z,\;cuts,\;t,\;q,\;\tau,\;adapter).
```

可行域 `P_safe` 至少同时满足：

```math
W_b(u,\tau_u)\leq C_b,\quad \forall b\in\{INP,WGT,ACC,OUT\},
```

```math
\sum_e q_eQ_e\leq M_{queue-budget},
```

```math
Contract_{producer}(e)
\xrightarrow{adapter_e}
Contract_{consumer}(e),
```

以及编译器支持、DAG dependency、量化语义和最大 VTA island 数等离散约束。在完整模型中，
CPU 核心数和 VTA mutex 作为 Timed Event Graph 中的共享资源 token，从而表示超配等待；当前
V1 为保持可实现性，改用实际 affinity 加静态 `sum(t_k)<=4`，并将所有 VTA island service 相加。

该问题是包含离散 assignment/cut/tile 和非线性最大环均值目标的组合优化问题。RAMPS 不
声称闭式求得全局最优，而是使用正确性约束过滤、DNNPipe 风格 admissible bound 剪枝和
risk-aware 排序，在有限 build/board budget 下返回近似最优候选集合。

当前 V1 实例把 `z`、`cuts` 和每个 CPU stage 的核心数 `t` 作为自由变量，并采用
`sum(t_k)<=C_cpu` 和实际 affinity 作为简化的核心资源约束。`q=2`、`tau=tau^0` 和固定 compiler
configuration 必须写入 protocol hash。tile、FIFO 和完整资源图联合求解保留为 V2 框架扩展，
不计入 V1 贡献。

## 3. DNN 静态工作量模型

本节全部字段由新 DNN 的 compiler/lowering 静态提取，不读取该候选的板端吞吐或 stage time。
每个 operator 必须输出通用 primitive、逻辑/物理 OP、memory/load/store bytes、DMA calls、
tile count、spill 和固定 schedule ID。`stem`、`residual block`、`YOLO head` 等网络语义不是
硬件服务特征。

### 3.1 OP 与物理 shape

CPU 使用编译后算子的逻辑 shape 计算 OP。VTA 必须使用 padding 和 block 后的物理 shape：

```math
C_{in}^{phy}=\left\lceil\frac{C_{in}}{B_{in}}\right\rceil B_{in},\qquad
C_{out}^{phy}=\left\lceil\frac{C_{out}}{B_{out}}\right\rceil B_{out}.
```

卷积的物理 OP 近似为：

```math
O_{conv}^{phy}=2NH_oW_oC_{out}^{phy}K_hK_wC_{in}^{phy}.
```

这里必须统一 OP/MAC 口径。本文使用 `1 MAC = 2 OP`，所有 GOP/s 都按 OP/s 计算。
YOLO head 的 `255` 通道在 VTA 上可能按 `256` 通道执行，因此不能用逻辑 shape 低估计算、
权重和输出流量。

### 3.2 数据流量与融合

未融合算子的 compulsory traffic 下界为：

```math
Q_u^{min}=Q_{input,u}+Q_{weight,u}+Q_{output,u}.
```

相邻算子融合后，中间 tensor 可留在 cache/SRAM 中，不能重复计入完整外部流量。因此 RAMPS
以编译器最终融合组或 VTA compute unit 为最小建模单位，而不是任意单 Relay op。该原则与
Timeloop、MAESTRO 对 mapping、tiling 和数据复用的建模一致 [R8, R9]。

## 4. CPU 服务时间：Roofline 与多线程缩放

### 4.1 Compute-bound 与 memory-bound

定义算术强度：

```math
I_u=\frac{O_u}{Q_u}\quad[OP/B].
```

CPU 算力上限为 `P_cpu(k,t)`，内存带宽为 `B_cpu(t)`。Roofline 模型给出 [R1]：

```math
P_{attainable}(u,t)=\min\left(P_{cpu}(k,t), B_{cpu}(t)I_u\right).
```

ridge point 为：

```math
I^*(k,t)=\frac{P_{cpu}(k,t)}{B_{cpu}(t)}.
```

因此 `I_u < I*` 时是 memory-bound，`I_u > I*` 时是 compute-bound。时间形式为：

```math
T_{cpu}(u,t)=
\max\left(
\frac{O_u}{P_{cpu}(k,t)},
\frac{Q_u}{B_{cpu}(t)}
\right)+L_{launch,cpu}(k,t).
```

单位换算为：

```text
compute_ms = Ops / (GOP/s * 1e9) * 1000
memory_ms  = Bytes / (GB/s * 1e9) * 1000
```

CPU stage 的服务时间按最终融合组聚合：

```math
T_{CPU}(S_k,t_k)=L_{set}(S_k)+
\sum_{g\in S_k}T_{cpu}(g,t_k)+L_{get}(S_k)+L_{bridge}(S_k).
```

### 4.2 线程数不是线性加速

Amdahl 定律给出固定工作量多线程加速的上界 [R2]：

```math
Speedup(t)\leq\frac{1}{s+(1-s)/t}.
```

memory-bound 算子还会在共享带宽达到饱和后停止缩放，ECM 模型对此给出了比简单 Roofline
更细的解释 [R3]。RAMPS 不假设 `P_cpu(k,t)=t*P_cpu(k,1)`，而是一次性实测每类算子在
`t=1,2,3,4` 时的性能和 CPU memory-bandwidth saturation curve。

请求线程数 `t_k` 不是 CPU 资源需求。一个配置为 4 线程的 stage 可能因串行区、同步或
memory stall 只使用部分核心，因此不能使用 `t_k T_CPU` 作为 core-time。定义可测量的
core demand：

```math
D_{core}(u,t)=\sum_{j\in threads}T^{cpu-time}_{u,j},
```

其中 `T_cpu-time` 是线程实际在 CPU 上执行的时间，可由 per-thread CPU clock 或 PMU
`task-clock/cycles` 获得。再定义平均有效并行度：

```math
p_{eff}(u,t)=\frac{D_{core}(u,t)}{T_{wall}(u,t)},
\qquad 1\leq p_{eff}\leq t.
```

这个定义同时有两类经典理论依据。Denning 与 Buzen 的 operational analysis 将资源
service demand 定义为每个完成请求消耗的资源 busy time，并给出 demand law [R19]：

```math
D_r=\frac{U_r}{X},
```

其中 `U_r` 是资源 `r` 的利用率，`X` 是系统吞吐。对 CPU core pool，所有 worker 的
thread CPU time 之和正是可归因于该请求的 core busy time；pipeline 运行时还可用
`总 CPU utilization / throughput` 交叉验证。Blumofe--Leiserson 的 work/span 模型把
总 work 记为 `T_1`、关键路径记为 `T_infinity`，并给出多处理器时间由 `T_1/P` 与
`T_infinity` 共同约束 [R20]。因此 RAMPS 必须同时保留 wall/critical-path stage constraint
和 `sum(D_core)/N_core` 资源约束，不能用请求线程数替代 work。

例如一个 4 线程 stage 的 wall time 为 100 ms，但各线程 CPU time 总和只有 220 ms，则
它的需求是 220 core-ms、平均占用 2.2 cores，而不是 400 core-ms。stage 的 core demand
按最终融合组聚合：

```math
D_{core}(S_k,t_k)=\sum_{g\in S_k}D_{core}(g,t_k)+D_{launch}(S_k).
```

多个 CPU stage 同时运行时，至少满足以下资源下界：

```math
T_{cycle}\geq\frac{\sum_k D_{core}(S_k,t_k)}{N_{core}},
\qquad N_{core}=4,
```

以及：

```math
T_{cycle}\geq\frac{\sum_k Q_{CPU}(S_k)}{B_{cpu,shared}}.
```

第一项表达实测 core-time 竞争，第二项表达共享内存带宽竞争。二者和每个 stage 自身的
wall service constraint 同时进入资源事件图。实际调度抖动由受控并发 microbenchmark
给出，而不是用固定的 `cpu_contention_penalty` 猜测。

### 4.3 可移植 CPU demand 校准

对一块新开发板只执行一次与 DNN 无关的 CPU 校准：

1. 对 conv3x3、conv1x1、elementwise、pool/dense 和 layout/bridge 等算子类别，在
   `t=1..N_core` 下采样不同 shape 和算术强度。
2. 同时记录 wall time、所有 worker 的 thread CPU time、cycles、instructions、cache miss
   和已知 tensor bytes。
3. 得到硬件表 `P_cpu(kind,t)`、`B_cpu(t)`、`D_core_per_op(kind,t,intensity)` 和
   `p_eff(kind,t,intensity)`。
4. 用两类合成 kernel 并发运行，覆盖 compute/compute、compute/memory 和 memory/memory，
   辨识共享 DDR 饱和曲线；该校准依赖硬件，不依赖 ResNet 或 YOLO 层名。

对新 DNN，只从编译后的融合组读取 kind、OP、physical bytes、shape 和线程数，查表或插值
得到 wall service、core demand 和 memory demand。只有硬件指纹变化时才重新校准；更换
DNN 不重新拟合这些参数。

## 5. VTA 服务时间：load/compute/store 与 SRAM

### 5.1 三通道服务分解

VTA 的 task ISA 显式组织 load、compute 和 store task，并允许这些 task 重叠 [R4]：

```math
T_c(u)=\frac{O_u^{phy}}{P_{vta}(k)},\qquad
T_l(u)=\frac{Q_{load}(u)}{B_{pl\rightarrow vta}(a)},\qquad
T_s(u)=\frac{Q_{store}(u)}{B_{vta\rightarrow pl}(a)}.
```

令 `gamma_k in [0,1]` 表示不能重叠的剩余比例，则：

```math
T_{VTA}(u)=T_{max}
+\gamma_k\left(T_c+T_l+T_s-T_{max}\right)
+L_{submit}+L_{sync}+T_{spill},
```

其中：

```math
T_{max}=\max(T_c,T_l,T_s).
```

`gamma=0` 表示完全重叠，`gamma=1` 表示完全串行。它应命名为
`nonoverlap_fraction`，不能称为 overlap factor。

正式校准必须把 compute-only 周期和 DMA service 分开。若 `effective_gops` 已经由包含 DMA
的 kernel 总时间计算，就不能再叠加 `T_l/T_s`，否则会双重计数。允许的实现只有两种：

```text
A. profiler 分解：compute-only GOP/s + 独立 DMA，使用上述公式；
B. 黑盒 effective kernel time，不再显式叠加该 kernel 内部 DMA。
```

RAMPS 正式模型采用 A，B 只作为消融基线。

### 5.2 SRAM 可行性与 spill

对 VTA buffer `b in {INP,WGT,ACC,OUT}` 和 tile `tau`：

```math
W_b(u,\tau)\leq C_b
```

是 tile 可行条件。如果不存在满足全部容量约束的 tile，候选 build-invalid。对可行 tile，
外部流量由 tile schedule 精确计数：

```math
Q_{spill}(u,\tau)=
\max\left(0,Q_{external}(u,\tau)-Q_{compulsory}(u)\right).
```

```math
T_{spill}=\frac{Q_{spill,load}}{B_{load}}
+\frac{Q_{spill,store}}{B_{store}}
+N_{spill,calls}\alpha_{dma}.
```

Hong-Kung Red-Blue Pebble Game 给出了有限快速存储导致 I/O 下界的理论基础 [R6]；Eyeriss、
MAESTRO 和 Timeloop 说明了 DNN tile、reuse、buffer 与外部流量之间的关系 [R7-R9]。
但理论不会自动给出本项目的具体 spill bytes，具体数值必须来自 VTA lowering/tile schedule。

在当前固定 tile 实例中，SRAM 不是切图评分变量：固定 tile 只执行容量合法性检查；其真实
load/store 和重复搬运必须由 lowering 或 profiler 给出并计入 VTA 服务时间。不得继续使用
`SRAM risk penalty`、peak-utilization penalty 或把静态 tile count 当作 spill bytes。由 stage
切点新增的中间 tensor materialization 统一计入 boundary communication，避免重复计费。

## 6. DMA、boundary 与 layout adapter

### 6.1 每次调用固定开销

对访问类型 `a`，DMA 时间采用 latency-bandwidth 模型：

```math
T_{dma}(e,a)=N_{calls}(e)\alpha_a
+\frac{Q_{load}(e)}{B_{load,a}}
+\frac{Q_{store}(e)}{B_{store,a}}.
```

这与 LogP 中 latency、overhead 和 communication gap 的思想一致 [R5]。访问类型至少区分：

```text
large_contiguous
small_tensor
strided
padded
```

每类分别实测 `alpha_a`、`B_load,a` 和 `B_store,a`。不能使用未经测量的固定 16 KiB
阈值或手写 small/strided/padded 权重。DMA fragmentation 的物理含义是同样字节数被拆成
更多调用后增加的 `N_calls * alpha_a`，而不是抽象分数。

### 6.2 Stage boundary

边界 tensor 的 contract 包含 logical/physical shape、dtype、layout、padding 和 quantization：

```math
T_{boundary}(e)=T_{pack}(e)+T_{dma}(e,a)+T_{unpack}(e).
```

```math
Q_{boundary}(e)=N\prod_i shape_i^{physical}\times bytes(dtype).
```

CPU-to-VTA 需要 quantize/pack，VTA-to-CPU 需要 unpack/dequantize/slice，VTA-to-VTA 在兼容时
保留 packed tensor。concat/route 的所有输入必须处于兼容 layout 和 quantization domain，
否则插入 adapter 或拒绝候选。Correctness contract 是搜索约束，不是性能 penalty。

## 7. FIFO、queue depth 与在途帧

每个输入帧在相邻 stage 间产生一个 token。对边界 `e`：

```math
0\leq Produced_e(t)-Consumed_e(t)\leq q_e.
```

当差值达到 `q_e` 时，producer 必须停止，这就是 backpressure。FIFO 内存占用为：

```math
M_{fifo}=\sum_e q_e Q_{boundary}(e).
```

queue depth 不会降低任何 stage 的服务时间，也不能突破真正瓶颈。它只可能吸收短期抖动、
调度延迟或共享资源造成的 burst。对完全确定、资源独立的线性 pipeline，深度 1 通常已经
可以达到 `1/max(stage_time)` 的渐近吞吐；更深队列主要改变启动、排空和抖动敏感性。

Little 定律给出维持目标吞吐所需的平均在途任务数 [R13]：

```math
N_{inflight}=\lambda L_{e2e}.
```

因此一个有用的总并发下界为：

```math
N_{min}=\left\lceil\frac{L_{e2e}}{T_{cycle}}\right\rceil.
```

它不能直接决定每条边的 `q_e`。RAMPS 根据 `N_min`、每条 boundary bytes 和 buffer budget
枚举可行 queue vectors，再由 Max-Plus 模型求解。固定测试 `1/2/4` 只能作为粗粒度消融，
不能声称是理论最优 queue depth。

## 8. 共享资源 Timed Event Graph 与 Max-Plus

Synchronous Dataflow 将 stage 表示为 actor，将 tensor/FIFO 表示为 token channel [R10]。
对于执行顺序和资源需求固定的周期 pipeline，事件时间可写成：

```math
x(n+1)=A\otimes x(n),
```

其中 Max-Plus 加法为 `max`，乘法为普通加法 [R11]。每个资源、worker 或 backpressure
关系形成一个带 token 的环 `c`：

```math
\mu(c)=\frac{\sum_{e\in c}T_e}{\sum_{e\in c}m_e}.
```

稳态 initiation interval 是最大环均值：

```math
T_{mp}=\rho_{max}(A)=\max_c\mu(c),
\qquad FPS=\frac{1000}{T_{mp}}.
```

最大环均值使用 Karp 算法计算 [R12]。由资源容量直接得到以下吞吐下界：

```math
T_{cycle}\geq\max\left\{
\max_k T_{worker,k},
\frac{W_{cpu-core}}{4},
T_{cpu-memory},
\sum_{k:d_k=VTA}T_{VTA,k},
T_{PSPL-load},
T_{PSPL-store},
T_{bridge},
T_{fifo-cycles}
\right\}.
```

多个 VTA island 在一个 VTA mutex token 下累加，而不是彼此并行。CPU stage 可与 VTA
处理不同帧，但仍共享四核 CPU pool、CPU memory 和 PS-PL/bridge 资源。

Max-Plus 的适用条件是确定性服务时间、固定 token rate 和固定资源顺序。Linux 调度、DVFS
和动态仲裁的随机性不属于精确 Max-Plus 线性系统，应通过固定 governor/affinity 降低，并用
重复实验给出不确定性，而不是伪装成确定性公式。

## 9. DNNPipe：独立设备模型、特例与剪枝依据

DNNPipe 针对低延迟、高带宽网络连接的异构 IoT 节点，将 DNN 看作顺序层链，并假设通信
时间小于最大 stage time，从而只优化 stage compute time [R14]。

令参考设备上的第 `i` 层时间为 `e_i`，设备 `d` 相对参考设备的速度为 `c_d`。连续 stage
`sigma=(f,l,d)` 的时间为：

```math
ST(f,l,d)=\frac{\sum_{i=f}^{l}e_i}{c_d}.
```

DNNPipe 的目标为：

```math
P^*=\arg\min_P MST(P),
\qquad MST(P)=\max_{\sigma\in P}ST(\sigma).
```

在设备顺序固定时，可用动态规划计算前 `l` 层分配到前 `d` 个设备的最小最大 stage
时间。令 `F(l,d)` 表示该值，则递推形式为：

```math
F(l,d)=\min_{0\leq f<l}\max\left(F(f,d-1),ST(f+1,l,d)\right).
```

枚举 `f` 就是在选择最后一个 stage 的起点；外层 `min` 寻找最优切点，内层 `max` 对应
pipeline 吞吐由最慢 stage 决定。在其独立设备和通信可忽略假设下，该递推具有最优子结构。

允许层被任意切分时，理想均衡 stage time 为：

```math
IST=\frac{\sum_i e_i}{\sum_d c_d}.
```

由于真实层不可任意切开，DNNPipe 给出最优最大 stage time 的安全上界：

```math
\widehat{MST^*}=IST+\frac{\max_i e_i}{\min_d c_d},
\qquad MST^*\leq\widehat{MST^*}.
```

动态规划扩展 partial plan 时，如果新 stage 已超过该上界，就可以安全剪枝。期刊版本进一步
加入 upper-bound-based 和 under-utilized-stage pruning，并保持其假设下的最优性 [R14]。

### 9.1 共同思想

RAMPS 保留 DNNPipe 的三个核心思想：

```text
以稳态吞吐而不是单帧延迟作为 partition 目标；
以最慢流水阶段为独立资源系统中的基本瓶颈；
用动态规划或 branch-and-bound 避免穷举全部 layer partitions。
```

因此 DNNPipe 不是无关工作的罗列，而是本文最直接的算法基线。实验中应实现
`DNNPipe-compute-only`：使用相同候选、相同设备服务时间，但删除通信边和共享资源约束。
RAMPS 与它的差值才能量化“通信和共享资源感知”本身带来的贡献。

### 9.2 通信可忽略假设为什么在本平台失效

DNNPipe 的 compute-only 目标无法区分 compute stage 完全相同、但 boundary 不同的两个
切分。设候选 `p_1` 和 `p_2` 满足：

```math
\{T_{compute,k}(p_1)\}=\{T_{compute,k}(p_2)\},
\qquad Q_{boundary}(p_1)\ne Q_{boundary}(p_2).
```

则 DNNPipe 给出：

```math
MST(p_1)=MST(p_2),
```

但本平台至少存在如下额外通信服务时间：

```math
T_{comm}(p)=T_{pack}(p)+T_{unpack}(p)
+N_{dma}(p)\alpha_{dma}
+\frac{Q_{load}(p)}{B_{load}}
+\frac{Q_{store}(p)}{B_{store}}.
```

只要 `T_comm(p_1) != T_comm(p_2)`，两个候选的真实周期就可能不同，甚至改变最优切点。
这不是一个可以统一加到最终分数上的常数，因为 boundary tensor 的 shape、物理 padding、
layout、量化域、DMA 调用数和生产者/消费者位置都随切点变化。

通信还可能同时产生三类二阶效应：

1. **阻塞和反压**：有限 FIFO 满后 producer 停止，通信延迟进入 Max-Plus 关键环。
2. **资源竞争**：CPU pack/unpack 与 CPU stage 争用核心和 DDR；DMA 与 VTA load/store 争用
   PS-PL 通道。
3. **语义转换**：CPU/VTA 边界可能需要 quantize、pack、unpack、dequantize 和 padded-channel
   slice；这些既是时间成本，也是 correctness 条件。

因此只满足 `T_comm < max(T_stage)` 并不足以证明通信可忽略。还需要证明通信与计算可以完全
重叠、不会竞争同一资源、不会造成 backpressure，并且不会改变可行切点；本平台不满足这些
条件。

### 9.3 DNNPipe 与 RAMPS 的模型关系

DNNPipe 是 RAMPS 在以下条件同时成立时的特例：

```text
DNN 是顺序层链；
每个 stage 使用互相独立的设备；
设备间通信可以忽略；
设备性能可由单一 scaling factor 表示；
没有 CPU memory、VTA mutex、PS-PL 等共享资源；
FIFO 足够深且服务时间确定。
```

在这个特例下，Max-Plus 关键环退化为最大 worker stage：

```math
T_{mp}=\max_k T_{stage,k}=MST(P).
```

本项目不满足 DNNPipe 的关键假设：CPU/VTA 位于同一 SoC，多个 VTA island 共享同一个
mutex，CPU stage 共享核心和 DDR，boundary/PS-PL 不可忽略，YOLO 还有 route/concat DAG。
因此不能直接用 `max(stage time)` 代替资源图，但可以继承它的优化目标和可证明剪枝思想。

| 维度 | DNNPipe | RAMPS |
|---|---|---|
| 执行环境 | 网络连接的独立异构设备 | 共享 DDR/PS-PL 的 CPU-FPGA SoC |
| 图结构 | 顺序 layer chain | 含 route/concat 的 DNN DAG |
| stage 时间 | 参考时间按设备速度缩放 | 按 OP、bytes、tile、线程和访问类型计算 |
| 通信 | 在假设下从目标中省略 | pack/DMA/unpack/sync 显式事件 |
| 共享资源 | 设备相互独立 | CPU core、DDR、VTA mutex、PS-PL token |
| 吞吐目标 | `min max(stage compute)` | `min rho_max(A(p))` |
| correctness | layer boundary | layout/padding/quantization contract |

本文对 DNNPipe 的批评应严格限定为：其通信可忽略和设备独立假设不适用于本项目平台；不能
声称 DNNPipe 在其目标环境中错误。论文需要通过同候选 paired ablation 证明，忽略通信会造成
排序错误、regret 增大或需要更多上板测试，而不是只通过文字宣称。

### 9.4 RAMPS 的安全 branch-and-bound

任何已构造的完整可行方案给出最优周期上界 `UB`。对 partial plan `r`，构造资源下界：

```math
LB(r)=\max\left\{
T_{mp}(r),
\frac{W_{cpu,remaining}}{4},
T_{vta,remaining},
T_{dma,remaining},
T_{minimum-required-boundary}
\right\}.
```

若：

```math
LB(r)\geq UB,
```

则该 partial plan 的所有扩展都不可能优于 incumbent，可安全剪枝。每个 lower-bound 项必须
证明不会高估 remaining cost；不能把经验 penalty 放入安全剪枝条件。经验分数只能用于
搜索顺序，不能用于删除候选。

## 10. 一次性硬件校准

### 10.1 需要直接测量的参数

```text
S_cpu[signature,threads]             fused-unit compute/core service surface
B_cpu[threads,working_set,policy]    isolated cache/DRAM bandwidth surface
S_vta[signature,schedule]            device compute service surface
B_dma[access,direction,bytes,calls]  effective PS-PL bandwidth surface
alpha_dma[access,direction]          per-call latency
adapter[conversion_signature,threads,access]
                                      pack/unpack/requantize service surface
submit/sync                           fixed and policy-dependent host overhead
C_INP/C_WGT/C_ACC/C_OUT              exact SRAM capacities
```

上述是 Stage 4 的单资源参数。`gamma_vta`、多 CPU stage 的 DDR/core contention、VTA mutex
ordering 和 FIFO/backpressure 参数不从单 kernel 校准中推断，而在 Stage 5 受控 pipeline 中辨识。

CPU/VTA 不再按 `stem/residual/skip projection` 等 DNN 语义分桶。校准自变量是通用 primitive
及数值 workload signature：`Cin/Cout/H/W/kernel/stride/groups/threads/OP/bytes/tile/calls`。
DMA strata 只表示 contiguous/small/strided/padded 等访问行为。每个服务点必须由真实
benchmark 支撑，不允许从其他 primitive 或默认 GOP/s 静默填充。

### 10.2 校准与新 DNN 的分离

硬件校准的正式输出：

```text
hardware_fingerprint.json
hardware_profile.json
calibration_raw.csv
calibration_uncertainty.json
```

`hardware_profile.json` 使用 schema version 3，并分别通过
`correctness_gate_passed`、`precision_gate_passed` 和 `service_fit_validated` 后才可用于正式
预测。旧 `service_model.json` 是 ResNet-shaped calibration point archive，不是可移植服务函数。

新 DNN 只提供：

```text
operator DAG
logical/physical tensor schema
OP/byte counts
fusion and tile candidates
boundary contracts
```

然后代入同一个 `hardware_profile.json`。这就是跨模型迁移的核心：迁移的是硬件资源模型，
不是用 ResNet18 throughput 标签训练出来的模型名称相关回归器。

### 10.3 完整画像协议是条件增强

protocol v4 的 212 个 case 是完整 HardwareProfile 路线的 semantic pool：它定义希望支持的 primitive、shape、threads、
bytes/calls、access geometry 和 conversion signatures。它既不是全量 compile 清单，也不是
全量上板清单。H2 只审计 20 个 builder equivalence-class 代表；数值点覆盖由实际 lowering
feature manifest 和越界拒绝共同约束。

该 pool、20-template H2 和 80-case H3 当前全部暂停。新的主路径先对已有候选完成数据泄漏审计
和低预算曲线；只有轻量 M0/M1 的误排能够归因于 service surface 缺失时，才恢复本节协议。
因此下面的 80-case 约束只适用于“决定执行完整画像”后的协议，不是当前候选搜索预算。

从 pool 中选择正式校准子集时，必须使用：

```text
parameter identifiability and Jacobian rank
D/G-optimal information gain
compiler-lowering signature coverage
predicted variance and pre-registered uncertainty thresholds
frozen grouped holdout coverage
```

上板预算固定为最多 80 个 case：默认最多 64 个 fit，至少 16 个 grouped holdout。
`measurement_plan.json` 必须绑定 observed feature manifest 和 protocol SHA256。只有 holdout
按预注册规则失败且用户确认后，才允许建立独立扩展阶段；不得根据目标 partition throughput
选点或临时补点。

论文必须同时报告 semantic-pool 规模、H2 模板成功率、实际校准 case 数、上板总时间、
holdout 误差和 profile 被复用的 DNN 数量。

## 11. 系统参数辨识与最优实验设计

### 11.1 不再使用固定 12 anchors

farthest-point 只近似最小化静态特征空间的 covering radius [R15]。它不能保证每个资源成为
关键瓶颈，也不能保证模型参数可辨识。固定“每个 island count 选 4 个”没有统计学保证，
仅保留为 space-filling baseline。

主物理参数不得使用 ResNet18、YOLOv3-tiny 或其他目标 DNN 的完整 partition throughput 标签
辨识。Stage 5 的配置由 Stage 4 已校准的通用 CPU/VTA/DMA/bridge bucket 组合成受控合成
pipeline；这些 pipeline 改变资源混合、依赖、线程和 FIFO，但不对应某个模型名称或切点。
ResNet18 历史候选只用于冻结后的回顾性验证，不能回流修改主模型参数。

对待辨识参数 `theta` 和通用实验配置
`x_i=(service_mix,dependency,q,threads)`，计算局部灵敏度：

```math
g_i=\frac{\partial T_{mp}(x_i;\theta)}{\partial\theta}.
```

加权信息矩阵为：

```math
M(X)=\sum_{i\in X}\frac{g_i g_i^T}{\sigma_i^2}.
```

配置选择使用最优实验设计 [R16]：

```math
D\text{-optimal}:\quad \max_X \log\det(M(X)+\lambda I),
```

```math
G\text{-optimal}:\quad
\min_X\max_{j\in\mathcal X}g_j^T(M(X)+\lambda I)^{-1}g_j.
```

前者减小参数联合置信椭球，后者减小候选空间中的最坏预测方差。实验开始前必须报告：

```text
rank(J) == number_of_free_parameters
condition number of information matrix
parameter confidence intervals
maximum standardized prediction variance
```

样本数由信息矩阵和目标置信度决定，不再由“12 anchors”预先指定。

### 11.2 Thread 与 queue 共同设计

线程候选来自受控合成 pipeline 的所有可行 CPU thread vectors：

```math
t_k\in\{1,2,3,4\}.
```

同时包含 `sum(t_k)<=4` 的非超配配置和少量超配 control。D/G-optimal 选择对 CPU scaling、
memory saturation 和 contention 参数最有信息量的组合，不再只比较 `tuned` 与 `single`。

queue 候选由 Little 下界和内存预算产生：

```math
N_{min}=\left\lceil L_{e2e}/T_{cycle}\right\rceil,
\qquad
\sum_e q_eQ_e\leq M_{queue-budget}.
```

然后把 queue vector 与 service-mix/thread vector 一起加入实验设计。重复 session 用于估计
噪声，不会增加设计矩阵秩；真正的参数可辨识性来自不同资源灵敏度的配置。只有当主物理模型
冻结后，才允许在单独消融中使用 ResNet18 标签拟合 monotonic residual；该 residual 不得参与
严格 cross-DNN zero-shot 主结果。

## 12. 不确定性与可选残差

物理参数的不确定性来自独立校准 session 的 median/MAD 或 bootstrap。对参数采样
`theta^(b)`，传播到候选周期：

```math
T^{(b)}(p)=T_{mp}(p;\theta^{(b)}).
```

由这些样本直接得到均值、标准差和预测区间。该过程不需要新 DNN 标签。

若系统性偏差仍存在，可增加只依赖物理量的单调残差：

```math
r(x)=\beta_0+\sum_j\sum_k\beta_{jk}\max(0,x_j-\kappa_{jk}),
\qquad \beta_{jk}\geq0.
```

shape-constrained GAM 的理论依据见 [R17]，split conformal prediction 可用于覆盖率校准 [R18]。
残差只能使用 fragmentation、spill、transaction size、oversubscription 等可迁移物理特征，
不得使用模型名、层名、候选 ID 或新 DNN 实测 stage time。它是可选修正项，不是系统模型
成立的前提。

当前论文以**无残差的物理模型**为主结果。Linux 调度、偶发 cache miss 和 DMA 抖动通过
固定运行环境、独立重复、median/MAD 和 bootstrap 区间处理，不允许由 residual 吸收。
单调 residual 仅作为单独消融；只有冻结后在跨模型 held-out 数据上显著改善误差、排序和
regret 时，才能报告为增强版本，不能替换物理主模型。

## 13. 新 DNN 的求解流程

完整框架的通用流程为：

```text
1. 完整图量化并验证 all-VTA/CPU reference correctness。
2. 从编译后图生成 correctness-safe compute units 和 boundary contracts。
3. 提取 logical/physical OP、bytes、layout、padding、quantization 和 tile 候选。
4. 枚举 CPU/VTA assignment、1-3 个 VTA islands、thread vectors 和 queue vectors。
5. 拒绝 SRAM 不可行、adapter 不可构造或 build schema 不一致的候选。
6. 用一次性硬件模型计算 CPU/VTA/DMA/boundary 服务时间。
7. 构造资源需求矩阵和 Timed Event Graph，求最大环均值。
8. 使用 DNNPipe 风格的 admissible lower bound 和 incumbent upper bound 剪枝。
9. 按 risk-aware cycle 排序，保留少量随机/边界/粒度 control。
10. package-only build 后，只上板验证有限 shortlist。
```

当前论文 V1 固定 schedule 图级实例采用以下受限流程：

```text
1. 冻结量化、compiler pass、VTA schedule/tile、queue depth 和 runtime。
2. 从编译后图生成 correctness-safe compute units 和 boundary contracts。
3. 对 DP 可达 segment 做 compile-only lowering，生成 fusion/materialization、boundary 和 DDR
   transaction manifest。
4. 对去重 CPU/VTA/boundary signature 建立局部成本表；CPU 覆盖 1--4 核心。
5. 使用 Tarnawski-style DP 联合选择 CPU/VTA cuts、VTA islands 和 CPU 核心分配。
6. 目标为 CPU 最大 stage、单物理 VTA 串行 service 加 boundary、共享 DDR demand 三者最大值。
7. 对 DP Top-K 执行 native compile/reference gate 和有限 Pipeline 上板验证。
8. 固定 tile、pairwise DDR contention、FIFO 和完整 Timed Event Graph 仅作为条件 V2 扩展。
```

每个候选必须输出：

```text
predicted_cycle_ms / predicted_fps
critical max-plus cycle and bottleneck resource
CPU compute/memory bound classification
VTA compute/load/store bound classification
PS-PL bytes, calls, bandwidth and fragmentation
INP/WGT/ACC/OUT tile utilization and spill bytes
boundary contracts and adapter costs
thread vector, queue vector and queue memory
uncertainty interval
```

## 14. 可验证主张

### 14.1 统一模型消融与最近工作对照

使用完全相同的候选集合、硬件服务参数和上板标签，依次比较：

```text
A0: DNNPipe-compute-only        max(stage compute)
A1: nominal-bandwidth           A0 + tensor bytes / nominal bandwidth
A2: calibrated-DMA              A0 + direction/bytes/calls/access-kind service
A3: system-aware serial         A2 + bridge/coherence/submit/sync 串行路径
A4: RAMPS max-plus              完整通信、共享资源、FIFO 和 backpressure
A5: PartitionTuner-style        grouped-subgraph aggregate profile + branch parallelism
```

`A5` 使用相同 legal candidates、native package 和 backend，不复现另一个编译器的代码质量；
它隔离的是 aggregate subgraph profile、顺序 partition 和同帧 branch parallelism 这一策略。
`A2/A3/A4` 用来区分“理想带宽不足”“漏加系统开销”和“通信进入资源时序后改变流水关键环”
三个问题。至少构造或从历史数据中匹配以下 counterfactual candidate pairs：compute balance 接近，
但 boundary bytes、DMA calls、layout adapter 或 FIFO 压力显著不同。报告每组模型的：

```text
cycle MAE / Spearman / Kendall
Top-K recall and regret
compute-only 预测并列但实测显著分离的 pair 数量
加入 communication 后被纠正的 pair 数量
达到 oracle 95% throughput 所需的 board evaluation 数
```

只有 `A4` 相对 `A0/A2/A3/A5` 的 paired bootstrap 改善显著，才能把“DNNPipe 式
compute-only 或 PartitionTuner 式 aggregate profile 不足，必须建模通信和共享资源时序”作为
论文贡献。若 `A3` 已达到相同性能，则证据只支持系统通信感知，不支持更复杂的 Max-Plus
共享资源模型。若 serial 最优与 pipeline 最优没有稳定差异，论文也必须把 pipeline-aware
search 降级为实现机制，而不是强行列为已成立贡献。

### 14.2 主张边界

RAMPS 可以主张：

```text
一次性校准的 CPU/VTA/DMA/存储资源模型可以跨 DNN 预测候选瓶颈；
Max-Plus 资源图比单纯 max(stage time) 更准确地描述共享 SoC pipeline；
DNNPipe 风格的可证明上下界能够安全缩小切分搜索空间；
资源模型在未见 DNN 上降低达到 oracle 95% 吞吐所需的上板次数。
```

RAMPS 不能在没有证据时主张：

```text
任意 DNN 或任意 TVM schedule 的精确周期预测；
固定 12 anchors、1/2/4 queue 或 tuned/single 是理论最优实验设计；
经验 DMA 权重等价于硬件传输模型；
更深 FIFO 必然提升稳态吞吐；
多个 VTA island 能在单 VTA mutex 下并发执行。
```

### 14.3 预注册的创新边界

若对应证据门槛通过，论文只主张以下三点：

1. **可移植的系统服务模型**：一次硬件校准得到 CPU core demand、VTA
   load/compute/store、DMA、bridge 和同步服务；新 DNN 只提供编译后物理图属性。
2. **面向稳态吞吐的共享资源 Max-Plus 模型**：通信是具有依赖、资源占用和可隐藏性的事件，
   而不是 stage time 上的手写 penalty。
3. **资源感知的合法图划分搜索**：在固定 schedule/runtime 和 correctness-safe boundary
   contract 下，使用 admissible bound 缩小声明的 assignment/cut 空间并降低上板 regret。

native executor、boundary adapter、VTA mutex、build cache 和自动失败记录作为系统实现贡献
支撑上述三点，不单独声称“首次实现 pipeline”或“首次支持多 request”。

## 15. 规范化可移植架构

RAMPS 分成三个严格解耦的接口，禁止把模型名或某个切点写入评分公式。

### 15.1 HardwareResourceModel

对每个硬件指纹执行一次自动校准并冻结 JSON：

```text
CPU: wall service, core-time demand, effective parallelism,
     cache/DDR traffic and saturation by generic op kind/thread/intensity
Accelerator: physical compute, load/store service, overlap and legal tiles
Communication: bytes/calls/stride/padding -> bandwidth/latency,
               pack/unpack/coherence/submit/sync
Topology: core count, memory domains, accelerator mutex/channels,
          SRAM capacities and runtime resource tokens
```

硬件指纹至少包含 CPU/DDR/PL 时钟、bitstream/VTA config、memory port/coherence、编译器、
runtime、target flags 和 OS 调度配置。指纹变化时重新校准参数，不修改求解算法。

### 15.2 DNNResourceGraph

对每个新 DNN 只做编译和静态提取，不读取候选吞吐标签：

```text
compiler-fused DAG and dependencies
logical/physical shape, OP and compulsory bytes
layout/padding/quantization/boundary contract
tile/SRAM/spill and DMA transaction estimates
legal CPU/accelerator assignment and stage cuts
```

TVM 已说明通过统一编译栈、图级融合和硬件后端实现 performance portability [R21]；RAMPS
在其编译结果之上增加 pipeline resource graph，而不是依赖 ResNet/YOLO 层名。

### 15.3 ResourceGraphSolver

固定算法完成：correctness-safe 候选枚举、服务时间计算、资源 token/event graph 构造、
maximum-cycle-mean、置信区间、branch-and-bound 和 Top-K 排序。输出包括 split、device、
threads、queue vector、tile、预测 FPS、关键资源环和不确定性。

### 15.4 两种部署协议

```text
严格 cross-DNN zero-shot:
  只使用通用硬件 calibration + 新 DNN 静态图，不运行新 DNN 的性能标签。

label-free target-aware calibration:
  允许从新 DNN 提取独特 operator signatures 并单独 microbenchmark，
  但不运行或拟合任何完整 partition candidate throughput。
```

后者更容易落地，但论文必须准确命名，不能称为严格 zero-shot。Habitat 证明了基于设备
执行模型和算子缩放进行跨设备 DNN 性能预测是可行研究路线 [R22]，但 RAMPS 的目标对象是
CPU--FPGA 稳态 pipeline 和共享资源，而非 GPU training iteration。

### 15.5 可移植性验收

```text
同板跨 DNN：冻结硬件模型后依次测试 ResNet18、YOLOv3-tiny、SqueezeNet；
跨硬件：更换第二种 VTA config/频率/端口或第二块板，只替换 calibration JSON；
代码审计：特征中不存在 model/layer/candidate 名称和目标候选 FPS；
效果：报告 Top-K recall/regret、达到 oracle 95% 所需上板次数和校准成本；
越界处理：新算子/shape 超出 calibration domain 时扩大不确定性或补 profile，不能静默外推。
```

对 GPU/DSP 或非 VTA FPGA，允许增加 backend-specific profiler/contract adapter；资源图、
Max-Plus 求解和统计协议必须保持不变。论文可移植性范围应明确为“支持已实现 adapter 的
异构后端”，不能声称零修改支持任意 accelerator。

## 16. 当前实现与模型缺口

### 16.1 Queue depth 已真实实现

native runner 为 `N` 个 stage 创建 `N+1` 个 `BoundedQueue<shared_ptr<Frame>>`。每个队列
容量等于 `--queue-depth`：`Push` 在 full 时等待，`Pop` 在 empty 时等待；每个 stage 有独立
worker thread。因此 queue depth 已影响真实调度和 backpressure，不是静态模型虚构参数。

当前限制是：所有边使用同一个 depth；没有分别记录 `Pop-empty wait` 和 `Push-full wait`；
`Frame` 会保留 live stage outputs，因此内存开销不能简单写成单个 boundary bytes 乘 depth。
正式 FIFO 建模前必须补等待计时和 live-tensor memory accounting。

### 16.2 CPU demand 已接入，正式数据仍待校准

`resource_aware_maxplus.py` 已为 `StageService` 增加 `core_demand_ms` 和 source，并用
`sum(core_demand_ms)/N_core` 构造 CPU core-pool constraint。旧 record 仍可回退到
`stage_service * requested_threads`，但会明确标记
`legacy_wall_time_x_requested_threads`；publication mode 会拒绝该回退。Stage 4b 仍需为全部
CPU bucket 生成正式 process/thread CPU-time 参数，之后 CPU contention 才能标为
publication-valid。

### 16.3 已成立与未成立

直接 CPU--VTA copy 通信项的价值已经由 200 个 ResNet18 profiler 记录显著验证；完整 A4、
CPU contention、FIFO 参数和 cross-DNN zero-shot 尚未通过各自 gate。详见
[PAPER_EVIDENCE.md](PAPER_EVIDENCE.md)。

## 17. 参考文献

- **[R1]** S. Williams, A. Waterman, and D. Patterson, "Roofline: An Insightful
  Visual Performance Model for Multicore Architectures," *Communications of the
  ACM*, 2009. [PDF](https://people.eecs.berkeley.edu/~kubitron/courses/cs252-S09/handouts/papers/RooflineVyNoYellow.pdf)
- **[R2]** G. M. Amdahl, "Validity of the Single Processor Approach to Achieving
  Large Scale Computing Capabilities," *AFIPS*, 1967.
  [DOI](https://doi.org/10.1145/1465482.1465560)
- **[R3]** G. Hager, J. Treibig, J. Habich, and G. Wellein, "Exploring Performance
  and Power Properties of Modern Multicore Chips via Simple Machine Models."
  [arXiv](https://arxiv.org/abs/1208.2908)
- **[R4]** T. Moreau et al., "A Hardware-Software Blueprint for Flexible Deep
  Learning Specialization," 2018. [arXiv](https://arxiv.org/abs/1807.04188)
- **[R5]** D. Culler et al., "LogP: Towards a Realistic Model of Parallel
  Computation," *PPoPP*, 1993. [DOI](https://doi.org/10.1145/155332.155333)
- **[R6]** J.-W. Hong and H. T. Kung, "I/O Complexity: The Red-Blue Pebble Game,"
  *STOC*, 1981. [DOI](https://doi.org/10.1145/800076.802486)
- **[R7]** Y.-H. Chen, T. Krishna, J. Emer, and V. Sze, "Eyeriss: An
  Energy-Efficient Reconfigurable Accelerator for Deep Convolutional Neural
  Networks," *JSSC*, 2017. [DOI](https://doi.org/10.1109/JSSC.2016.2616357)
- **[R8]** H. Kwon et al., "Understanding Reuse, Performance, and Hardware Cost of
  DNN Dataflows: A Data-Centric Approach," *MICRO*, 2019.
  [PDF](https://users.cs.duke.edu/~lkw34/papers/maestro-micro2019.pdf)
- **[R9]** A. Parashar et al., "Timeloop: A Systematic Approach to DNN Accelerator
  Evaluation," *ISPASS*, 2019.
  [PDF](https://people.csail.mit.edu/anurag_m/papers/2019.timeloop.ispass.pdf)
- **[R10]** E. A. Lee and D. G. Messerschmitt, "Static Scheduling of Synchronous
  Data Flow Programs for Digital Signal Processing," *IEEE Transactions on
  Computers*, 1987. [Paper](https://ptolemy.berkeley.edu/publications/papers/87/staticscheduling/)
- **[R11]** F. Baccelli, G. Cohen, G. J. Olsder, and J.-P. Quadrat,
  *Synchronization and Linearity: An Algebra for Discrete Event Systems*, 1992.
  [Book](https://www.rocq.inria.fr/metalau/cohen/SED/book-online.html)
- **[R12]** R. M. Karp, "A Characterization of the Minimum Cycle Mean in a
  Digraph," *Discrete Mathematics*, 1978.
  [DOI](https://doi.org/10.1016/0012-365X(78)90011-0)
- **[R13]** J. D. C. Little, "A Proof for the Queuing Formula: L = lambda W,"
  *Operations Research*, 1961. [DOI](https://doi.org/10.1287/opre.9.3.383)
- **[R14]** W. Seo, S. Kim, and S. Hong, "DNNPipe: Dynamic Programming-Based
  Optimal DNN Partitioning for Pipelined Inference on IoT Networks," *Journal of
  Systems Architecture*, vol. 166, 103462, 2025.
  [DOI](https://doi.org/10.1016/j.sysarc.2025.103462)
- **[R15]** T. F. Gonzalez, "Clustering to Minimize the Maximum Intercluster
  Distance," *Theoretical Computer Science*, 1985.
  [DOI](https://doi.org/10.1016/0304-3975(85)90224-5)
- **[R16]** J. Kiefer, "Optimum Experimental Designs," *JRSS Series B*, 1959.
  [DOI](https://doi.org/10.1111/j.2517-6161.1959.tb00338.x)
- **[R17]** N. Pya and S. N. Wood, "Shape Constrained Additive Models,"
  *Statistics and Computing*, 2015.
  [DOI](https://doi.org/10.1007/s11222-013-9448-7)
- **[R18]** J. Lei, M. G'Sell, A. Rinaldo, R. Tibshirani, and L. Wasserman,
  "Distribution-Free Predictive Inference for Regression," *JASA*, 2018.
  [PDF](https://www.stat.berkeley.edu/~ryantibs/papers/conformal-jasa.pdf)
- **[R19]** P. J. Denning and J. P. Buzen, "The Operational Analysis of
  Queueing Network Models," *ACM Computing Surveys*, 1978.
  [DOI](https://doi.org/10.1145/356733.356735)
- **[R20]** R. D. Blumofe and C. E. Leiserson, "Scheduling Multithreaded
  Computations by Work Stealing," *Journal of the ACM*, 1999.
  [DOI](https://doi.org/10.1145/324133.324234)
- **[R21]** T. Chen et al., "TVM: An Automated End-to-End Optimizing Compiler
  for Deep Learning," *OSDI*, 2018.
  [Paper](https://www.usenix.org/conference/osdi18/presentation/chen)
- **[R22]** G. Yu et al., "Habitat: A Runtime-Based Computational Performance
  Predictor for Deep Neural Network Training," *USENIX ATC*, 2021.
  [PDF](https://www.usenix.org/system/files/atc21-yu.pdf)
- **[R23]** M. Yu et al., "PartitionTuner: An Operator Scheduler for
  Deep-Learning Compilers Supporting Multiple Heterogeneous Processing Units,"
  *ETRI Journal*, vol. 45, no. 2, pp. 318--328, 2023.
  [DOI](https://doi.org/10.4218/etrij.2021-0446)
- **[R24]** G. Zhong et al., "Synergy: A HW/SW Framework for High Throughput
  CNNs on Embedded Heterogeneous SoC," *ACM TECS*, vol. 18, no. 2, 2019.
  [DOI](https://doi.org/10.1145/3301278)
- **[R25]** J. Tarnawski et al., "Efficient Algorithms for Device Placement of
  DNN Graph Operators," 2020. [arXiv](https://arxiv.org/abs/2006.16423)
