# 硕士论文研究主线、既有实验与新创新点整合总结

更新日期：2026-09-11。

本文档面向论文《面向深度神经网络的嵌入式异构计算共享内存技术研究》，以仓库中已经冻结的实验记录为依据，重新区分：

- 已经有实验支撑的主要创新；
- 应并入主要创新的子机制和工程证据；
- 只能作为负结果、边界或后续工作的探索；
- 新提出但尚未完成验证的候选创新。

本文档不把“做过一个实验”自动等同于“形成一个创新点”，也不把未通过门槛的结果写成已成立结论。

## 1. 总体判断

现阶段最适合硕士论文的结构是“两项已有主创新 + 一项新候选创新”：

1. **面向共享资源与通信代价的 CPU--VTA 流水线 Top-K 划分方法**：已有证据最完整，解决“网络切在哪里、CPU/VTA怎样分工”。
2. **面向跨 Executor 多帧执行的生命周期与需求感知共享内存运行时**：已有零拷贝、双 slot、输入池复用和命令队列缩容证据，解决“切开以后如何安全交接，以及需要占用多少共享内存”。
3. **面向显式 DMA 共享内存的 VTA 驻留--请求--命令协同调优方法**：这是本轮新提出的候选创新。它以复现输入优先和片上权重复用 schedule 为起点，进一步解决“不同 workload 应驻留什么、以什么请求形态访问 u-dma-buf、生成多少命令，以及局部优化能否改变 stage/流水性能”。

三者构成同一条共享内存研究主线，而不是三个互不相关的小修补：

```text
整网层：哪些算子放 CPU/VTA，产生哪些共享内存边界
   ↓
运行时层：边界张量如何零拷贝、安全轮转并控制空间占用
   ↓
加速器层：VTA 如何分块并复用片上数据，减少对共享内存的 DMA 请求
```

最简洁的论文故事是：

> 本文以 u-dma-buf 提供的 CPU--FPGA 物理共享内存为基础，分别从异构任务划分、跨执行器数据生命周期和加速器访存组织三个层次，减少不必要的共享内存边界、消除边界冗余物化、压缩过量空间预留，并降低 VTA 对共享 DDR 的重复和碎片化访问。

## 2. 统一问题模型

### 2.1 时间上限

单帧时延可分解为：

```math
L = T_{CPU}+T_{VTA}+T_{boundary}+T_{DMA}+T_{sync}.
```

连续多帧执行时，FPS 不再由各项简单相加决定，而由稳态最慢资源或最慢阶段限制：

```math
II(P) \ge
\max\left(
D_{CPU\text{-}pool}(P),
D_{single\text{-}VTA}(P),
D_{shared\text{-}DDR}(P),
\max_j T_{stage,j}(P)
\right),
```

```math
FPS(P) \le \frac{1}{II(P)}.
```

因此，减少某一处 copy 或 DMA 只说明局部服务量下降；只有该服务位于单帧关键路径，或超过流水线 slack 并成为稳态瓶颈时，才会转化为 latency 或 FPS 提升。

### 2.2 空间上限

u-dma-buf 池内高水位可分为：

```math
M_{peak}=
M_{graph\ pool}+
M_{boundary\ slot}+
M_{instruction}+
M_{uop}+
M_{other\ runtime}.
```

双 slot 解决正确性和复制问题；输入池复用减少 `M_boundary slot`；编译期/运行期峰值定容减少 `M_instruction+M_uop`。三者评价分母不同，必须分别报告新增 slot 降幅、总池高水位降幅和系统固定预留，不能把其中一个比例替代另一个。

### 2.3 VTA 访问共享内存的时间

VTA 执行 LOAD/STORE 时，通过 AXI 在 u-dma-buf 和片上 SRAM 之间搬运数据。访存成本不仅由总字节数决定，还受请求数量和粒度影响：

```math
T_{DMA}\approx
\frac{B_{DMA}}{BW}
+N_{req}\tau_{req}
+N_{small}\tau_{small}
+N_{stride}\tau_{stride}.
```

因此，降低总 payload、减少输入/权重重复载入、合并小请求和避免不利 stride 都可能改善时间；仅报告“DMA 字节减少”不足以解释性能。

## 3. 创新点一：面向共享资源与通信代价的 CPU--VTA 流水线 Top-K 划分

### 3.1 研究问题

在固定 VTA 硬件、量化和基础 schedule 的条件下，联合决定：

- 连续 CPU/VTA segment；
- VTA island 数量；
- 每个 CPU stage 的 TVM 线程参数；
- CPU--VTA 边界及其共享内存通信代价。

传统的“逐层选择最快设备”忽略了四个事实：多个逻辑 VTA stage 共享一个物理 VTA、多个 CPU stage 共享四个 CPU 核、异构切换产生边界服务、连续推理由稳态资源瓶颈决定。

### 3.2 方法增量

1. 将 ResNet18 表示为 21 个中小粒度、依赖闭合的计算单元。
2. 枚举编译合法的连续 CPU/VTA segment、VTA island 和 per-stage thread。
3. 显式建模单 VTA 串行 demand、CPU core-time、方向化 CPU--VTA 边界和共享 DDR demand。
4. 利用非负服务量的单调下界执行 k-best 动态规划，仅输出 Top-K 上板候选。
5. 使用编译上下文、workload 和 DMA signature 去重，避免对相同 VTA workload 按 stage 重复测量。

FuseOps、量化、packing 和 AutoTVM 是复用的编译基础，不属于该创新点本身。

### 3.3 已有实验与证据

| 实验 | 结果 | 当前能支持的结论 |
|---|---:|---|
| ResNet18 搜索空间 | 4623 种合法 topology、972528 个线程化配置 | 问题规模和联合决策真实存在 |
| DP 对完整枚举 | Top-20 候选与分数逐项一致 | 固定静态目标下 k-best 搜索正确 |
| 自然 Top-20 上板 | 10.220--11.511 FPS，前 10 含实测池最佳 | shortlist 能覆盖当前高性能区域 |
| Top-20 细排 | Spearman 仅 0.155 | 尚不能声称准确预测绝对 FPS 或精确细排 |
| 200 条通信消融 | MAE 17.3617→12.8220 ms，改善 26.15%；Spearman 0.7700→0.9310 | CPU--VTA direct boundary copy 必须建模 |
| island 数与 copy | 1/2/3 island 的串行 copy 中位数约 3.431/6.212/10.959 ms | 多次重入 VTA 会增加边界代价 |
| 同 topology 线程扫描 | t1/t2/t3/t4 为 5.413/6.237/9.312/10.449 FPS | CPU stage 线程数必须进入搜索，不能假设线性缩放 |
| YOLOv3-tiny 回放 | 105696 个配置；6 组目标组件 profile 后，历史最优 topology 静态第 6 | 提供跨模型可迁移线索，但仍是回顾性、非零样本前瞻证据 |

### 3.4 建议的正式创新表述

> 针对嵌入式 CPU--FPGA 共享内存平台中逻辑阶段多于物理计算资源、CPU 核与 VTA 被多个阶段复用且异构边界代价不可忽略的问题，本文提出资源与通信代价感知的 Top-K 流水线划分方法。该方法联合搜索连续 CPU/VTA segment、VTA island 和 CPU stage 线程参数，以单物理 VTA、CPU 核心池、方向化边界和共享 DDR 服务需求构造稳态资源下界，并采用 k-best 动态规划从大规模合法配置中生成少量候选供实机验证。

### 3.5 主张边界

可以声称：静态目标下 DP 正确；通信项显著改善历史数据解释；Top-20 覆盖高性能区域。

不能声称：找到硬件全局最优；准确预测绝对 FPS；已经实现完整 stage--tile 全空间联合搜索；已经完成跨 DNN 零样本泛化。

## 4. 创新点二：面向跨 Executor 多帧执行的生命周期与需求感知共享内存运行时

该创新点由一个核心协议和两个空间增强组成。三者不应拆成三个独立创新点。

### 4.1 核心机制：跨 Executor 双 slot 零拷贝

TVM 已有 `set_input_zero_copy` 等接口，双缓冲也不是本文首次提出。本文的增量是将以下对象连接成一个可执行协议：

- 切图 manifest 与边界 tensor contract；
- u-dma-buf 中的物理共享 slot；
- CPU/VTA 同址 `DLTensor` 视图；
- 多个独立 GraphExecutor；
- `{edge, slot, generation}` 令牌队列；
- `FREE→WRITING→READY→USING→FREE` 所有权状态机；
- 双 slot 跨帧轮转；
- 残差边界多张量的原子发布与回收。

它同时解决跨 Executor 重复物化和多帧覆盖，而不是仅把一次 `memcpy` 换成指针赋值。

### 4.2 空间增强一：复用独占 consumer 输入池作为 slot0

普通 K2 双 slot 为一条边额外申请两个 external slot。若静态审计和启动期物理检查证明某个 CPU→VTA consumer 输入池独占、非参数、无冲突、接口和对齐一致，则把该池作为 slot0，只新增 external slot1；任何检查失败时整边回退 external-K2。

这不是重新实现 USMP：没有重排算子内部 workspace 或求解通用静态地址规划。本文复用既有 graph storage 信息，补的是跨 GraphExecutor、异步 VTA 和多帧 generation 条件下的安全绑定与所有权规则。

实测结果：

| topology | 新增 slot 分配下降 | 总池内高水位下降 |
|---|---:|---:|
| A | 44.44% | 1.030% |
| B | 36.36% | 1.027% |
| C | 40.00% | 1.028% |
| D | 38.46% | 1.215% |

四 topology 每个 1000 帧，合计 4000 帧、16000000 B 最终输出逐字节一致，warmup 后池内高水位无增长，并观察到真实 slot0/slot1 跨帧交错。

### 4.3 空间增强二：按真实峰值确定 instruction/UOP 队列容量

原 VTA runtime 为 instruction 和 UOP backing 分别固定预留 32 MiB，总计 64 MiB。诊断得到四 topology 的默认峰值仅为：

- instruction：5312 B；
- UOP：2792 B。

Q1 采用两倍余量和 256 B 对齐，把总队列缩到 16 KiB；Q2 进一步在实际切批、依赖闭合、FINISH 和重放检查后冻结 T2656：

```text
instruction backing = 5376 B
UOP backing         = 5632 B
total               = 11008 B
```

四 topology 的 64 MiB→16 KiB 实验使总池内高水位下降 82.23%--86.96%；T2656 又比 16 KiB 少 5376 B。Q1 与 T2656 均完成四 topology 各 1000 帧自然交错正确性测试。首个完整性能 boot 中各 topology 的 II 点变化均低于 2%，但三 boot 非劣效检验没有完成，所以不能声称统计意义上的性能无损。

### 4.4 零拷贝核心实验

| 实验 | 结果 | 解释 |
|---|---:|---|
| 固定代表候选三 boot | 框架物化 1806336 B/frame→0 | 100% 消除该层冗余复制 |
| 边界 API service | 1.446899→0.041830 ms，下降 97.11% | 核心机制稳定有效 |
| 稳态 II | 95% 区间包含 0 | copy 位于流水线 slack 内时，不能宣称 FPS 提升 |
| 三 VTA-island 串行压力 | 9031680 B/frame 物化；167.399→162.662 ms，观察下降 2.83% | 边界位于单帧关键路径时可转化为 latency 收益；只有一个 boot |
| 异常退出 | 直接重启输出错误；重载相同 bitstream 后 A/B/C/D 恢复 | 正确恢复语义是 fail-stop→重配置 FPGA→重建运行时，不是进程级自动恢复 |

VTA 内部 DDR↔SRAM LOAD/STORE 在零拷贝前后不变，不能计入被消除流量。

### 4.5 建议的正式创新表述

> 针对多阶段 CPU--VTA 推理中独立 GraphExecutor 产生中间张量重复物化、双缓冲增加额外物理分配以及 VTA 命令队列固定过量预留的问题，本文提出生命周期与需求感知的有界共享内存运行时。该方法以 manifest 驱动的双 slot 所有权和帧代次协议实现跨 Executor 多张量安全零拷贝；在满足别名、对齐、物理范围和跨帧使用约束时复用独占输入池作为一个 slot；并依据实际指令、微指令、FINISH 与重放需求确定安全队列容量，从而同时降低边界物化开销和共享内存峰值占用。

### 4.6 主张边界

可以声称：零拷贝稳定消除框架物化；受限输入池复用实现并验证；命令队列大幅缩容且长程正确性通过。

不能声称：自然 Top-20 的稳态 FPS 显著提升；所有图和所有别名情形均安全；已经实现通用 USMP；已经释放系统固定的 192 MiB u-dma-buf 预留；队列缩容已通过三 boot 统计非劣效。

## 5. 新候选创新点三：面向显式 DMA 共享内存的 VTA 驻留--请求--命令协同调优

后续实现、实验 gate、板端预算和 subagent 交接以 `C3_DMA_RESIDENCY_AUTOTUNING_MASTER_PLAN.md` 为唯一主控计划。

### 5.1 它与既有 stage--tile 实验的关系

这不是已有创新点的改名，而是在既有负结果上提出的新搜索维度。

已经完成的一次有界 AutoTVM 只在当前 schedule 结构中改变 `tile_h/tile_w/tile_ci/tile_co` 和 virtual-thread。80 个 TopHub incumbent/单轴邻域测量中：

- 32 个正确；
- 41 个编译失败；
- 7 个数值失败；
- 0 个达到至少 2% 加速的安全替换。

合法配对中，`tile_w` 变化使 LOAD 请求、LOAD payload 和运行时间中位数分别变为 4.5×、4.173× 和 3.374×。一个较差的新 schedule 相对恢复的旧 topology-B incumbent，LOAD 请求、payload 和 device wait 分别达到 14.91×、3.78× 和 3.39×。

这些结果不能证明“AutoTVM 优化成功”，但证明：

1. tile 会显著改变 DMA 碎片和重复载入；
2. TopHub 必须作为不可删除 incumbent；
3. 编译合法性、数值正确性和传输签名可以用于安全过滤；
4. 仅在当前模板的单轴邻域继续搜索，发现大幅性能提升的机会有限。

新创新点的实质是修改 schedule 搜索空间，把“优先驻留输入还是权重”加入决策，而不是只给现有候选重新打分。

### 5.2 相关论文已经做到什么程度

这个方向不是从零开始。与第三点最相关的工作可分为四层：

| 层次 | 代表工作 | 已经覆盖的内容 | 本文不能重复声称的内容 |
|---|---|---|---|
| 张量程序自动调优 | AutoTVM、Ansor、MetaSchedule | tile、loop、virtual thread、学习成本模型、任务预算分配 | “自动搜索 tile”或“硬件参数进入调优”本身 |
| 空间加速器 mapping | Timeloop、MAESTRO、Marvel、CoSA、ZigZag、AMOS、ROLLER | 硬件约束、dataflow、片上复用、访存量估计和合法空间构造 | “用 SRAM 容量过滤”和“减少片外访问量”本身 |
| VTA 软硬件栈/TPS | Banerjee 等 2021 | tiling parameter search、DRAM bytes 削减、virtual-thread 冗余 LOAD 消除、FSim/TSim/DE10 分层验证 | “VTA 硬件约束剪枝、减少 DMA 或分层正确性”本身 |
| VTA 有效性调优 | 2022 Hardware-aware Initialization、2025 ML²Tuner | VTA 非法配置过滤、有效性预测、编译隐藏特征、减少板端样本 | “VTA 无效候选很多”或“用编译特征减少试验”本身 |
| VTA 数据访问 schedule | Cheng 等的 FGCS 工作 | 输入优先 schedule、片上 weight-memory reuse、最小数据访问调优；公开摘要报告 YOLOv3 推理时间约下降 10% | 输入优先、权重驻留或最小总访问量不能再单独宣称首次提出 |
| 指令存储定容 | AEx 等 | 联合处理器/ISA 综合缩减 instruction width 与片上 instruction memory | 不能泛称“首次 instruction-memory sizing”；本文只研究固定 VTA 的 host instruction/UOP backing |

关键参考包括：

- Rieber 等，*HW-Aware Initialization of DNN Auto-Tuning to Improve Exploration Time and Robustness*，2022；
- Cha 等，*Multi-level Machine Learning-Guided Autotuning for Efficient Code Generation on a Deep Learning Accelerator*，2025；
- Cheng 等，*Design of Data Access and Schedule Optimization for VTA Compiled Instruction Streams*，FGCS 176，2026，108165；
- Timeloop、MAESTRO、CoSA、ZigZag、AMOS 和 ROLLER 等空间加速器 mapping 工作。

其中 Cheng 等的工作与本方向最接近。它正式排在 2026 年卷，但 DOI 为 `10.1016/j.future.2025.108165`，且存在 DOI 为 `10.2139/ssrn.5157285` 的 2025 年公开预印本记录。因此，课题在 2025 年启动和已有带日期的代码、实验记录可以证明独立形成过程，但论文中仍应引用该工作，不能仅按期刊卷年将其当成不存在。

### 5.3 如何利用 2026 工作，而不是被它堵死

建议把第三点拆成“复现层 R”和“本文增强层 E”。

#### R：复现输入优先和权重复用 schedule

这一层的目标是获得可信基线，不承担最终 novelty：

1. 获取论文全文、伪代码和实验设置，确认其所谓 input-prioritized、weight-memory reuse 和 tuning parameters 的精确定义；
2. 在当前 VTA 版本中复现 loop reorder、`compute_at` 层级、cache lifetime 和相应参数；
3. 用 FSim、量化 LLVM reference 和完整 Relay stage 分别验证正确性；
4. 报告复现结果与论文结果的差异来源，包括开发板、VTA config、频率、网络、TVM 版本和 TopHub；
5. 将复现 schedule 作为 B2026 基线，与原 VTA/TopHub 在同一硬件指纹下比较。

原样复现可以成为论文的一节、工程贡献或复现实证，但通常不能单独命名为“本文提出的创新方法”。它的价值在于用较低风险打开当前模板之外的新 schedule 空间，并为后面的增量提供强基线。

#### E1：从固定输入优先扩展为按层选择驻留模式

如果原工作采用固定的输入优先与权重复用组合，本文不机械地套到所有卷积，而将数据驻留方式变成按 workload 选择的变量：

```math
p\in\{original,input\text{-}stationary,weight\text{-}stationary,hybrid\text{-}reuse\}.
```

选择依据不使用硬编码层名，而由输入/权重/累加工作集、卷积几何形状和片上容量决定。需要验证不同层是否出现不同的最优模式；如果所有层最终都选同一个模式，这一增量就不成立。

#### E2：从“最小总访问量”扩展为“最小请求服务代价”

2026 工作公开描述的重点是最小数据访问量；本文已有数据表明总字节数不足以解释性能，因此把目标扩展为：

```math
C_{transfer}(x)=
\alpha B_{DMA}(x)+
\beta N_{req}(x)+
\gamma N_{small}(x)+
\delta N_{stride}(x)+
\eta R_{inp}(x)+
\theta R_{wgt}(x).
```

其中 `R_inp/R_wgt` 是输入和权重重复载入量。系数不应凭经验随意拟合；小样本阶段优先采用 Pareto 支配、分层过滤或硬件微基准标定。当前 lowered-TIR extractor 已能精确得到逻辑描述符，优势是针对实际编译结果，而不是只从 tile 公式估计。

#### E3：把数据请求和命令空间统一为资源安全证书

对每个候选同时生成：

```text
TransferSignature = {
  load/store calls, payload,
  input/weight/acc 分类,
  small/stride 请求,
  input/weight reload
}

CommandSignature = {
  instruction bytes,
  serialized UOP bytes,
  FINISH/replay requirement,
  submit count
}
```

在性能等价集合中进一步选择命令 backing 更小的配置：

```math
x^*=\arg\min_x(I(x)+U(x))
\quad s.t.\quad
T(x)\le(1+\epsilon)T_{best},\ Correct(x)=1.
```

这与 USMP 不同：USMP 规划已有 buffer 的 lifetime 和 offset；这里决定生成哪一种 VTA schedule，以及该程序产生多少 DDR↔SRAM 请求和多少 JIT instruction/UOP 工作集。

#### E4：TopHub/2026 双基线保护与失败隔离

每个 workload 同时保留：

- 原始 TopHub incumbent；
- 复现的 2026 schedule 最优配置；
- 本文增强方法产生的候选。

候选必须依次通过 compile、FSim、量化参考、完整 stage 和板端计时。只有正确且相对最强 incumbent 至少快 2% 才允许覆盖；非法候选批次结束后重载固定 bitstream 并重启 RPC，再验证完整 stage，防止已观察到的设备状态污染。

#### E5：由 workload 改善反馈到 stage 和多帧流水

2026 工作主要解决 VTA 编译指令流；本文已有外层切图和流水模型，可以进一步回答局部 schedule 改善是否改变整网决策：

```math
priority(w)\propto
occurrence(w)\cdot
sensitivity(w)\cdot
criticality_{pipeline}(w).
```

只对出现频繁、对 VTA segment service 敏感且位于流水瓶颈附近的 workload 分配更多调优预算，再将安全 overlay 的 stage service 反馈一次给 k-best DP。这一层把单算子访存优化与论文已有的异构流水线主线连接起来。

### 5.4 第三点的完整研究问题

第三点不再只问“输入驻留还是权重驻留”，而是同时回答：

1. 如何复现并参数化输入优先、权重复用的数据访问 schedule？
2. 对固定 VTA，哪一种驻留模式适合当前 workload？
3. 总访问量相近时，DMA 请求数量、粒度和 stride 能否区分性能？
4. 候选能否完整放入片上 SRAM 和 host instruction/UOP backing？
5. 相比 TopHub、2026 方法、随机/XGB和仅合法性过滤，是否能用更少板端测量找到安全配置？
6. 单 workload 的改善是否传递到完整 stage、切图排名和多帧 FPS？

### 5.5 新的优化变量

当前卷积模板把输入 cache 和权重 cache 都放在 `k_o` 层级，未显式区分驻留方向。建议引入：

```math
p\in\{original,input\text{-}stationary,weight\text{-}stationary,hybrid\text{-}reuse\},
```

并联合搜索：

```math
x=(p,tile_h,tile_w,tile_{ci},tile_{co},nthread_h,nthread_{co}).
```

- input-stationary：使输入 tile 跨更多输出通道计算复用，减少 input LOAD；
- weight-stationary：使权重 tile 跨更多空间位置复用，减少 weight LOAD；
- hybrid-reuse：复现并参数化 2026 工作的输入优先与片上权重复用组合；
- original：保留当前 TopHub 对应模板作为安全回退。

### 5.6 固定 FPGA 约束

当前硬件的片上存储不对称：input SRAM 32 KiB、weight SRAM 256 KiB、accumulator SRAM 128 KiB、UOP SRAM 32 KiB。所有候选必须满足：

```math
M_{inp}(x)\le 32\,KiB,
\quad M_{wgt}(x)\le 256\,KiB,
\quad M_{acc}(x)\le 128\,KiB,
```

并通过 tensorize 整除、地址对齐、指令/UOP 容量和数值正确性检查。

“硬件设计完成后把硬件信息加入 AutoTune”不应表述成泛化到所有 FPGA 的万能规则。更稳妥的适用范围是：

> 固定阵列规模、显式 DMA、软件管理片上 scratchpad、片上容量非对称的 DNN 加速器。

硬件更换时只替换容量、带宽、启动开销和合法性参数，算法流程不变。

### 5.7 可复用的现有资产

1. Top-20 已合并为 10 类唯一 packed-convolution workload，可按 workload 复用配置。
2. 静态 lowered-TIR DMA 提取器在 8 个 direct-template workload 上，与 runtime 的 LOAD/STORE calls、input/weight/store payload 精确一致。
3. 10 类 workload 已观察到 input reload 1.72--4.29×、weight reload 1.00--7.00×，说明存在优化空间。
4. 完整 stage 的主要 input、weight 和 STORE payload 可由 workload occurrence 相加；LOAD call 的 1.92%--6.34% 残差主要来自 ACC、ALU 和图级控制，可单独修正。
5. 现有 runner 已具备 TopHub incumbent、逐元素 correctness、runtime DMA profile 和 bitstream 重载协议。

### 5.8 推荐算法流程

```text
卷积 workload
  ↓
生成 original / input-stationary / weight-stationary / hybrid-reuse schedule
  ↓
L0：SRAM、tensorize、对齐、instruction/UOP 合法性过滤
  ↓
Lowered TIR 静态提取 LOAD/STORE、payload、reload、stride、小请求
  ↓
L1：按传输签名做 Pareto 筛选，始终保留 TopHub incumbent
  ↓
少量 FSim/板端逐元素正确性和计时
  ↓
正确且至少快 2% 才形成安全 overlay
  ↓
将更新后的 VTA segment service 反馈一次给创新点一的 DP
```

### 5.9 必须比较的基线和消融

正式实验至少包含：

| 编号 | 基线/方法 |
|---|---|
| B0 | 原始 VTA + TopHub incumbent |
| B1 | 复现的 2026 input-prioritized + weight-reuse 方法 |
| B2 | Random AutoTVM，相同有效板端预算 |
| B3 | XGB AutoTVM，相同有效板端预算 |
| B4 | 只做 compile-valid/SRAM 过滤 |
| B5 | B4 + DMA total bytes |
| B6 | B4 + 完整 request-shape signature |
| B7 | B6 + residence-mode choice + command resource + incumbent protection |
| B8 | B7 + pipeline-criticality 预算和一次 DP 反馈 |

关键消融依次删除：驻留模式选择、请求数、small/stride、input/weight reload、命令足迹、TopHub 保护和 pipeline criticality。这样才能回答增量究竟来自复现的 schedule，还是本文新增的请求/资源/系统反馈机制。

### 5.10 何时能够成为独立创新点

至少需要同时满足：

1. 完成 2026 方法的同平台复现，并把它而不是弱 fallback 作为直接基线；
2. 实现的驻留模式产生真正不同的 loop order、cache lifetime 和 lowered TIR，而不是给同一 schedule 加标签；
3. 至少两类卷积呈现不同最优驻留方向，证明不存在一个固定规则覆盖全部层；
4. 完整 DMA signature 相比总访问量、合法性过滤或 knob-only 基线具有额外筛选能力；
5. 在 grouped holdout 或前瞻候选上减少板端测量，且 TopHub/2026 双 incumbent regret 受保护；
6. 至少在若干 workload、完整 VTA stage 或端到端流水之一取得可重复的正向结果。

推荐的最低验收口径：

- 板端候选测量数减少至少 50%；
- 最终选择性能不低于 TopHub 2% 容忍线；
- 至少 2--3 个代表 workload 获得 5% 左右的稳定提升，或完整 stage 获得可重复提升；
- 正确性、编译失败率、DMA 请求与 payload 同时报告。

如果只完成“原样复现 2026 schedule”或“根据 SRAM 容量过滤编译失败配置”，它应作为复现实验或创新点一的编译优化子模块；如果在复现基线上进一步证明跨层驻留策略切换、请求形态额外预测力、命令资源安全和 stage/流水收益，则可以升格为第三项独立创新。

### 5.11 可行性评估

| 子任务 | 工程可行性 |
|---|---:|
| 获得全文并准确复现 2026 schedule | 50%--75%，取决于公开实现和细节完整性 |
| 容量/合法性过滤 | 85%--95% |
| 静态 DMA signature 排序 | 85%--90% |
| 构造两种真正不同的驻留 schedule | 55%--70% |
| 板端测量预算减少 50% | 70%--85% |
| 单 workload 超过 TopHub 5% | 35%--50% |
| 端到端 FPS 提升 5% | 25%--40% |
| 最终支撑硕士论文独立创新点 | 65%--80%，取决于完整实验闭环 |

这些数值是结合当前代码结构、TopHub 强度和既有失败率作出的工程判断，不是统计概率。复现完成并不自动提高创新性；它主要降低 schedule 实现风险，并使本文的增量能够与最接近工作进行公平对比。

### 5.12 截至 2026-09-11 的实现与板端证据

第三点已经不再只是方案设计，并完成了 P7Q、P7R 和 E03 的真实 FPGA 资格实验；但尚未取得
可替换 TopHub 的未见性能胜者，因此不能声称完整性能闭环已经成立。

1. 固定 `16×16@100 MHz` VTA 的理论峰值为 **25.6 GMAC/s**，128-bit、100 MHz AXI 的理想单向 payload 为 **1.6 GB/s**。冻结 10 类 workload 在最乐观 full-duplex 模型中均为 compute-bound，说明减少总字节不能自动降低理想下界；请求启动、碎片、同步和不完全重叠必须进入解释。
2. input-stationary 在 W00/W02/W09 分别减少 **50%/75%/50%** 静态 input DMA，并通过代表 workload FSim。
3. 最初的 safe weight-stationary 在扩展池中 **0/43** 产生 weight-byte 降幅。根因不是“TVM 不会复用”，而是外提 weight lifetime 后形成 VTA 物理 token 通路无法表达的 `STORE(3)→LOAD(1)` 循环依赖。
4. 本文补充了显式 full-sync 的编译器语义：使用注册的 `tir.vta.coproc_sync`，并使依赖检测器在 full drain 处封闭当前 token 段、重置后续依赖状态。由此，外层 weight-residency 在 W00/W02 上把 weight bytes 分别减少 **85.71%/75%**；W09 的 bytes 不变，但 weight LOAD calls 从 **32 降到 2**。三个 workload 均无非法 `1↔3`，9/9 个 FSim seed 逐元素正确。扩展到 60 个 tile 实体后，32 个通过 lower/依赖审计且全部通过三 seed FSim及 AXU5EVB build/export；22/32 减少 weight bytes，中位降幅 50%、最大 87.5%。新增 residency drain 为 1--8 次、中位数 2，故其 full-sync 代价尚未上板前不能称为加速。
5. 80 个历史来源映射为 60 个唯一 tile 实体，并构造 250 个候选记录。85 个在 lower 阶段被容量、DMA 2-D、compact 或 padding 规则过滤；其余 **165/165** 均有三 seed FSim 证书并通过 AXU5EVB build/export，其中 155 个为本轮独立交叉编译。
6. 无标签 B4--B7 shortlist 已按预算 4/8/16 冻结，只使用板前 lower 和请求形态特征，始终保留 TopHub incumbent。W00/W02/W09 的 12 项通用探索合同，以及每类恰好包含 original、最佳 input 降幅、最佳 barrier-weight 降幅的 9 项机制 canary 合同均已冻结；后者 9/9 绑定 FSim 与 AXU 资格，并预注册 30 轮平衡交叉顺序。由于没有板端 latency label，G5/G6、Regret 和搜索效率仍未评价。
7. command dry-run 表明驻留收益伴随可量化代价：代表 W00/W02/W09 的 original 均为 1 次提交，而显式屏障模式为 **3/5/3** 次；后者降低单次 instruction 峰值，却把累计 UOP bytes 从 **236/208/68** 增至 **1624/5536/976**。全池扩展后，197/197 个本地合格候选均取得无模拟耗时字段的结构画像；mode-4 提交次数中位数 3、最大 9，全池 instruction/UOP 单次峰值最大 **23,008/10,820 B**，mode-4 累计 UOP 最大 **43,552 B**。因此候选不能只按 DMA bytes 排序，还要联合比较请求次数、同步、提交和命令空间。该数据来自 FSim，FINISH 为源码推导、replay 不可观测，不能作为板端性能或安全定容结论。
8. P5c 已把 165 个原资格候选与 32 个 mode-4 候选合为 **197** 项无标签全池，严格只用静态 DMA 与无 timing 的命令结构做 Pareto/分层排序；budget 4/8/16 的所有 workload 均保护 incumbent，且 label-poison invariance 测试通过。相对 P5b，各 workload 的派发前缀均改变，但这只表明命令/新模式特征进入了决策，不能在没有板端标签时声称搜索效率或性能改善。
9. P7Q 在四类 workload 上取得 79 个真实 FPGA 正确候选和 395 个计时样本。同 tile 的 56 个
   驻留配对中 42 个更快、中位 +3.31%，但四个 pool oracle 仍全是 TopHub；full request-shape
   排序未优于 bytes-only，证明局部驻留收益存在，却不足以直接替换强 incumbent。
10. P7R 的 E00/E01/E02 联合空间从 2368 个经解析/lowering 证书缩到 209 个，板前过滤
    **91.17%**。E02 的 input-stationary 虽把输入字节减半，却让总 DMA 字节增加约 42.65%/
    37.32%，真实延迟分别退化 14.80%/13.18%。由此形成的 DMA Pareto 证书在 P7Q 回看选择
    11 个 input-stationary 配对，11/11 更快、中位 +8.31%，并对 E02 两点 abstain；这些仍是
    回顾性结果。
11. 在任何 E03 FPGA 标签前冻结的新验证把 864 个配置缩到 74 个，双路径 FSim 4/4 通过，
    但 Pareto 首选 residency config17 在 FPGA 上 0/3 正确，合同在计时前自动停止，而同 tile
    original 3/3 正确。这一负结果不支持性能预测，却直接证明真实硬件 correctness canary 必须
    位于 cost-model/计时之前，失败候选只能回退 original/TopHub。
12. 随后在此前没有 C3 驻留板端标签的 W03/W05/W06 上冻结映射结构迁移。三个目标的双路径
    FSim 共 18/18 seed 正确；W03/W06 的 original 在 FPGA 均 0/3，故在 residency 与计时前停止。
    W05 的 `original -> input-stationary -> original` 共 9/9 正确，7 个随机完整区组中驻留相对
    同 tile original 的成对中位加速为 **12.59%**、7/7 获胜，首次取得一项前瞻迁移正证据；
    但仍比 TopHub config575 慢 **34.15%**，所以最终派发安全回退 TopHub。exact 硬件证书账本
    已扩至 10 项，并已接入测量前 allowlist。
13. 针对 W05 的 TopHub 邻域进一步保留 `oc_nthread=2`：纯 input-stationary 的 27 个输入降流量
    候选全部因总 DMA bytes 增加而 abstain；bounded hybrid 改为“虚线程跨组、组内顺序复用”
    后，从 400 个配置中筛出 19 个全 DMA Pareto，并在动态标签前按 TopHub tile 距离冻结 2 个。
    两候选通过 12/12 FSim、4/4 交叉编译和 18/18 FPGA seed。config574/494 相对同 tile original
    分别加速 **4.96%/9.86%**、均 7/7 获胜；预冻结的 config574 最快预测成立，其延迟
    5.148322 ms，距 TopHub 5.132598 ms 仅 **0.31%**。因此在候选派发空间 400→2（99.5%）下
    得到 strongest-incumbent 2% 等价带结果，同时严格回退仍选择 TopHub。
14. 对最终 W05 派发集补充了与共享内存空间直接相关的命令资源证书。无 timing FSim 采集得到
    TopHub575 的 instruction/UOP 峰值为 **3744/1480 B**，hybrid574 为 **3808/1424 B**，均已
    包含提交前 FINISH；按二者最大值和 4 KiB 页对齐后，只需 **4096+4096 B**。在冻结的 8 KiB
    总容量下两种部署身份重新执行正确，另一个 same-tile 对照也在其独立 12 KiB 计划下通过，
    共 3/3。相对原 runtime 为两个队列各请求 32 MiB，精确 W05 身份的 requested backing 减少
    **99.9878%**。随后把统一 24 KiB instruction + 12 KiB UOP 容量施加到 W00--W09、五模式的
    197 个冻结身份，197/197 通过；逐身份容量中位 8 KiB、P90 12 KiB。留一 workload 只有 9/10
    能直接迁移，W08 需要额外三个 instruction 页，说明容量必须随精确编译集合重算。W05 的
    4096+4096 B 计划又在真实 FPGA/u-dma-buf 路径取得 6/6 seed 正确，同会话报告峰值
    3808/1480 B 和 6 次提交，默认 RPC 随后恢复。该证书绑定静态形状、ConfigEntity、TIR 和
    runtime 源码，不能外推整网或动态形状。
15. 为避免把上述 8 KiB 退化成手工常数，现已闭合
    `semantic dispatch -> AutoTune pre-measure -> command certificate -> board qualification`：派发
    v2 记录完整 `ConfigEntity`，AutoTune 按实体语义而非 `config_index` 过滤，并同时校验
    `candidate_id` 和资源证书哈希。两个可计时 hybrid 加 TopHub fallback 的精确集合由峰值推导为
    **8192+4096 B=12 KiB**；最终 `{TopHub575, hybrid574}` 子集重新推导为
    **4096+4096 B=8 KiB**。两者数值不同正说明容量是 allowlist 的函数。最终子集已利用既有同会话
    6/6 FPGA 证据升级为 `qualified_reduced_capacity`；包含 config494 的调优集合仍保持
    `local_provisional`，不得用于正式缩容。当前 replay 无可观测资源记录，证书明确声明
    `replay_policy=disabled`，不再声称已覆盖 replay。
16. 上述闭环已进一步完成当前版本的 prospective deployment attestation。FSim 结构证据不再只
    记录搜索得到的库路径，而是由每个 worker 从 `/proc/self/maps` 绑定实际加载二进制；部署前
    生成的 pending manifest 固定 identity set、资源证书、并发队列数和 replay policy。板端隔离
    runtime 运行 6/6 seed 后返回同一 manifest ID、真实峰值 **3808/1480 B**、容量
    **4096/4096 B**、6 次提交及 `replay=disabled`。qualifier 将板端二进制哈希和结果附加为
    attestation，ready manifest 仍保持原 pending ID。因此这里证明的是一个可检查的部署规则：
    `allowlist -> max-per-submit -> allocator alignment -> concurrency sum -> runtime attestation`，
    而不是 AXU5EVB 上“8 KiB 总是最好”的经验常数。独立 FSim 负向控制又主动调用 capture 和
    replay API，两者均被当前 runtime 拒绝，证明 `replay=disabled` 是 fail-closed 执行约束。

因此目前可以把第三点描述为“固定 FPGA 的分层候选证书与安全调优方法”：解析证书减少无效
派发，双路径命令/FSim 排除可模拟错误，真实 FPGA canary 隔离模拟器不可见错误，DMA Pareto
避免只优化局部张量，强 incumbent 提供零退化回退，并对最终精确身份生成缩容命令资源证书。
不能描述为“已在未见几何提高推理速度或
FPS”。目前已经获得“新工作负载同 tile 局部收益 + 极小板端候选预算接近强基线 + 最终零回归
回退”的闭环，但还没有新 incumbent，398 个未测点也不能当作已知 pool oracle。详细证据见
[`c3_dma_residency_autotune/00_governance/P7R_UNSEEN_VALIDATION_RESULTS.md`](c3_dma_residency_autotune/00_governance/P7R_UNSEEN_VALIDATION_RESULTS.md)
和 [`c3_dma_residency_autotune/00_governance/P7R_SUPPORT_TRANSFER_RESULTS.md`](c3_dma_residency_autotune/00_governance/P7R_SUPPORT_TRANSFER_RESULTS.md)。

## 6. 既有实验应该放到哪里

| 已完成工作 | 论文中的归属 | 是否单独算创新 |
|---|---|---|
| ResNet18/YOLO CPU--VTA--CPU 切图搜索 | 创新点一的动机、搜索空间和跨模型边界 | 否 |
| 原生多阶段流水线、单 VTA mutex、per-stage threads | 创新点一的系统实现基础 | 否 |
| RAMPS 资源模型、max-plus/资源下界、k-best DP | 创新点一的核心方法 | 是，合并表述 |
| direct boundary copy 消融、island scaling | 创新点一的通信代价证据 | 否 |
| HP/HPC、cache flush/invalidate、四 HP 端口 | 平台背景和通信机制实验 | 否 |
| compile-context audit、workload 去重 | 创新点一到三的编译桥梁 | 否 |
| stage--tile 2×2 排名反转 pilot | 新创新点的动机；不是正向性能结论 | 否 |
| 80 个 TopHub 邻域及失败样本 | 新创新点的 incumbent/过滤依据 | 否 |
| lowered-TIR 静态 DMA 与 runtime 精确对齐 | 新创新点的重要测量方法 | 可作为方法子贡献，不能单独立点 |
| 跨 Executor shared-slot、owner、generation、bundle | 创新点二核心机制 | 是，合并表述 |
| 输入池作为 slot0 | 创新点二的空间增强 | 否 |
| runtime instruction/UOP 容量检查与提前提交 | 创新点二的运行时安全基础 | 否 |
| exact allowlist 命令资源证书、AutoTune 语义派发与板端 attestation | 创新点三的共享内存/部署方法 | 是，合并表述 |
| DDR contention、wait/poll、去屏障、硬件预取 | 负结果或后续工作 | 否 |
| 异常退出后重载 bitstream 恢复 | 创新点二的失败语义边界 | 否 |

## 7. 不应包装成创新点的内容

以下内容有研究价值，但当前不能单独声称创新：

1. **使用 u-dma-buf。** 它是物理连续共享内存工具和实验基础，不是论文创新本身。
2. **一般性的 zero-copy、双缓冲或 USMP。** 已有大量工作；本文只能主张它们在跨 Executor、多帧和异步 VTA 语义下的组合协议与实证。
3. **单纯增加 FPGA 参数过滤 AutoTune。** 若没有新驻留搜索维度、DMA signature 和 holdout 证据，创新性不足。
4. **权重 tile 复用或预取概念本身。** 相关思想已有工作；本文应强调 VTA schedule 中驻留策略、非对称 SRAM 和真实 TIR-DMA 的联合。
5. **提前于 LOAD 指令发 DMA。** 这通常需要修改 ISA、DMA engine、依赖跟踪和硬件队列，当前未实现，不进入主要创新。
6. **队列缩容等于 DMA 减少。** instruction/UOP backing 是空间预留，张量 LOAD/STORE 是时间和流量，两者必须分开。
7. **局部工作量下降等于 FPS 提升。** 零拷贝和队列实验均表明是否影响 FPS 取决于关键路径和流水线 slack。

## 8. 三个创新点的区别与接口

| 项目 | 创新点一 | 创新点二 | 新创新点三 |
|---|---|---|---|
| 决策层次 | 整网/stage | 跨 Executor runtime | VTA workload/schedule |
| 核心问题 | 切在哪里、线程怎样配 | 如何交接、何时复用、分配多少 | 驻留什么、tile 怎样选 |
| 主要对象 | topology、island、thread、edge | slot、owner、generation、queue capacity | input/weight SRAM、tile、DMA request |
| 主要时间指标 | 稳态 II、FPS | boundary service、latency、II | workload/stage latency、DMA service |
| 主要空间指标 | 边界数量和 DDR demand | slot/high-water/command backing | 片上 working set |
| 主要算法 | 资源下界 + k-best DP | manifest + 状态机 + 安全定容 | 驻留模式 + 约束过滤 + DMA shortlist |
| 当前状态 | 已完成，主张需守边界 | 核心完成，性能/通用性有边界 | 已有 400→2 小预算、同 tile 正收益和 TopHub 2% 等价带闭环；尚无新 TopHub/stage 胜者 |

三者采用分层而不是笛卡尔积联合搜索：创新点一生成有限 VTA workload 集；创新点三按去重 workload 优化并把安全结果反馈一次；创新点二在最终固定映射上部署共享 slot 和队列容量。这样既控制编译/上板成本，也避免把所有变量一次性展开成不可完成的搜索空间。

## 9. 推荐论文章节结构

1. 绪论：共享内存不是“地址可达即免费”，提出切图、交接和片上搬运三层问题。
2. 平台与数据路径：CPU、VTA、u-dma-buf、GraphExecutor、HPC/AXI、LOAD/STORE。
3. 共享资源与通信代价感知的流水线划分：资源模型、k-best DP、ResNet/YOLO 证据。
4. 生命周期与需求感知的共享内存运行时：双 slot 零拷贝、输入池复用、命令/UOP 定容。
5. VTA 驻留--请求--命令协同调优：先复现最接近的 VTA 数据访问 schedule，再验证本文的按层驻留选择、请求形态、命令资源和流水反馈；仅在新创新点通过门槛后作为正式方法章。
6. 原型系统与统一实验：正确性、性能、内存高水位、DMA、消融和失败样本。
7. 总结与展望。

## 10. 可直接用于论文的总贡献表述

> 本文围绕嵌入式 CPU--FPGA 共享内存推理中的“划分、交接和访问”三个层次开展研究。首先，针对多个逻辑阶段共享有限 CPU 核与单物理 VTA、异构边界代价不可忽略的问题，提出资源与通信代价感知的 Top-K 流水线划分方法，以稳态资源下界和 k-best 动态规划从大规模合法配置中筛选高性能候选。其次，针对独立 GraphExecutor 间中间张量重复物化和多帧覆盖问题，提出 manifest 驱动的有界共享 slot 协议，并通过独占输入池复用和指令/UOP 安全定容降低额外 slot 与运行时队列占用。最后，面向 VTA 非对称片上存储和显式 DMA 特征，在复现输入优先及片上权重复用 schedule 的基础上，研究驻留模式、分块参数、DMA 请求形态和命令资源的协同选择，通过硬件容量约束、Lowered TIR 传输签名、双 incumbent 保护和流水关键度反馈，减少高风险候选及重复、碎片化共享内存访问。三项工作分别优化共享内存边界的产生、边界数据的安全交接与空间占用，以及加速器访问共享内存的方式。

其中前两项已有较完整实验支撑；第三项已经完成驻留 schedule、依赖合法化、分层证书、测量前
硬件 allowlist 和一次前瞻同 tile 正收益验证。它目前可作为“安全、硬件约束的调优方法”写入，
但若要升级为性能型独立创新，仍需在 TopHub 邻域取得最终候选或证明固定预算搜索效率优势。

## 11. 第三点主要参考文献

1. Rieber et al., “HW-Aware Initialization of DNN Auto-Tuning to Improve Exploration Time and Robustness,” 2022. <https://arxiv.org/abs/2205.15568>
2. Cha et al., “Multi-level Machine Learning-Guided Autotuning for Efficient Code Generation on a Deep Learning Accelerator,” LCTES 2025. <https://ksp.etri.re.kr/ksp/article/file/70710.pdf>
3. Cheng et al., “Design of Data Access and Schedule Optimization for VTA Compiled Instruction Streams,” *Future Generation Computer Systems*, vol. 176, 2026, 108165. <https://doi.org/10.1016/j.future.2025.108165>
4. Parashar et al., “Timeloop: A Systematic Approach to DNN Accelerator Evaluation,” ISPASS 2019. <https://accelergy.mit.edu/timeloop.pdf>
5. Kwon et al., “Understanding Reuse, Performance, and Hardware Cost of DNN Dataflows: A Data-Centric Approach Using MAESTRO,” MICRO 2019. <https://arxiv.org/abs/1805.02566>
6. Huang et al., “CoSA: Scheduling by Constrained Optimization for Spatial Accelerators,” ISCA 2021. <https://arxiv.org/abs/2105.01898>
7. Zheng et al., “AMOS: Enabling Automatic Mapping for Tensor Computations on Spatial Accelerators with Hardware Abstraction,” ISCA 2022. <https://sizezheng.github.io/files/AMOS_ISCA_22_Final.pdf>
8. Zhu et al., “ROLLER: Fast and Efficient Tensor Compilation for Deep Learning,” OSDI 2022. <https://www.usenix.org/conference/osdi22/presentation/zhu>
9. Banerjee et al., “A Highly Configurable Hardware/Software Stack for DNN Inference Acceleration,” 2021. <https://arxiv.org/abs/2111.15024>
10. Viitanen et al., “AEx: Automated High-Level Synthesis of Compiler Programmable Co-Processors,” 2023. <https://doi.org/10.1007/s11265-023-01841-3>

完整的 22 篇相关工作定位与微调建议见同目录 `FPGA_HARDWARE_AWARE_AUTOTUNING_LITERATURE_REVIEW.md`。
