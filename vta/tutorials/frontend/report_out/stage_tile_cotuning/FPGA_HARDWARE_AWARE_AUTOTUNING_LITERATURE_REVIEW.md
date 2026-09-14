# 固定 FPGA 位流下硬件感知自动调优：相关工作、创新边界与可执行微调

日期：2026-09-09  
面向论文：《面向深度神经网络的嵌入式异构计算共享内存技术研究》

## 0. 结论先行

“FPGA 硬件设计完成后，把阵列规模、片上 SRAM、总线宽度等信息加入 AutoTune，以减少搜索空间”这个大方向已经有大量工作，不能单独作为创新点。最接近本项目的直接先例至少包括：

1. Banerjee 等 2021 年的 VTA 软硬件栈已经包含 tiling parameter search、减少 DRAM bytes、修改 virtual-thread injection 消除冗余 LOAD，并用 FSim/TSim/DE10 做分层正确性验证。[^23]
2. 2022 年针对 **VTA + AutoTVM** 的硬件感知初始化，专门处理大量非法 tile；其方法平均只需基线 41.6% 的板端测量即可找到最优配置。[^5]
3. 2025 年的 **ML²Tuner** 也在扩展 VTA 上工作，用独立的有效性模型过滤非法配置，再把编译器内部特征加入性能模型；达到相同性能只用了类似 TVM 方法 12.3% 的样本，并把无效测量平均减少 60.8%。[^6]
4. 2026 年 FGCS 的 VTA 工作已经直接提出“输入优先调度、片上权重复用、最小数据访问量和相应参数调优”，YOLOv3 推理时间约下降 10%。[^9]
5. Timeloop、MAESTRO、Marvel、CoSA、ZigZag、AMOS、ROLLER 和 DORY 已分别覆盖硬件约束建模、数据复用/流量分析、有效映射生成、解析模型和搜索空间剪枝。[^10][^11][^12][^13][^14][^15][^16][^17]

因此，最适合你的不是重新发明“硬件感知 AutoTune”，也不是再做一次权重驻留，而是在已有工作之间吃一块很具体的小肉：

> **面向固定 VTA 位流和显式 DMA 共享内存路径的资源安全分层调优：从候选 lowered TIR 中精确提取 DMA 请求形态和命令足迹，以确定性约束排除不可执行候选，以请求粒度特征排序合法候选，最后只将少量候选交给开发板测量，并始终保留已验证的 TopHub incumbent。**

建议把它定位成第一创新点的“编译器侧增强”或一个小节，而不是仓促再立一个完全独立的大创新点。其可答辩的差异不是“用了硬件信息”，而是下面四项的组合：

- 面向 VTA 的**后端精确特征**，不是只看 tile knob 或粗略总字节；
- 同时观察 `LOAD/STORE` 请求数、payload、输入/权重重复载入、小请求和 stride，而不是只最小化总访问量；
- 把 host 侧 instruction/UOP backing capacity 纳入资源安全和性能等价选择，而不是只检查片上 SRAM；
- 用正确性、固定 bitstream、强 incumbent 和端到端流水 FPS 保护搜索结果，防止“调优后反而丢掉好配置”。

这里仍需谨慎：单独拿出任何一项都不够新，贡献来自它们在 **VTA 显式 DMA、u-dma-buf 共享内存、跨 stage 多帧流水**中的闭环实现与实证。

## 1. 先把研究问题说准确

### 1.1 不是再设计 FPGA，而是固定硬件后的软件映射

设固定硬件契约为：

```math
H = \{P_{MAC}, f, C_{inp}, C_{wgt}, C_{acc}, C_{uop}, W_{AXI}, A, B_I, B_U\},
```

其中分别表示 MAC 并行度、频率、四类片上存储、AXI 数据宽度/对齐，以及 host 侧指令和 UOP backing buffer 容量。给定算子或 stage `W`，AutoTVM 配置为：

```math
x=(tile_b,tile_h,tile_w,tile_{ci},tile_{co},vthread_h,vthread_{co},\ldots).
```

你的问题应表述为：

```math
\min_{x\in\mathcal X} T_{board}(x)
```

满足：

```math
Correct(x)=1,
\quad M_j(H,W,x)\le C_j(H),
\quad I(x)\le B_I,
\quad U(x)\le B_U.
```

这里的 `M_j` 是 VTA 片上 inp/wgt/acc/uop 工作集；`I(x)`、`U(x)` 是本次提交需要的指令和微指令 backing 足迹。合法空间是：

VTA 官方架构说明明确把 JIT instruction stream、共享内存管理与同步放在 runtime 一侧，因而
这里研究的是 host--FPGA 共享内存中的控制工作集，而不是泛称片上 instruction memory。[^25]

```math
\mathcal X_{valid}(H,W)=\{x\in\mathcal X\mid C_i(H,W,x)\ \forall i\}.
```

这一定义具有有限的通用性：它适用于“固定阵列、软件管理 scratchpad、显式 DMA、静态 shape”的空间加速器，不应外推到所有 FPGA 或所有 DNN 编译器。

### 1.2 为什么不能只按 DMA 总字节排序

对显式 DMA 加速器，一个候选的内存时间下界更接近：

```math
T_{mem}^{lb}(x)=
\frac{B_{dma}(x)}{BW_{eff}}
+N_{req}(x)\tau_{req}
+N_{small}(x)\tau_{small}
+N_{stride}(x)\tau_{stride}.
```

总字节只覆盖第一项；请求启动、短请求、二维 stride 访问会产生额外代价。完整的粗粒度下界可写成：

```math
T^{lb}(x)=\max\left(T_{compute}^{lb}(x),T_{mem}^{lb}(x)\right)+T_{control}(x).
```

但在没有硬件 stall counter 和可归因 AXI 计数之前，这只能用于排序、分层和解释，不能声称算出了真实执行时间。VTA 自身已经用 LOAD/COMPUTE/STORE task、依赖 token 和 virtual thread 隐藏部分访存延迟；计算与访存并不是简单相加。[^1]

## 2. 相关工作到底做到了哪一层

### 2.1 通用张量自动调优

| 工作 | 核心能力 | 已覆盖你的哪部分 | 没覆盖什么 |
|---|---|---|---|
| TVM/AutoTVM | 专家定义 schedule template 和 knob，学习成本模型后做硬件在环搜索 | tile、loop、thread、tensorize；VTA 后端原始基础 | 模板之外的候选、VTA 特有命令容量、多 stage 共享内存目标[^2] |
| Ansor | 自动生成层次化大搜索空间，用进化搜索和学习模型调优；task scheduler 把预算倾向更影响整网性能的子图 | “不应给所有 workload 相同预算”已有先例 | 论文主要评估 CPU/GPU，不直接建模 VTA DMA descriptor、命令队列或跨 Executor 流水[^3] |
| MetaSchedule | 用可组合的 probabilistic program、schedule rule、postprocessor 和 cost model 构造调优流程 | 为添加硬件规则提供成熟框架 | “框架允许加入规则”不是创新；仍需证明你的 VTA 特征和选择机制有效[^4] |
| ROLLER | 用与执行单元、内存事务长度、bank 和 tensor shape 对齐的 rTile 限制形状，并用微性能模型快速构造程序 | “硬件对齐可缩小 tile 空间”已被系统化 | 面向 GPU/IPU 等，不是 VTA JIT 指令/UOP、共享 DDR 或跨帧执行[^16] |

结论：不能把“加入硬件参数”“缩小搜索空间”“按硬件对齐 tile”写成新意。可借鉴的是 MetaSchedule 的模块化接口和 ROLLER 的构造式思想；你的落点必须是 VTA 后端可验证的具体信息。

### 2.2 空间加速器映射与内存模型

| 工作 | 方法 | 与本项目的关系 | 边界 |
|---|---|---|---|
| Timeloop | 描述 PE、存储层次、互连、带宽和架构约束；mapper 搜索合法 mapping，并估计性能/能耗 | `HardwareContract + mapping constraints` 的经典基线 | 抽象 mapping/DSE；不读取 TVM lowered TIR，也不负责 VTA runtime queue[^10] |
| MAESTRO | 以 data-centric directives 表示 dataflow，解析估计 reuse、性能、能耗和硬件代价 | 证明 reuse/流量是 mapping 的核心特征 | 重点是解析 dataflow 模型及 HW DSE，不是实际 VTA 编译候选正确性[^11] |
| Marvel | 先优化 off-chip mapping，再优化 on-chip mapping，以显著缩小搜索空间 | “优先处理片外数据移动”已有直接先例 | 仍是抽象数据流空间；没有 VTA 请求 API 和 host 命令足迹[^12] |
| CoSA | 把算子规则和架构约束写成混合整数规划，一次求出高效 schedule | “用确定性约束得到合法空间”已被覆盖 | 不是 AutoTVM 的测量闭环，也不检查真实板端数值正确性[^13] |
| ZigZag | 统一表示算法、硬件和 mapping，重点探索多级存储与 uneven mapping | “memory-centric mapping”不是空白 | 更偏架构—mapping DSE，不处理 VTA JIT/runtime backing allocation[^14] |
| AMOS | 正式描述空间硬件的计算/存储行为，自动生成 ISA-aware mapping，并验证 mapping 语义 | “硬件抽象 + mapping validity”已有强工作 | 范围远大于本论文；重做硬件抽象不现实，也偏离共享内存主线[^15] |
| MATCH | 在 TVM 上用可定制硬件模型和 cost model 映射异构边缘 SoC，并用 temporal mapping engine 搜索 layer schedule | “TVM + 硬件模型 + 异构设备映射”已有近期实现 | 面向 MCU/多执行单元的可移植部署，不处理 VTA lowered DMA descriptor、JIT 命令 backing 和你的多帧 slot 协议[^18] |

结论：解析模型和硬约束本身早已成熟。你的优势是已经拥有**实际编译器最终 TIR 与板端 runtime 的逐字段对齐**，可以把抽象估计变成 VTA 后端的精确逻辑描述符统计。

### 2.3 FPGA/HLS 设计空间探索

τ-VTA 把精度、频率、buffer、HLS directive 等硬件选择和软件调优联合起来，并报告最高 2.5 倍性能和最高 8.1 倍收敛加速。[^7] DCOC 进一步在 VTA++ 上使用多智能体强化学习联合优化 DNN、软件 mapping 和硬件配置，报告平均约 1.17 倍吞吐提升及最高 42.2% 优化时间下降。[^8] Chimera、AutoDSE、HASCO 等也分别用多目标学习、瓶颈引导或贝叶斯/强化学习探索 HLS 和软硬件空间。[^19][^21][^22]

这些工作回答的是“位流应该设计成什么样”或“软硬件怎样共同变化”。你的当前前提是 bitstream、频率、16×16 tensorize、片上 SRAM 和 AXI 口均冻结，因此不应该与它们正面比较最终硬件 Pareto，也不需要重新综合 FPGA。你的合理比较对象是固定硬件上的 AutoTVM、TopHub 和 VTA 直接调优工作。

### 2.4 与 VTA/AutoTVM 最接近的直接工作

#### A. Banerjee 等的可配置 VTA 软硬件栈，2021

该工作扩展 VTA 微架构、ISA 和编译栈，同时实现 tiling parameter search；论文与开源 fork 还修改
virtual-thread injection，通过调整 UOP 访问顺序复用输入/权重，避免双缓冲线程中的部分冗余
LOAD，并采用 FSim、TSim 和 DE10 的分层 CI/正确性验证。[^23]

因此，本文不能声称首次“根据固定 VTA 硬件剪枝”、首次“减少 DRAM bytes”、首次“消除冗余
LOAD”或首次“分层验证 VTA 候选”。可以保留的差异是：把候选完整身份、三态板级准入、失败
预算和 incumbent 回退连成确定性派发协议；并对最终 lowering 的逐 submit instruction/UOP
控制工作集生成 allowlist 容量证书和运行时 fail-closed 契约。

#### B. Hardware-aware Initialization，2022

该文扩展 VTA Conv2D 搜索空间后，13 个 DeepBench workload 的合法比例平均只有 8.3%，最低 0.8%，最高 23.1%。其方法先调用现有 VTA 编译器判断候选是否有效，利用合法点在 knob 网格中的局部聚集做邻域预采样，再构造合法/非法较均衡的初始训练集，并给模拟退火已知有效性偏置。平均只用基线 0.416 倍 trial 找到最优点。[^5]

它对你的约束非常直接：

- “VTA 非法配置很多”不是新发现；
- “编译前/测量前过滤非法候选”也不能单独作为创新；
- 该文的 validity check 仍需要编译候选，且主要优化模型初始化；它没有对合法候选的 DMA 请求形态和命令资源做精确、可解释排序；
- 它使用预先穷举得到的 ground-truth 数据代理硬件测量做重复实验，未覆盖你已经遇到的非法候选污染设备状态、数值错误和 bitstream 恢复协议。

#### C. ML²Tuner，2025

ML²Tuner 把模型拆为性能预测 P、有效性预测 V 和带隐藏特征的性能细化 A。隐藏特征来自编译过程，例如 loop 数、branch、partial tile、dummy thread 和 output-store loop count。其 VTA 实验达到相同性能仅需类似 TVM 方法 12.3% 的样本，无效测量平均下降 60.8%，加入隐藏特征后的平均 RMSE 比率为 0.919。[^6]

这篇论文说明“把编译器内部特征加入 XGBoost”也已经有人做了。你的差异不能只是把 `LOAD calls` 再加进一个模型，而应强调：

- 你的 DMA descriptor 是从最终 lowered TIR 精确展开得到，并已与 runtime 逐字段验证；
- 特征有明确硬件含义，能分解 input/weight/ACC、请求数、字节、small、stride、reload，而不只是自动抓取大量内部变量；
- 你的目标不只减少调优样本，还要提供资源安全证书、命令容量和端到端系统保护；
- 小数据条件下可以先采用确定性 Pareto/filter，再把 XGBoost 作为可选层，避免为了“像论文”而强行上复杂模型。

#### D. VTA 最小数据访问调度，2026

该工作提出输入优先 schedule、片上 weight-memory reuse 和面向最小数据访问的调优策略；它指出原 VTA schedule 会因覆盖而无法复用已经加载的权重块，并在 YOLOv3 上报告约 10% 推理时间下降。[^9]

这几乎直接覆盖“权重 tile 驻留复用”路线。除非你能获得全文、复现其 schedule，并明确做出不同机制，否则不建议把该路线作为当前主要创新。仍可利用的空隙是：

- 它的公开摘要强调减少总数据访问和输入/权重驻留；你的数据表明请求数、平均 payload、小请求和 stride 也必须联合观察；
- 它改变 schedule 以制造复用；你可以不改变 VTA ISA/RTL，只把已经生成的实际 DMA 请求形态用于安全 shortlist；
- 它没有把 host instruction/UOP backing size、跨 Executor u-dma-buf slot 和多帧流水目标放入同一选择规则。

## 3. 你的现有证据在文献坐标中的位置

仓库现状并不是从零开始，而是已经完成了最昂贵的“特征是否可信”准备：

1. [`vta_conv2d.py`](../../../../python/vta/top/vta_conv2d.py) 暴露 `tile_b/h/w/ci/co` 和两个 virtual-thread knob；这是现有 AutoTVM 搜索空间。
2. [`tune_resnet18_vta.py`](../../tune_resnet18_vta.py) 支持 XGB、GA、random 和 gridsearch，但板端 cost 仍主要是执行时间。
3. [`vta_tuning_history.py`](../../vta_tuning_history.py) 已实现稀疏实验记录叠加 TopHub，避免缺失 workload 丢掉强基线。
4. [`TILE_DMA_ANALYSIS.md`](iteration6_safe_overlay_final/TILE_DMA_ANALYSIS.md) 的 80 个邻域候选中，32 个正确、41 个编译失败、7 个数值失败、0 个满足至少 2% 加速的安全替换。改变 `tile_w` 后 LOAD calls 中位数为 4.5 倍、payload 为 4.173 倍、运行时间为 3.374 倍。
5. [`STATIC_DMA_EXTRACTION.md`](stage_memory_experiments/STATIC_DMA_EXTRACTION.md) 已对 8/8 个 direct-template workload 实现六个 DMA 字段与 runtime bit-exact；10 类 workload 的 input reload 为 1.72–4.29 倍、weight reload 为 1–7 倍。
6. [`EXPERIMENT_QUEUE.md`](queue_capacity/EXPERIMENT_QUEUE.md) 已测得默认两块 32 MiB 命令 backing，而每次实际峰值只有 instruction 5312 B、UOP 2792 B；缩到合计 16 KiB 后，四拓扑总池高水位下降 82.23%–86.96%，且每拓扑 1000 帧自然压力正确。
7. [`README.md`](README.md) 还记录了一个关键反例：一次较差的新调度相对历史高性能二进制产生 14.91 倍 LOAD 请求、3.78 倍 LOAD payload 和 3.39 倍 device wait，FPS 从约 10.724 降到 3.482。

这些结果使你区别于仅做模型/仿真的工作：你可以证明静态特征与真实 VTA runtime 请求一致，也有正确性和流水实测。但目前只在 TopHub 和小邻域验证，尚不能声称该特征能在完整 AutoTVM 空间中稳定找到更快配置。

## 4. 推荐微调：资源安全的两级候选筛选

### 4.1 方法名称

推荐中文名称：

> **面向 VTA 显式 DMA 与命令资源的编译期特征引导安全调优**

更保守的英文名称：

> **Compile-time Transfer-Signature Guided Safe Tuning for Fixed Explicit-DMA Accelerators**

不要叫“通用 FPGA 硬件感知 AutoTune”，因为支持范围和实验都没有那么大。

### 4.2 分层流程

```text
AutoTVM 原始候选 X
        │
        ├─ L0：硬件契约/shape 的廉价合法性规则
        │       tensorize 整除、地址字段、对齐、明显 SRAM 上界
        │
        ├─ L1：本地 instantiate + lower
        │       编译失败淘汰；提取精确 logical DMA signature
        │
        ├─ L2：本地 JIT/simulator dry-run
        │       instruction/UOP 足迹；FINISH 由源码绑定推导
        │       replay 未观测时必须禁用，不能宣称已认证
        │
        ├─ Pareto shortlist
        │       不用单一“总字节分数”粗暴删点
        │
        └─ 开发板测量
                数值正确 → latency → stage → pipeline FPS
                TopHub incumbent 始终在候选集中
```

候选静态特征定义为：

```math
\phi(x)=\big[
N_{load},B_{inp},B_{wgt},B_{acc},N_{small},N_{stride},
N_{store},B_{store},R_{inp},R_{wgt},I,U
\big].
```

第一版不必训练新的深度模型。可按以下原则生成 shortlist：

1. 确定性非法点直接删除；
2. 若候选 A 在计算利用率不低于 B 的前提下，其所有 DMA/命令维度均不大于 B，至少一项严格更小，则 A 支配 B，可删除 B；
3. 对剩余 Pareto 候选分层采样，保留少量特征多样点，防止解析模型不准；
4. 强制加入 TopHub/历史最优配置，不允许任何稀疏日志覆盖未测 workload；
5. 候选只有通过独立数值参考且显著快于 incumbent，才替换性能配置；
6. 对性能等价集合再选最小资源：

```math
x^*=\arg\min_x (I(x)+U(x))
\quad\text{s.t.}\quad
T(x)\le(1+\epsilon)T_{best},\ Correct(x)=1.
```

这最后一步把“时间优化”和“空间优化”接成一个故事：先用性能约束限定可接受区域，再在等价性能内减少共享内存预留。它不是 USMP，因为 USMP 根据 buffer 大小和生命周期给 workspace/constant pool 分配 offset，并不选择 VTA tile，也不计算 VTA JIT 指令/UOP 批次。[^20]

### 4.3 它与既有工作的精确差异

| 对比对象 | 对方已有能力 | 你的增量 |
|---|---|---|
| Banerjee 2021 VTA stack | TPS、DRAM bytes 优化、冗余 LOAD 消除、分层正确性 | exact-identity 三态准入、equal budget、逐 submit command backing 与 runtime attestation |
| HW-aware initialization | 编译器 validity check + 邻域预采样 | 对**合法候选**继续提取精确 DMA/命令向量；板端错误隔离和 incumbent 保护 |
| ML²Tuner | 学习 validity；收集一般编译隐藏特征 | 少数据、可解释的 VTA descriptor；请求类别和 reload；资源证书与系统目标 |
| Timeloop/MAESTRO/ZigZag | 解析估计 mapping 的流量、reuse、性能/能耗 | 对实际 TVM/VTA lowered 程序统计，不另建抽象 mapping 语言 |
| ROLLER | 对齐硬件特征后构造 rTile | 在既有 AutoTVM template 内做轻量筛选，不重写 tensor compiler |
| 2026 VTA 最小访问工作 | 修改 schedule，做输入优先和权重复用 | 不把权重驻留作为新意；关注请求形态、命令资源和安全选择 |
| USMP | 静态 workspace/constant pool 的全局 offset 与复用 | 选择生成什么 VTA 程序及其 DMA/命令足迹；二者是“程序选择”与“地址规划”的不同层 |
| AEx | 裁剪/综合处理器、ISA 宽度与片上 instruction memory | 固定 VTA/ISA 下的 host 共享 instruction/UOP backing，按 exact allowlist 定容[^24] |

## 5. 推荐实验设计

### 5.1 研究问题

- RQ1：硬件契约和本地 lowering 能减少多少无效的板端试验？
- RQ2：精确 DMA signature 相对 knob-only、bytes-only 是否提高小预算下的 top-k 命中率和 best-so-far latency？
- RQ3：在相同板端预算下，是否能恢复或超过 TopHub，并避免历史上 14.91 倍 LOAD 请求的灾难性候选？
- RQ4：在不超过 2% 性能退化的配置中，能否选择 instruction/UOP backing 更小的候选？
- RQ5：算子级改善能否传递到真实 stage 和多帧流水 FPS？

### 5.2 必须比较的基线

| 编号 | 基线 |
|---|---|
| B0 | 原始 TopHub incumbent |
| B1 | Random AutoTVM，相同板端 trial 数 |
| B2 | XGB AutoTVM，相同板端 trial 数 |
| B3 | 只做 compile-valid 过滤，不使用 DMA 特征 |
| B4 | compile-valid + DMA total bytes |
| B5 | compile-valid + 完整 DMA request-shape vector |
| B6 | B5 + instruction/UOP resource selection + incumbent protection |

如果时间有限，最少完成 B0/B1/B3/B4/B5；B2 可以只在 XGBoost 环境稳定时加入。不能只和随机搜索比，也不能继续把较差的新编译 baseline 当成 incumbent。

### 5.3 数据集与预算

- 先用 ResNet18 的 10 个唯一 VTA workload；不要按 39 次 occurrence 重复调优。
- 再加入至少 3–4 个不同几何形态的 holdout，例如 5×5、1×7、7×1 或 depthwise 中当前 VTA 真正支持的子集，以免只得到 ResNet18 规则。
- 板端预算建议为每 workload `{8,16,32,64}` 个正确候选；每种方法至少 5 个搜索 seed。
- 报告本地预处理时间、编译失败数、数值失败数、板端有效测量数，不能只报告 trial 数。
- 最终入选配置必须放回完整 Relay stage 和四个冻结 topology 中测，而不是停留在裸模板 microbenchmark。

### 5.4 指标

```math
Regret@b=\frac{T_{best@b}-T_{oracle}}{T_{oracle}},
```

以及：

- valid-board-trial ratio；
- 到达 TopHub `±2%` 所需的板端测量数和墙钟时间；
- top-k recall 或 nDCG；
- LOAD/STORE calls、bytes、small/stride、input/weight reload；
- instruction/UOP peak 与实际 backing allocation；
- 单 workload latency、完整 stage service、稳态 II/FPS；
- 数值正确率、设备恢复次数、跨 boot 方差。

### 5.5 消融

依次删除：请求数、small/stride、input/weight 分类、reload、命令足迹、incumbent 保护。你的已有四 topology 结果已经证明总 bytes 不能单独解释 FPS，因此完整向量应使用 Pareto、分层或学习排序，而不是随意设一组固定线性权重。

### 5.6 成立与失败都能写的门槛

建议预注册为：

- 若 B5 在多数 workload 上以不超过一半板端测量达到 B1/B2 的相同 best latency，并且最终 stage/FPS 不退化，则“搜索效率”成立；
- 若 B5 只减少非法测量但不改善合法候选排序，则贡献降级为 resource-safe filtering，不声称 DMA 模型提高搜索质量；
- 若命令足迹在所有候选都远低于 16 KiB，容量约束是 inactive guard，只作为安全证书和性能等价 tie-break，不能包装成性能瓶颈；
- 若请求特征只能解释当前 ResNet18，不在 holdout 上成立，则结论限定为 VTA/当前 template 的 case study。

## 6. 三个可选“微调包”及优先级

### 方案 A：精确 DMA 请求形态引导 shortlist——推荐

工作量：约 2–3 周；风险：中；与现有代码复用度：高。

需要把 [`extract_static_vta_dma.py`](../../extract_static_vta_dma.py) 从“读取已选 TopHub 配置做分析”扩展为“接受任意 AutoTVM ConfigEntity，lower 后返回特征向量”，再在 [`tune_resnet18_vta.py`](../../tune_resnet18_vta.py) 前加入本地候选预筛选。

优点是与你的共享内存主题直接相连，也避开了 2026 论文的“改 schedule 做权重驻留”。缺点是最终可能只降低调优成本、不提高已有 TopHub FPS；论文目标应预先允许这个结果。

### 方案 B：性能等价下的最小命令内存配置——作为 A 的子目标

工作量：约 1 周；风险：低到中；新意：单独看较小。

把已经完成的 queue diagnostics 变成调优候选的资源标签，在 `≤2%` 非劣性能集合内最小化 `instruction + UOP` backing。它与论文空间维度最贴切，但不能单独宣称通用内存规划；更适合成为 A 的第二目标或 C2 资源优化的编译器接口。

### 方案 C：按真实 stage/流水影响分配调优预算——可选增强

工作量：约 1–2 周；风险：中高。

Ansor 已有 task scheduler，因此“重要层多调几次”不是新意。你的差异只能是用已有 stage 切图、workload occurrence、单 VTA 串行和跨帧稳态 II 计算权重，例如：

```math
priority(w)\propto occurrence(w)\cdot sensitivity(w)\cdot criticality_{pipeline}(w).
```

如果没有完成 A 的候选级特征和稳定板端反馈，先不要做 C；否则很容易再次变成“看到哪里能优化就优化哪里”。

## 7. 明确不推荐的路线

1. **只给 AutoTVM 增加 `if tile_size <= SRAM`。** CoSA、DORY、AMOS 和 VTA hardware-aware initialization 已覆盖得更系统，硕士论文贡献太薄。
2. **只训练一个新的 XGBoost/RL tuner。** AutoTVM、Ansor、ML²Tuner 和 DCOC 的基线都很强；样本少时很难证明算法新意。
3. **权重 tile 驻留/输入优先 schedule。** 2026 VTA 论文已经直接完成；除非做复现后的明确增量，否则撞题风险最高。
4. **重新联合搜索 FPGA 参数和软件 tile。** τ-VTA、HASCO、DCOC 等已有工作，而且需要大量综合，与你“固定 FPGA 已设计好”的现实相反。
5. **把 USMP 当成 VTA tile optimizer。** USMP 处理 buffer lifetime、pool 和 offset，不决定 VTA DDR→SRAM 请求；可以复用其接口思想，但不能混为同一问题。
6. **把逻辑 DMA calls 直接称为 AXI transaction 或 compute stall。** 当前 extractor 和 runtime 只能证明软件发出的描述符；未有 APM/RTL counter 时不能外推物理总线细节。
7. **硬编码“AXU5EVB + ResNet18 的 tile_w=1 最好”。** 这只能是经验规则。可发表的形式必须让规则由 `H` 和 `W` 导出，并在不同 shape 或配置 holdout 上验证。

## 8. 最适合写进论文的故事

可以把三个已有方向统一成“共享内存数据和控制工作集的全路径收缩”，而不是三个零散补丁：

```text
网络跨 CPU/VTA stage 执行
        │
        ├─ 数据面：跨 Executor tensor 是否必须复制？
        │          → 双 slot 零拷贝保证跨帧不覆盖
        │          → 独占 GraphExecutor 输入池复用为 slot0，减少额外物理分配
        │
        └─ 控制面：VTA 运行这一 stage 需要搬多少数据、生成多少命令？
                   → 编译期 DMA/命令签名
                   → 合法候选筛选与请求形态引导调优
                   → 按实际峰值缩小 instruction/UOP backing
```

一句话版本：

> 本文面向 CPU–FPGA 共享物理内存上的多帧 DNN 流水，从数据交接和加速器控制两个层面减少共享内存工作集：运行时以双槽所有权协议消除跨 Executor 物化并安全复用既有存储；编译时从 VTA 最终指令生成路径提取 DMA 与命令资源需求，过滤不可执行映射、优先验证低搬运开销候选，并按实测峰值配置控制缓冲，在保持正确性与吞吐的条件下降低 u-dma-buf 占用。

这个故事中：

- 双 slot 解决“同一地址怎样跨 Executor、跨帧安全交给下游”；
- slot0 借池解决“安全交接所需的双槽是否必须额外买两份”；
- DMA/命令感知调优解决“将什么样的 VTA 程序放进这片共享内存，以及为它预留多少控制空间”。

它们共同研究的是共享内存中的**数据对象、所有权和控制流足迹**，因而不再是哪里能省就省。

## 9. 最终建议

在当前代码和时间约束下，建议顺序如下：

1. 先完成正在进行的队列和 slot 实验收尾，保住两个已经有完整正确性链的贡献；
2. 将方案 A 做成第一创新点的增强：先实现 compile-valid + exact DMA shortlist，不急于引入新 ML 模型；
3. 把方案 B 作为方案 A 的资源约束/tie-break，从而与“共享内存空间维度”闭环；
4. 只有 A 在 10 个 workload 上确实减少板端测量后，再做 C 的 pipeline-aware budget；
5. 论文中主动引用 2022 VTA hardware-aware initialization、2025 ML²Tuner 和 2026 VTA data-access schedule，并明确你的增量，反而比回避它们更可信。

如果只能做一个最小可交付版本，应选择：

> **TopHub 保底 + 本地 lowering 合法性检查 + 精确 DMA Pareto shortlist + 32 次以内板端正确性/性能验证。**

它不追求在算法名上超过所有自动调优论文，而是把你已经掌握的 VTA runtime 和 u-dma-buf 证据变成一个可重复、可解释、不会把好基线调坏的工程研究闭环。这正适合“别人吃大块、你吃一块扎实小肉”的硕士论文策略。

## Sources

[^1]: Moreau et al., “A Hardware–Software Blueprint for Flexible Deep Learning Specialization,” IEEE Micro, 2019. <https://arxiv.org/abs/1807.04188>
[^2]: Chen et al., “TVM: An Automated End-to-End Optimizing Compiler for Deep Learning,” OSDI 2018. <https://www.usenix.org/system/files/osdi18-chen.pdf>
[^3]: Zheng et al., “Ansor: Generating High-Performance Tensor Programs for Deep Learning,” OSDI 2020. <https://www.usenix.org/system/files/osdi20-zheng.pdf>
[^4]: Shao et al., “Tensor Program Optimization with Probabilistic Programs,” NeurIPS 2022. <https://proceedings.neurips.cc/paper_files/paper/2022/file/e894eafae43e68b4c8dfdacf742bcbf3-Paper-Conference.pdf>
[^5]: Rieber et al., “HW-Aware Initialization of DNN Auto-Tuning to Improve Exploration Time and Robustness,” 2022. <https://arxiv.org/pdf/2205.15568>
[^6]: Cha et al., “Multi-level Machine Learning-Guided Autotuning for Efficient Code Generation on a Deep Learning Accelerator,” LCTES 2025. <https://ksp.etri.re.kr/ksp/article/file/70710.pdf>
[^7]: Diamantopoulos et al., “Agile Autotuning of a Transprecision Tensor Accelerator Overlay for TVM Compiler Stack,” FPL 2020. <https://arxiv.org/abs/2004.10854>
[^8]: Fayyazi et al., “Dynamic Co-Optimization Compiler: Leveraging Multi-Agent Reinforcement Learning for Enhanced DNN Accelerator Performance,” ASP-DAC 2025. <https://doi.org/10.1145/3658617.3697547>
[^9]: Cheng et al., “Design of Data Access and Schedule Optimization for VTA Compiled Instruction Streams,” Future Generation Computer Systems, vol. 176, 2026, article 108165. <https://doi.org/10.1016/j.future.2025.108165>
[^10]: Parashar et al., “Timeloop: A Systematic Approach to DNN Accelerator Evaluation,” ISPASS 2019. <https://accelergy.mit.edu/timeloop.pdf>
[^11]: Kwon et al., “Understanding Reuse, Performance, and Hardware Cost of DNN Dataflows: A Data-Centric Approach Using MAESTRO,” MICRO 2019. <https://arxiv.org/abs/1805.02566>
[^12]: Chatarasi et al., “Marvel: A Data-centric Compiler for DNN Operators on Spatial Accelerators,” 2020. <https://arxiv.org/abs/2002.07752>
[^13]: Huang et al., “CoSA: Scheduling by Constrained Optimization for Spatial Accelerators,” ISCA 2021. <https://arxiv.org/abs/2105.01898>
[^14]: Mei et al., “ZigZag: A Memory-Centric Rapid DNN Accelerator Design Space Exploration Framework,” DATE 2021. <https://arxiv.org/abs/2007.11360>
[^15]: Zheng et al., “AMOS: Enabling Automatic Mapping for Tensor Computations on Spatial Accelerators with Hardware Abstraction,” ISCA 2022. <https://sizezheng.github.io/files/AMOS_ISCA_22_Final.pdf>
[^16]: Zhu et al., “ROLLER: Fast and Efficient Tensor Compilation for Deep Learning,” OSDI 2022. <https://www.usenix.org/conference/osdi22/presentation/zhu>
[^17]: Burrello et al., “DORY: Automatic End-to-End Deployment of Real-World DNNs on Low-Cost IoT MCUs,” IEEE Transactions on Computers, 2021. <https://arxiv.org/abs/2008.07127>
[^18]: Hamdi et al., “MATCH: Model-Aware TVM-based Compilation for Heterogeneous Edge Devices,” IEEE TCAD, 2025. <https://arxiv.org/abs/2410.08855>
[^19]: Yu, Huang, and Chen, “Chimera: A Hybrid Machine Learning Driven Multi-Objective Design Space Exploration Tool for FPGA High-Level Synthesis,” 2022. <https://arxiv.org/abs/2207.07917>
[^20]: Apache TVM RFC, “Unified Static Memory Planning,” 2021. <https://discuss.tvm.apache.org/t/rfc-unified-static-memory-planning/10099>
[^21]: Sohrabizadeh et al., “AutoDSE: Enabling Software Programmers to Design Efficient FPGA Accelerators,” FPGA 2021 / ACM TODAES 2022 extended version. <https://arxiv.org/abs/2009.14381>
[^22]: Xiao et al., “HASCO: Towards Agile Hardware and Software Co-design for Tensor Computation,” ISCA 2021. <https://arxiv.org/abs/2105.01585>
[^23]: Banerjee et al., “A Highly Configurable Hardware/Software Stack for DNN Inference Acceleration,” 2021. Paper: <https://arxiv.org/abs/2111.15024>; author TVM fork: <https://github.com/pasqoc/incubator-tvm/tree/il_contrib_0421>; author VTA fork: <https://github.com/pasqoc/incubator-tvm-vta/tree/il_contrib_0421>.
[^24]: Viitanen et al., “AEx: Automated High-Level Synthesis of Compiler Programmable Co-Processors,” *Journal of Signal Processing Systems*, 2023. <https://doi.org/10.1007/s11265-023-01841-3>
[^25]: Apache TVM, “VTA: Deep Learning Accelerator Stack,” release announcement, 2018. <https://tvm.apache.org/2018/07/12/vta-release-announcement.html>
