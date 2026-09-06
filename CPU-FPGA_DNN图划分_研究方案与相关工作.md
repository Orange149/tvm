# 面向 CPU–FPGA 异构 SoC 的通信与流水感知 DNN 图划分

## 研究方案、相关工作与实验设计

> **论文定位与相关工作文档。** 本文用于组织研究问题、最近工作、实验基线和创新边界；
> 当前 V1 设计以 `vta/tutorials/frontend/report_out/resource_aware_maxplus/HARDWARE_LAYER_DESIGN_REVIEW.md`
> 为准，执行状态以同目录 `RAMPS_EXECUTION_ROADMAP.md` 为准；`RAMPS_SYSTEM_MODEL.md` 只保留为
> 条件 V2 的理论参考。

## 摘要

随着无人机等飞行器向高速化、自主化和复杂任务方向发展，机载视觉系统需要在有限功耗与计算资源下持续处理高帧率图像。Xilinx Zynq 等异构 SoC 将 ARM 处理器系统（Processing System，PS）与 FPGA 可编程逻辑（Programmable Logic，PL）集成在同一芯片中，为 DNN 推理提供了通用处理与定制并行计算相结合的平台。然而，DNN 的不同算子在 CPU 与 FPGA 上具有不同的执行效率；当相邻算子被分配到不同设备时，还会产生 DMA 数据传输、缓存维护、同步等待和运行时控制等开销。对于连续图像流，部分计算和通信又可能通过流水执行发生重叠。因此，仅依据算子计算时间进行设备映射，或把计算与通信简单串行相加，都可能错误预测真实端到端性能。

本文拟研究共享内存 CPU–FPGA SoC 上的 DNN 图划分问题。V1 固定 VTA schedule/tile，只测搜索直接需要的 CPU/VTA segment、复合边界和持续 DDR demand；根据编译后的合法 segment 自动决定 CPU/VTA 放置、stage cut 和 CPU 核心分配，并以连续输入的稳态吞吐为目标输出 Top-K 上板候选。完整 component service、FIFO 和 Max-Plus 只在 V1 排序误差证明有必要时扩展。研究主线可概括为：

> **Profile → Model → Partition → Execute → Validate**

本文不声称已有研究从未考虑图划分、通信或流水，而是关注这些维度在紧耦合 CPU–FPGA DNN 推理中的联合建模。

---

## 1. 研究背景与问题定位

### 1.1 应用背景

飞行器的视觉感知链路通常包含图像预处理、特征提取、目标检测、跟踪与识别等阶段。飞行速度、输入帧率和任务复杂度的提高，会压缩感知—决策闭环可用的处理时间。单独使用嵌入式 CPU 虽然编程灵活，但在卷积、矩阵乘等计算密集型算子上往往难以同时满足吞吐率和能效要求；单独使用 FPGA 则会受到硬件资源、算子支持范围、开发成本以及控制类算子效率的限制。

Zynq 系列 SoC 在同一芯片上集成 ARM CPU 与 FPGA 逻辑。CPU 适合控制复杂、规模较小或 FPGA 不支持的算子，FPGA 适合具有规则数据并行性的计算密集型算子。这种结构使软硬件协同执行成为可能，同时也引入三个彼此关联但不能混为一谈的问题：

1. **图划分与设备放置**：哪些算子或子图在 CPU 上执行，哪些在 FPGA 上执行？
2. **跨设备执行代价**：设备边界产生多少数据移动、缓存维护、同步和驱动开销？
3. **执行时序**：连续输入时，CPU、DMA 与 FPGA 能否流水并行，通信能隐藏多少？

### 1.2 本文研究的不是一般任务调度

通用 CPU–FPGA 调度主要回答多个任务何时运行、使用多少 CPU 核或 FPGA 资源，以及如何处理动态到达、优先级和重配置等问题。本文首先回答的是一个更具体的 DNN 编译与执行问题：

> 给定一个 DNN 计算图，如何决定每个算子或子图的执行设备，使真实硬件上的端到端延迟或稳态吞吐率最优？

因此，本文的主问题是 **DNN Graph Partitioning / Operator Placement**。运行时执行与流水机制不是第二个彼此独立的研究主题，而是影响划分决策是否正确的执行语义和代价来源。

---

## 2. 概念区分

### 2.1 图划分：决定“在哪里执行”

图划分决定 CPU/FPGA 的子图边界。例如：

```text
输入 → CPU 子图 → FPGA 子图 → CPU 子图 → 输出
```

它需要考虑设备支持能力、算子计算性能、子图融合机会、中间张量大小以及边界切换代价。

### 2.2 分支并行：同一输入中的独立路径并行

对于含有无依赖分支的 DNN：

```text
             ┌→ 分支 B：CPU  ─┐
输入 → Split │                ├→ Merge
             └→ 分支 C：FPGA ─┘
```

两个分支可以处理同一个输入并同时执行。这属于 **DAG branch parallelism**。

### 2.3 工作量协同：同一算子或内核由 CPU 与 FPGA 分担

FastFit、Synergy 等工作可以把一个可切分的 workload 或矩阵乘 tile 同时分发给 CPU、NEON 与 FPGA。这属于 **intra-kernel workload partitioning / co-execution**，并不等价于把 DNN 的不同算子映射到不同设备。

### 2.4 跨输入流水：不同输入在不同阶段并行

对于连续图像流，不同帧可以同时处于 CPU、通信和 FPGA 阶段：

```text
时间向右

图像 1：CPU 计算 → DMA/同步 → FPGA 计算
图像 2：          CPU 计算 → DMA/同步 → FPGA 计算
图像 3：                    CPU 计算 → DMA/同步 → FPGA 计算
```

这属于 **inter-frame pipeline / overlap**。它与分支并行、同一内核的工作量协同均不是同一种执行模型。

---

## 3. 相关工作

### 3.1 固定软硬件分工与 CPU–FPGA 协同推理

NEURAghe 在 Zynq SoC 上构建卷积专用处理器，由 FPGA 处理 CNN 的主要计算负载，ARM CPU 处理难以加速的图部分，证明了 CPU 与 FPGA 协同执行完整 CNN 的可行性 [1]。它的核心贡献是端到端软硬件体系结构与加速器设计，而不是依据真实跨设备代价搜索任意算子到 CPU/FPGA 的映射。

Synergy 在 Zynq 上统一使用 ARM、NEON 和 FPGA 资源，将卷积转换为可分发的矩阵乘 tile，并通过 work stealing 平衡异构加速器负载 [2]。更重要的是，Synergy 使用多线程生产者—消费者结构，让不同网络层处理不同输入帧，实现了真正的跨帧流水。它说明 CPU–FPGA pipeline 在真实 Zynq 系统上能够提高吞吐率，但其算子职责主要由算子类型和既定体系结构决定，并未把任意 DNN 子图放置、实测跨设备通信和流水关键路径作为一个联合搜索问题。

EcoSys 面向 CPU–FPGA 视频分析进行硬件/软件协同设计和设计空间探索，并利用一致性互连等平台能力优化系统 [3]。这类工作表明体系结构、加速器设计和内存系统会显著影响最终性能，但它与“给定现有 CPU、FPGA 加速器和 DNN 图，自动搜索设备边界”的问题仍有层次差异。

### 3.2 共享内存与 PS–PL 通信剖析

Xiao 等面向 Zynq 上的 CNN 加速器研究统一虚拟内存、零拷贝式数据共享和一致性管理，指出 CPU 与加速器共享 DRAM 时，内存分配、地址转换和缓存一致性是系统性能的重要组成部分 [4]。

Rios-Navarro 等在 Zynq 上系统测量 PS 与 PL 之间的 AXI-DMA 传输，比较用户态轮询、内核态中断驱动以及不同数据切分策略，说明数据包大小、驱动路径和传输管理方式会改变 CNN 端到端性能 [5]。这类研究提供了通信模型所需的底层证据，但它本身不决定 DNN 图应如何映射到 CPU 和 FPGA。

这条研究线说明跨设备代价不能总被简化为：

$$
T_{\mathrm{comm}} = \frac{\text{Tensor Size}}{\text{Peak Bandwidth}}
$$

真实代价还会受到固定调用开销、缓存维护、同步、数据驻留、DMA 粒度和总线竞争影响。

### 3.3 通用 CPU–FPGA 调度与异步协同

Nunez-Yanez 等研究软件定义异构 FPGA 上的同时多处理，使 CPU 和 FPGA 能协同处理可分解 workload [6]。Vaishnav 等提出 resource-elastic scheduling，根据运行负载动态调整任务使用的 CPU 与 FPGA 资源 [7]。Zhu 等进一步研究边缘场景下软硬件任务调度和 FPGA 动态部分重配置 [8]。这些工作解决的是任务、资源和动态负载层面的调度，不是 DNN 图中的 operator/subgraph placement。

FastFit 与 MultiFastFit 使用轻量自动调优确定 CPU–FPGA workload 的近优 chunk size，并通过异步调度支持多 CPU 核与多个 FPGA IP [9]。它充分说明 CPU 与 FPGA 的异步协同和工作量划分已经被研究，但划分单位是可切分 workload/chunk，而不是具有张量依赖和算子支持约束的 DNN 子图。

Shi 等在共享总线 CPU–FPGA 异构系统上显式建模动态通信竞争，并联合处理 workflow task mapping 与总线传输调度 [10]。这是通信感知调度方面非常接近的相邻工作，但其对象是通用 workflow DAG；本文还需要处理 DNN 特有的算子支持、张量形状、子图融合、数据驻留和跨输入流水语义。

Hernandez-Yañez 等在 Zynq-7000 上将 PyTorch 矩阵运算委托给 FPGA，并让 CPU 同时处理预处理、控制和 I/O，明确研究 PS–PL DMA 与异步 overlap [11]。该工作证明了通信与计算重叠的实际价值，但没有自动搜索 DNN graph placement。

### 3.4 DNN 图优化、代码生成与设备映射

PCGC 通过运行时硬件反馈、代价模型和多层 fusion-splitting 规则优化 DNN 计算图 [12]。它会递归切分计算图并搜索适合目标硬件的融合子图，但其主要优化变量是图融合与图结构，而不是已经明确证实的同一 SoC 内 CPU/FPGA device placement。

AKGF 基于 TVM Halide IR，在 IR 中标注模型层算子对应的硬件核心，并分别使用 ARM 函数库和多面体模型优化 CPU 与 FPGA 代码 [13]。它与本文在平台、对象和目标上高度相似，是必须重点对比的工作。不过，公开摘要能够确认的是硬件标注、kernel generation 和硬件相关优化，不能仅凭摘要进一步断言其完成了“基于实测通信与流水代价的全图自动 CPU/FPGA 放置搜索”。因此，本文与 AKGF 的差异不能写成泛泛的“自动化”，而应落在划分目标函数、跨设备代价和执行模型上。

PartitionTuner 是目前最接近本文问题的工作之一 [14]。它在 Xilinx ZCU102 上使用 ARM CPU 和实现于 FPGA PL 中的 EVTA NPU，剖析单算子及算子组生成代码的 aggregate execution time，并据此进行 backend mapping、图划分和 partition scheduling。该 aggregate profile 会反映最终机器代码路径，但论文没有把 DMA、cache、sync 和 compute 分别作为可迁移资源参数报告；当 DNN 中两个无依赖分支被映射到不同 backend 时，它还会用线程并行执行。

但是，PartitionTuner 也明确规定：顺序分支上的 partitions 仍然顺序执行，只有无依赖分支才可能并行。因此，它的并行属于同一输入内的 **branch parallelism**，不是连续输入下 CPU、DMA 与 FPGA 的 **inter-frame pipeline**。aggregate execution time 也不等于显式描述 cache、DMA、同步及其可重叠部分。本文若要建立创新边界，必须证明这种 aggregate cost 在流水执行下不足以稳定选择最优划分。

### 3.5 面向吞吐率的 DNN 流水划分

DNNPipe 将 DNN 切分成顺序 pipeline stages，并将各 stage 分配给异构 IoT 节点，以最小化最大 stage time、提高连续推理吞吐率 [15]。它是“DNN partition + pipeline”最直接的对照工作。

但 DNNPipe 的问题定义假设相邻 stage 的最大通信延迟小于最大 stage 执行时间，因此只优化 stage computation time。这个假设适用于其低时延、高带宽网络目标环境，却不能直接套用于共享 DDR 的 CPU–FPGA SoC：DMA、cache flush/invalidate、同步等待和共享总线竞争可能与计算时间处于同一数量级。

这说明：

> **考虑流水不等于考虑通信；考虑通信也不等于建模通信与计算的重叠。**

### 3.6 计算与通信联合划分

Neurosurgeon 根据移动端、云端计算代价和中间数据传输代价自动选择 DNN 切分点 [16]；Edgent 则进一步根据网络带宽动态选择设备—边缘协同方案 [17]。它们证明“DNN partition + communication”本身早已不是空白。

然而，这类工作主要面向设备与边缘/云之间的网络通信，优化单次推理延迟或动态带宽下的切分点。Zynq 中的 CPU 与 FPGA 通过共享 DDR、AXI、DMA 和缓存维护路径交互，通信机制与可重叠条件不同。因此，这些工作可用于说明计算—通信联合优化的必要性，却不能直接替代同一 SoC 内的系统代价模型。

### 3.7 相关工作对照矩阵

下表中的“流水”严格指不同输入或不同 stage 在时间上重叠，不把 DAG 分支并行或单内核工作量并行混入其中。

| 工作 | 自动 DNN 设备放置 | 硬件剖析 / Cost | 显式通信代价 | 分支并行 | 跨输入流水 / Overlap | CPU–FPGA 同 SoC | 与本文的关键区别 |
|---|---:|---:|---:|---:|---:|---:|---|
| NEURAghe | 固定分工 | 部分 | data marshaling | 非核心 | 是，固定 stage | 是 | 有真实多帧流水，但不搜索任意设备边界 |
| Synergy | 否，职责基本固定 | 部分 | 未纳入放置目标 | tile/work stealing | 是 | 是 | 有真实流水，但没有通信感知的自动子图放置 |
| EcoSys | 设计空间协同，非本文粒度 | 是 | 一致性内存相关 | 部分 | 系统级 | 是 | 重点是 HW/SW co-design 与系统构造 |
| Xiao 等 | 否 | 是 | 是，内存与一致性 | 否 | 否 | 是 | 解决共享内存机制，不做 DNN 图划分 |
| Rios-Navarro 等 | 否 | 是 | 是，AXI-DMA | 否 | 否 | 是 | 详细通信剖析，不做自动 placement |
| FastFit / MultiFastFit | 否，划分 workload | 是 | 简化或隐含 | 不适用 | 异步协同 | 是 | chunk/workload 调度，不是 DNN 子图映射 |
| PCGC | 图切分/融合 | 是 | 未明确作为跨设备项 | 否 | 否 | 可面向 FPGA | 优化图融合结构，不是已证实的 CPU/FPGA placement |
| AKGF | 硬件标注已确认；全局搜索未确认 | kernel tuning | 未确认 | 否 | 否 | 是 | 强项是 kernel generation，划分目标与执行模型仍需全文核对 |
| PartitionTuner | 是 | 是 | 未分解，aggregate profile | 是 | 否 | 是 | 最接近；无顺序阶段的跨输入流水模型 |
| CoDL [19] | 单算子协同分块 | 是 | 是，转换/同步/mapping | 单算子 CPU-GPU 并行 | 否 | 统一内存 SoC，非 FPGA | 通信模型接近，但目标是单次延迟而非跨帧图划分 |
| HaX-CoNN [20] | 是 | 是 | 是，transition + contention | 多 DNN 并发 | 否，同一 DNN 不做跨帧 stage | 共享内存 SoC，非 FPGA | 已解决共享内存竞争感知映射，但无 VTA lowering 和单 VTA Pipeline |
| DNNPipe | 是 | 是 | 通过低时延假设弱化 | 否 | 是 | 否，面向 IoT 节点 | 有 partition + pipeline，但不显式优化通信 |
| Tarnawski 等 [18] | 是 | 是 | 是，设备间 edge cost | 支持一般图 | 是 | 否，服务端/多设备抽象 | 有最优 placement 算法，但无单 VTA、CPU 核池和共享 DDR |
| Shi 等 | 通用 workflow mapping | 是 | 是，含总线竞争 | 通用 DAG 调度 | 非 DNN 跨帧模型 | 是 | 通信模型强，但不是 DNN 专用图划分 |
| PyTorch/Zynq 异步协同 | 否 | 是 | 是，DMA | 否 | 是 | 是 | 有 DMA + overlap，不搜索 DNN 图映射 |
| Neurosurgeon / Edgent | 是 | 是 | 是，网络传输 | 否 | 否 | 否 | 通信环境和优化目标不同 |

### 3.8 相关工作结论

已有研究分别覆盖了以下能力：

- 自动 DNN 图划分与设备映射；
- 硬件性能剖析和代价模型；
- CPU–FPGA 通信与共享内存机制；
- 同一输入中的独立分支并行；
- 同一 workload 的 CPU–FPGA 协同执行；
- 连续输入下的跨阶段流水；
- 共享总线上的动态通信竞争；
- 统一内存 SoC 上的数据转换、同步和 contention-aware optimal mapping。

因此，本文不能把创新表述成“首次考虑通信”“首次并行执行”或“首次进行 CPU–FPGA DNN 划分”。更稳妥的研究空缺是：

> **现有工作的代价抽象和执行模型不同。PartitionTuner 已在 ARM+EVTA 上完成 fusion-aware 图划分，Synergy/NEURAghe 已在 Zynq 上完成跨输入流水，CoDL 已处理统一内存下的转换与同步，HaX-CoNN 已处理共享内存 contention-aware mapping，Tarnawski/DNNPipe 已给出 Pipeline placement 算法；但本次审查尚未发现一个方案同时处理 TVM/VTA compiler-valid segment、单物理 VTA 串行复用、CPU stage 核心分配、共享 DDR 约束、跨帧 II 和低预算 Top-K 上板。**

本文拟研究这些维度在紧耦合 CPU–FPGA DNN 推理中的联合建模，而不是把其中任一维度单独宣称为新问题。

---

## 4. 问题形式化

### 4.1 DNN 计算图与设备映射

将 DNN 表示为有向无环图：

$$
G=(V,E)
$$

其中，$V$ 是算子或候选子图集合，$E$ 是张量依赖边集合。对于每个节点 $v_i$，定义设备映射变量：

$$
x_i \in \{\text{CPU},\text{FPGA}\}
$$

完整划分方案为：

$$
X=(x_1,x_2,\ldots,x_{|V|})
$$

可行方案集合 $\mathcal{X}$ 需要满足算子支持、量化精度、FPGA 片上资源、缓冲区容量、数据依赖和子图合法性等约束。

### 4.2 跨设备边界

给定划分 $X$，跨设备边集合为：

$$
E_{\mathrm{cut}}(X)
=
\{(v_i,v_j)\in E \mid x_i \ne x_j\}
$$

跨设备边的代价与张量大小相关，但不能只由张量大小决定。更一般地：

$$
C_{ij}^{\mathrm{switch}}
=
f(S_{ij},D_{ij},R_{ij},K_{ij},M_{ij},Q)
$$

其中：

- $S_{ij}$：中间张量大小和形状；
- $D_{ij}$：传输方向，例如 CPU→FPGA 或 FPGA→CPU；
- $R_{ij}$：数据驻留位置与是否可复用；
- $K_{ij}$：缓存状态及一致性操作；
- $M_{ij}$：DMA 模式、调用粒度和缓冲策略；
- $Q$：总线、DMA 队列和运行时资源状态。

在完全串行、各项不重叠的测量路径中，可进一步分解为：

$$
C_{ij}^{\mathrm{switch}}
=
T_{ij}^{\mathrm{dma}}
+T_{ij}^{\mathrm{cache}}
+T_{ij}^{\mathrm{sync}}
+T_{ij}^{\mathrm{submit}}
+T_{ij}^{\mathrm{wait}}
$$

该加法式只表示串行路径的代价分解。在流水执行中，这些分量不一定全部位于关键路径，也不能不加区分地再次相加。

### 4.3 Compute-only 基线

如果仅考虑算子计算时间：

$$
T_{\mathrm{comp}}(X)
=
\sum_{v_i\in V} T_i(x_i)
$$

则对应的划分为：

$$
X_{\mathrm{comp}}^*
=
\underset{X\in\mathcal{X}}{\operatorname{argmin}}
\; T_{\mathrm{comp}}(X)
$$

该模型隐含了“切换设备没有代价”的强假设，适合作为实验基线，不适合作为真实系统的最终模型。

### 4.4 通信感知的串行模型

当执行过程近似串行时，可以使用：

$$
\widehat{T}_{\mathrm{serial}}(X)
=
\sum_{v_i\in V} T_i(x_i)
+
\sum_{(v_i,v_j)\in E_{\mathrm{cut}}(X)} C_{ij}^{\mathrm{switch}}
+T_{\mathrm{runtime}}(X)
$$

相应的串行延迟优化目标为：

$$
X_{\mathrm{serial}}^*
=
\underset{X\in\mathcal{X}}{\operatorname{argmin}}
\; \widehat{T}_{\mathrm{serial}}(X)
$$

该模型能够回答“一个划分本身产生了多少计算与切换开销”，却还不能描述连续输入的稳态吞吐率。

### 4.5 流水模型

设批量输入数为 $B$，给定划分 $X$ 后，流水执行的总完成时间记为：

$$
T_{\mathrm{pipe}}(X,B)
$$

稳态启动间隔（Initiation Interval，II）定义为：

$$
\operatorname{II}(X)
=
\lim_{B\to\infty}
\frac{T_{\mathrm{pipe}}(X,B)}{B}
$$

稳态吞吐率为：

$$
\operatorname{Throughput}(X)
=
\frac{1}{\operatorname{II}(X)}
$$

只有在一个简单三阶段流水中，且 CPU、通信和 FPGA 使用相互独立的资源、缓冲充分、没有总线竞争、没有额外依赖时，才可近似写为：

$$
\operatorname{II}_{\mathrm{ideal}}
=
\max\left(
T_{\mathrm{CPU}},
T_{\mathrm{comm}},
T_{\mathrm{FPGA}}
\right)
$$

这个公式不是任意 DNN 图划分的通用性能模型。若划分形成 CPU→FPGA→CPU→FPGA 的多次切换，或 cache maintenance、DMA channel、共享 DDR、缓冲区容量和依赖关系限制了重叠，则必须根据真实执行 DAG 或事件模拟计算 $T_{\mathrm{pipe}}$。

因此，面向流式视觉输入的优化目标应写为：

$$
X_{\mathrm{pipe}}^*
=
\underset{X\in\mathcal{X}}{\operatorname{argmin}}
\; \widehat{\operatorname{II}}(G,X,\mathcal{R})
$$

其中，$\mathcal{R}$ 表示 DMA 通道、缓存状态、缓冲区、总线竞争和运行时同步等资源约束。

### 4.6 两类目标必须分开报告

单次推理场景主要优化：

$$
\operatorname{Latency}(X)
$$

连续视频流场景主要优化：

$$
\operatorname{II}(X)
\quad \text{或} \quad
\operatorname{Throughput}(X)
$$

一个划分可能具有较低的单次延迟，却因阶段失衡而具有较差吞吐率；另一个划分可能增加单次延迟，但在流水稳定后取得更高吞吐率。因此，论文不应把 serial 最优和 pipeline 最优默认视为同一个方案。

---

## 5. 拟议方法

### 5.1 图分析与候选子图生成

首先从 TVM Relay、Relax 或等价 IR 中提取：

- 算子类型、属性和拓扑依赖；
- 输入输出张量的形状、数据类型和字节数；
- CPU 与 FPGA backend 的支持集合；
- 可融合算子组和必须共同映射的约束；
- residual、concat 等分支与汇合结构；
- 可形成 pipeline stage 的连续子图。

搜索单位不应被限制为单层，也不宜允许任意碎片化映射。较合理的基本单位是满足 backend 合法性与融合约束的候选子图。

### 5.2 最小局部剖析

V1 不建立全硬件 service surface。它先从目标 DNN 的真实 lowering 中提取 signature，再只测去重后的
缺失项：

1. **CPU segment**：lowering/fusion signature、physical shape 和 `threads=1..4` 下的 wall time；
2. **VTA segment**：冻结 schedule/tile 下、单 VTA mutex 内的 native service；
3. **复合边界**：实际 adapter、set/get、layout/quantization 和必要 cache 路径；
4. **共享 DDR demand**：由唯一 transaction id 汇总 physical bytes，并使用少量 sustained bandwidth。

相同 signature 只测一次，不按层名或完整切图重复 profile。完整候选吞吐只用于最终 Top-K 验证，
不得回填局部成本。cache/coherence 只有在实际 runtime 路径触发并被观测时才计费。

### 5.3 基线与 V1 模型

统一使用 B0-B5：

| 模型 | 组成 | 隔离的问题 |
|---|---|---|
| B0 | random / stratified random | 无模型搜索预算基线 |
| B1 | Synergy/NEURAghe-like type-fixed mapping | 固定软硬件分工是否已经足够 |
| B2 | PartitionTuner-like grouped cost + single-frame latency | compiler-valid group profile 的能力 |
| B3 | Tarnawski/DNNPipe-like Pipeline max load | 跨帧目标的增量能力 |
| B4 | B3 + CPU 核分配 + 单 VTA + composite boundary + shared DDR demand | 本文 V1 的 SoC 修正 |
| B5 | measured-pool oracle | 回顾性 regret 分母，不是可部署方法 |

完整 contention slowdown、FIFO 和 Max-Plus 不属于 B4。只有 B4 的稳定排序残差明确指向这些因素，
才将其作为 V2 消融，而不是 V1 的前置条件。

### 5.4 划分搜索

V1 先生成满足 backend、fusion、quantization 和 boundary contract 的有序 compute units。k-best DP
状态保存已覆盖前缀、VTA island 数、已分配 CPU 核数和最后设备；标签保存最大 CPU stage time、
单 VTA 总 demand、复合 boundary 和 DDR demand。终点按这些资源负载的最大值排序并回溯 Top-K。

搜索器输出包括 CPU/VTA segment、每个 CPU stage 的核心数、异构 boundary、共享 DDR demand、
预测 II、成本 provenance 和 native compile manifest。对 ResNet18 使用主机端静态枚举验证 DP，
枚举不等于编译或上板全部候选。没有新的 pruning 正确性或复杂度结果前，该求解器称为约束 DP
实例化，不称为新算法。

### 5.5 运行时执行

运行时应提供两种明确分离的模式：

- **Serial Executor**：用于验证划分的纯计算与跨设备代价；
- **Pipeline Executor**：用于验证多输入时的 overlap、关键路径和稳态吞吐率。

两种模式使用同一划分方案和相同算子实现，避免因代码路径不同而把实现差异误判为 pipeline 收益。

---

## 6. 与现有 ResNet18 数据的对应关系

当前约 200 个 ResNet18 候选首先是历史 pipeline ranking 和资源诊断数据。并非每个候选都已经
具备协议一致、可直接比较的完整 serial 标签；使用前必须由数据审计给出 serial/pipeline
coverage，不能把缺失的 serial profile 默认为已测。已有 serial 记录和后续 executor-matched
复测分别承担以下角色：

### 6.1 Serial profile 回答的问题

- compute-only 模型为何预测错误？
- CPU→FPGA 与 FPGA→CPU 的边界代价是否对称？
- tensor size / bandwidth 模型遗漏了哪些固定开销？
- cache、sync 和 runtime 分量对不同候选的排序有多大影响？

### 6.2 Pipeline profile 回答的问题

- serial 更快的候选是否仍然具有更高稳态吞吐率？
- 不同候选能够隐藏多少通信？
- CPU、DMA、FPGA 中谁是瓶颈资源？
- aggregate `compute + transfer` cost 是否能够预测 pipeline 排序？
- pipeline 最优划分是否与 serial 最优划分不同？

可以定义候选 $X$ 的隐藏时间：

$$
T_{\mathrm{hidden}}(X)
=
T_{\mathrm{serial}}(X)
-T_{\mathrm{pipe,equiv}}(X)
$$

但比较时必须保证 $T_{\mathrm{pipe,equiv}}$ 已按相同输入数转换为每输入平均时间或稳态 II，不能直接拿 batch 总时间与单次 serial 时间相减。

---

## 7. 实验设计

### 7.1 研究问题

建议围绕以下问题组织实验：

**RQ1：Compute-only 是否会选错划分？**

验证：

$$
X_{\mathrm{comp}}^* \ne X_{\mathrm{real}}^*
$$

并测量 compute-only 最优候选相对真实最优候选的 regret。

**RQ2：哪些 SoC 约束必须进入 Pipeline ranking？**

通过 B3 与 B4 及 B4 内部消融，分别评估 CPU 核分配、复合 boundary 和共享 DDR demand 对低 K
regret 的贡献。只改善局部 MAE 而不改变 shortlist 的因素不进入主模型。

**RQ3：串行最优是否等于流水最优？**

比较：

$$
X_{\mathrm{serial}}^*
\quad \text{与} \quad
X_{\mathrm{pipe}}^*
$$

若二者经常相同，则 pipeline-aware partition 的增量价值有限；若稳定不同，才构成进一步研究的实验证据。

**RQ4：通信量与可隐藏通信是否需要分开建模？**

比较具有相似 serial time 或相似通信字节数、但 pipeline throughput 不同的候选，分析差异是否来自依赖、阶段平衡、DMA 队列、缓冲或竞争。

**RQ5：模型能否泛化到不同 DNN？**

主物理参数由 DNN 无关校准和受控合成 pipeline 冻结；ResNet18 用于回顾验证，YOLOv3-tiny 用于冻结后的跨模型验证，投稿前再以 SqueezeNet 作为完全未见模型。若新模型出现校准域外 signature，只允许 label-free microbenchmark，不得使用完整候选吞吐调参。

### 7.2 基线

至少包含：

- CPU-only；
- FPGA-only 或最大可下沉方案；
- B0：random / stratified random；
- B1：Synergy/NEURAghe-like 按算子类型固定映射和固定 CPU thread policy；
- B2：PartitionTuner-like grouped aggregate profile，优化单帧 latency；
- B3：Tarnawski/DNNPipe-like Pipeline max load，但不联合 CPU 核和共享 DDR；
- B4：本文 V1，联合单 VTA、CPU 核分配、复合边界和共享 DDR demand；
- B5：historical/exhaustive measured-pool oracle，仅用于回顾评价。

所有基线必须使用相同 legal candidates、native package、correctness gate 和 executor；
all-VTA baseline 也必须 executor-matched，不能拿 RPC graph executor throughput 与 native pipeline
直接比较。

### 7.3 指标

代价模型需要同时评估“数值预测”和“决策质量”：

- MAE、MAPE、RMSE、$R^2$；
- Spearman 或 Kendall 排名相关系数；
- Top-$k$ 命中率；
- 选择 regret；
- 真实端到端 latency；
- 稳态 II、吞吐率和 pipeline fill/drain 开销；
- p50、p95、p99 抖动；
- CPU 利用率、FPGA 利用率、DMA/总线利用率；
- 能耗或每帧能量（若具备可靠测量条件）。

其中，仅有较高 $R^2$ 不能证明模型适合划分。优化器真正依赖的是候选排序和最终选择，因此 rank correlation 与 regret 更关键。

### 7.4 消融实验

V1 依次移除以下因素：

- CPU stage 核心联合分配；
- composite boundary，退化为 tensor bytes / bandwidth；
- boundary 方向和固定调用开销；
- shared DDR aggregate demand；
- segment grouped cost，退化为单 unit 求和。

buffer、overlap、pairwise contention 和 Max-Plus 属于 V2 条件消融，不得在 V1 排名失败时一次性全部加入。

消融目标不是证明模型越复杂越好，而是确认哪些因素能显著改善预测、排序或最终划分质量。

### 7.5 数据划分与防止过拟合

主 RAMPS 参数只能来自硬件 microbenchmark 和 DNN 无关受控 pipeline。ResNet18/YOLOv3-tiny
候选不得参与主参数或阈值选择；ResNet 标签仅可用于明确标记的 residual 消融。模型冻结、协议
SHA256 和候选组必须在读取验证吞吐前生成。除候选级 grouped validation 外，还要按模型进行
严格的 cross-DNN 验证。

---

## 8. 预期创新点与成立条件

### 8.1 可主张的研究方向

V1 不追求一次性完成完整 Max-Plus 服务模型。若实验结果支持，只主张以下分层贡献：

1. **compiler-valid CPU-VTA Pipeline 系统**：在 TVM/VTA 上联合选择连续 segment 和 CPU stage 核心数，并让多个逻辑 VTA island 在一个物理 VTA 上跨帧时分复用。
2. **面向排名的 SoC 资源修正**：在 grouped segment cost 之外加入复合 CPU-VTA boundary 与共享 DDR demand，验证它们是否改善低 K regret，而不是只改善局部 MAPE。
3. **预算化 k-best 搜索与验证**：不编译或上板全部合法候选，输出 Top-K，并报告达到 95% measured-pool oracle 所需的 profile、build、board 次数和 wall-clock。
4. **可复现的经验规律**：识别何时新增 VTA island 或 CPU stage 会被单 VTA 总负载、边界代价或 DDR demand 抵消，并在冻结 holdout 或第二个 DNN 上验证。

第 1 项是系统实现贡献；第 2、4 项是否成为方法贡献取决于消融；第 3 项只有提出新的状态压缩、
下界或 pruning 证明时才称为算法创新，否则只是标准 DP 在当前约束下的实例化。native executor、
boundary adapter、VTA mutex、build cache 和失败隔离不单独声称“首次实现”。

### 8.2 不能提前写成既定创新的内容

以下结论必须由实验支持后才能写入论文贡献：

- communication-aware 模型一定改变最优划分；
- pipeline-aware 模型一定优于 serial model；
- cache、sync 或 contention 一定是主要误差来源；
- 本文方法一定优于 PartitionTuner 风格策略；
- 本文首次实现 CPU–FPGA Pipeline、首次考虑共享内存或首次考虑 layout/sync；
- 200 个 ResNet18 候选上的结果一定能泛化到其他 DNN。
- 在没有声明搜索空间大小和 coverage 的情况下获得全局最优；
- 当前工作联合优化了 tile、fusion、VTA schedule 或算子级 schedule；V1 只联合 CPU 核心分配。

### 8.3 最关键的创新判据

本文能否站住，最终取决于四个证据：

1. B4 的 DP 与相同成本表上的静态枚举 oracle 一致，native runtime 也执行了对应 core/stage plan；
2. B4 相对 B1/B2 的提升证明目标从固定分工或单帧 latency 改为跨帧 II 有实际价值；
3. B4 相对 B3 的提升证明 CPU 核联合分配、复合边界或共享 DDR 至少一项改善低 K regret；
4. 冻结方法在第二个 DNN 上仍减少达到近优吞吐所需的上板次数。

若第三点不成立，论文收敛为 PartitionTuner-style graph placement 与 Synergy-style Pipeline 在
TVM/VTA 上的系统集成，不声称新的资源耦合优化；若第四点不成立，则只能表述为 ResNet18 的
target-aware implementation，不能声称跨 DNN 规律。

---

## 9. 论文叙事建议

整篇论文可以按以下逻辑展开：

1. 飞行器视觉负载要求端侧 DNN 具备低延迟和高吞吐；
2. CPU 与 FPGA 适合不同算子，但设备切换不是免费的；
3. 现有工作分别解决了固定协同、图划分、通信剖析、分支并行或流水执行；
4. 这些工作的代价抽象和执行模型并不相同；
5. 通过真实 serial/pipeline profiling，证明简单模型在候选预测或排序上存在不足；
6. 建立面向稳态 II 的 V1 cost model，并根据排序残差决定是否加入 contention/FIFO/Max-Plus；
7. 在真实 Zynq/VTA 平台上验证预测准确度和最终划分质量。

一句话的问题定义可以写为：

> **在单 VTA、共享 DDR 的 CPU–FPGA SoC 上，针对连续 DNN 推理，联合选择 compiler-valid CPU/VTA segment 与 CPU stage 核心数，并利用复合边界和 DDR demand 预测稳态 II，以少量 Top-K 上板找到近优吞吐方案。**

---

## 参考文献

[1] P. Meloni et al., [NEURAghe: Exploiting CPU-FPGA Synergies for Efficient and Flexible CNN Inference Acceleration on Zynq SoCs](https://doi.org/10.1145/3284357), ACM Transactions on Reconfigurable Technology and Systems, 2018.

[2] G. Zhong et al., [Synergy: A HW/SW Framework for High Throughput CNNs on Embedded Heterogeneous SoC](https://doi.org/10.1145/3301278), ACM Transactions on Embedded Computing Systems, 2019.

[3] X. Zhang et al., [Exploring HW/SW Co-Design for Video Analysis on CPU-FPGA Heterogeneous Systems](https://ieeexplore.ieee.org/document/9467325), IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems, 2022.

[4] T. Xiao et al., [Unified Virtual Memory Support for Deep CNN Accelerator on SoC FPGA](https://link.springer.com/book/10.1007/978-3-319-27119-4), ICA3PP, 2015, pp. 64–76.

[5] A. Rios-Navarro et al., [Performance Evaluation over HW/SW Co-design SoC Memory Transfers for a CNN Accelerator](https://arxiv.org/abs/1806.01106), 2018.

[6] J. Nunez-Yanez et al., [Simultaneous Multiprocessing in a Software-Defined Heterogeneous FPGA](https://doi.org/10.1007/s11227-018-2367-9), The Journal of Supercomputing, 2019.

[7] A. Vaishnav, K. D. Pham, and D. Koch, [Heterogeneous Resource-Elastic Scheduling for CPU+FPGA Architectures](https://doi.org/10.1145/3337801.3337819), HEART, 2019.

[8] Z. Zhu et al., [A Hardware and Software Task-Scheduling Framework Based on CPU+FPGA Heterogeneous Architecture in Edge Computing](https://doi.org/10.1109/ACCESS.2019.2943179), IEEE Access, 2019.

[9] A. Rodríguez-Moreno et al., [Lightweight Asynchronous Scheduling in Heterogeneous Reconfigurable Systems](https://doi.org/10.1016/j.sysarc.2022.102398), Journal of Systems Architecture, 2022.

[10] H. Shi et al., [Dynamic Contention-Aware Workflow Scheduling on Shared Bus-Based CPU-FPGA Heterogeneous Computing Systems](https://doi.org/10.1016/j.eswa.2025.130421), Expert Systems with Applications, 2026.

[11] O. Hernandez-Yañez et al., [Asynchronous Co-Execution of PyTorch on Zynq-7000: FPGA Matrix Delegation and PS–PL Overlap for End-to-End Inference Throughput](https://doi.org/10.3390/electronics15153308), Electronics, 2026.

[12] D. Dong et al., [PCGC: A Performance Compact Graph Compiler Based on Multilevel Fusion-Splitting Rules](https://doi.org/10.1007/s11227-023-05298-w), The Journal of Supercomputing, 2023.

[13] D. Dong, H. Jiang, and B. Diao, [AKGF: Automatic Kernel Generation for DNN on CPU-FPGA](https://doi.org/10.1093/comjnl/bxad086), The Computer Journal, 2024.

[14] M. Yu et al., [PartitionTuner: An Operator Scheduler for Deep-Learning Compilers Supporting Multiple Heterogeneous Processing Units](https://doi.org/10.4218/etrij.2021-0446), ETRI Journal, 2023.

[15] W. Seo, S. Kim, and S. Hong, [DNNPipe: Dynamic Programming-Based Optimal DNN Partitioning for Pipelined Inference on IoT Networks](https://doi.org/10.1016/j.sysarc.2025.103462), Journal of Systems Architecture, 2025.

[16] Y. Kang et al., [Neurosurgeon: Collaborative Intelligence Between the Cloud and Mobile Edge](https://doi.org/10.1145/3037697.3037698), ASPLOS, 2017.

[17] E. Li, Z. Zhou, and X. Chen, [Edge Intelligence: On-Demand Deep Learning Model Co-Inference with Device-Edge Synergy](https://doi.org/10.1145/3229556.3229562), MECOMM, 2018.

[18] J. Tarnawski et al., [Efficient Algorithms for Device Placement of DNN Graph Operators](https://proceedings.neurips.cc/paper/2020/file/b14680dec683e744ada1f2fe08614086-Paper.pdf), NeurIPS, 2020.

[19] F. Jia et al., [CoDL: Efficient CPU-GPU Co-execution for Deep Learning Inference on Mobile Devices](https://doi.org/10.1145/3498361.3538932), MobiSys, 2022.

[20] I. Dagli and M. E. Belviranli, [Shared Memory-contention-aware Concurrent DNN Execution for Diversely Heterogeneous SoCs](https://doi.org/10.1145/3627535.3638502), PPoPP, 2024.

[21] J. Haris et al., [SECDA: Efficient Hardware/Software Co-Design of FPGA-based DNN Accelerators for Edge Inference](https://arxiv.org/abs/2110.00478), ISPASS, 2022.

---

## 文档使用说明

本文采用 Markdown 公式格式：行内公式使用一对美元符号，独立公式使用两对美元符号。全文没有使用 LaTeX 文档级公式环境或公式标签命令，可直接在支持 MathJax/KaTeX 的 Markdown 编辑器中渲染。
