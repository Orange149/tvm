# 面向嵌入式 CPU-FPGA 共享内存平台的深度神经网络流水线运行时优化研究

> 学位论文初稿，Markdown 版本

| 项目 | 内容 |
|---|---|
| 学校 | 【待填写】 |
| 学院 | 【待填写】 |
| 专业 | 【待填写】 |
| 学位类型 | 专业硕士 |
| 作者 | 【待填写】 |
| 学号 | 【待填写】 |
| 指导教师 | 【待填写】 |
| 完成日期 | 【待填写】 |

## 提交前说明

本稿依据当前源码、冻结协议和实验产物整理，主线已经覆盖图划分搜索与共享内存运行时两部分，
但仍是送审前技术初稿。转入 Word 排版前需要完成以下工作：

1. 按学校模板补充封面、原创性声明、授权书、页眉页脚和中图分类号。
2. 将文中的“图 X-X 建议内容”绘制成正式图片，并把自动编号交给 Word 题注管理。
3. 根据学校参考文献格式，将文末条目统一转换为 GB/T 7714。
4. 若要把三 VTA-island 的 `2.83%` 单帧时延下降写成正式稳定提升，应再补两个独立 boot；
   当前正文严格将其标为单 boot stress-case 观察。
5. 当前没有能耗数据，不在摘要和结论中声称能效提升；当前静态模型的绝对吞吐预测 gate 未通过，
   不把资源下界倒数表述为预测 FPS。

# 摘要

面向高速图像感知的嵌入式系统既要求较高吞吐，也受功耗和资源约束。CPU 具有完整的算子覆盖与
控制能力，FPGA 上的专用加速器适合执行规则的计算密集型子图，将一个深度神经网络划分为多个
CPU/FPGA 阶段并跨帧流水执行，可以提高两类资源的并行利用率。然而，共享 DDR 并不会自动带来
低代价的数据交接：多个逻辑 FPGA 子图复用一个物理加速器，多个 CPU 阶段竞争四个核心，CPU 与
VTA 同时访问 DDR，框架还可能在共享物理内存上重复物化同一中间张量。由此产生两个相互关联的
问题：如何在大量编译合法方案中筛选高吞吐切图，以及如何在保证多帧生命周期安全的前提下消除
跨执行器冗余复制。

本文面向集成四核 ARM 与 Versatile Tensor Accelerator（VTA）的 Zynq UltraScale+ MPSoC，提出
一套由静态划分和共享 slot 运行时组成的流水线优化方法。首先，将 ResNet18 转换为 21 个比单算子
更大、比完整残差块更细且保持依赖闭合的计算单元；在 VTA schedule、tile 和量化策略固定的条件下，
联合搜索 CPU/VTA 连续 segment、VTA island 数和每个 CPU stage 的 TVM 线程参数。成本函数由
CPU stage 服务、单物理 VTA 串行服务、CPU core-time、CPU-VTA 复合边界与共享 DDR demand 构成，
使用各稳态资源负载的最大值估计流水线启动间隔。随后，利用非负服务需求形成的单调下界实现
k-best 动态规划，只输出 Top-K 候选供编译和上板验证。其次，针对普通 `get_output/set_input`
在跨 GraphExecutor 边界产生的重复物化，本文复用 TVM 已有 zero-copy 接口，设计 manifest 驱动的
有界共享 slot 协议：从 u-dma-buf 预分配同址 CPU/VTA 视图，队列仅传递 slot 标识和帧代次，
通过所有权状态机及双 slot 防止跨帧覆盖，并支持残差边界的多张量原子交接。

实验得到以下结果。ResNet18 共形成 4623 种合法拓扑和 972528 个线程化执行配置，k-best 搜索的
Top-20 与完整枚举逐项一致。自然 Top-20 的板端吞吐分布为 10.220--11.511 FPS，其中前 10 个已
包含该实测池最优方案，但 Top-20 内 Spearman 相关系数仅为 0.155，说明当前模型能够筛出高性能
区域，尚不能准确细排或预测绝对 FPS。对 200 条历史原生流水线记录的机制消融显示，加入实测
CPU-VTA 直接边界复制后，周期 MAE 由 17.3617 ms 降至 12.8220 ms，改善 26.15%，Spearman 从
0.7700 提升到 0.9310。共享 slot 的三 boot 实验将每帧框架物化字节从 1806336 B 降至 0，边界
API 服务由 1.446899 ms 降至 0.041830 ms，减少 97.11%；但启动间隔差值的 95% 置信区间包含 0，
故不声称自然 Top-20 的吞吐提升。在一个含 3 个 VTA island、6 条异构边的串行压力方案中，相对
优化 `memcpy` 基线，边界 API 服务减少 4.373 ms，单帧中位时延由 167.399 ms 降至 162.662 ms，
观察下降 2.83%；该结果来自一个 boot，仅作为机制观察。实验由此揭示：零拷贝可稳定消除边界内存
工作，但只有被消除的工作位于关键路径或超过流水线余量时，才会转化为端到端吞吐提升。

**关键词：** 深度神经网络；CPU-FPGA 异构计算；共享内存；流水线并行；图划分；动态规划；零拷贝

# Abstract

Embedded visual perception demands high inference throughput under strict power and resource constraints.
CPUs provide flexible control and broad operator coverage, whereas FPGA accelerators efficiently execute
regular compute-intensive subgraphs. Partitioning a DNN into CPU/FPGA stages and processing successive inputs
as a pipeline can exploit both resources. Shared physical DDR, however, does not imply free communication.
Logical FPGA segments serialize on one accelerator, CPU stages contend for four cores, and framework executors
may materialize the same intermediate tensor repeatedly. This creates two coupled problems: selecting promising
compiler-valid partitions from a large configuration space and eliminating redundant cross-executor copies
without corrupting in-flight frames.

This thesis presents a pipeline runtime optimization method for a Zynq UltraScale+ MPSoC integrating a
quad-core ARM processor and a Versatile Tensor Accelerator (VTA). ResNet18 is represented by 21 dependency-
closed compute units. With VTA scheduling, tiling, and quantization fixed, the method jointly searches
contiguous CPU/VTA segments, the number of VTA islands, and the TVM thread parameter of every CPU stage. Its
cost objective combines CPU-stage service, serialized demand on the single VTA, CPU core-time, composite
CPU-VTA boundary service, and shared-DDR demand; the maximum steady-state resource load estimates the pipeline
initiation interval. A monotonic-bound k-best dynamic program emits only a Top-K shortlist. The runtime part
uses TVM's existing zero-copy APIs but adds a manifest-driven bounded-slot protocol: u-dma-buf storage is
exposed as same-address CPU/VTA tensor views, queues carry slot and generation tokens rather than tensor data,
and an ownership state machine with two slots prevents cross-frame overwrite and supports multi-tensor residual
boundaries.

ResNet18 produces 4,623 legal topologies and 972,528 threaded execution configurations. The dynamic program's
Top-20 exactly matches full enumeration under the static objective. The natural Top-20 achieves 10.220--11.511
FPS on the board and contains the measured-pool best within the first ten evaluations, while its internal
Spearman correlation is only 0.155; the method therefore identifies a high-performance region but does not yet
accurately fine-rank candidates or predict absolute FPS. In an ablation over 200 archived native profiles,
adding measured direct CPU-VTA boundary copy reduces cycle MAE from 17.3617 ms to 12.8220 ms (26.15%) and
raises Spearman correlation from 0.7700 to 0.9310. Across three independent boots, shared slots reduce framework
materialization from 1,806,336 bytes per frame to zero and boundary API service from 1.446899 ms to 0.041830 ms
(97.11%), but the 95% confidence interval of the initiation-interval difference includes zero, so no steady-
state throughput gain is claimed. In a one-boot serial stress case with three VTA islands and six heterogeneous
boundaries, zero-copy reduces median latency from 167.399 ms to 162.662 ms (2.83%) against an optimized memcpy
baseline. These results show that zero-copy reliably removes boundary memory work, but affects throughput only
when the removed work lies on the critical resource path or exceeds available pipeline slack.

**Keywords:** deep neural network; CPU-FPGA heterogeneous computing; shared memory; pipeline parallelism;
graph partitioning; dynamic programming; zero-copy

# 目录

【转入 Word 后，在此插入自动目录。建议目录显示到三级标题，并另建图目录和表目录。】

# 主要符号与缩略语

| 符号或缩略语 | 含义 |
|---|---|
| CPU | Central Processing Unit，中央处理器 |
| FPGA | Field-Programmable Gate Array，现场可编程门阵列 |
| VTA | Versatile Tensor Accelerator |
| TVM | 面向深度学习编译与部署的编译栈 |
| DNN | Deep Neural Network，深度神经网络 |
| DDR | Double Data Rate SDRAM，本文指 SoC 共享外部内存系统 |
| PS/PL | Processing System / Programmable Logic |
| DMA | Direct Memory Access |
| II | Initiation Interval，稳态流水线相邻两帧的启动或完成间隔 |
| FPS | Frames Per Second，吞吐率 |
| `u_i` | 第 `i` 个不可再切的计算单元 |
| `S_j` | 第 `j` 个连续流水线阶段 |
| `t_j` | CPU 阶段 `S_j` 的 TVM 线程参数 |
| `R` | 允许的最大 VTA island 数，本文 V1 固定为 3 |
| `K` | 输出候选列表大小，本文主要取 20 |
| `D_core` | CPU 核心池服务需求下界 |
| `D_ddr` | 共享 DDR 服务需求下界 |
| slot | 为一条 stage edge 预分配、可被 CPU/VTA 同址访问的有界中间缓冲区 |
| generation | 同一 slot 循环复用时用于区分新旧帧的代次编号 |
| B0/B1/B2 | 普通复制、单 slot 串行零拷贝、双 slot 流水零拷贝三类实验路径 |
| oracle | 指定候选池或指定静态目标函数内的最优解，不等同于全局硬件最优 |
| regret@K | Top-K 中最佳实测吞吐相对候选池最优吞吐的损失 |

# 第1章 绪论

## 1.1 研究背景

深度神经网络已广泛用于图像分类、目标检测和飞行器视觉感知。随着输入频率和任务实时性要求
提高，只优化单张图片的执行时间并不充分，系统还必须持续处理图像流。把全部数据上传云端会受到
链路时延、带宽、隐私和任务可用性的限制，因此推理逐渐下沉到功耗和资源受限的嵌入式平台。

CPU-FPGA SoC 在一颗芯片内同时提供通用 ARM 核、可编程逻辑和共享外部内存。CPU 适合执行控制
复杂、后端不支持或收益较低的算子，FPGA 加速器适合规则且计算密集的卷积子图。两者功能互补，
但“把所有可支持算子放入 FPGA”并不必然得到最高端到端性能：CPU 前后处理可能成为瓶颈；连续
VTA 子图的形状和量化代价不同；每次 CPU-VTA 切换还会触发框架张量物化、布局或量化转换、设备
LOAD/STORE、提交与同步。

对连续输入而言，CPU 和 VTA 可以处理不同帧，优化目标由各段串行延迟之和转为流水线稳态启动
间隔。这个变化同时带来“划分”和“数据交接”两个问题。划分过粗可能无法平衡 CPU/VTA，划分过
细则增加边界与同步；即使 PS 与 PL 访问同一 DDR，两个独立 GraphExecutor 仍可能把中间张量从
一个框架缓冲区复制到另一个共享缓冲区。共享地址能力只提供零拷贝的硬件基础，并不自动解决
张量契约、缓冲区所有权、设备完成同步和多帧覆盖。

因此，本文不把共享内存视为一个孤立带宽优化，而是研究共享内存平台上“切在哪里”和“切开以后
如何交接”这两个连续问题：先通过资源感知搜索减少不必要的异构边界，再对保留边界使用有界共享
slot 消除框架冗余复制。

## 1.2 研究问题

本文围绕两个研究问题展开。

**问题一：共享资源条件下的流水线图划分。** 在固定 VTA 硬件、schedule、tile 与量化策略下，
如何从大量编译合法的 CPU/VTA segment、VTA island 数和 CPU stage 线程配置中，以有限的 profile、
编译和上板成本筛选高吞吐候选？该问题要求同时处理残差依赖闭合、单 VTA 串行、四核 CPU 共享、
异构边界以及共享 DDR demand，而不能只比较算子 FLOPs。

**问题二：固定划分下的跨执行器共享内存交接。** 在 CPU/VTA 映射运行前已知、PS/PL 共享 DDR
且使用硬件一致性通路的条件下，如何使 producer 和 consumer 直接读写同一物理中间缓冲区，同时
避免上一帧尚未消费、下一帧已覆盖该缓冲区，以及残差边界多个张量只交接一部分的问题？

两个问题的共同目标不是穷举上板。静态搜索将近百万配置压缩为 Top-K；共享 slot 运行时则减少
每个入选划分的边界内存工作。对于绝对时间暂时难以准确预测的情形，本文优先评价 shortlist 的
实测质量，并把无法通过统计检验的吞吐提升作为负结果公开。

## 1.3 研究内容

围绕上述问题，本文完成以下工作：

1. 在 TVM/VTA 上实现 CPU-VTA 多阶段原生流水线，支持每 stage 独立 TVM 线程参数、有限队列和
   单 VTA 全局互斥，并建立可复现的 native compile/reference/measurement 流程。
2. 建立中小粒度、依赖闭合的计算单元和 tensor contract，从编译结果生成合法 segment、boundary
   与资源 manifest。
3. 通过组件 profile 构建 CPU stage、单 VTA、CPU core-time、复合边界和共享 DDR demand 组成的
   资源下界，使用 k-best 动态规划生成 Top-20。
4. 基于 u-dma-buf 和 TVM zero-copy API 实现跨 Executor 共享 slot，增加 generation、owner、
   completion 与双 slot 状态机，并处理残差边界的多张量原子所有权。
5. 使用完整枚举、200 条历史 ResNet18 记录、自然 Top-20、三 boot 零拷贝实验、三-island 串行
   压力实验和 YOLOv3-tiny 回顾性数据，分别验证算法正确性、机制有效性与适用边界。

## 1.4 本文贡献与创新边界

本文形成两项主要创新和一项工程化贡献。

1. **面向单 VTA 共享资源 SoC 的编译约束 Top-K 流水线划分方法。** 本文使用中小粒度依赖闭合
   单元，在同一搜索中联合决定连续 CPU/VTA segment、VTA island 数和每个 CPU stage 的 TVM
   线程参数；成本函数显式累加单物理 VTA demand，并加入共享四核 core-time、方向化复合边界和
   DDR demand。利用非负资源服务形成的单调下界实现 k-best 标签搜索，在 972528 个 ResNet18
   配置上用完整枚举验证 Top-20 精确性。与仅按节点执行时间或独立设备最大负载建模相比，该方法
   针对的是逻辑 stage 多于物理设备、CPU/VTA 共享资源的嵌入式运行时。
2. **面向固定 CPU-VTA 流水线的 manifest 驱动共享 slot 零拷贝机制。** TVM 已提供单个 Executor
   的 zero-copy 绑定接口，双缓冲也不是本文单独提出的概念；本文的增量在于把切图 manifest、
   u-dma-buf 物理内存和多个 GraphExecutor 连接成一套有界协议。运行时为每条兼容边预分配双
   slot，建立同址 CPU/VTA `DLTensor` view，队列只传 `{edge, slot, generation}`，并以
   `FREE→WRITING→READY→USING→FREE` 管理完成与复用。该机制进一步支持残差 tuple 的多张量
   原子发布，避免多帧流水中的覆盖、旧帧和悬空引用。
3. **从组件工作量到端到端效果的可归因验证方法。** 本文将 framework materialization、VTA
   内部 LOAD/STORE、边界 API service、串行 latency 和流水线 II 分开统计，不把减少的字节自动
   换算成吞吐提升。实验同时给出正结果与负结果，并总结“零拷贝收益只有超过非瓶颈余量时才改变
   II”的适用规律。

本文不声称首次提出 CPU-FPGA 流水线、动态规划、zero-copy API 或双缓冲。当前实验也未证明准确
的绝对 FPS 预测、完整 DDR contention、tile 联合搜索和跨网络前瞻泛化。论文的创新边界是上述
机制在固定 CPU-VTA 共享 DDR 流水线中的联合设计、实现与实证。

## 1.5 论文结构

第2章介绍异构图划分、嵌入式流水线、共享内存管理和 TVM 执行器相关研究。第3章给出硬件平台、
数据路径、候选表示及两个优化问题。第4章介绍组件 profile 与资源成本模型。第5章介绍 k-best
动态规划。第6章说明 native pipeline 与共享 slot 零拷贝实现。第7章报告划分、通信、Top-20、
跨模型和零拷贝实验。第8章总结全文并讨论局限与后续工作。

# 第2章 技术背景与相关工作

## 2.1 TVM 与 VTA

TVM 将前端网络转换为 Relay 中间表示，并通过算子融合、量化、调度和代码生成面向不同后端生成可执行模块[11]。VTA 是与 TVM 集成的可参数化深度学习加速器，使用显式 LOAD、计算和 STORE 任务以及依赖队列组织数据搬运和执行[10]。这一结构意味着 VTA 的阶段时间不仅取决于逻辑运算量，还取决于物理通道填充、片上 buffer、LOAD/STORE 指令和 host runtime。

本文不搜索 VTA 硬件参数，也不在 V1 中联合搜索 tile。实验固定 bitstream、量化、graph packing、schedule 和 runtime policy，只改变网络划分与 CPU 线程配置。这一选择先隔离系统划分问题，并避免把 AutoTVM 的调度空间与图划分空间一次性做笛卡尔积。

## 2.2 DNN 图放置与划分

Tarnawski 等人把 DNN operator placement 抽象为带节点计算和边通信代价的结构化优化问题，针对
推理、训练和流水线目标给出最优算法[1]。该工作说明“设备负载与跨设备通信共同决定 placement”，
也是本文最直接的算法基础。但其抽象设备通常具有各自的负载，不能直接表达多个逻辑 VTA stage
串行复用同一个物理 VTA、CPU stage 共享同一四核池，以及不同 CPU stage 具有独立线程参数的情形。

PartitionTuner 面向包含 CPU 和 NPU 的深度学习编译器，根据单算子与 grouped operator 的实测 profile 生成调度方案，并考虑编译融合后的分组代价[2]。本文继承“按编译后 segment 查询成本”的思想，但研究目标由单次推理调度扩展为同一 DNN 的跨帧稳态吞吐，并加入单 VTA 与共享 CPU 核心池约束。

DNNPipe 以最大 stage time 为目标，使用保持最优性的动态规划和上界剪枝生成异构 IoT 节点流水线
方案[7]。其问题强调通过网络连接的独立节点。本文借鉴其最大负载和安全剪枝思想，但处理的是一颗
SoC 内多个逻辑 stage 共享物理 CPU、VTA 和 DDR 的问题，状态必须保留资源累计量而不只是最后设备。

## 2.3 嵌入式 CPU-FPGA 流水线

NEURAghe 在 Zynq SoC 上利用 ARM 执行难以加速的部分，并由 FPGA 卷积处理器承担主要卷积工作，证明了 CPU-FPGA 协同执行的实际价值[3]。Synergy 进一步利用 ARM、NEON 与 FPGA 构建多线程流水线，并通过 work stealing 平衡卷积 tile[4]。这些系统说明嵌入式 SoC 上的跨帧流水线可以提高资源利用率，但其设备分工或 tile 工作分配由系统结构决定，并不直接搜索本文所需的任意 compiler-valid CPU/VTA segment 与各 CPU stage 线程数。

VTA 与上述定制 CNN 引擎的差别在于，其后端行为受 TVM lowering、量化和张量布局共同影响。本文因此把 native compile 与 reference correctness 放在性能验证之前，不把仅在抽象图上合法的切点直接当作可执行方案。

## 2.4 统一内存异构执行与通信

CoDL 面向移动 CPU-GPU 统一内存系统，在算子内部联合使用 CPU 和 GPU，并显式处理数据类型、数据共享以及并发非线性[5]。HaX-CoNN 面向共享内存 SoC 上多个并发 DNN，结合 layer mapping、加速器间转换与 memory contention 选择调度[6]。二者表明“统一物理内存”并不意味着通信和竞争可以忽略。

本文场景与它们仍有差异。本文处理同一 DNN 的跨帧 CPU-VTA stage 流水线，VTA 是单实例互斥资源，边界涉及 VTA LOAD/STORE 和 TVM runtime 的 set/get 路径。本文 V1 只使用共享 DDR 聚合服务下界，不拟合 pairwise slowdown，因此不能声称已经达到 HaX-CoNN 式 contention 建模的完整程度。

Rios-Navarro 等人与 SECDA 对 Zynq PS-PL 数据通路的研究表明，DMA、驱动、数据准备和软件栈开销会显著影响加速收益[8-9]。这构成本文将通信拆成方向、字节数、host copy 和 lowered DMA，而不是只用 tensor bytes 除以理论带宽的依据。

需要特别区分“共享物理内存”和“没有数据移动”。在本文平台上，CPU 与 VTA 可以访问同一 DDR，
但 VTA 仍需把权重和激活从 DDR LOAD 到片上 SRAM，并在必要时 STORE 回 DDR；两个框架 Executor
也可能各自拥有逻辑输入输出张量，从而在同一 DDR 内发生一次冗余 CPU copy。因此共享内存优化的
对象至少包括地址可达性、cache coherence、框架物化和缓冲区生命周期，不能只报告 DMA 峰值带宽。

## 2.5 异构运行时内存管理与零拷贝

TVM Pipeline Executor 通过多个子图 Executor、worker、队列和通知实现任务级流水并行，但其 RFC
明确把自动切图排除在初始范围之外[16]。在本文所使用的实现路径中，stage 间数据 forwarding 仍
采用 owning tensor 和普通 `SetInput` 语义，因而控制流水线并不等于跨 CPU/VTA Executor 的共享
物理 slot。TVM GraphExecutor 已提供 `set_input_zero_copy` 和 `set_output_zero_copy`；这些 API
可以重绑定外部张量，却不负责取得 u-dma-buf 物理内存，也不管理多 Executor 间的 producer/
consumer 所有权、帧代次和完成事件[18]。因此，本文不把 API 本身作为创新，而研究 API 上层的固定
映射共享 slot 协议。

RIMMS 面向 task-to-processing-element 映射在运行时动态变化的 CPU/GPU/FPGA 系统，使用运行时
数据位置跟踪和内存分配来避免不必要迁移[17]。其核心难题是运行前不知道下一任务位于哪个设备，
因此需要通用位置管理。本文的 CPU/VTA 划分在运行前已冻结，且两端共享 DDR，不需要照搬完整的
动态 location manager；可借鉴的是“显式记录数据状态，只执行必要迁移”。本文将这一思想收窄为
每条静态 edge 的 slot owner、frame generation、consumer count 和 completion。

CoDL 也指出统一内存系统仍需处理同步、映射和数据类型转换，数据共享开销可能抵消并发收益[5]。
与 CoDL 的算子内部 CPU-GPU 协同不同，本文优化连续 DNN stage 之间的跨帧交接；与 RIMMS 的动态
映射不同，本文利用固定拓扑预先规划缓冲区，从而减少运行时元数据与位置判断，但相应牺牲了动态
设备映射能力。

## 2.6 国内外研究现状小结

从研究地域看，国内外工作都已覆盖 DNN 编译、异构执行和统一内存优化。例如 TVM/VTA 形成了开放
的编译与 FPGA 加速基础[10-11]；CoDL 由国内研究团队提出，面向移动 CPU-GPU 统一内存中的协同
执行与转换开销[5]；国外的 Tarnawski、DNNPipe 分别研究一般设备放置和异构节点流水线搜索[1,7]，
NEURAghe、Synergy研究 Zynq 上的 CPU-FPGA 协同[3-4]，HaX-CoNN 与 RIMMS 分别研究共享内存竞争
和动态异构内存管理[6,17]。现有研究并非缺少某个单项技术，而是尚未直接覆盖本文约束的交集：

```text
编译器约束的中小粒度 segment
+ 单个物理 VTA 被多个逻辑 stage 复用
+ 每个 CPU stage 独立线程参数但共享四核
+ CPU/VTA 共享 DDR 与复合边界
+ 固定映射下跨 GraphExecutor 的有界 zero-copy slot
```

因此，本文的研究策略不是另造通用异构运行时，而是在固定嵌入式 CPU-FPGA 场景中利用已知映射，
把已有的 placement、pipeline 和 zero-copy 基础能力组合为可验证的专用优化。

## 2.7 现有工作的覆盖关系

| 工作 | 主要目标 | 与本文最接近的部分 | 本文额外处理的问题 |
|---|---|---|---|
| Tarnawski 等[1] | 多设备 operator placement | 计算与通信联合成本、流水线目标 | 单 VTA 复用、共享 DDR、TVM 线程与 lowering |
| PartitionTuner[2] | 异构编译器调度 | grouped operator profile、编译器合法性 | 跨帧 II、共享资源负载 |
| NEURAghe[3] | Zynq CPU-FPGA CNN 推理 | ARM 与 FPGA 协同 | 自动切图与线程联合搜索 |
| Synergy[4] | Zynq 高吞吐流水线 | 多线程跨帧流水线 | 任意 segment placement、复合边界 |
| CoDL[5] | 移动 CPU-GPU 算子协同 | 统一内存、转换和并发非线性 | DNN stage 划分、单 VTA mutex |
| HaX-CoNN[6] | 多 DNN 共享内存调度 | memory contention 与转换 | 单 DNN 跨帧流水和 FPGA lowering |
| DNNPipe[7] | IoT 节点流水线划分 | 最大 stage time、DP 剪枝 | 片上共享资源与单加速器复用 |
| TVM Pipeline Executor[16] | 子图任务级流水 | worker、queue、stage DAG | 自动切图和跨 Executor 共享 slot |
| RIMMS[17] | 动态异构内存管理 | 显式数据状态、减少冗余迁移 | 利用静态映射预规划有界 slot |

综上，本文的研究价值不来自简单更换硬件，而来自这些约束在当前 TVM/VTA SoC 上的交集，以及对该交集进行可运行的系统实现和实验检验。

# 第3章 系统架构与问题定义

## 3.1 硬件与软件平台

实验平台为 AXU5EVB Zynq UltraScale+ MPSoC 开发板。PS 侧包含 4 个同构 ARM CPU 核，实验固定为 userspace governor 和 1066666 kHz。PL 侧加载 VTA HPC/coherent bitstream，CPU 与 VTA 通过共享 DDR 交换数据。VTA 配置的 batch 为 1、block 为 16，输入和权重采用 8 bit，累加采用 32 bit。软件栈基于 TVM 0.13.0 源码版本和 VTA runtime，AArch64 交叉编译器为 GCC 9.2.0。

运行时为每个流水线 stage 建立一个持久 host worker。CPU stage 调用 TVM graph runtime；VTA stage 的 set input、run 和 get output 由一个全局 mutex 串行保护。相邻 stage 之间使用深度为 2 的有界队列。实验固定 `poll_sleep_ns=1000`，不把 queue depth 和轮询策略作为 V1 搜索变量。

**图3-1建议内容：** 画出 ARM 四核、共享 DDR、HPC 端口、VTA 和 CPU/VTA stage workers。用一条全局 VTA mutex 包围所有 VTA stage，并标出不同帧可在 CPU 与 VTA 上并行。

## 3.2 共享内存数据路径

本文将一次 CPU-VTA 边界的数据活动分为三层。

1. **框架物化。** producer Executor 的输出先读到普通 CPU 张量，再由 consumer 的普通
   `set_input` 复制到其内部输入或 VTA 可访问缓冲区。这一层在共享 DDR 内发生，可以通过跨
   Executor 的同址绑定消除。
2. **数据适配。** 当两侧 dtype、layout、padding 或量化参数不同，必须执行 pack、quantize、
   requantize 或 slice。它属于真实计算，不能仅靠指针绑定消除，但可以直接写入目标共享 slot，
   避免再生成中间副本。
3. **VTA 内部搬运。** VTA LOAD/STORE 在共享 DDR 与片上 SRAM 之间传输物理 tile。这是 VTA
   执行所必需的数据流，边界 zero-copy 不会消除。

**图3-2建议内容：** 左侧画普通跨 Executor 路径，依次经过 producer 内部张量、runner 临时张量、
consumer 输入和 VTA SRAM；右侧画共享 slot 路径，producer 与 consumer 绑定同一 u-dma-buf 地址。
分别用虚线框标出“可消除的框架物化”“可能需要保留的 layout/量化 adapter”和“不可由边界
zero-copy 消除的 VTA LOAD/STORE”。

普通路径可简化为：

```text
producer internal tensor -> CPU temporary tensor -> shared/VTA input buffer
```

共享 slot 路径为：

```text
producer and consumer bind the same u-dma-buf slot -> queue passes only slot token
```

二者都使用 DDR，但前者多了一次框架层读取和写入。本文固定 HPC/coherent bitstream 与
`VTA_COHERENT_ACCESSES=1` runtime，正式路径中显式软件 flush/invalidate 为 0。硬件 cache
coherence、线程之间的 release/acquire 同步、VTA 完成等待是三个不同问题；使用一致性端口不能
替代 slot 生命周期和设备完成同步。

## 3.3 计算单元与连续阶段

以 ResNet18[12] 为例，将编译前后的 DNN 图转换为有序计算单元：

```math
U=(u_1,u_2,\ldots,u_n).
```

计算单元是 V1 允许切分的最小粒度。ResNet18 被拆为 21 个单元，包括 stem、各残差块的 main 分支、skip projection、add-ReLU tail 和 head。残差 tuple 边界保留每个 slot 的 shape、dtype 与 ordinal，防止只比较名称造成错误连接。

21 个单元的组成如下：

| 单元编号 | 组成 | 数量 | 输出契约 |
|---|---|---:|---|
| 00 | stem | 1 | 单 tensor |
| 01,03,05,08,10,13,15,18 | residual main path 到 add 前 | 8 | main 与 residual tuple |
| 06,11,16 | 下采样块 skip projection | 3 | 更新后的 tuple |
| 02,04,07,09,12,14,17,19 | add + ReLU tail | 8 | 单 tensor |
| 20 | global pool + flatten + dense head | 1 | 分类输出 |

该粒度并非“任何残差块内部都禁止切分”。真正的约束是依赖闭合：不能只把 main tensor 传给下游
而丢失 skip tensor，也不能在尚未形成可描述 tensor contract 的融合算子内部切开。对无 projection
的 block，main-preadd 与 add-ReLU tail 可在二元 tuple 边界切分；对下采样 block，skip projection
作为独立单元保留。这样比整 block 切分更细，又避免单算子级搜索破坏残差语义或产生大量无法编译
的切点。

一个 stage `S_j=[u_a,\ldots,u_b]` 必须满足：

1. 覆盖连续单元且非空；
2. 与相邻 stage 的 tensor contract 兼容；
3. 若放到 VTA，则起止位置满足 VTA backend 规则并通过 native lowering；
4. 相邻同设备 stage 自动合并，消除等价表示；
5. 第一个和最后一个 stage 固定为 CPU，VTA island 数为 1 至 3。

**图3-3建议内容：** 展开一个带 projection 的 ResNet18 残差块。将 main-preadd、skip projection
和 add-ReLU tail 画成三个依赖闭合单元；用红色叉号表示只传 main tensor 的非法切法，用绿色边界
表示同时携带 main/skip tuple contract 的合法切法。

## 3.4 划分决策变量

候选方案记为：

```math
p=(z,c,t),
```

其中 `z` 表示各 segment 的设备类型，`c` 表示 stage cut，`t=(t_1,\ldots,t_m)` 表示各 CPU stage 的 TVM 线程参数，且：

```math
t_j\in\{1,2,3,4\}.
```

线程参数不是对四个物理核心的静态切块，因此不存在 `sum(t_j)<=4`。实验运行时每个 CPU stage 独立使用 `[0,t_j)` affinity mask，不同 stage 的 mask 可以重叠。与此同时，各 stage 会竞争相同核心，所以必须使用 process CPU time 建立共享容量约束。

V1 固定以下变量：

```text
VTA bitstream、schedule 与 tile
量化与 graph packing
CPU affinity 规则
queue depth 与 polling policy
单 VTA mutex 范围
最大 VTA island 数
```

## 3.5 稳态流水线目标

单帧串行延迟为各 stage 时间之和，但多帧填满流水线后，吞吐由稳态 II 决定。若每个资源每帧需要的服务时间分别为 `D_r(p)`，则资源容量下界为：

```math
II(p)\ge \max_r D_r(p).
```

本文使用以下 V1 估计：

```math
\widehat{II}(p)=\max\left\{
D_{CPU-stage}(p),
D_{single-VTA}(p),
D_{core}(p),
D_{DDR}(p)
\right\},
```

```math
\widehat{FPS}(p)=\frac{1000}{\widehat{II}(p)}.
```

这里的“取最大值”表示资源可处理不同帧并发生重叠，并非把 CPU、VTA 与通信简单串行相加。同一 CPU stage 内的 run 与 CPU 所属边界顺序执行；全部 VTA stage 和 VTA mutex 所属边界顺序执行；不同资源之间允许跨帧并行。

## 3.6 共享 slot 约束

对每条可零拷贝异构边 `e` 预分配 `q_e=2` 个 slot。每个 token 写为：

```math
x=(e,s,g,f),
```

其中 `s` 是 slot 编号，`g` 是复用代次，`f` 是帧编号。合法运行必须满足：同一时刻一个 slot 只有
一个写者；consumer 只能读取已经发布且 generation 匹配的 slot；同步 `run()` 返回或 completion
到达后才可释放；多张量边界必须作为一个逻辑 slot 同时发布和释放。

双 slot 的目的不是降低单次 copy，而是允许 consumer 使用第 `k` 帧 slot 时 producer 准备第
`k+1` 帧。slot 数固定为 2，不作为当前搜索变量；若 producer 追上 consumer，则通过 condition
variable 形成有界 backpressure，而不能覆盖尚未消费的数据。

## 3.7 评价指标

静态目标函数的完整枚举最优只用于验证搜索算法。硬件实验使用候选池最优：

```math
FPS_{oracle}=\max_{p\in P_{measured}} FPS_{measured}(p).
```

Top-K regret 定义为：

```math
regret@K=1-\frac{\max_{p\in TopK}FPS_{measured}(p)}{FPS_{oracle}}.
```

`regret@K=0` 表示 Top-K 中已经包含该实测池的最优候选。该 oracle 只对已测池有效，不等于整个搜索空间的未知硬件全局最优。本文同时报告 Spearman 秩相关、Top-K recall、MAE/MAPE，以及达到候选池最优吞吐 95% 所需的评价次数。

# 第4章 资源感知成本模型与最小 Profile

## 4.1 设计原则

本文不追求一次性拟合全部硬件细节，而是只测量能够改变候选排序的主要资源。成本模型遵循四项原则：

1. 参数必须对应明确的物理工作或运行时服务；
2. 同一 transaction 只能由一个模块计费；
3. profile 不读取完整候选的流水线吞吐；
4. 更复杂的 contention、FIFO 或 tile 模型只有在残差能明确归因时才加入。

这一设计将问题分成“网络产生多少工作”和“硬件完成单位工作需要多久”。切换 DNN 时重新生成 workload 和必要的新 signature，不直接复用候选吞吐标签。

## 4.2 CPU 阶段服务

对 CPU segment `S_j` 和线程参数 `t_j`，局部表给出：

```math
\widehat T_{cpu}(S_j,t_j)=Cost_{cpu}(S_j,t_j).
```

早期版本参考 Roofline 模型以运算量和内存流量描述 CPU 服务下界[14]，并使用统一的 `ms/logical-GOP` 初始化计算价格。该价格在不同 segment 上的 MAPE 为 20.66% 至 37.78%。板端实验表明它显著低估 stem、transition 和 head 等 CPU 段。因此最终 V1 使用 21 个原子单元的 wall time 与 process CPU time，并对连续 segment 做可加组合。

CPU 线程加速并不线性。以固定三阶段拓扑为例，首 CPU stage 使用 `t=1,2,3,4` 时实测吞吐分别为 5.413、6.237、9.312 和 10.449 FPS。因此 `t_j` 必须进入成本查询键，而不能用 `T(1)/t_j` 近似。

## 4.3 CPU 核心池约束

令 `W_j` 为 CPU stage 的 process CPU core-ms，`W_host,total` 还包括 VTA host 调用和边界 adapter 的 CPU work。四核 work-conservation 下界为：

```math
D_{pool}=\frac{W_{host,total}}{4}.
```

由于 stage 使用嵌套前缀 affinity `[0,t_j)`，所有 `t_j<=k` 的 stage 只能在前 `k` 个核心上运行。因此还需要：

```math
D_{prefix}=\max_{k=1,2,3,4}
\frac{\sum_{j:t_j\le k}W_j}{k}.
```

最终：

```math
D_{core}=\max(D_{pool},D_{prefix}).
```

该式没有把线程数当作互斥配额，也不假设 worker 永久绑定到某一个具体核心。它只是任何可行调度都必须满足的容量下界。

## 4.4 单物理 VTA 服务

设候选中共有 `r` 个 VTA island。由于 native runner 使用一个全局 VTA mutex，各 island 不能同时运行：

```math
D_{single-VTA}(p)=
\sum_{j=1}^{r}\widehat T_{vta}(S_j)
+\sum_{e\in E_{vta-owner}}\widehat T_{boundary,vta}(e).
```

`T_vta(S_j)` 使用固定 schedule 下的 VTA run service，已经包含设备内部 LOAD、compute 与 STORE 的实际交叠。本文不再把 DMA service 额外加到 VTA run 上，否则会产生重复计费。

该约束还解释了一个可用于剪枝的规律：增加 VTA island 并不会增加物理 VTA 并行度，却一定增加或保持 VTA 总服务与异构边界数量。只有中间 CPU 段显著缓解其他瓶颈时，额外 island 才可能有收益。

## 4.5 CPU-VTA 复合边界

跨设备边界不是单一的 `bytes/bandwidth`。借鉴 LogGP 将固定调用开销与按字节开销分离的思想[15]，本文将实际边界写为：

```math
T_{boundary}(e)=T_{adapter}(e)+T_{set/get}(e).
```

边界 signature 至少包含方向、逻辑字节数、dtype、layout、padding/slice 和量化状态。根据 runner 的实际锁范围进一步分配：

```text
CPU -> VTA：CPU producer get 归 CPU stage，VTA set 归 VTA mutex；
VTA -> CPU：VTA get 归 VTA mutex，CPU consumer set 归 CPU stage。
```

VTA lowered LOAD/STORE 只属于 VTA stage 的物理 DMA transaction，不在 boundary 再计费。每笔工作使用稳定 accounting id，例如：

```text
boundary:03:host_adapter:0
vta_stage:1:load_instruction:17
vta_stage:1:store_instruction:4
```

P2 grouped holdout 中，CPU->VTA 和 VTA->CPU 边界总时间 APE 分别为 0.84% 和 1.66%。

## 4.6 共享 DDR 服务下界

CPU stage 的输入输出、VTA physical LOAD/STORE 和 adapter 最终访问共享 DDR。本文按唯一 transaction owner 汇总：

```math
D_{DDR}(p)=\sum_{a\in A}\frac{Q_a(p)}{B_a},
```

其中 `Q_a` 是访问类 `a` 的物理字节数，`B_a` 是对应的可持续带宽。P2 实测 DDR 带宽如下。

| 线程数 | Read (GB/s) | Write (GB/s) | Copy (GB/s) |
|---:|---:|---:|---:|
| 1 | 2.163 | 4.100 | 4.275 |
| 2 | 4.167 | 7.749 | 6.293 |
| 3 | 5.879 | 7.943 | 6.867 |
| 4 | 7.336 | 7.757 | 7.529 |

`D_DDR` 与已经包含内存 stall 的 isolated stage wall time取最大值，不再相加。该项表达总带宽容量，但没有描述 CPU 与 VTA 同时访问 DDR 时的相互降速。因此它是必要但不充分的共享资源模型。

## 4.7 最小 Profile 与验证

初始 profile 包含 19 个冻结性能 case，其中 7 个为 native stage 模式、12 个为 DDR case。共获得 84 个 native stage frame 和 96 个 DDR sample，耗时 262.90 s，未测量任何搜索候选吞吐。

初始加性成本为：

| 参数 | 测量值 |
|---|---:|
| CPU t1 | 225.166 ms/logical-GOP |
| CPU t2 | 114.763 ms/logical-GOP |
| CPU t3 | 95.856 ms/logical-GOP |
| CPU t4 | 67.244 ms/logical-GOP |
| VTA | 27.551 ms/logical-GOP |
| CPU->VTA boundary | 3.790 ms/MiB |
| VTA->CPU boundary | 3.536 ms/MiB |

统一 CPU 斜率在前瞻候选上失效后，本文补测 21 个 atomic CPU unit 的 `threads=1..4` wall time 与 core-ms，并冻结 23 个 grouped checks。其 median、P95 和 max APE 分别为 0.89%、4.17% 和 17.96%，通过预先设定的 5%、15% 和 25% gate。该迭代说明，增加 profile 的依据应是已定位的排序误差，而不是追求无限细化的硬件画像。

# 第5章 k-best 动态规划搜索

## 5.1 搜索空间

对于 `n` 个计算单元，朴素方法需要同时枚举所有合法 cuts、设备标签和 CPU 线程组合。即使 VTA tile 固定，ResNet18 仍有 4623 个 canonical topology；各 CPU stage 独立选 1 至 4 线程后，共有 972528 个执行配置。未来加入 tile 后，空间还会进一步扩大。

V1 不进行候选上板枚举。搜索只查询局部成本表，输出较小的 Top-K，再对 shortlist 执行 native compile、reference 和性能测试。

## 5.2 DP 状态与标签

对线性化单元，状态定义为：

```text
DP[i,r,d]
```

其中 `i` 表示已覆盖前 `i` 个单元，`r` 是已使用的 VTA island 数，`d` 是最后一个 stage 的设备。

由于目标由多个资源最大值组成，每个状态标签保存：

```text
最大 CPU stage 时间
最后一个 CPU stage 的当前时间
VTA service 累计值
VTA-owned boundary 累计值
CPU core work 总量
按 threads 分组的 CPU core work
boundary host core work
共享 DDR demand
前驱与完整路径
```

保留“最后一个 CPU stage 时间”是为了正确处理一个 CPU stage 两侧均与 VTA 相邻时的边界归属。若只保存全局最大 CPU 时间，第二条边界可能被错误地加到另一个 stage。

## 5.3 状态转移

从当前位置 `i` 选择下一个合法连续 segment `[i,j)`：

1. 放到 CPU 时，枚举 `t in {1,2,3,4}`，更新 stage wall time、core-ms 和 DDR demand；
2. 放到 VTA 时，增加 VTA island 计数，累加 VTA service 与 VTA mutex 所属边界；
3. 设备发生变化时查询对应方向的 boundary cost；
4. 相邻同设备 stage 不生成新状态，而是在 canonical representation 中合并；
5. 任一资源需求只增不减，因此当前 partial score 是完整候选 score 的下界。

## 5.4 k-best label-setting

算法使用按下界排序的优先队列。其伪代码如下。

```text
输入：计算单元 U、合法 segment、局部成本表、最大 island 数 R、候选数 K
输出：预测 II 最小的 K 个完整候选

1  将空标签放入最小堆 Q
2  complete <- empty
3  while Q 非空：
4      L <- 弹出下界最小的标签
5      若已有 K 个完整候选，且 bound(L) 大于第 K 个完整 score：
6          终止搜索
7      若 L 覆盖全部单元：
8          按 (score, candidate_id) 插入 complete 并去重
9      否则：
10         枚举所有合法下一 segment、设备和 CPU threads
11         更新资源累计量，生成 successor
12         将 successor 放入 Q
13 返回 complete 的前 K 项
```

终止条件是安全的，因为所有新增 stage、边界和资源 work 非负，任何 successor 的最终 score 不小于当前下界。本文没有在 Top-K 模式使用未经证明的 Pareto 删除，避免不同路径在相同数值目标下因 candidate id tie-break 而丢失正确顺序。

**图5-1建议内容：** 横轴为已覆盖 unit 位置，节点按“已用 VTA island 数、末 stage 设备”分层；
从节点向右连接合法 CPU/VTA segment，并在边上标出 threads 与局部成本。右侧同时画最小堆，说明
算法每次扩展当前资源下界最小的标签，在队列最小下界超过第 K 个完整解时停止。

**命题5-1：** 当所有 segment、boundary 与 resource demand 均非负时，上述终止条件不会遗漏 score 小于当前第 K 个完整候选的路径。

**证明：** 设优先队列当前最小下界为 `b`，已找到的第 K 个完整候选 score 为 `q`，且 `b>q`。任一未完成路径都位于队列中的某个标签或其后继下。由于每次转移只增加非负服务需求，各资源累计值不减，后继完整路径的 score 不小于其祖先标签下界，因而均不小于 `b`，进一步大于 `q`。所以队列中不存在可进入当前 Top-K 的路径，算法可以终止。证毕。

## 5.5 正确性与复杂度边界

在最终 atomic CPU 成本下，搜索扩展 61255 个标签、生成 203585 个标签，并在得到 28 个完整标签后确认 Top-20 阈值并停止。完整枚举则遍历 972528 个执行配置。两者 Top-20 candidate id 和预测周期逐项一致，最大数值差为 0 ms。

该结果证明当前固定 tile、线性化 ResNet18 实例上的实现正确性，但不意味着任意 DAG 与 tile 联合搜索已经获得多项式复杂度。最坏情况下，k-best 标签数仍可能随路径组合快速增长。YOLOv3-tiny 的当前实现因为 105696 个配置仍可承受，采用完整枚举作为 branch-aware DP 的 oracle；可扩展 DAG 搜索仍属于后续工作。

## 5.6 候选多样性

纯数值 Top-20 可能包含同一 topology 的多个线程变体。最终 ResNet18 自然 Top-20 只有 4 种 topology。若把这些近重复方案全部上板，会浪费有限预算。因此本文区分：

1. raw Top-K：用于验证优化目标的精确顺序；
2. topology-diverse Top-K：每种 topology 先保留最佳线程配置，再加入少量预注册 thread control；
3. compile/reference failure 仍消耗预算，不依据中途吞吐替换候选。

# 第6章 原型系统设计与实现

## 6.1 系统流程

原型系统包含七个步骤：

```text
DNN 导入与量化
  -> 计算单元和 tensor contract 提取
  -> compiler-valid segment manifest
  -> 最小局部 profile 与成本表
  -> k-best 静态搜索
  -> Top-K native compile/reference/board measurement
  -> manifest 驱动的边界共享 slot 绑定与执行
```

**图6-1建议内容：** 以两种颜色区分静态搜索和板端运行时。明确 candidate throughput 只在排名
冻结后出现，不能反向进入同一轮 score；共享 slot 由已冻结 candidate 的 edge manifest 生成。

## 6.2 原生流水线运行时

运行时使用 C++ 实现多 stage pipeline。每个 stage 拥有持久 worker，从输入 FIFO 取 frame，执行 set/run/get，再把结果推入输出 FIFO。CPU stage 可分别设置 TVM thread-pool 参数和 affinity。VTA stage 在调用前获取全局 mutex，确保多个逻辑 island 与单物理设备语义一致。

吞吐测量重复使用同一输入图片，以消除文件读取对稳态执行的影响。对 22 帧运行丢弃最先完成的 2 帧，使用剩余 20 帧 `stage_last_end` 首尾跨度估计周期，并同时报告完成间隔 median 和 P95。重复图片只影响输入内容多样性，不影响固定计算图的吞吐测量。

旧流水线的边界数据通路使用普通 `get_output/set_input`。producer 先把内部输出复制到 runner
拥有的中间张量，consumer 再复制到自己的输入。因此，即使两个 stage 位于同一 SoC 并共享 DDR，
Executor 所有权边界仍会引入框架物化。该路径作为 B0 普通复制基线保留。

## 6.3 共享 slot 零拷贝设计

### 6.3.1 设计分工

本文复用三项已有能力，但补齐其间缺少的协议：

| 层次 | 已有能力 | 本文实现 |
|---|---|---|
| 物理内存 | VTA driver/u-dma-buf 提供 CPU 与 VTA 可访问区域 | 按 edge/tensor contract 分配、对齐并审计 slot 区间 |
| Executor API | TVM `set_input_zero_copy/set_output_zero_copy` 可重绑定外部 `DLTensor` | 为同一物理 slot 构造 CPU/VTA 双 view 并绑定两侧 Executor |
| 流水控制 | worker、queue、mutex 可调度 stage | queue 只传 slot token，管理 generation、owner、completion 与回收 |

GraphExecutor API 不会自动保证外部内存来自正确的物理池，也不会知道另一 Executor 是否仍在使用
该地址。反过来，仅有 u-dma-buf 也不能改变 Executor 内部输入输出指针。本文的
`BoundarySlotManager` 位于二者之间，其贡献是让编译期 edge contract 转换为运行期有界所有权。

### 6.3.2 Slot 布局与双视图

每个逻辑 slot 记录：

```text
edge id、slot id、frame id、generation
virtual/physical address、allocated bytes、alignment
shape、dtype、layout、producer/consumer
CPU DLTensor view、VTA DLTensor view
state、remaining consumers、error/completion status
```

CPU view 与 VTA view 指向相同 `data` 地址，但 device type 分别满足 CPU 与 ext_dev Executor 的
检查。管理器持有底层 NDArray 和 view 生命周期，Executor 只借用指针。绑定前检查连续性、shape、
dtype、对齐和物理范围；转换不兼容的 edge 不能冒充 zero-copy。

### 6.3.3 生命周期与双缓冲

单生产者/单消费者状态机为：

```text
FREE -> PRODUCER_WRITING -> READY -> CONSUMER_USING -> FREE
```

producer 在运行前取得空闲 slot 并绑定输出，完成后发布 `{edge_id, slot_id, generation, frame_id}`；
consumer 校验 token 后绑定同一 slot，执行完成再释放。状态转换由 mutex 和 condition variable
保护，超时或任一 worker 错误会唤醒并终止所有等待者。generation 用于阻止 slot 编号循环复用时
旧 token 被误认成新帧。

每条边固定两个 slot。slot 0 被 consumer 处理时，producer 可写 slot 1，从而保留跨帧重叠。
双 slot 解决的是生命周期安全和流水并发，不意味着边界一定成为吞吐瓶颈。若 consumer 较慢，
producer 等待空闲 slot 是正确的 backpressure，而不是应额外加到 stage 时间上的独立开销。

**图6-2建议内容：** 画两条时间轴表示 producer 与 consumer，slot 0/1 交替承载 frame k、k+1、
k+2；在下方画 `FREE→WRITING→READY→USING→FREE` 状态机，并标出 token 中的 slot、frame 和
generation。对比单 slot 必须等待与双 slot 可重叠的时序。

### 6.3.4 残差多张量边界

部分 ResNet18 切点横跨尚未完成合流的残差依赖，一条 stage edge 同时包含主分支与 skip 张量。
初始“每条边一个 tensor buffer”的实现会产生一部分张量属于新帧、另一部分仍属于旧帧的风险。
最终实现把一条边的全部 live tensor 组成一个原子逻辑 slot：共享 frame/generation 状态，一起发布、
消费和释放。该修正使自然 Top-20 中包含 tuple 边界的 20/20 候选都通过逐帧输出等价检查。

## 6.4 编译与可执行性审计

ResNet18 冻结 shortlist 的 20 个候选全部成功生成 AArch64 native package。它们共引用 88 个 stage，按完整 compile key 去重后为 33 个，其中 15 个 CPU stage、18 个 VTA stage。每个 VTA 二进制均引用原生 `VTAPushGEMMOp` 或 `VTAPushALUOp`，未使用 CPU fallback。

首次完整构建与审计耗时 626.303 s；缓存复审耗时 0.895 s。利用硬链接复用不可变产物后，逻辑 3.165 GB 的 package 数据对应约 540 MB 唯一 inode 数据，减少了 SD 卡部署压力。

## 6.5 Correctness gate

每个候选必须先通过以下检查，随后才允许计时：

1. serial 与 pipeline 输出 tensor 的 shape 和 dtype 一致；
2. serial 与 pipeline 完整 `[1,1000] float32` logits 一致；
3. 与独立 MXNet reference 比较 cosine、normalized RMSE、Top-K overlap 和 top1；
4. 多帧输出 hash 稳定；
5. module 不包含 fallback，VTA runtime 无 timeout。

共享 slot 另外执行以下 gate：所有物理区间对齐且互不重叠；两 slot 均被实际使用；A/B 交替输入
与 ordinary-copy 逐帧一致；frame/generation 顺序正确；流水排空后全部 slot 回到 `FREE`；B2
目标边界的 framework materialization bytes 为 0；VTA 内部 LOAD/STORE 与 B0 保持一致。

在 qualification smoke 中，serial 与 pipeline logits 逐字节一致，cosine 为 0.995778、normalized RMSE 为 0.093204、Top-10 overlap 为 9/10。量化 TVM/VTA 输出 top1 为 285，浮点 MXNet reference 为 282，因此本文将其表述为通过预注册的量化语义 gate，而非 bit-exact 浮点等价。后续五阶段候选的 top1 与 reference 均为 282。

## 6.6 实验治理与可复现性

系统为协议、硬件指纹、候选清单、score 与 measurement 分别生成 JSON artifact 和 SHA256。硬件指纹包含 bitstream、TVM/VTA source、runtime library、编译器、CPU 频率和运行时策略；IP 地址只作为连接信息，不作为稳定硬件身份。

评分阶段读取的候选池文件不包含实测吞吐。只有 score 封存后，evaluation 阶段才按 candidate id 连接标签。该物理隔离避免在已有历史数据上反复调参后再把结果表述为前瞻搜索。

性能实验按 boot 分组。一个 boot 指开发板从一次系统启动到下一次重启之间的同一运行环境；同一
boot 内大量帧只能刻画帧内抖动，不能替代独立启动带来的温度、内存状态和系统服务差异。正式稳定
提升需要至少三个独立 boot，并在 boot-level 配对置信区间上判断。单 boot 结果只称 qualification
或 stress-case observation。

流水线 II 使用每个计分 block 的完整完成窗口：

```math
II_{block}=\frac{t_{last\ completion}-t_{first\ completion}}{N-1}.
```

旧实现使用相邻完成间隔的中位数，在完成事件成批出现时会忽略批次之间的长空洞并产生虚假提升；
该口径已经废止。逐帧间隔中位数仅用于调度抖动诊断。

# 第7章 实验与结果分析

## 7.1 实验问题

本章回答以下问题：

1. **RQ1：** CPU-VTA 直接通信是否会显著影响周期预测与候选排序？
2. **RQ2：** k-best 搜索能否在当前成本函数下正确恢复静态 Top-K？
3. **RQ3：** 自然 Top-20 的板端性能如何，CPU 线程和原子 segment profile 有何影响？
4. **RQ4：** 当前模型在历史 ResNet18 池和 YOLOv3-tiny 上能提供多强的筛选证据？
5. **RQ5：** 组件 profile 能否解释当前静态模型遗漏的固定残差和 DDR contention？
6. **RQ6：** 共享 slot 能消除多少边界内存工作，并在什么条件下改善 latency 或 throughput？

## 7.2 实验设置

ResNet18 输入为单张图像，并在多帧流水线中重复使用。VTA schedule、tile、量化和 bitstream 全部固定。主要搜索预算为 Top-20；性能试验运行 native-only package，先做 correctness，再测 22 帧并丢弃 2 帧 warmup。

实验数据分为五类：

1. 200 条历史 ResNet18 pipeline profile，用于通信机制消融和离线回放；
2. 当前 runtime 下新测的 ResNet18 线程对照与自然 Top-20，用于候选筛选和细排检查；
3. 169 条历史 YOLOv3-tiny pipeline 记录及 6 组串行组件 profile，用于回顾性跨模型分析；
4. P7 CPU 并发、runtime 固定开销和 CPU-VTA 共享 DDR matched control，用于物理公式重建；
5. P8 三 boot B0/B2、自然 Top-20 单 boot 扫描和三-island 串行 stress case，用于共享 slot 消融。

这些数据的 runtime 指纹和候选语法并不完全相同，本文只在明确兼容的层级做比较。

## 7.3 RQ1：通信代价消融

对 200 条 ResNet18 记录，比较只使用共享 compute 的模型与加入 direct Host/VTA boundary copy 的模型：

```math
T_{compute}=\max(\max T_{cpu-stage},\sum T_{vta-stage}),
```

```math
T_{comm}=\max(\max T_{cpu-stage},
\sum T_{vta-stage}+T_{direct-copy}+T_{coherence}).
```

结果如下。

| 模型 | MAE (ms) | RMSE (ms) | Spearman | Kendall | Top-10 regret |
|---|---:|---:|---:|---:|---:|
| 不含直接边界复制 | 17.3617 | 20.1302 | 0.7700 | 0.6255 | 0.0000 |
| 含直接边界复制 | 12.8220 | 13.6655 | 0.9310 | 0.7911 | 0.0000 |

MAE 降低 4.5397 ms，相对改善 26.15%。按 23 个 VTA outer-span group 做 5000 次 grouped bootstrap，改善量 95% 置信区间为 `[1.8054,7.6966] ms`，单侧 `p=0.000200`。因此，在当前平台上显式建模 direct boundary communication 是有必要的。

通信开销还随 island 数增加：

| VTA island 数 | 样本数 | 单帧 direct copy 中位数 | 均值 | P95 |
|---:|---:|---:|---:|---:|
| 1 | 13 | 3.431 ms | 3.324 ms | 3.879 ms |
| 2 | 78 | 6.212 ms | 6.479 ms | 8.820 ms |
| 3 | 109 | 10.959 ms | 10.788 ms | 14.286 ms |

所有 200 条记录的 direct copy 中位数为 7.883 ms，P95 为 13.831 ms，最大值为 15.392 ms。这支持“更多 island 往往产生更多边界代价”的经验规律，但不能单独证明 island 数与总吞吐严格单调。

**图7-1建议内容：** 用箱线图按 1、2、3 个 VTA island 展示 200 条记录的 direct-copy 时间，
在图上标注各组中位数 3.431、6.212 和 10.959 ms；图注强调这是边界数量相关的统计规律，不是
island 数对总吞吐的严格单调因果关系。

`three_stage_f` 的 H2D copy 在串行执行时为 2.875 ms，在流水线中按每帧归一后增至 18.569 ms。该现象表明并发访问会改变有效通信时间，也说明 V1 的聚合 DDR 下界尚未完全描述 contention。

## 7.4 RQ2：静态搜索正确性

ResNet18 搜索得到：

| 项目 | 数量 |
|---|---:|
| 原子计算单元 | 21 |
| 合法 canonical topology | 4623 |
| CPU 线程加入后的执行配置 | 972528 |
| 最终 DP 扩展标签 | 61255 |
| 最终 DP 生成标签 | 203585 |
| 输出候选数 | 20 |

k-best 搜索 Top-20 与流式完整枚举在 candidate id 和 score 上完全一致。该结果验证的是算法实现与成本函数的一致性，不验证成本函数等于真实硬件。

最终静态 rank-1 为：

```text
CPU[00..02] t4 -> VTA[03..17] -> CPU[18..20] t1
```

当前资源下界为：

| 资源项 | 数值 |
|---|---:|
| 最大 CPU stage（含所属边界） | 66.708 ms |
| 单 VTA service（含 mutex 所属边界） | 70.807 ms |
| CPU core pool 下界 | 67.327 ms |
| 共享 DDR 下界 | 9.536 ms |
| 原始资源下界 II | 70.807 ms |
| 原始下界倒数（仅排序分数） | 14.123 |
| 正式绝对 FPS 预测 | 尚未建立 |

这里的 `14.123` 是资源下界倒数，只用于静态排序，不能称为最终预测 FPS。历史 200-case 的两个
批次互作训练集和测试集时，可观察到约 `1.42` 的周期比例偏差；加法形式则得到 33.764--36.459 ms
残差并获得 5.535%--6.046% 对侧批次 MAPE。二者都读取了完整候选的实测吞吐，只能诊断“公式
遗漏系统开销”，不得写回正式 score。

冻结 Top-20 已全部完成 native compile、独立 reference 和板端测量。静态前 10 项如下：

| 静态名次 | 候选简写 | 资源下界 II (ms) | 实测 II (ms) | 实测 FPS | 池内实测名次 |
|---:|---|---:|---:|---:|---:|
| 1 | CPU00--02 t4 / VTA03--17 / CPU18--20 t1 | 70.807 | 91.850 | 10.887 | 5 |
| 2 | CPU00--02 t4 / VTA03--17 / CPU18--20 t2 | 70.807 | 96.862 | 10.324 | 19 |
| 3 | CPU00--02 t4 / VTA03--17 / CPU18--20 t3 | 70.807 | 93.264 | 10.722 | 9 |
| 4 | CPU00--02 t4 / VTA03--17 / CPU18--20 t4 | 70.807 | 91.889 | 10.883 | 6 |
| 5 | CPU00--02 t4 / VTA03--15 / CPU16--20 t2 | 71.059 | 93.840 | 10.656 | 11 |
| 6 | CPU00--02 t4 / VTA03--15 / CPU16--20 t3 | 71.059 | 95.490 | 10.472 | 16 |
| 7 | CPU00--02 t4 / VTA03--16 / CPU17--20 t1 | 71.110 | 91.525 | 10.926 | 4 |
| 8 | CPU00--02 t4 / VTA03--16 / CPU17--20 t2 | 71.110 | 93.717 | 10.670 | 10 |
| 9 | CPU00--02 t4 / VTA03--16 / CPU17--20 t3 | 71.110 | 86.872 | 11.511 | 1 |
| 10 | CPU00--02 t4 / VTA03--16 / CPU17--20 t4 | 71.110 | 96.532 | 10.359 | 18 |

20 个候选全部通过 correctness，实测范围为 10.220--11.511 FPS。按冻结顺序，`regret@1/3/5`
均为 5.419%，`regret@10/20=0`；也就是说第 9 次测量首次遇到该 Top-20 实测池的最优候选。
静态名次与池内实测名次的 Spearman 只有 0.155。合理结论是算法筛出了高性能区域，但不能根据
70--72 ms 的窄下界准确区分这些相近候选。

移除 direct boundary 后，Top-20 与完整模型的 Top-20 重合为 0/20，且 Top-1 发生变化。这进一步说明通信项不仅改善绝对误差，也实际改变搜索结果。

## 7.5 RQ3：线程与 CPU 成本迭代

### 7.5.1 同拓扑线程对照

固定拓扑 `CPU[00..02] -> VTA[03..17] -> CPU[18..20]`，尾 CPU stage 固定 `t2`，改变首 stage 线程数：

| 首 CPU threads | 实测 II (ms) | 实测 FPS |
|---:|---:|---:|
| 1 | 184.757 | 5.413 |
| 2 | 160.345 | 6.237 |
| 3 | 107.388 | 9.312 |
| 4 | 95.707 | 10.449 |

从 t1 到 t4 吞吐提升约 93.04%，但各相邻线程点的收益并不均匀。这验证了联合搜索 per-stage TVM threads 的必要性。

### 7.5.2 统一 GOP 斜率失败

初始模型把不同 CPU segment 统一映射为每线程 `ms/GOP`。其修正后未见 Top-1 是一个五阶段、
两个 VTA island 候选，原始资源下界倒数为 13.240，实际仅 8.869 FPS，throughput regret 相对
当时 5-candidate oracle 为 15.11%。三个 CPU stage 的预测 run 约为 62.95、50.68 和 65.29 ms，
实测中位数约为 117.12、115.23 和 88.82 ms。

该结果否定了“一个 CPU GOP/s 可跨 stem、residual transition 与 head 使用”的假设。误差首先应归因于 CPU segment shape 与 backend efficiency，而不是在没有证据时加入 DDR 或 FIFO 参数。

### 7.5.3 Atomic CPU 成本修正

补测 21 个 atomic unit 后，模型对 5 个已测候选进行盲回放，预测顺序与实测顺序完全一致：

| 排名 | 候选简述 | 原始资源分数倒数 | 实测 FPS |
|---:|---|---:|---:|
| 1 | 三阶段，首段 t4，尾段 t2 | 14.123 | 10.449 |
| 2 | 三阶段，首段 t3，尾段 t2 | 11.980 | 9.312 |
| 3 | 五阶段、两个 VTA island | 8.830 | 8.869 |
| 4 | 三阶段，首段 t2，尾段 t2 | 8.062 | 6.237 |
| 5 | 三阶段，首段 t1，尾段 t2 | 5.518 | 5.413 |

表中第三列用于比较修正前后的排序，不是经过历史 200-case 比例校准的最终 FPS。5 个候选的
cycle MAPE 为 14.66%。因此当前模型已能解释这个小池中的顺序，但原始资源下界仍系统性高估
高性能三阶段配置的绝对 FPS。不能把这 5 个点表述为大规模前瞻 Top-K 成功。

## 7.6 ResNet18 历史池回放

200 条历史记录对应 199 个唯一执行方案，其中 69 个唯一方案符合当前异设备交替 stage 语法。使用最终嵌套前缀核心容量公式回放得到：

| 范围 | Cycle MAPE | 平均有符号误差 | Spearman | regret@5 |
|---|---:|---:|---:|---:|
| 199 unique | 28.02% | -35.14 ms | 0.775 | 10.28% |
| 69 canonical | 26.70% | -34.27 ms | 0.786 | 7.30% |

在 199 个历史方案内部重新排序时，预测 Top-1 的实测排名为 50；在 69 个 canonical 方案中为 18。该公式能够捕获总体趋势，但距离可靠恢复排序前部仍有差距。

在只连接历史 200 条记录时，自然 DP Top-20 没有 exact execution match，因为新候选首段主要选择
t4，而旧数据同 topology 主要测量 t3。Topology-level 上，4 种新 topology 中有 3 种被历史池
覆盖，其旧线程配置为 8.769--9.461 FPS。该回放完成后，本文另行冻结并实测了自然 Top-20 全部
当前配置，结果已在第7.4节报告；新测标签没有回流修改原排序。

历史数据还显示绝对周期存在系统性少估。采用跨批次训练/测试时，纯比例模型的测试 MAPE 为
7.65% 和 8.19%；纯加法模型从两个训练批次得到 36.459 ms 和 33.764 ms 的 offset，在对侧批次
的测试 MAPE 为 5.535% 和 6.046%。加法残差目前比 `1.42` 比例缩放更符合数据，但它仍来自候选级
标签，只能用于提出物理 profile 假设，不能直接写入正式预测。后续需要用 empty/short/long
runtime、submit/sync、cache maintenance 和共享 DDR 并发 matched control 确定这约 34 至 37 ms
残差的物理所有权。

## 7.7 回顾性基线比较

在 69 个语义兼容历史配置上，本文冻结以下模型后再连接吞吐标签：B2 为 single-frame grouped cost，B3 为忽略 CPU contention 与 DDR coupling 的 pipeline max-load，B4 为本文 V1 模型。

| 模型 | regret@1 | regret@3 | regret@5 | 到 95% pool oracle 次数 | Spearman |
|---|---:|---:|---:|---:|---:|
| B2 | 24.34% | 23.22% | 20.53% | 15 | 0.081 |
| B3 | 30.58% | 25.66% | 25.66% | 27 | 0.271 |
| B4 | 9.60% | 7.62% | 7.62% | 12 | 0.482 |

B4 在很小的 K 上优于 B2 和 B3，也优于 uniform random 的 regret 中位数；但 B4 达到 95% pool oracle 需要 12 次，uniform random 中位数只需 9 次。因此当前结果不能支持“本文一定比随机搜索使用更少上板次数”的强结论。该实验池仅覆盖三种固定线程向量，也不足以单独验证线程联合搜索。

这一负面结果说明，论文评价不能只选择有利的 regret@1，也必须报告完整预算曲线和随机基线。

## 7.8 RQ4：YOLOv3-tiny 跨模型检查

YOLOv3-tiny[13] 具有两个检测头和 route 分支。本文按 12 个 branch-aware split point 生成 684 个合法 topology；加入 CPU 线程后得到 105696 个配置。当前规模可完整枚举，因而先把枚举作为未来 branch-aware DP 的正确性 oracle。

### 7.8.1 直接迁移失败

直接使用 ResNet18 CPU/VTA 每 GOP 价格时，topology-diverse Top-20 只有 2 个 topology 被历史池覆盖，最佳覆盖项仅排在 169 个历史候选的第 81 位，regret 为 31.60%。主要原因是 ResNet18 atomic t4 约 18.14 GOP/s，高估了 YOLO 高分辨率 CPU 前缀的后端执行效率。

这一结果非常重要：硬件相同不等于所有 DNN 可以无条件共享一个逻辑 GOP/s。可移植 profile 必须至少考虑 shape、算子族或实际 lowering signature。

### 7.8.2 六组串行组件校准

本文从旧实验中读取固定 topology 的 6 组 serial component profile，只使用 stage `run_ms` 和 threads，不使用 pipeline FPS。校准区分普通卷积与 255-channel logits 两类 CPU slope，并校准 VTA 主干 slope。

校准后的 topology-diverse Top-20 中，13/20 个 topology 在 169 条历史记录中有测量，命中历史 Top-20 的 7/14 个唯一 topology。关键结果如下：

| 静态排名 | Topology 简述 | 历史最佳排名 | 历史 FPS | 证据性质 |
|---:|---|---:|---:|---|
| 1 | pool2/shared13 + dual-pre-head 双 island | 22 | 4.558 | topology match |
| 3 | pool2/small-pre18 + dual-pre-head | 38 | 4.407 | topology match |
| 4 | pool2 -> dual-pre-logits | 4 | 5.113 | 校准 topology |
| 6 | pool2 -> logits | 1 | 5.440 | topology match，历史池 oracle |

按已覆盖 topology 计算，`regret@1/3/5/6/20` 分别为 16.22%、16.22%、6.01%、0% 和 0%。排除用于组件校准的整个 `pool2 -> dual-pre-logits` topology 后，剩余 162 个候选、80 个 topology 的历史 oracle 仍为 `pool2 -> logits`，并仍被静态第 6 名覆盖。

### 7.8.3 证据边界

上述结果只能说明，少量目标网络组件 profile 后，同一个资源目标能把历史高性能 YOLO 划分拓扑放入较小 shortlist。当前推荐的 CPU threads 与旧记录没有 exact execution match，不能把 5.440 FPS 当作新配置实测吞吐。Top-20 中还有 4 个历史 compile failure 和 3 个未编译 topology。后续必须先做 native compile/reference，再按冻结顺序测量，才能形成前瞻证据。

因此，当前实验支持“模型结构可迁移、价格需要少量目标域校准”，不支持“ResNet18 profile 对 YOLOv3-tiny 零样本精确泛化”。

## 7.9 RQ5：固定残差与共享 DDR 物理补测

自然 Top-20 的原始周期 MAPE 为 23.523%，统一加 34 ms 后降为 12.859%。这一现象有诊断价值：
资源下界系统性遗漏了某些服务；但 `+34 ms` 来自完整候选结果，不能作为可迁移硬件参数。为此，
P7 分别补测 CPU stage 并发、empty/short/long runtime 和 CPU-VTA 共享 DDR matched control，且
Top-20 标签不参与参数拟合。

CPU 并发实验使用两个独立进程和 ready/start barrier，分别记录 isolated 与 concurrent wall time、
process CPU time 及可用 PMU。固定开销实验拆分 frame scheduling、set/get、host copy、submit 与
poll wait。共享 DDR 实验比较 CPU cache-resident/64 MiB streaming 与 VTA compute-heavy/
DMA-heavy 组合。参数只有在 slowdown 超过 5%、区间排除 1 且跨 boot 可重复时才准入公式。

P7 得到两点结论。第一，单 boot 中 CPU streaming 压力下 VTA slowdown 约为 1.052 倍和 1.066 倍，
说明轻度共享内存干扰存在；但证据未满足三个独立 boot 的准入规则，故没有写入正式 score。第二，
聚合 DDR demand 在当前候选上明显低于 CPU/VTA 关键服务，没有改变自然 Top-20；加入已辨识组件后，
冻结验证指标仍未优于 `+34 ms` 诊断基线，Top-20 Spearman 和 regret@5 也未改善。因此 P7D 拒绝
了“这些组件已解释固定残差”的假设，保留原资源下界做 ranking，不再把未归属残差命名为硬件常量。

该负结果限定了静态算法的能力：当前模型可用于压缩候选空间，不能提供可信的绝对 FPS。共享 DDR
竞争仍是潜在机制，但在现有一 VTA-island 高性能区域中不是已证明的主要排序因素。

## 7.10 RQ6：共享 slot 零拷贝实验

### 7.10.1 Baseline 与归因口径

实验区分以下路径：

| 代号 | 数据路径 | 回答的问题 |
|---|---|---|
| B0 | native pipeline + 最快正确普通 `get_output/set_input` copy | 当前框架物化基线 |
| B1 | TVM zero-copy API + 手工单 slot 串行 | 指针重绑定能否消除单边界物化 |
| B2 | B1 + `BoundarySlotManager` + 双 slot pipeline | 能否安全用于多帧流水并保留重叠 |

所有路径使用相同模型、stage、线程、VTA schedule 和输入。只把 Host/Executor 间 materialization
计作可消除字节，VTA 在 DDR 与 SRAM 之间的 LOAD/STORE 保持原有 owner。P8C 采用三个独立 boot，
每个 boot 使用 `B0/B1/B2/B2/B1/B0` 对称顺序；II 按完整 block 完成窗口计算。

### 7.10.2 三 boot 组件与吞吐结果

固定三阶段候选为：

```text
CPU[00..02] t4 -> VTA[03..17] -> CPU[18..20] t1
```

三个 boot 的 B0/B2 结果为：

| Boot | B0 II (ms) | B2 II (ms) | B0 FPS | B2 FPS | FPS 相对变化 |
|---:|---:|---:|---:|---:|---:|
| 1 | 90.712 | 91.392 | 11.024 | 10.942 | -0.74% |
| 2 | 92.013 | 91.743 | 10.868 | 10.900 | +0.29% |
| 3 | 90.453 | 91.714 | 11.055 | 10.903 | -1.37% |

组件指标如下：

| 指标 | B0 | B2 | 结果 |
|---|---:|---:|---:|
| framework materialization | 1806336 B/frame | 0 B/frame | 消除 100% |
| boundary API service 均值 | 1.446899 ms | 0.041830 ms | 减少 1.405069 ms，97.11% |
| VTA internal LOAD | 12025856 B/frame | 12025856 B/frame | 不变 |
| VTA internal STORE | 1229312 B/frame | 1229312 B/frame | 不变 |

边界 API 减少量的 boot-level 95% 区间为 `[1.368642,1.441496] ms`，排除 0，因此组件收益成立。
但 `B0-B2` 的 II 差值均值为 -0.556938 ms，95% 区间为 `[-2.477096,1.363220] ms`，包含 0，
不能声明稳态吞吐提升。自然 Top-20 的单 boot 外部扫描也得到相同趋势：20/20 功能通过，每帧消除
1.81--2.61 MB 物化，边界 API 平均减少 96.59%，但 FPS 增量均值为 -1.20%，只有 6/20 为正。

### 7.10.3 三 VTA-island 串行压力实验

为检验多次边界交接的累计时延，本文固定一个非生产最优的压力方案：

```text
CPU(0)-VTA(1..3)-CPU(4)-VTA(5..8)-CPU(9)-VTA(10..17)-CPU(18..20)
```

该方案含 7 个 stage、3 个 VTA island 和 6 条异构边。普通路径与 zero-copy 路径均逐 stage 串行，
唯一变量是边界是否物化。相对最快正确 `memcpy` 基线的单 boot 结果为：

| 指标 | 普通复制 L0 | 单 slot zero-copy L1 | 差值 |
|---|---:|---:|---:|
| 单帧中位时延 | 167.399 ms | 162.662 ms | 减少 4.737 ms，2.83% |
| 边界 API 服务 | 4.484 ms | 0.111 ms | 减少 4.373 ms，97.52% |
| framework materialization | 9031680 B/frame | 0 B/frame | 消除 100% |

为解释历史 200-case 数据，实验还复现了 runner 默认安全复制：边界 API 为 16.324 ms，zero-copy
为 0.111 ms；单帧时延由 178.855 ms 降至 163.093 ms，观察下降 8.81%。这个较大数字只用于说明
历史同一方案的 17.239 ms direct-copy 来源，评价本文优化相对合理普通实现的增益必须使用更严格
的 `memcpy` 基线和 2.83%。两组都只有一个 boot，不能写成稳定端到端提升。

### 7.10.4 Latency 与 throughput 的收益条件

串行单帧执行时，各边界位于同一关键路径：

```math
L_{serial}=\sum_i T_{stage,i}+\sum_e C_{boundary,e}+T_{runtime},
\qquad
\Delta L\approx\sum_e(C_{copy,e}-C_{zc,e}).
```

因此多个边界的 copy 消除会近似累加为单帧时延下降。稳态流水线则由最慢 stage 或最忙共享资源
决定：

```math
II=\max_r D_r,
\qquad
\Delta II\approx\max(0,\Delta C_{boundary}-slack_{boundary}).
```

当边界工位比瓶颈工位更快时，copy 处于可隐藏余量中；即使 materialization bytes 降为 0，也不
改变 II。只有边界或 DDR 已位于关键资源路径，或被消除时间超过该余量时，zero-copy 才提高 FPS。
这一规律解释了“串行 stress case 时延下降”和“自然 Top-20 吞吐无显著变化”同时成立。

## 7.11 Profile、构建与搜索成本

| 已完整计时的阶段 | 当前成本或规模 |
|---|---:|
| 最小局部 profile | 262.90 s |
| ResNet18 初次 Top-20 build/audit | 626.303 s |
| 缓存复审 | 0.895 s |
| 合法执行配置 | 972528 |
| Top-K board 上限 | 20 |
| 自然 Top-20 已测候选 | 20 |

P2 最小 profile 与初次构建合计约 14.82 min，但该数字不含 P5B 后补的 atomic CPU profile、板端候选运行和前期系统开发，不能作为全文完整实验成本。候选规模已由 97 万静态配置缩小到 Top-20，数量级上避免了逐一上板；但这并不能直接证明比所有随机或启发式方法更省板端评价。公平的总成本应写为：

```math
C_{method}(m)=\frac{C_{profile}}{m}+C_{static}+K C_{board},
```

其中 `m` 是同一硬件 profile 被多少个 DNN 复用。只有当跨 DNN 复用成立，且 Top-K regret 足够低时，复杂 profile 才真正具有成本优势。

## 7.12 实验结论

为避免把“实现完成”“局部指标改善”和“端到端性能提升”混为一谈，本文将两项主要创新及其证据
边界汇总如下。

| 创新点 | 主要对比基线 | 已完成证据 | 可支持的结论 | 尚不能支持的结论 |
|---|---|---|---|---|
| 资源感知 k-best 图划分 | 仅计算成本、完整枚举、历史人工方案 | 通信消融、DP/枚举一致性、自然 Top-20 上板、YOLO 历史回放 | 通信项改善历史池估计；DP 精确恢复静态目标 Top-20；shortlist 覆盖 ResNet18 高性能区域 | 已找到硬件全局最优；绝对 FPS 已准确预测；零样本跨网络泛化 |
| 固定映射共享 slot 零拷贝 | 普通 GraphExecutor `get_output/set_input` 物化路径 | 三 boot B0/B2 配对、Top-20 扫描、三-island串行压力方案 | 框架物化字节降为 0，边界 API 服务显著下降；边界密集串行方案时延下降 | 自然 Top-20 稳态吞吐显著提高；VTA 内部 DDR-SRAM 流量被消除 |

其中，第一项回答“哪些阶段放到 CPU/VTA、CPU stage 使用多少线程”，第二项回答“划分确定后，
相邻 Executor 如何在共享 DDR 上交接张量”。二者属于同一流水线运行时流程，但优化对象和评价
指标不同，论文不以零拷贝结果反向证明划分模型，也不以 shortlist 质量替代内存路径消融。

当前实验可以支持以下结论：

1. CPU-VTA direct boundary copy 显著影响当前 SoC 的周期预测，且 island 数增加时通信通常上升。
2. 单 VTA、per-stage threads、CPU core work 与边界所有权能够组成一致的静态搜索目标。
3. k-best 搜索在 ResNet18 固定实例上精确恢复了完整枚举 Top-20。
4. 自然 Top-20 全部通过板端 correctness，实测范围为 10.220--11.511 FPS，前 10 个包含池内最优；
   但内部 Spearman 仅 0.155，细排能力有限。
5. Atomic CPU profile 能改善小池排序解释，但 P7 组件补测未能替代候选级 `+34 ms` 诊断，故当前
   绝对 FPS 预测不成立。
6. YOLOv3-tiny 需要少量目标域组件校准；校准后历史最优 topology 进入静态前 6，但仍是回顾性证据。
7. 共享 slot 在三个 boot 中消除 100% 框架物化并减少 97.11% 边界 API 服务，但自然候选的吞吐
   区间不支持正提升；三-island串行压力方案观察到 2.83% 单帧时延下降。

当前实验不能支持以下结论：

1. 已经找到完整搜索空间的硬件全局最优；
2. 已经证明比随机搜索更少上板；
3. 已经建立准确的 CPU-VTA 并发 DDR slowdown 模型；
4. 已经完成 tile 与切图联合优化；
5. 已经完成 YOLOv3-tiny 前瞻板端泛化验证；
6. zero-copy 已消除 VTA 内部 DDR-SRAM LOAD/STORE，或对所有切图都提高 FPS；
7. 单 boot 的 2.83% 串行时延下降已经具备跨 boot 统计显著性。

# 第8章 总结与展望

## 8.1 工作总结

本文研究了嵌入式 CPU-FPGA 共享内存平台上的 DNN 流水线运行时优化。围绕“切在哪里”和“切开后
如何交接”两个问题，本文完成了从编译合法单元、组件 profile、资源感知 k-best 搜索、native
流水线，到 u-dma-buf 共享 slot 零拷贝的完整原型。

图划分部分没有把每个逻辑 stage 当作独立硬件，而是显式表示单物理 VTA 串行、CPU core-time、
方向化边界和共享 DDR demand；搜索部分利用单调资源下界从近百万配置中精确生成 Top-20；共享
内存部分没有把 TVM zero-copy API 本身作为创新，而是在 API 上增加编译 manifest 驱动的物理
slot、双视图、帧代次和所有权协议，使多个 Executor 能够安全复用同一中间缓冲区。

本文采用分层证据：算法最优性由完整枚举验证，候选质量由冻结 Top-20 上板验证，通信项由 200 条
历史 profile 做消融，zero-copy 由物化字节、边界 API、串行时延和流水线 II 分层验证。这一方法
既得到正结果，也保留了 P7 未能解释固定残差、P8 未提高自然候选稳态吞吐等负结果。

## 8.2 主要结论

1. 对当前 Zynq/VTA 平台，通信不是可忽略常数。显式加入 measured direct boundary copy 后，
   200-case 周期预测 MAE 相对下降 26.15%，秩相关由 0.770 提高到 0.931。
2. CPU stage 线程数必须作为决策变量。同一 topology 的首段 `t1--t4` 实测吞吐由 5.413 FPS 变化
   到 10.449 FPS，且不是简单线性加速。
3. k-best DP 在当前固定 tile 问题上精确恢复完整枚举 Top-20。自然 Top-20 的板端吞吐集中在
   10.220--11.511 FPS，说明 shortlist 能覆盖高性能区域；Spearman 仅 0.155，说明细排仍不足。
4. 跨 Executor 共享 slot 将固定候选的框架物化从 1806336 B/frame 降到 0，三 boot 边界 API
   service 减少 97.11%，证明冗余内存工作可被稳定消除。
5. 减少边界工作不等于必然提高吞吐。自然候选 B0/B2 的 II 差值区间包含 0；当 copy 位于流水线
   slack 内时，只降低内存工作而不改变瓶颈。串行三-island方案中边界均在关键路径，因而观察到
   4.737 ms、2.83% 的单帧时延下降。
6. 可迁移的是计算单元、资源 owner、搜索和 slot 协议，不是单一 GOP/s。YOLOv3-tiny 经 6 组目标
   组件 profile 后，历史最优 topology 进入 105696 个配置的静态第 6，但仍需前瞻验证。

## 8.3 局限性

第一，当前静态 score 是资源服务下界。P7 组件实验没有解释候选周期中约 34 ms 的诊断残差，也
没有通过绝对 MAPE 和细排 gate，因此不能用资源下界倒数报告预测 FPS。

第二，`D_DDR` 只表达每帧聚合服务需求。P7C 单 boot 观察到 streaming CPU 压力下 VTA 约
1.052--1.066 倍 slowdown，但尚未形成跨 boot、可准入的 contention 函数。

第三，V1 固定 VTA tile、queue depth、poll policy 和 HPC/coherent 数据通路。当前结论不能外推到
切图与 tile 联合搜索，也不能代表 HP/non-coherent 路径。

第四，ResNet18 历史 200-case 池并非从 97 万配置均匀采样。自然 Top-20 虽全部实测，但主要由
4 种 topology 的线程变体构成；不能据此声称已找到整个空间的硬件全局最优。

第五，YOLOv3-tiny 结果是 topology-level 回顾性连接，推荐线程没有 exact match，不能作为零样本
泛化或前瞻绝对性能结论。

第六，P8 三-island 串行时延只测量一个独立 boot；2.83% 只能作为 stress-case观察。自然 Top-20
零拷贝扫描也主要是单 boot，只有固定代表候选的组件收益具备三 boot 证据。

## 8.4 后续工作

1. 若论文需要把三-island的 2.83% 时延结果升格为正式性能结论，补两个独立 boot，并只使用优化
   `memcpy` 基线做配对区间；否则保持当前机制观察表述。
2. 用 topology-diverse 而非纯 score Top-20 分配上板预算，避免把大量预算消耗在同 topology 的
   近重复线程配置；同时保留预注册 thread controls 用于辨识并发影响。
3. 对 CPU-only、VTA-only 和 CPU+VTA matched control 补充跨 boot DDR contention；只有冻结修正
   改善 Spearman 和 regret 时才纳入正式模型。
4. 为 YOLOv3-tiny 实现 branch-aware k-best DP，并以 105696 配置完整枚举验证，再对未参与六组
   组件校准的 topology-diverse shortlist 做 native compile/reference 和前瞻测量。
5. 在划分排序稳定后，为每个 VTA segment 引入小型合法 tile 集，采用分层搜索避免一次展开切图与
   AutoTVM 全空间；只有它减少真实 DDR traffic 或改善 Top-K 才保留。
6. 研究 adapter direct-write：当 dtype/layout 不匹配时直接把转换结果写入 consumer slot，减少
   一次中间物化；该机制需与纯 zero-copy 分开消融。
7. 增加功耗、峰值共享内存占用和 profile 摊销成本，形成 latency、throughput、memory footprint
   与 energy 的多目标评价。

# 参考文献

[1] TARNAWSKI J, PHANISHAYEE A, DEVANUR N, et al. Efficient Algorithms for Device Placement of DNN Graph Operators[C]//Advances in Neural Information Processing Systems. 2020, 33. <https://proceedings.neurips.cc/paper_files/paper/2020/hash/b14680dec683e744ada1f2fe08614086-Abstract.html>.

[2] YU M, KWON Y, LEE J, et al. PartitionTuner: An operator scheduler for deep-learning compilers supporting multiple heterogeneous processing units[J]. ETRI Journal, 2023, 45(2): 318-328. DOI: 10.4218/etrij.2021-0446.

[3] MELONI P, CAPOTONDI A, DERIU G, et al. NEURAghe: Exploiting CPU-FPGA synergies for efficient and flexible CNN inference acceleration on Zynq SoCs[J]. ACM Transactions on Reconfigurable Technology and Systems, 2018, 11(3): 18:1-18:24. DOI: 10.1145/3284357.

[4] ZHONG G, DUBEY A, TAN C, et al. Synergy: A HW/SW framework for high throughput CNNs on embedded heterogeneous SoC[J]. ACM Transactions on Embedded Computing Systems, 2019, 18(2): 1-23. DOI: 10.1145/3301278.

[5] JIA F, ZHANG D, CAO T, et al. CoDL: Efficient CPU-GPU co-execution for deep learning inference on mobile devices[C]//Proceedings of the 20th Annual International Conference on Mobile Systems, Applications and Services. 2022. DOI: 10.1145/3498361.3538932.

[6] DAGLI I, BELVIRANLI M E. Shared memory-contention-aware concurrent DNN execution for diversely heterogeneous system-on-chips[C]//Proceedings of the 29th ACM SIGPLAN Annual Symposium on Principles and Practice of Parallel Programming. 2024: 243-256. DOI: 10.1145/3627535.3638502.

[7] SEO W, KIM S, HONG S. DNNPipe: Dynamic programming-based optimal DNN partitioning for pipelined inference on IoT networks[J]. Journal of Systems Architecture, 2025, 166: 103462. DOI: 10.1016/j.sysarc.2025.103462.

[8] RIOS-NAVARRO A, TAPIADOR-MORALES R, JIMENEZ-FERNANDEZ A, et al. Performance evaluation over HW/SW co-design SoC memory transfers for a CNN accelerator[C]//2018 IEEE 18th International Conference on Nanotechnology. 2018. DOI: 10.1109/NANO.2018.8626313.

[9] HARIS J, GIBSON P, CANO J, et al. SECDA: Efficient hardware/software co-design of FPGA-based DNN accelerators for edge inference[C]//2021 IEEE 33rd International Symposium on Computer Architecture and High Performance Computing. 2021: 33-43. DOI: 10.1109/SBAC-PAD53543.2021.00015.

[10] MOREAU T, CHEN T, VEGA L, et al. A hardware-software blueprint for flexible deep learning specialization[J]. IEEE Micro, 2019. DOI: 10.1109/MM.2019.2928962.

[11] CHEN T, MOREAU T, JIANG Z, et al. TVM: An automated end-to-end optimizing compiler for deep learning[C]//13th USENIX Symposium on Operating Systems Design and Implementation. 2018: 578-594.

[12] HE K, ZHANG X, REN S, et al. Deep residual learning for image recognition[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016: 770-778.

[13] REDMON J, FARHADI A. YOLOv3: An incremental improvement[EB/OL]. 2018. <https://arxiv.org/abs/1804.02767>.

[14] WILLIAMS S, WATERMAN A, PATTERSON D. Roofline: An insightful visual performance model for multicore architectures[J]. Communications of the ACM, 2009, 52(4): 65-76.

[15] ALEXANDROV A, IONESCU M F, SCHAEUSER K E, et al. LogGP: Incorporating long messages into the LogP model[C]//Proceedings of the Seventh Annual ACM Symposium on Parallel Algorithms and Architectures. 1995: 95-105.

[16] APACHE TVM. Compute graph pipeline with Pipeline Executor[R/OL]. TVM RFC 0014, 2021. <https://apache.googlesource.com/tvm-rfcs/+/6990e1363c96945b6c9d19dc2d331306d2be2a17/rfcs/0014-pipeline-executor.md>.

[17] GENER S, UKARANDE A, MURTHY S M S, et al. RIMMS: Runtime Integrated Memory Management System for Heterogeneous Computing[J]. ACM Transactions on Embedded Computing Systems, 2025, 24(5s): 91:1-91:24. DOI: 10.1145/3760257.

[18] APACHE TVM. Graph Executor zero-copy input/output interface[CP/OL]. <https://github.com/apache/tvm/blob/main/src/runtime/graph_executor/graph_executor.cc>.

# 附录A 关键实验产物

| 产物 | 作用 |
|---|---|
| `v1_hardware_fingerprint.json` | 固定 bitstream、runtime、compiler 与 CPU policy |
| `v1_unit_and_boundary_schema.json` | 21 个单元、cut contract 与候选语法 |
| `v1_profile_manifest.json` | 去重 CPU/VTA segment 与 boundary signature |
| `v1_local_cost_table.json` | P2 最小局部成本表 |
| `v1_p5b_iteration2_ranked_candidates.json` | 使用 atomic CPU 成本后的最终 ResNet18 静态 Top-20 |
| `v1_p5b_iteration2_review.json` | 最终 DP 与 972528 个执行配置完整枚举的一致性及 5-candidate 盲回放 |
| `v1_p5a_compile_audit.json` | 20 个候选 native build/lowering 审计 |
| `v1_p5b_iteration1_review.json` | 线程曲线与五阶段候选失败分析 |
| `v1_p7_top20_board_20260904/top20_board_summary.json` | 自然 Top-20 correctness 与板端吞吐 |
| `v1_historical200_affinity_evaluation.json` | 历史 200-case 公式回放 |
| `yolov3_tiny_static_v1_evaluation.json` | YOLOv3-tiny 静态与历史 topology 审计 |
| `v1_p8c_cross_boot_summary.json` | 三 boot B0/B2 边界服务与 II 配对统计 |
| `v1_p8_top20_zero_copy/v1_p8_top20_summary.json` | 自然 Top-20 零拷贝外部扫描 |
| `v1_p8_latency_island_scaling/three_island_sessions/` | 三-island优化复制基线串行时延实验 |

# 附录B 论文结论用语检查表

| 可以使用 | 不应使用 |
|---|---|
| “在当前 200 条 ResNet18 记录上，边界通信使 MAE 相对下降 26.15%” | “通信模型对所有 SoC 都提高 26.15%” |
| “DP 在当前静态成本函数上与 972528 配置枚举 Top-20 一致” | “DP 找到了硬件全局最优” |
| “自然 Top-20 实测范围为 10.220--11.511 FPS” | “原始资源分数 14.123 就是预测或实测 FPS” |
| “YOLO 历史最优 topology 被排到第 6” | “YOLO 新配置实测达到 5.440 FPS” |
| “少量目标域组件 profile 后具有拓扑级泛化迹象” | “ResNet profile 对 YOLO 零样本泛化成功” |
| “共享 DDR 下界已进入目标函数” | “共享 DDR contention 已被精确建模” |
| “三 boot 边界 API 服务减少 97.11%” | “zero-copy 稳态 FPS 提升 4.42%” |
| “三-island单 boot观察到时延下降 2.83%” | “zero-copy 已稳定提升端到端时延 2.83%” |

# 致谢

【待填写。建议包含导师、实验室同学、硬件平台支持者和家人，不在技术正文中展开。】
