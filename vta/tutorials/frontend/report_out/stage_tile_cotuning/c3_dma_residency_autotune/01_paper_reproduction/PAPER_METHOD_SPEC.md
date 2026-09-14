# Cheng 2026 完整方法规格与本地复现边界

对象：Ruohan Cheng、Yanshuo Gao、Chenglong Zeng、Yinghai Zhao、Kuizhi Mei，
*Design of Data Access and Schedule Optimization for VTA Compiled Instruction Streams*，FGCS 176 (2026) 108165。

## 1. 全文身份

- 本地文件：`/home/orange/code/tvm/Design of data access and schedule optimization for VTA compiled.pdf`
- SHA-256：`9fdbe490db46d319fb992ff016f99c0f7bd64718f079f84749ff21d3c8839a02`
- 页数：9；PDF 元数据创建时间：2025-11-28；正式 DOI：`10.1016/j.future.2025.108165`
- 论文记录：received 2025-02-18，revised 2025-09-16，accepted 2025-09-21，online 2025-09-24。

此前“只能依据摘要重实现”的限制已经解除，可以按正文、Algorithm 1/2、Fig. 1--3、Table 1/2 核对
方法。但论文仍未提供源码、TVM commit 和完整自动调优器，因此不能称 code-level exact reproduction。

## 2. 论文真正提出的四种方案

| 方案 | schedule | weight loading | virtual thread |
|---|---|---|---|
| Scheme 1 | TVM 原始 output-prioritized | 原始覆盖式加载 | output-channel 方向 |
| Scheme 2 | input-prioritized | 原始加载 | 关闭 `oc_nthread`，启用 `h_nthread` |
| Scheme 3 | 原始 output-prioritized | on-chip weight-memory reuse | 保持原方案语义 |
| Scheme 4 | input-prioritized | on-chip weight-memory reuse，即 minimum-data-access | height 方向 virtual thread |

### 2.1 Input-prioritized schedule

Algorithm 1 的原始外层顺序是 output height → output channel → output width；Algorithm 2 改成 output
height → output width → output channel，使 input block 不再随 output-channel 外层重复加载。示例还把
`tile_h × tile_w` 从 8×16 改为 4×32，关闭 `oc_nthread` 并启用 `h_nthread`。示例 input access 从
4,527,360 B 降为 1,180,608 B，但 weight access 仍为 2,359,296 B；论文明确指出只做该变换可能因
冗余 weight access 使性能下降。

### 2.2 On-chip weight-memory reuse

适用前提是单层卷积全部权重能够放入 weight SRAM。权重分块按不同地址顺序加载且不再覆盖，后续 GEMM
根据 UOP 地址索引读取已驻留权重，而不是再次生成 LOAD WGT。论文示例中 weight access 从
2,359,296 B 降为 73,728 B；LOAD UOP 类指令由 12 增到 18，因为新权重地址需要新的 GEMM UOP。

正文 Fig. 3 给出的实现点包括：

1. `StorageFlatten` 处理 `BufferRealizeNode` 时初始化目标地址 offset；
2. `InjectVirtualThread` 不再让线程数改变目标 weight load 地址，并用 loop-variable 条件只在首次
   使用时产生 LOAD WGT；
3. `SchedulePostprocToPrimFunc` 对 `ProducerStoreNode` 生成条件节点；
4. `InjectCopyIntrin` 解释这些控制节点并在优化/原始加载间选择；
5. VTA runtime `PushGEMMOp` 在产生新 weight load/UOP 时生成相应 LOAD UOP，并处理重复 UOP。

因此论文的 weight reuse 不只是调整一个 schedule 原语，而是 TVM TIR pass、地址分配、copy intrinsic
和 VTA runtime 的联合修改。本地 `weight_resident_barrier` 目前是实现相同高层目标的另一种机制，
不能直接称为论文实现的逐行复现。

### 2.3 Minimum-data-access

Scheme 4 同时采用 input-prioritized 和 weight reuse。Table 1 的示例数据为：

| 指标 | Scheme 1 | Scheme 2 | Scheme 3 | Scheme 4 |
|---|---:|---:|---:|---:|
| input LOAD B | 4,527,360 | 1,180,608 | 4,527,360 | 1,180,608 |
| weight LOAD B | 2,359,296 | 2,359,296 | 73,728 | 147,456 |
| ACC LOAD B | 16,384 | 16,384 | 16,384 | 16,384 |
| UOP LOAD B | 232 | 448 | 808 | 832 |

Scheme 4 的 weight bytes 是理论最小的两倍，因为 height virtual thread 为隐藏延迟使全部权重多加载一次。
这证明论文并非简单追求单张量绝对最小，而是在复用与 virtual-thread 性能之间保留实现权衡。

## 3. 资源适用条件与回退

- output-prioritized 示例要求 `LOG_ACC_BUFF_SIZE > 15`；
- input-prioritized 无 virtual thread 时要求 `LOG_ACC_BUFF_SIZE > 16`；
- input-prioritized + height virtual thread 时要求 `LOG_ACC_BUFF_SIZE > 17`；
- weight reuse 要求 weight SRAM 大于该层全部权重；
- 论文按卷积参数和 stride 计算适用范围；资源不足时在 TVM compilation layer 自动禁用定制策略并
  回退 naive TVM flow；
- 默认 `15_16_19_18 (uop_inp_wgt_acc)` 配置下，YOLOv3 的 22 种卷积尺寸中 12 种满足 weight reuse。

论文已经包含硬件资源感知和 fallback，因此“根据 SRAM 条件决定是否启用驻留”不是本文的新意。

## 4. 实验结果

- 平台：Xilinx ZCU104，VTA naive HLS，32×32 array，200 MHz，默认 `15_16_19_18` buffer；
- 输入：主要为单帧 256×256；指标是单层/模型 pure-instruction inference time；
- 网络：YOLOv3、YOLOv5m、ResNet50。

Table 2 的 9 个 YOLOv3 卷积说明 Scheme 4 并非总是最快：

- c1/c4/c5/c7：Scheme 4 最快；
- c2：Scheme 3/4 同为 1.34 ms；
- c3：Scheme 4 最快但只略优于 Scheme 3；
- c6：Scheme 3 为 2.43 ms，优于 Scheme 4 的 2.54 ms；Scheme 2 恶化到 4.50 ms；
- c8：Scheme 3/4 同为 0.64 ms；
- c9：Scheme 3 为 2.08 ms，优于 Scheme 4 的 2.62 ms；Scheme 2 恶化到 6.90 ms。

模型 pure-instruction time：YOLOv3 75 个卷积从 195.632 降到 175.321 ms（约 10%）；YOLOv5m
从 85.21 降到 65.14 ms（约 23%）；ResNet50 从 102.22 降到 76.19 ms（约 25%）。

## 5. 全文确认的搜索方法边界

全文没有给出以下内容：

- AutoTVM 候选空间大小、ConfigEntity 生成或跨 tile 搜索算法；
- Random/XGB/SA 等 tuner 主实验；
- trial、sample、time-to-target、regret、搜索停止条件或搜索墙钟；
- 有效性/性能预测模型；
- 相同搜索预算的策略比较；
- 多 seed 正确性、错误候选分类或错误对 cost model 的处理。

论文中的“tuning strategy for schedule parameters”具体落在 input-prioritized 的块参数/virtual-thread
配置、资源适用条件、四方案硬件性能比较以及按层选择最优方案。它是 schedule/data-access 设计与
应用策略，不是本文所比较的完整 AutoTVM search process。

## 6. 本文可以和不可以主张什么

不能作为本文原创：

- input-prioritized loop reorder；
- 全层权重驻留、不覆盖地址和去除重复 LOAD WGT；
- 四种 original/input/weight/combined 方案；
- 用 data-access bytes 分析冗余；
- SRAM 条件判断和资源不足回退；
- 修改 TIR passes 与 runtime 支撑 weight reuse。

仍可形成本文方法贡献：

- 将驻留方案与 AutoTVM tile ConfigEntity 组成联合候选空间；
- 在板端 latency 未知时，用最终 lowered 程序的全张量 DMA 对跨 tile 候选排序；
- 以完整 FPGA-correct pool、Random/stock-XGB/方案内 minimum-access、20 seeds 和固定 budget 评价
  达到 pool oracle 所需试验，而不是只比较四种方案的最终 latency；
- 将 compiler/FSim/FPGA/measurement 的不同成本和搜索期共享内存流量纳入 time-to-target；
- 对真实 FPGA 数值错误 fail closed，并把最终候选绑定到 u-dma-buf 物理布局和命令容量合同。

这里前四项才可能承担“搜索方法”创新；最后一项是系统可靠性支撑。当前 frozen-pool 离线回放已经是
前瞻证据，但若要形成更强算法贡献，还需在线、成本感知的多保真搜索，而不是把全部候选预资格后再排序。
