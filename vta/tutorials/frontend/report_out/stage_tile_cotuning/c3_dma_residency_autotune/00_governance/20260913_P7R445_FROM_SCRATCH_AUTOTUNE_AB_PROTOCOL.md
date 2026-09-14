# P7R445：原生 AutoTVM 与本文方法从零调优—整图部署 A/B 协议

状态：`SYSTEM_FLOW_THREE_BY_THREE_COMPLETE; SAME_SPACE_ONLINE_ABLATION_PENDING`

## 1. 为什么必须补这个实验

现有实验能够证明本文方法在冻结候选池内减少候选构建、FPGA 派发和逻辑 DMA，并能将入选配置
部署到完整图。但它还不能直接回答论文答辩最自然的问题：

> 对同一个从未加载历史调优记录的卷积任务，从开始调优到得到可部署完整图，原生 AutoTVM 花费
> 多少时间和硬件试验？本文方法花费多少？二者最终完整网络的实测 latency 和正确性怎样？

当前不能代替该实验的已有数字包括：

- P7R131/P7R144/P7R154 等 `stock-XGB` 是冻结完整池上的顺序 replay，不是重新运行 AutoTVM；
- P7R435 的 212.727 s 是本文完整图在线执行段，不含候选生成、static lowering 和 FSim 资格；
- Y10 穷举的 143.665 s candidate-build 与 436.283 s invocation host wall 是已测组件之和，不是原生
  AutoTVM 从零调优的完整进程墙钟；
- TopHub 是预调优记录，不能充当“从零开始”的成本基线。

因此在该 A/B 完成前，论文只能主张候选池内、阶段级或本文在线执行段的成本下降，不能主张
“相对从头 AutoTVM，端到端调优时间降低 X%”。

## 2. 共同起点和终点

### 共同起点 T0

- FPGA bitstream 已加载、默认 RPC 健康；
- 编译器、交叉工具链、模型 cfg/weights 已安装；
- 只给出同一个目标卷积 workload 和同一个完整 YOLO 图；
- 不加载 TopHub、历史 AutoTVM log、现有候选性能标签或完整池 oracle；
- 使用新的输出目录和空调优日志。

模型下载、工具链安装和 bitstream 综合属于一次性部署准备，单列但不计入每次调优墙钟。任务提取、
ConfigSpace 构造及此后全部工作必须计时。

### 共同终点 T1

搜索器选出的配置已经：

1. 构建进同一个完整 YOLO 图；
2. 通过三个确定性输入、全部图输出的 FPGA 正确性；
3. 完成七轮 stock/AutoTVM-selected/ours-selected 平衡交错计时；
4. 保存选中配置、TIR、图、参数、DSO、bitstream/runtime 和日志哈希。

主成本为 `T1-T0`，而不是只计算 tuner 内部的 `tuner.tune()`。

## 3. 两类对照必须同时报告

### A. 实际使用流程比较（主结果）

`Stock AutoTVM-XGB`：

- 原始 `conv2d_packed.vta` 模板和其完整 1280 点 ConfigSpace；
- 空 history，从 XGBoost 的随机初始化开始；
- 每个被提出候选真实编译、上传、数值检查并在 FPGA 上测量孤立算子；
- 固定最大 300 trials，同时保留 60/120/300/600 s 墙钟检查点；
- 到达固定上限后选择其最好正确记录，再构建并测量完整图。

`本文方法`：

- 同一 1280 点 tile 域，扩展 original/input-stationary/weight-resident-barrier 三种访问语义；
- 从解析容量、真实 lowering、FSim、cheap operator-TIR bytes/calls/submission 前沿开始计时；
- 仅对晋级点按需构建完整图，真实 FPGA 三 seed correctness 后才获得 latency；
- 采用已经冻结的存活率分支和 invalid-dominator peeling，不得根据本次标签调参；
- 同样保留 60/120/300/600 s 检查点，并在固定 600 s 或预注册动作上限结束。

这个比较回答实际用户使用原生 AutoTVM 和本文系统各自会得到什么。因为本文增加了驻留模式，两者
搜索空间不同，结果必须称为“系统流程比较”，不能把差异全部归因于搜索算法。

### B. 相同空间搜索器消融（归因结果）

为了分离“新增驻留机制”和“搜索过程优化”，再做一个相同候选宇宙的比较：

- 两边都只使用 original 模式，或都使用同一预注册的 `tile × mode` 有限候选集合；
- 一边由 stock-XGB 逐步提出候选，另一边使用本文 cheap-proxy/multi-fidelity 顺序；
- 两边调用同一 builder、同一三 seed correctness gate 和同一完整图计时器；
- 固定相同的 gross candidate、FPGA invocation 和墙钟预算，分别给出三种口径，而不是只挑有利口径。

只有这个消融才能支持“搜索策略本身降低搜索成本”；主结果则支持“完整系统更快得到可部署方案”。

## 4. 冻结时定义的最终结果表

以下 `待测` 单元保留为预注册痕迹；实际三对三结果见第 7 节与 P7R459 聚合产物。

### 表 A：从零调优成本

| 指标 | Stock AutoTVM-XGB | 本文方法 | 变化 |
|---|---:|---:|---:|
| ConfigSpace 大小 | 待测 | 待测 | — |
| gross proposals | 待测 | 待测 | 待测 |
| 编译尝试/成功 | 待测 | 待测 | 待测 |
| FSim 次数 | 0 | 待测 | 额外安全成本 |
| FPGA correctness invocation | 待测 | 待测 | 待测 |
| FPGA timing invocation | 待测 | 待测 | 待测 |
| 搜索期逻辑 LOAD/STORE bytes | 待测 | 待测 | 待测 |
| 搜索期 DMA calls | 待测 | 待测 | 待测 |
| 首个正确配置时间 | 待测 | 待测 | 待测 |
| 首次达到最终较优者 +2% 的时间 | 待测 | 待测 | 待测 |
| tuner 结束时间 | 待测 | 待测 | 待测 |
| 从 T0 到完整图验证结束 | 待测 | 待测 | 核心结果 |

### 表 B：最终完整图结果

| 部署图 | 目标层模式/tile | 三 seed 正确性 | 全图 latency 中位数 | 相对 stock 图 | 7 轮胜场 |
|---|---|---:|---:|---:|---:|
| 未调优 stock graph | 官方当前配置 | 待测 | 待测 | 0% | — |
| AutoTVM-selected | original + XGB 选中 tile | 待测 | 待测 | 待测 | 待测 |
| Ours-selected | tile + residency | 待测 | 待测 | 待测 | 待测 |

主质量指标是完整图 latency。孤立算子 latency 只解释 AutoTVM 为什么选择该点，不能代替整图结论。

### 图：anytime 曲线

横轴分别画：

1. 从 T0 开始的墙钟；
2. gross candidate 数；
3. FPGA invocation 数。

纵轴画“截至当前得到的最好、FPGA-correct 配置的最终完整图 latency”。对于尚未完成整图构建的
候选不能偷偷填入孤立算子 latency。最终给出 `time-to-best+2%` 和曲线下面积。

## 5. 随机性与执行顺序

- Stock-XGB 至少运行三个独立随机种子；本文确定性策略也重复三个 clean-start session，以区分算法
  随机性和板端漂移；
- 运行顺序采用 `XGB—Ours—Ours—XGB—XGB—Ours` 或等价平衡顺序；
- 每个 session 开始前重载同一 bitstream、启动新的默认 tmpfs RPC；
- 发生重启、RPC 污染或存储错误时，该 session 整体作废，不拼接墙钟；
- 所有失败候选仍计入 gross proposal、编译成本和已经发生的 FPGA 调用。

## 6. 可以和不可以得到的结论

若本文方法以更少 `T1-T0` 时间达到不差于 AutoTVM-selected 2% 的完整图 latency，可以主张：

> 在该固定 VTA FPGA 和目标网络上，共享内存语义驱动的多保真方法，以更低的从零调优代价获得了
> 与原生 AutoTVM 相当或更好的可部署完整图。

即使 AutoTVM 最终图更快，只要本文明显更早进入其 2% 等价带，也可以形成“time-to-quality”贡献。
如果本文更慢，则必须分解原因是 FSim、完整图构建、三 seed gate 还是候选排序，并将其报告为安全性
成本，不能继续使用冻结池节省数据替代端到端结论。

单一 workload 的正结果只能称 case study。为了较强的外部有效性，最终应至少覆盖一个 YOLO 层和
一个 ResNet50 层；其中至少一个应是协议冻结后才产生任何 A/B 标签的新 workload。

## 7. P7R452--P7R460 实测结果

已按 `XGB—Ours—Ours—XGB—XGB—Ours` 顺序完成三对三 clean-start 系统流程比较。两类运行都在
重载同一 bitstream、启动新 RPC 后记录 T0；T1 均要求选中程序进入同一 YOLOv3-tiny-320 完整图，
通过三个确定性输入的全部八个输出等价，并完成七轮 selected/stock 配对计时。目标层搜索不加载
TopHub 或历史 AutoTVM log；完整图其余层使用两边相同的正常部署上下文。

| 指标（三轮中位数；括号为 min--max） | AutoTVM-XGB | 本文方法 | 变化 |
|---|---:|---:|---:|
| T0→T1 | 696.427 s（695.965--699.170） | 244.844 s（244.835--245.926） | -64.84%，2.84× |
| gross proposal / 生成身份 | 173（157--173） | 32 | 口径不同，不能直接作百分比 |
| 候选程序板端派发 | 115 个孤立算子 | 3 个完整图 | -97.39% |
| FSim | 0 | 12 候选 × 3 seed = 36 | 本文额外安全成本 |
| 最早得到 XGB 最终图 +2% 内的可部署完整图 | 696.427 s | 120.279 s（119.931--121.732） | -82.73%，5.79× |
| 最终完整图 latency | 1054.773 ms | 1057.774 ms | 本文慢 0.285%，在 2% 带内 |
| selected/同轮 stock 配对比 | 0.987875 | 0.990676 | 相差 0.280 个百分点 |
| 正确性与计时 | 每轮 3/3 输入、8 输出，7/7 胜 | 每轮 3/3 输入、8 输出，7/7 胜 | 全部通过 |

更细的三轮中位成本分解如下。XGB 的板端调用数以 112 个成功孤立候选为中位：每个候选三次正确性、
三次计时和一次 profiler；另有中位 3 个设备失败尝试。本文的 18/42 是三个候选各自与 stock 配对的
完整图调用数，因此不能把两个 invocation 数当成同一种粒度。

| 成本项 | AutoTVM-XGB | 本文方法 |
|---|---:|---:|
| task/ConfigSpace 构造 | 0.008 s | 已含候选生成 |
| 空历史 tuner / 候选生成 | 604.036 s | 2.562 s |
| 目标候选编译 | 173 次尝试，115 个产生板端产物，112 个正确（中位） | 24 次算子 lowering、12 次成功；3 次完整图候选 build、3 次成功 |
| FSim | 0 | 36 次（12 候选 × 3 seed） |
| 资格 + 前沿 | 无独立阶段 | 26.249 + 1.566 s |
| 搜索期 FPGA 调用 | 336 correctness + 336 timing + 112 profile，另约 3 次失败尝试 | 18 correctness + 42 timing 个完整图调用（含 paired stock） |
| 搜索后完整图阶段 | 模型/stock/selected build 与验证合计约 92.4 s | 已含在 214.453 s 在线完整图阶段 |
| 可审计 driver kernel calls | 下界 1,192 | 1,326 |
| 可审计逻辑 DMA calls | 下界 4,814,886 | 13,875,927 |
| 可审计逻辑 DMA bytes | 下界 5.052 GB | 14.486 GB |

XGB 三个随机种子均选中 original `config 543`。本文三轮都从同一 1280 点 tile 域确定性生成 32 个
四模式身份，正式资格化 24 个三模式身份；12 个通过 lowering/FSim，三个代理前沿完整图均正确，
最终均选中相同 `input_stationary` 身份。P7R460 的完整聚合报告、逐轮 CSV、runner 口径说明和哈希
清单位于 `../07_grouped_holdout/20260913_p7r460_y10_from_scratch_autotune_ab_aggregate_with_scope_run01/`。

必须同时报告负向成本：本文对三个候选执行完整图并反复配对 stock，每轮逻辑 DMA 约 14.486 GB；
XGB 对成功孤立候选的七次调用及最终图可审计下界约 5.052 GB，本文高 186.7%。两者执行粒度不同，
该数值只说明完整图安全门的成本，而且均不是物理 AXI 流量，不能写成本文降低了所有搜索资源。

这里“原生 AutoTVM-XGB”精确指官方空历史 `XGBTuner`、原始 `conv2d_packed.vta` 模板与 1280 点
ConfigSpace；板端 runner 为本实验增加的三 seed 正确性和逐候选 clean-start 安全适配器。P7R447
已证明共享 RPC 遇到 VTA runtime fatal 配置后会污染后续测量而无法完成整轮，因此正式三次运行均
在每个真实板端候选前重载 bitstream 并启动新 RPC，相关时间全部计入 XGB。本文三个完整图候选也
采用相同 clean-start 原则。故 64.84% 是相对**安全、可完成的部署调优流程**，不是未经保护的 TVM
默认 runner 微基准，也不能假设别的平台具有相同重启代价。

协议有两项未完全闭合。第一，实际执行采用每个方法各自与 stock 的七轮配对，而不是同一进程内
stock/XGB/ours 三方交错；三轮同轮 stock 中位只相差约 0.01%，正式比较优先使用 selected/stock
配对比。第二，相同空间搜索器消融仍只有既有冻结池/replay 与独立控制证据，尚未完成新的在线
T0→T1 A/B。因此当前可以主张 Y10 上的**系统流程**以 64.84% 更少端到端墙钟得到与空历史 XGB
相差 0.285% 的可部署完整图，不能把全部差异单独归因于搜索算法，也不能外推为跨网络结论。
