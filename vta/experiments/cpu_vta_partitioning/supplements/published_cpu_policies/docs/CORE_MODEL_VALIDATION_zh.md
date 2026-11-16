# 逐核负载与调用线程放置：小范围验证

入口`run_core_model.py`，批次`resnet18_core_model_validation_01`。
固定切图为CPU0–2→VTA3–4→CPU5–7→VTA8–19→CPU20；CPU20单线程核心0，
VTA主机及两个控制线程共享四核，CPU阶段门控关闭，普通复制、队列深度2、输入0。

## 两项独立改变

核心集合`mask`仍按升序表示允许核心；`worker_core_map`保留逻辑线程顺序。
逻辑线程0是调用线程。例如mask为`[0,1,2]`而map为`[2,0,1]`时，调用线程位于核心2。
其余helper按剩余核心升序分配，未搜索全部worker排列。
新执行器通过`--stageN-cpu-worker-map`传递有序映射，缺省行为与原执行器相同。

11项共同隔离成本包括两个重CPU子图各1–4线程、CPU20单线程及两个VTA阶段。
6项多线程CPU成本采8个run范围的线程计数。各逻辑线程核时按原始增量总和除以8取得，
总核时取向量之和，权重为向量归一化；不用每帧百分比的平均值。
服务时间为set＋排除前后探针工作的实际run＋get，交接核时来自隔离set/get测量。
三个单线程CPU签名使用普通隔离CPU时钟、权重为1；VTA保持普通隔离口径。

均分与逐线程评分共用上述同一成本。均分将总计算核时按线程数分配，逐线程版本按
实测逻辑线程分布分配。均按有序map映射物理核心，交接工作归caller_core。
同一物理VTA累计、DDR及15个非空核心子集约束保留原公式。控制工作未单独建模，不能写成零。
其他物理核心及映射沿用明确的同构核置换近似；不把流水数据当作profile输入。

## 完整的小域与测量池

固定调用线程为核组最小核心时有225项；允许选择调用线程时有1024项。
两种评分与两种候选域组成四组对照，独立标量实现检查全部2048个分数和四份完整排序。
池由四个静态第一名、4/4线程caller为0/0、0/3、3/0、3/3的四格、
2/2和3/1不重叠计算核组，以及两项最大正负预测分歧组成，去重后最多12项。
并列按完整配置的canonical顺序，不给历史结果保送名额。

11项组件各资格1帧、预热2帧并计入8帧，共22个任务、121次推理。
全部成本与排名、池、任务顺序冻结后，才测原程序基准及其余流水。
每项流水资格1帧，正式预热8帧、保留16个完成时间戳、尾部4帧；正式线程采样关闭。
额外旧程序基准用同一28帧性能窗口。池大小N时，共23＋2N项任务、149＋29N次推理，
上限47项、497次。只有一次启动、一张图片、一轮，无锚点、漂移拒收、自动加测或自动重启。

## 命令与恢复

```bash
python run_core_model.py prepare
python run_core_model.py validate --bundle artifacts/bundles/<bundle-id>
python run_core_model.py board run --bundle artifacts/bundles/<bundle-id> --execute-board
python run_core_model.py board resume --run-dir runs/resnet18_core_model_validation_01 --execute-board
python run_core_model.py status --run-dir runs/resnet18_core_model_validation_01
python run_core_model.py rank --run-dir runs/resnet18_core_model_validation_01
python run_core_model.py analyze --run-dir runs/resnet18_core_model_validation_01
```

准备只编译独立执行器，不连接板子、不重新编译阶段或运行库。
断线恢复按配置、映射、执行器、原始凭据和boot身份接续，已完成任务不重测。
缺失必要线程核时则停止排名并保留原始记录，不静默退回均分。
输出错误、哈希错误、错误映射及缺／重复帧仍作为无效执行处理。

## 产物与解释

交付四组静态第一名实测、同池两种评分的误差和排序、调用线程四格、不重叠核组对照、
全部组件核时向量、身份与原始证据索引。读取包络和Linux调度记账限制仍保留。
一启动短窗口不用于宣称显著性或等效，不把小域结果说成完整切图DP结论。
本批不自动启动全域profile或新增搜索机制；后续完整DP域和成本依据本轮证据另行冻结。
