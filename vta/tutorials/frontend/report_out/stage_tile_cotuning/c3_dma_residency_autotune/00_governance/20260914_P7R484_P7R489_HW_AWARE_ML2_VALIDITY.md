# P7R484--P7R489：HW-Aware 初始化与 ML²Tuner Model V 合法性实验

状态：`COMPLETE_LOCAL_VALIDITY_ONLY; NO_BOARD_OR_PERFORMANCE_LABELS`

## 1. 实验边界

本组实验只研究 original 模式完整 ConfigSpace 的 **真实 VTA lowering 合法性**。它不接触开发板，
不读取算子或整图 latency，也不把 FSim/FPGA 数值正确性冒充成编译合法性。三个真实空间的规模不是
预案中统一的 1280，而是 H1/H2/H3 分别为 2304/1600/480。

P7R484--P7R486 对每个 ConfigEntity 实际执行 instantiate/lower，逐项记录编译墙钟和失败类型。
P7R488 在完整标签上做 20 seeds 的 HW-Aware 方法一致复现；P7R489 用另外两个几何训练 Model V，
在完全留出的目标几何上评价。所有结果来自只读完整扫描，未重复消耗板端资源。

P7R487 是一次失败且不可拼接的分析 session：并行邻域扩展时，批次内已处理节点仍可能残留在
frontier，导致 H3 个别 seed 只遍历 479/480 点。该目录保留 `invalid_session.json`；修复后新增奇数
空间耗尽测试，P7R488 从头重跑，未拼接 P7R487 的部分结果。

## 2. 完整空间真实 lowering

| workload | compiler calls | valid | invalid | invalid ratio | 完整扫描墙钟 |
|---|---:|---:|---:|---:|---:|
| R18-H1 | 2304 | 290 | 2014 | 87.41% | 54.503 s |
| R18-H2 | 1600 | 147 | 1453 | 90.81% | 35.559 s |
| R18-H3 | 480 | 132 | 348 | 72.50% | 8.781 s |
| 合计 | 4384 | 569 | 3815 | 87.02% | 98.842 s |

主要失败不是一条手写容量公式能够覆盖：H1/H2 的 padding innermost 失败分别为 1226/788，另有
allocation、二维 DMA pattern 和其他 lowering 错误。因此 HW-Aware/Model V 在这里解决的是昂贵
真实 lowering 之前的优先级或粗筛问题，而不是替代编译器最终裁决。

## 3. HW-Aware 四级消融

协议固定 `E0=50`、最多 25 valid + 25 invalid、`presampling=min(1000, |S|)`，种子为
57001--57020。离散邻域定义为“只改变一个 knob，且只移动到该 knob 观察域中的相邻取值”，不能用
数值差 1 代替，因为 tile 因子域并不等距。

| workload | Random E0 valid yield（中位/IQR） | 邻域 E0 valid yield（中位/IQR） | 找到前25 valid的 calls：Random→邻域 | 对应墙钟：Random→邻域 |
|---|---:|---:|---:|---:|
| H1 | 12% / 2.5pp | 42% / 14pp | 212→57.5 | 4.978→1.544 s |
| H2 | 10% / 4.5pp | 41% / 14pp | 285.5→62 | 6.173→1.578 s |
| H3 | 26% / 9pp | 55% / 6.5pp | 97.5→46.5 | 1.742→0.916 s |

邻域 E0 对 Random 的 valid yield 在 H1/H3 为 20/20 seeds 更高，在 H2 为 19/20；双侧 sign-test
分别为 `1.91e-6/4.01e-5/1.91e-6`。平衡 E0 按定义稳定得到 50% valid yield 和 0 IQR。

但固定预采样成本不能被隐藏：H1/H2 的 1000 次 lowering 中位墙钟为 24.901/22.503 s，H3 因
`|S|=480` 扫完整域需要 8.621 s；相同 seed 的 Random 前 50 次仅为 1.159/1.101/0.898 s。
所以“找到前25个合法点所需 calls/墙钟下降”只对允许早停的邻域发现过程成立；若忠实执行固定
1000 点 presampling，初始化总墙钟反而显著增加。邻域 IQR 也只在 H3 下降，在 H1/H2 上变大，
不能笼统声称稳定性全面提高。

在完成 presampling 后，validity-biased 的下一批 50 点合法率中位数为 H1 58%、H2 11%；H3 已
扫描完整空间，没有 unseen 点，不能生成该项。这是平台/空间差异应保留的边界。

## 4. ML²Tuner Model V 留一几何验证

Model V 只使用 workload、tile、mode 等 lowering 前可见特征；每次用两个完整几何训练，在第三个
几何测试。目标几何标签不参与训练。当前 XGBoost 参数没有随机子采样，所以 20 个 seed 得到同一
模型结果；零 IQR 是确定性实现属性，不应包装为随机稳定性证据。

| target | 全空间合法率 | F1 | recall | validity nDCG | top-20合法率 | top-50合法率 | top-100合法率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| H1 | 12.59% | 0.679 | 0.759 | 0.957 | 80% | 92% | 90% |
| H2 | 9.19% | 0.428 | 0.680 | 0.893 | 100% | 100% | 52% |
| H3 | 27.50% | 0.267 | 0.182 | 0.861 | 70% | 50% | 44% |

Model V 的可用结论是：在本组几何间迁移时，高置信候选的前部排序明显富集合法点，因此有希望减少
进入真实 lowering 的无效 profiling。不能据此声称分类器对整个新几何可靠；H3 的低 recall/F1
就是明确反例。该实验也没有 Model P/Model A 的 latency RMSE/nDCG、FPGA invalid 比例或
time-to-target，这些必须等待 P7R482 的 50 点完整板池获得真实 correctness/timing 标签后再做。

## 5. 当前结论与下一步

1. HW-Aware 的“合法点存在邻域聚集”在三个 ResNet18 几何上得到支持；固定大规模 presampling 的
   总成本却可能超过直接 Random 初始化，必须同时报告早停成本与完整协议成本。
2. ML²Tuner Model V 在少量 top-ranked 候选上显著降低预计无效编译，但跨几何全局分类能力不均，
   适合做低成本粗筛，不能替代 real lowering、FSim 和 FPGA correctness。
3. 当前 SSH host key 尚未由串口重新认证，P7R482/P7R483 的 50 点池不能安全上板。获得当前
   `/etc/dropbear/dropbear_rsa_host_key` 的完整 `dropbearkey -y` 输出后，才能从头启动不可拼接的
   correctness/timing session；随后才可闭合 ML²Tuner P/A、Cheng 四方案性能和六策略等预算比较。

