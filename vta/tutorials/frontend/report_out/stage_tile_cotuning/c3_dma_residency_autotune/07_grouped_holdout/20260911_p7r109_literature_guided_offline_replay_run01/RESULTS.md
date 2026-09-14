# 论文启发式搜索离线消融

> `SUPERSEDED`：本次初始运行未加入 exact request histogram/max-request 特征；正式可消费结果为
> `../20260911_p7r109_literature_guided_offline_replay_run02/`。

> 性质：冻结历史数据上的开发性 replay；不是论文原样复现，不是新的 FPGA 确认实验。

## 合法性初始化（310 个候选，按 workload 留一）

| 方法 | budget=4 合法数 | budget=8 合法数 | budget=16 合法数 |
|---|---:|---:|---:|
| Random | 2.54 | 5.10 | 10.18 |
| Rieber-inspired 邻域 | 2.80 | 6.00 | 11.80 |
| ML²-inspired 基础 V 模型 | 3.70 | 7.50 | 14.20 |
| ML²-inspired 硬件 V 模型 | 4.00 | 7.60 | 14.50 |

这里的 label 仅为本地 lowering 成功/失败，不能外推为真实 FPGA correctness。

## 驻留增益排序（56 个真实 FPGA same-tile 配对，按 workload 留一）

| 特征 | 符号准确率 | regret@1 | regret@2 | regret@4 | regret@8 |
|---|---:|---:|---:|---:|---:|
| 模式+计算上下文 | 80.5% | 11.38 | 3.87 | 1.94 | 0.00 |
| +总 DMA bytes | 75.5% | 0.02 | 0.00 | 0.00 | 0.00 |
| +完整请求形态 | 75.5% | 0.02 | 0.00 | 0.00 | 0.00 |
| +命令/硬件联合 | 75.1% | 0.02 | 0.00 | 0.00 | 0.00 |

完整请求形态在 0/4 个预算点严格优于 bytes-only；这只是跨四个已曝光 workload 的开发证据。
