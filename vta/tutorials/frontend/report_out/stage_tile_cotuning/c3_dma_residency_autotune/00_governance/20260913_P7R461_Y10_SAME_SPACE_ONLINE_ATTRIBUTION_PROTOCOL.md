# P7R461：Y10 同空间在线选择归因实验协议

状态：`COMPLETE_BY_P7R462_P7R468; TARGET_LABELS_ALREADY_EXIST`

## 执行结果（P7R462--P7R468）

冻结顺序的六次真实运行均已完成：DMA prior 和 mode-aware XGB 各三个 seed、每次固定派发六个
候选，全部使用同一 12 点候选池和同一完整图执行路径。36/36 个候选派发均通过三输入、八输出
正确性门。完整池 exact oracle 为
`dc939aafdfbb9e7e84172f7ce12b651551b75e95753ca7e33c30967750f7abb2`，历史 paired ratio 为
`0.987878482`。

| 指标，中位数（范围） | DMA prior | mode-aware XGB |
|---|---:|---:|
| 完整外层墙钟/s | 400.933（398.963--401.787） | 398.963（397.041--401.549） |
| exact oracle 首次 dispatch | 3（3--3） | 6（4--6） |
| exact oracle 首次墙钟/s | 212.564（212.066--213.443） | 396.797（276.261--398.726） |
| oracle+2% 首次 dispatch | 1（1--1） | 1（1--1） |
| oracle+2% 首次墙钟/s | 87.697（87.629--88.064） | 87.997（86.964--89.838） |

DMA prior 的 exact time-to-quality 中位数相对 XGB 减少 46.43%，即提前 1.87 倍。由于协议要求
两种方法都执行满六个候选，最终总墙钟接近是预期结果，不能据此声称固定终止预算下总成本下降；
算法差异体现在达到 exact oracle 的时间和派发位置。两种方法第一次派发均进入 +2% 带，因此本池
不支持“DMA prior 更早进入 +2% 带”的主张。

完整聚合、逐次曲线和成本分解见
`../07_grouped_holdout/20260914_p7r468_y10_same_space_online_ablation_aggregate_run01/REPORT.md`。
本结果是标签已暴露 Y10 上的同空间选择器归因消融，不是新的 prospective holdout。

## 目的

P7R460 比较了两个真实的从零系统流程，但两者的候选空间和测量粒度不同，64.84% 的 T0→T1
缩短不能全部归因于搜索算法。本实验固定同一个 Y10 12 点 `tile × residence-mode` 候选池、同一个
完整 YOLOv3-tiny-320 构建器、同一个逐候选 clean-start 板端执行器、三输入八输出正确性门和七轮
candidate/stock 配对计时，只改变下一个候选的选择方法。

## 两种选择器

1. `dma_prior`：使用 P7R431 在 Y10 板端标签产生前冻结的 total-DMA-bytes 顺序，依次选择候选；
2. `mode_aware_xgb`：特征为七个 ConfigEntity tile knob 加三类 residence mode one-hot。前四点按
   `sha256(seed:Y10:candidate_id)` 确定性冷启动，此后每轮以已通过正确性门的完整图 paired-ratio
   标签重新拟合 XGBoost，并选择预测值最低的未测候选。失败候选消耗 dispatch，但不进入回归标签。

XGB 固定为 64 trees、depth 3、learning rate 0.1、hist、单线程；种子固定为 44501、44502、44503。
两种方法每次均执行 6 个候选。执行顺序冻结为：

```text
DMA(44501) -> XGB(44501) -> XGB(44502) -> DMA(44502)
-> DMA(44503) -> XGB(44503)
```

## 评价

- 每次运行 6 个真实完整图候选构建/板端 dispatch；
- 报告完整外层墙钟、候选构建时间、正确性/计时调用、逻辑 DMA bytes/calls；
- 运行后才以候选身份连接既有 completed-pool oracle，报告 exact-oracle 与 oracle+2% 首次命中位置；
- 三个种子分别报告，并汇总中位数与范围。

## 边界

Y10 全池标签在本协议前已经存在，所以这不是新的 prospective holdout；它是一次选择器不读取旧标签、
但在相同真实执行路径上重新付费的归因消融。它可以回答“相同候选空间和相同测量预算下，两种选择
逻辑的实际代价和结果”，不能消除实验设计者已知 Y10 的风险。随机输入只验证确定性全输出等价，
不代表 COCO mAP；runtime 的逻辑 LOAD/STORE 计数不冒充物理 AXI 流量。
