# P7R378：Y08 冻结全池分析

Y08 是 YOLOv3-tiny conv12（CI=512、CO=1024、H=W=13、K=3）。P7R357 在读取任何 Y08
FPGA 正确性或 latency 前冻结了六候选和五个顺序；P7R377 在 boot
`4d232b6f-6e2a-4395-aafe-853a021403e8` 上得到 6/6 三 seed 正确、42/42 计时正确，完整池 oracle 为
`132.920140 ms`。

| family | mode | logical DMA bytes | DMA calls | median latency (ms) | regret |
|---|---|---:|---:|---:|---:|
| Y08F01 | input_stationary | 798,316,032 | 13,520 | 1253.649 | 843.16% |
| Y08F01 | original | 808,829,952 | 175,760 | 1734.253 | 1204.73% |
| Y08F02 | input_stationary | 61,761,024 | 1,040 | 133.240 | 0.24% |
| Y08F02 | original | 65,455,104 | 13,520 | 208.913 | 57.17% |
| Y08F06 | input_stationary | 61,761,024 | 858 | 132.920 | 0.00% |
| Y08F06 | original | 62,007,296 | 1,690 | 138.023 | 3.84% |

## 同 tile 驻留消融

| family | DMA bytes reduction | DMA calls reduction | latency improvement |
|---|---:|---:|---:|
| Y08F01 | 1.30% | 92.31% | 27.71% |
| Y08F02 | 5.64% | 92.31% | 36.22% |
| Y08F06 | 0.40% | 49.23% | 3.70% |

三个 input-stationary 配对全部加速（3/3），但加速幅度从 3.70% 到 36.22%。F02 与 F06 的
input-stationary 总 DMA bytes 都是 61,761,024 B，延迟仍为 133.240 ms 与 132.920 ms；因此
DMA bytes 是有效的冷启动优先级，不是精确 latency 模型。F01 只降低 1.30% bytes，却因请求数
下降 92.31% 得到 27.71% 加速，也说明请求粒度会影响 bytes 与时间之间的映射。

## 冻结顺序回放

| policy | trials to exact oracle | trials to oracle+2% |
|---|---:|---:|
| random_lazy_build | 4 | 1 |
| operator_bytes_lazy_build | 1 | 1 |
| operator_calls_lazy_build | 1 | 1 |
| operator_pareto_hash_lazy_build | 1 | 1 |
| operator_pareto_hash_prebuild_ablation | 1 | 1 |

bytes、calls 和 Pareto 顺序都在第一次测量得到精确 oracle。不过，预注册的单个 Random 顺序首先
测到 F02 input-stationary，也已进入 oracle+2% 带，只是到第 4 次才得到精确 oracle。因此本池不能
宣称本文方法在 success@2% 上优于随机；它提供的是第三个 YOLO 网络几何上的机制复现和
`exact-oracle time-to-target` 证据。这里的策略成本是对完整全池测量所得逐候选成本的事后前缀累计，
不是五套分别执行并计时的在线 tuner，不能写成独立 online wall-clock 实验。

## 口径

- 逻辑 DMA 来自最终 lowered VTA LOAD/STORE，不等同于物理 AXI burst。
- 每个通过候选的 admission 成本含三次 correctness 与七轮计时；每轮 time evaluator 实际执行两次，
  因而是 17 次 FPGA kernel invocation。
- 全池外层进程墙钟为 `135.309 s`，包含交叉编译、clean start、
  健康门、上传、分配、正确性和计时；不能按候选前缀精确分摊。
