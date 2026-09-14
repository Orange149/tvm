# P7R380：R50C 全图驻留收益跨启动复验

同一组冻结的预训练 ResNet50 图、参数、A/B DSO 与 fused-TIR DMA 预测在两个独立 boot 上执行：

| boot | original (ms) | input-stationary (ms) | throughput improvement | paired wins |
|---|---:|---:|---:|---:|
| `aa7a3e5c-021d-4d6b-88ae-7f696faa567c` | 346.687 | 330.277 | 4.968% | 7/7 |
| `4d232b6f-6e2a-4395-aafe-853a021403e8` | 346.912 | 330.290 | 5.033% | 7/7 |

两启动均为正向，合计 14/14 配对轮获胜；速度提升范围 4.968%--5.033%，
均值 5.001%。12 次 correctness 与 28 次 timing 调用全部 A/B 输出相等且非零，
预注册的十项 fused-TIR LOAD/STORE 差值在两启动上都精确命中。

这支持“该 frozen R50C 全图驻留收益跨启动方向与量级稳定”，但只有一个 workload 的两个 boot，
不能把 14 个配对轮当成 14 个独立启动，也不能外推为 ResNet50 通用加速率、ImageNet accuracy 或
物理 AXI 流量结论。
