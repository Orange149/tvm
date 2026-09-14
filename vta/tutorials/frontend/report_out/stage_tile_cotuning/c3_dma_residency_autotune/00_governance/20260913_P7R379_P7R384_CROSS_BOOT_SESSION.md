# P7R379--P7R384：R50C/R50D 跨启动会话

当前 boot：`4d232b6f-6e2a-4395-aafe-853a021403e8`。全部实验只使用 RAM runtime，结束后
FPGA=`operating`、u-dma-buf=`201326592`、RPC 工作目录正确，未出现新 EXT4/mmc 日志。

## R50C：正向整图收益稳定

P7R379 原样复用 P7R309/P7R311 冻结的预训练 ResNet50 A/B 图和 fused-TIR 预测：

- 当前 boot：346.912→330.290 ms，吞吐提升 5.033%，7/7；
- 旧 boot：346.687→330.277 ms，吞吐提升 4.968%，7/7；
- 两启动合计 12 次 correctness、28 次 timing，输出全部相等非零；
- 十项最终融合 LOAD/STORE 差在两启动上均精确命中。

因此 R50CF00 的整图机制收益可以写成“两独立启动方向和量级稳定”，但仍只代表一个 workload。

## R50D：精确 oracle 不稳定，2% 质量稳定

P7R381/P7R382 在当前 boot 重跑同一个 P7R332 18 点最终融合池：

- 正确性仍为 15/18；固定四点前沿仍为 2/4 正确；
- 旧 boot 的前沿最佳 R50DF07 input 是精确 oracle；
- 当前 boot 的精确 oracle 改为前沿外 R50DF04 weight-barrier；
- 当前前沿最佳的绝对 latency regret 为 0.0327%，paired-ratio regret 为 0.1706%，仍在 2% 带。

当前 oracle 只被前沿中一个 FPGA-invalid weight 候选静态支配。删除两个失败前沿点后重新计算前沿，
唯一新增点恰为当前 oracle。这是 `invalid-dominator peeling` 的开发依据，不是已确认新方法；下一次
必须在新 workload 标签前冻结该动作。

论文必须保留此负结果：R50D 固定 Pareto 前沿只能主张 2/2 boot 保持 oracle+2%，不能再主张跨启动
保证精确 oracle。首 boot 的候选/流量缩减事实仍成立，但其 exact-oracle 结论仅限首 boot。

证据目录：

- `07_grouped_holdout/20260913_p7r379_r50cf00_fullgraph_cross_boot_run01/`
- `07_grouped_holdout/20260913_p7r380_r50cf00_fullgraph_cross_boot_audit_run01/`
- `07_grouped_holdout/20260913_p7r381_r50d_pareto_front_cross_boot_run01/`
- `07_grouped_holdout/20260913_p7r382_r50d_remaining_cross_boot_run01/`
- `07_grouped_holdout/20260913_p7r383_r50d_pareto_cross_boot_analysis_run01/`
- `07_grouped_holdout/20260913_p7r384_r50d_cross_boot_stability_run01/`

