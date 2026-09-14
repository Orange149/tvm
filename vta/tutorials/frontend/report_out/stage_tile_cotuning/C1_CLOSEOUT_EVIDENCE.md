# C1 最小收尾证据与主张限制

2026-09-08；本轮只核对已有结果，不补新的切图/调优实验。

| 项目 | 已有事实 | 允许的结论 / 限制 |
|---|---|---|
| 搜索 | 4623 拓扑、972528 配置，k-best Top-20 与枚举逐项一致 | 固定静态目标下搜索正确，不代表绝对 FPS 准确 |
| 历史边界消融 | 200条记录 MAE 17.3617→12.8220 ms，Spearman .7700→.9310 | 历史 copy 路径的离线解释，不是新增吞吐收益 |
| 自然 Top-20 | 10.220–11.511 FPS；前10含池内最佳，池内相关 .155 | 已测候选池覆盖有价值，不能保证全域最优或准确细排 |
| A/B/C/D | M0 regret@1 7.39%，M1/M2 0；验证集仅4种 | 描述性对照，非前瞻泛化证明 |
| M2 | 0/972528 为预测瓶颈，最接近为 M1 的17.27%，Top20不变 | 当前 DMA 带宽下界未改变排名，不能据此证明物理 DDR 无争用 |

主要引用：论文第7章既有主结果、`stage_memory_experiments/MEMORY_MODEL_ABLATION.md` 及其 JSON；
冻结 package 数值参考沿用旧路径，不将 E3 新 CPU tail 换进旧性能数据。

## 成本模式核查

`run_stage_memory_model_ablation.py` 从 `v1_local_cost_table.json`、`v1_profile_manifest.json`
经 `solve_cpu_vta_pipeline_v1_p3.py:build_context/fit_boundary_ownership` 读取直接边界及其 CPU/VTA owner。
`freeze_cpu_vta_pipeline_v1.py` 将其定义为 `host_adapter_set_get_copy`。
这套 M1/M2 服务不应写成已经重新标定的 shared-K2 成本。C2 的 shared 测量是固定切图下的
运行方式对照；C3-S 在同一 all-shared K2 基线上固定 topology，只改变额外缓冲来源，不借此重新评价 C1 排名。

## 数值边界与未完成项

各 stage 对自身量化 LLVM 参考的正确性不等于全网络不同切图数值等价。
E3 曾发现 int8 回绕及独立量化差异；已完成的 once-quantized tail 资格化只覆盖对应局部试验。
历史输出/参考应按原记录呈现，不宣称新增 ImageNet 精度或全域跨切图语义通过。
最终论文展示配置的统一精度政策/反例逐帧回归仍未完全关闭，保持未完成标记。
该缺口限制跨切图的共同精度性能主张，不自动否定固定目标下的 DP/枚举一致性。

## 本轮文字收敛

中英文摘要与贡献段已撤下 stage–tile 联合优化有效的暗示，明确 FuseOps/量化/packing/AutoTVM
的职责；相邻切点的字节差异改为描述证据，不再写成已实现安全剪枝。
旧 tile 实验保留为补充；第三项尚未通过验收，不提前加入论文已完成创新列表。
