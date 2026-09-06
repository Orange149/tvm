# P7D 公式收口 Review

更新日期：2026-09-06

## 结论

P7D 已完成参数准入审计、DP 重跑和冻结标签评价。三 boot 的 CPU streaming read/write
带宽进入共享 DDR 下界；CPU pair slowdown 因尚无可泛化 surface 不进入全局搜索，P7B-2/P7C
只有单 boot，也不进入。候选拟合的 `+34 ms` 与比例系数继续禁用。

本轮没有强行生成新的修正项。质量改善 gate 未通过，因此 V1 冻结为高性能区域筛选模型，
不能称为经过物理校准的绝对 FPS 预测器。P8A 可以把它作为不再变化的规划 baseline，独立验证
共享 slot zero-copy；P8 不得反向修改本轮排名。

## 参数准入

| 参数 | 是否进入公式 | 原因 |
|---|---:|---|
| `candidate_fitted_constant_or_scale` | 否 | +34 ms and proportional calibration consume complete-candidate outcomes and cannot enter a transferable score |
| `cpu_memory_streaming_bandwidth` | 是 | three independent boots passed the frozen component gate |
| `cpu_pair_slowdown` | 否 | generalized_slowdown_surface_ready is false; 16 selected pairs cannot be extrapolated to every DP transition |
| `cpu_vta_contention` | 否 | single-boot evidence is not a pure DDR parameter and has not passed the required three-boot gate |
| `p7b2_boundary_slope` | 否 | single-boot observation and replacing the existing owner-split boundary model would increase the known underprediction |
| `runtime_frame_or_stage_intercept` | 否 | P7B-2 found an unidentifiable/de-duplicated host queue floor below 0.1 ms; stage run_ms already owns invocation and wait |

## 冻结评价

| 范围 | cycle MAPE | Spearman | regret@5 |
|---|---:|---:|---:|
| 历史 200 条（绝对误差） | 28.022% | 不适用 | 不适用 |
| 历史 199 unique | 28.023% | 0.775 | 10.279% |
| 历史 69 canonical | 26.750% | 0.786 | 7.303% |
| 自然 Top-20 | 23.523% | 0.155 | 5.419% |

历史 200 条统一 `+34 ms` 的诊断 MAPE 为 `5.896%`；自然 Top-20 为 `12.859%`。
自然 Top-20 的并列分数感知 Spearman 为 `0.053`；表中 `0.155` 沿用冻结上板报告的
固定顺序口径，用于与 `0.155` 基线直接比较。新旧 Top-20 的候选、顺序和预测周期完全一致。
更新 streaming 带宽后 Top-20 仍由计算/core/VTA资源决定，DDR不是其关键瓶颈，因此排名
和现有误差没有得到实质改善。这是负结果，但避免把局部 slowdown 或单 boot 干扰错误外推。

失败 gate：`historical_200_mape_below_fixed34_baseline`, `natural_top20_mape_below_fixed34_baseline`, `natural_top20_regret_at_5_below_5_419pct`, `natural_top20_spearman_above_0_155`。

## 下一步

进入 P8A，只验证单边界 zero-copy 可行性：审计普通 set/get 的真实复制字节，比较最快正确
普通复制与 u-dma-buf 直接访问，并对一个 CPU->VTA 边界完成同址绑定和交替输入正确性检查。
P8A 完成后停止 review，不自动进入双 slot Pipeline。

## 产物

- `v1_p7_physical_profile.json`: `b4d36712ef7baac6c33179de7d2767fc4405015fadc68f20774521f598b42557`
- `v1_p7d_ranked_candidates.json`: `bfa0e9ecba2eaafbc188e33752961ec54edebc3feaa019faa2a93b97a78edd35`
- `v1_p7_formula_validation.json`: `983afc8d7e0f6f919aae7c0d40c2c8a041203f670c9366f01d84c7ad60664aa6`
