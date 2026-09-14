# C3 P2-local handoff

任务编号：C3-P2-local-run01  
状态：`partial_completed_concurrent_source_change`

## 已完成

- 冻结 TopHub 文件 hash 与 P0 一致；`ApplyHistoryBest` 对 W00--W09 为 10/10 index 命中、0 fallback。
- 保存每个 workload 的 resolved ConfigEntity。TopHub 的完整 split 外因子与 ConfigSpace 中 `-1` 推导占位不同，但 inner factors 相同，实例化所得 TIR 结构相等。
- 原 hash 尚为 `b7fc752b...` 时重跑静态 DMA，输出与历史 `static_workload_dma.json` 逐字节一致，10 workloads、8 direct profiles 六字段精确。
- 10/10 原 TopHub 配置通过 AXU5EVB 交叉编译。
- 10 workloads × 3 个冻结 seed 全部通过本地 FSim 与独立 NumPy 卷积逐元素比较，共 30/30 正确。
- `test_vta_insn.py` 的 7 个入口通过；`test_stage_tile_cotuning.py` 为 19 passed。

## 偏差与停止原因

执行期间 schedule 文件被并发任务于 22:42:44 修改。此变更并非本代理产生；本代理未回退或覆盖它。根据预注册 source guard 停止继续验证。`lowered_tir/` 和 `lowered_tir_manifest.json` 是变更后生成的 10 份 original-path TIR，已明确标记为 post-change，不得混入冻结原模板结论。

## Gate 建议

可以接受为 `P2_LOCAL_EVIDENCE`，但不能接受完整 G2。后续应由主代理在新 schedule 重构稳定后创建不可覆盖的 local run02，专门证明 original path 相对本 run 的冻结 `static_dma.json`、ConfigEntity 和必要 TIR 结构不变。开发板恢复后仍必须执行 current-boot manifest、TopHub canary 和 topology-B 5% 门槛。

本任务未 SSH、未上板、未声称恢复 10.724 FPS、未修改任何源码。
