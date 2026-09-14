# C3-P4-local 状态

任务：稳定候选身份、失败分类和历史静态 DMA 聚合适配。  
状态：`completed_local_only`  
日期：2026-09-10

## 结果

- 规范 JSON 的 SHA-256 候选身份已实现；`config.index/config_index/debug_index` 不参与语义哈希。
- 7 个单元测试通过，覆盖 key 顺序、config-space 重编号、实体变化、mode/硬件变化、非有限浮点和失败词表。
- 冻结的 `static_workload_dma.json` 10 条 workload 均与 `selected_tasks.json` 的完整 ConfigEntity 匹配，生成 10 个互异 `candidate_id`。
- 聚合结果保留分类 DMA totals、唯一 tensor bytes、reload、最大请求、精确 request-size histogram、padding 与 per-memory 聚合。
- 所有 command footprint 均明确为 `not_measured`；本任务没有推测 instruction/UOP/FINISH/replay/submit。

## 边界

本 run 只是历史证据适配器，没有重新 lower、compile、运行正确性或联系开发板。`legality.status=archived_success` 只表示历史静态提取存在，不是当前源码或当前 boot 的重新资格化结果。

## 已知非结果性失败

第一次以 `python -m unittest -v vta/tutorials/frontend/test_c3_candidate_identity.py` 运行时，因 `vta/tutorials` 不是可导入包而报 `ModuleNotFoundError`。随后使用 discovery/file 入口运行同一测试并通过 7/7。失败不涉及代码语义或开发板。
