# C3-P4-local Handoff

任务编号：C3-P4-local-A1  
阶段：P4 无板基础工具  
状态：`completed`

## 代码变更

- `vta/tutorials/frontend/c3_candidate_identity.py`：严格规范 JSON、稳定 candidate ID、ConfigEntity debug-index 剔除、冻结失败类别。
- `vta/tutorials/frontend/extract_vta_candidate_features.py`：独立历史 archive adapter；未修改既有 `extract_static_vta_dma.py`。
- `vta/tutorials/frontend/test_c3_candidate_identity.py`：7 个身份/adapter 单测。

未修改 schedule、runtime、driver、hardware、`tune_resnet18_vta.py` 或 P0/P1 文件；没有 SSH/RPC/板端操作。

## 验证

- `py_compile`：3/3 文件通过。
- `unittest discover`：7/7 通过。
- 真实历史适配：10/10 行成功、10 个唯一 ID、command status 唯一值为 `not_measured`。

## 输出

- `results.jsonl`：10 条 `c3_candidate_feature_v1`。
- `summary.json`：输入/输出哈希、记录数、失败分类 schema。
- 本目录其余文件记录命令、环境、预注册边界和产物哈希。

## 未解决项

- 当前 boot 硬件指纹仍未资格化；本 run 的 hardware fingerprint 明确使用冻结的“expected bitstream”值，只用于历史记录身份。
- SRAM working set 不存在于旧 archive，记录为 `not_available_in_archived_source`。
- 新 residence mode 的 lower/runtime 对齐属于正式 P4，需 P3 schedule gate 后执行。
- command footprint 属于 P7 候选诊断，本 run 不测量。

## 建议

建议主代理验收本地 identity/schema 基础。不要据此接受 G4；G4 仍需要新模式的实际 lower/runtime 核对和当前环境资格化。
