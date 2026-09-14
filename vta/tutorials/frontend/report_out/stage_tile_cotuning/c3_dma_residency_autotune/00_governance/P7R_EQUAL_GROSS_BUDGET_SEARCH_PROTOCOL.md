# P7R 等 gross-budget、no-leak 搜索协议

## 目的与状态

本协议比较五个策略家族、七条实际曲线：Random、stock-knob XGB、规则 only、合法性模型 V，
以及 bytes/calls、+request、+command 三档 `V + same-tile ΔT`。离线 full-label replay 只用于开发和接口回归；未来确认结论必须来自候选与
判定阈值均已冻结后的前瞻逐次派发记录。

## 候选池接口

规范 schema 为 `c3_no_leak_search_pool_v1`。每个 workload 包含：

- 独立的 `sealed_reference`，其 `reference_kind` 只能是 `tophub`、
  `long_budget_stock_xgb` 或 `full_pool_oracle`；
- 不含 sealed reference 的 `candidates`；
- 每个候选的 `predispatch`：规则分数、stock knob、V 特征、same-tile ΔT 特征；
- 可选 `oracle`：仅离线 replay 使用，按 lower、compile、FPGA、measure 四阶段记录状态与 wall-ms。

候选池必须声明 `frozen_before_target_labels=true`。未来 pool 可以逐步补充阶段 outcome/cost；
前瞻模式不要求目标候选带 `oracle`。所有 feature map 在池内必须同键，缺失、非有限值或来源声明
不合格时 fail closed。

## 无泄漏约束

1. selector 只接收由固定字段构成的 `public_candidate` 投影，永远不接收 `oracle`。
2. 每次先选一个候选，再揭示 lower→compile→FPGA→measure 结果；首个失败后的阶段必须为
   `not_run`。
3. 任意失败都消耗一个 gross dispatch；invalid 不进入 XGB latency 或 ΔT 回归标签。
4. V 与 ΔT 的预训练记录必须排除当前 target workload；当前 workload 只能使用此前已经派发并揭示
   的结果。
5. sealed reference 不得出现在 candidate list、首候选、随机 warmup 或任何训练标签中。其 latency
   只允许在整条顺序生成完以后计算 trials/success-to-2%/5%。若声明 `reference_kind=tophub`，必须有
   精确冻结的 TopHub identity；YOLO 等没有精确 TopHub 的 workload 不得冒充，可使用冻结的
   long-budget stock-XGB reference。
6. `full_pool_oracle` 不允许在 pool 中预存 latency；只能在完整派发顺序生成后由全部成功标签计算。

## 五个策略家族、七条曲线

- Random：由 seed、workload 和 candidate identity 哈希得到可复现随机排列。
- stock-knob XGB：前四个 gross dispatch 为 seeded warmup，之后只用当前 workload 已观测成功点的
  七个 ConfigEntity knob 拟合 64 棵、depth=3 的 XGBRegressor；失败点不训练。
- 规则 only：按冻结的、越小越优的 `rule_score` 排序。
- validity V：other-workload 加当前已揭示记录训练标准化 logistic classifier，选择合法概率最高点。
- `V + same-tile ΔT`：ΔT 使用模式截距和非负物理特征系数；把 invalid-risk rank 与预测 ΔT rank
  等权 Borda 融合，避免以延迟单位手调权重。冻结三档累积 feature set：`bytes_calls`、
  `request`（bytes/calls + request shape）、`command`（再加 command）；分别由
  `v_plus_delta_t`（兼容别名）、`v_plus_delta_t_request`、`v_plus_delta_t_command` 汇总。它们都是
  标定代理，不是理论模型或理论上界。

## 公平预算与统计

主预算为 4/8/12 gross dispatch，固定 20 seeds。每个策略、workload、seed 都从空 target history
开始，使用相同候选池和相同 failure oracle。报告：

- gross dispatch 数；lower-invalid、compile-invalid、FPGA-invalid、measured-ok；
- lower、compile、FPGA、measure wall-clock 分项及完整率；缺失 cost 保持 null，不按零伪造；
- 相对非 reference 全标签 pool oracle 的 simple regret@budget；若尚无合法测量则为 null，并单独报告
  defined rate；
- 相对 sealed-reference latency 的 success@2%/5% 和 trials-to-2%/5%；失败 run 另以
  `pool_size+1` 做删失
  统计，同时保留成功率和成功样本统计；
- workload×seed 的 median、Q1、Q3、IQR。

## 两种执行模式

- `offline-replay`：全标签只通过 dispatch 后的 reveal 边界进入 history。可用 P7Q adapter 做
  development 回归，但 P7Q 的 phase wall-clock 不完整，不能输出确认结论。
- `prospective`：读取无 oracle 的目标 pool、不可变历史和可选旧 training pools，只输出一个下一
  candidate identity。下一步必须把真实阶段结果另存为新 history 后重新调用；不得就地改写旧记录。

## 失败策略与主张边界

候选重复、reference 混入候选、伪造 TopHub、target 泄漏进训练、阶段顺序不合法、feature schema
漂移、非有限数或输出路径已存在时立即失败。离线 replay 只能证明实现和历史开发数据上的行为；
它不能证明新 workload 性能、板端搜索收敛、跨 boot 稳定性或 FPS 提升。
