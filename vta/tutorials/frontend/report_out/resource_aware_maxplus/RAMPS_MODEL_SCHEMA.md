# RAMPS Physical Model Schema

更新日期：2026-09-01

本文档定义完整 M2/M3 publication-mode 的数据契约。当前 M0/M1 最小实例只复用
`PartitionWorkload -> StageService` 边界，不要求先生成本文件中的完整 HardwareProfile；
启用条件见 [HARDWARE_LAYER_DESIGN_REVIEW.md](HARDWARE_LAYER_DESIGN_REVIEW.md)。理论公式见
[RAMPS_SYSTEM_MODEL.md](RAMPS_SYSTEM_MODEL.md)，执行状态见
[RAMPS_EXECUTION_ROADMAP.md](RAMPS_EXECUTION_ROADMAP.md)。

## StageService

每个固定 schedule stage 必须提供以下物理服务分量，单位均为 ms/item：

| 字段 | 含义 | 正式来源 |
|---|---|---|
| `compute_ms` | CPU/VTA 计算分量之和 | primitive calibration + lowering |
| `memory_ms` | CPU memory roofline 相位 | bytes / calibrated CPU memory bandwidth |
| `sequential_unit_service_ms` | CPU `sum(max(unit_compute,unit_memory))` | 对每个 backend fused unit 分别应用 Roofline |
| `load_ms` | VTA PS-PL load 相位 | profiler bytes/calls + calibrated DMA model |
| `store_ms` | VTA PS-PL store 相位 | profiler bytes/calls + calibrated DMA model |
| `launch_ms` | submit/sync 或 CPU launch | 独立 calibration |
| `bridge_ms` | pack/unpack adapter | boundary calibration |
| `spill_ms` | 兼容字段，publication profile 中固定为 0 | spill 已包含在 lowering-derived total physical DMA 中，不得再次相加 |
| `core_demand_ms` | CPU 使用的总 core-time | serial exclusive process CPU time |
| `core_demand_source` | demand 证据来源 | publication mode 不得为 legacy fallback |

CPU wall time 与 CPU core demand 是两个不同量。例如一个 20 ms wall-time stage 可能只消耗
10 ms core-time；不能按请求线程数把它写成 `20 * 4 = 80 ms`。

## Fixed-Schedule SRAM Rule

- SRAM capacity 仍是硬合法性条件。
- 固定 tile 的 total physical load/store（包含真实 spill）由 lowering/profiler 给出。
- `sram_risk_penalty_ms`、peak utilization penalty、静态 tile count 不进入主周期预测。
- partition 造成的中间 tensor materialization 记入 boundary bytes/calls。
- `spill_bytes/calls` 只用于解释和审计；正式周期不得再增加独立 `spill_ms`。

## Resource Matrix

每个 candidate 生成 `D[actor, resource]`，资源至少包括：

```text
cpu_core_pool[N tokens]
cpu_memory[M tokens]
vta_mutex[V tokens]
ps_pl_load[L tokens]
ps_pl_store[S tokens]
bridge_pack_unpack[B tokens]
```

`N/M/V/L/S/B` 全部来自 `HardwareProfile.resource_capacities`，求解器不得硬编码当前板子的
四核/单 VTA 配置。stage data channel 初始 token 为 0；每条真实 boundary 的反向 FIFO
free-slot channel 使用该边的 `queue_depth`。branch/route 必须保留真实 producer/consumer，不能
按 stage 列表伪造相邻边。worker 是每个 actor 自身的一 token reuse cycle，不是所有 stage
共享的一列资源。CPU-to-CPU materialization 归入 CPU memory；异构 boundary 的 PS-PL bytes/calls
只由 consumer/producer 的 lowered VTA LOAD/STORE 持有，boundary actor 只计 host adapter，避免同一
物理传输重复建立资源 demand。

## Prediction Modes

主结果必须满足：

```json
{
  "prediction_options": {
    "include_residual": false,
    "require_measured_core_demand": true
  }
}
```

`physical_plus_residual` 只能作为单独消融。旧记录可以用
`legacy_wall_time_x_requested_threads` 做历史兼容分析，但 publication mode 会直接拒绝。

## Example Artifact

[stage3_event_graph_validation/validation.json](stage3_event_graph_validation/validation.json)
包含完整示例：stage actor、boundary/FIFO channel、资源需求矩阵、cycle constraints、展开后的
状态图、分析最大环均值和 Karp 谱半径。

## Evidence Boundary

当前 schema 和合成测试只证明实现语义自洽。Karp 测试验证的是已构造 resource-cycle
constraints 的数学编码，不单独证明这些约束已完整描述真实 runtime contention。
`compute_ms`、CPU core demand、VTA overlap、DMA
带宽与 bridge latency 的数值准确性，必须由阶段 4 的无 fallback 硬件校准建立。

## Layer H: Optional Full HardwareProfile

当低预算消融证明必须启用完整 M2/M3 时，`hardware_profile.json` 必须满足以下条件。Stage 4 的 isolated-service draft 不包含
Stage 5 system overlap surfaces，因此不能通过 publication gate：

```json
{
  "schema_version": 3,
  "profile_kind": "portable_hardware_service_profile",
  "optimization_objective": "top_k_candidate_triage",
  "profile_sha256": "...",
  "compiler_config_sha256": "...",
  "allowed_schedule_ids": ["..."],
  "resource_capacities": {
    "cpu_core_pool": 4,
    "cpu_memory": 1,
    "vta_mutex": 1,
    "ps_pl_load": 1,
    "ps_pl_store": 1,
    "bridge_pack_unpack": 1
  },
  "fallback_used": false,
  "estimated_from_used": false,
  "calibration_budget": {
    "support_pool_case_count": 212,
    "measured_case_count": 80,
    "holdout_case_count": 16,
    "full_pool_measured": false
  },
  "validation_evidence": {
    "protocol_sha256": "...",
    "measurement_plan_sha256": "...",
    "calibration_cost_ledger_sha256": "...",
    "raw_data_sha256": "...",
    "fit_report_sha256": "...",
    "reference_correctness_report_sha256": "..."
  },
  "validation_gates": {
    "correctness_gate_passed": true,
    "determinism_gate_passed": true,
    "precision_gate_passed": true,
    "service_fit_validated": true
  }
}
```

这里的 212/80/16 是旧 protocol v4 完整画像分支的冻结值，不是当前 `B_search`。只有恢复该
分支时，`support_pool_case_count` 才记录语义候选池规模；publication profile 必须满足
`measured_case_count < support_pool_case_count`、`measured_case_count <= 80` 和
`holdout_case_count >= 16`。这三个约束用于保证硬件画像服务于减少搜索成本，而不是形成
另一种全量穷举。

其 CPU/VTA key 只能是 `conv2d/elementwise/dense/pool_reduction/concat_copy/resize` 等通用
primitive 和数值参数，不得包含模型名、层名、候选 ID 或候选 throughput。

## Layer D: PartitionWorkload

新 DNN frontend 输出：

```text
StageWorkload[]: device, threads, backend-fused OperatorWorkload[]
OperatorWorkload: primitive, fusion group, schedule, logical/physical OP,
                    total physical DMA bytes/calls, fixed-tile SRAM peaks,
                    explanatory spill bytes/calls, resource accounting IDs
BoundaryWorkload[]: true DAG endpoints, per-edge queue depth, transfer resource,
                    producer/transport/consumer dtype+layout, quantization,
                    padding/slice, adapter, physical bytes and accounting IDs
```

`graph_id/candidate_id` 仅是 identity；`prediction_payload()` 会删除 identity metadata 和实测
标签。workload 必须携带 hashed static boundary validation；硬件 fingerprint、compiler hash、
schedule ID 或校准数值域不匹配时拒绝预测。静态 contract 通过不等于 runtime correctness
通过；新预测 record 标记 `runtime_correctness_status=not_measured`。

## BoundaryService

每条真实 graph edge 单独形成 boundary actor：

```text
CPU-to-CPU materialization:
  service = adapter + bytes / calibrated_cpu_memory_bandwidth + transaction overhead

heterogeneous stage_dma edge:
  service = host pack/requantize/unpack adapter only
  PS-PL transfer = lowered VTA LOAD/STORE service owned by adjacent VTA stage
```

完整 adapter lookup key 还包含 `access_kind`，因此连续、strided 和 padded host conversion 不会
共享同一服务点。CPU-to-CPU materialized edge 的 bytes 至少覆盖 physical tensor，`bytes/calls` 必须与平均
transaction size 一致。异构 `stage_dma` edge 仍记录 physical tensor schema/bytes 供审计，但
boundary 的 `call_count/avg_bytes_per_call` 必须为 0，PS-PL calls 由 lowered VTA operator 提供。
`identity` adapter 不允许暗中改变 dtype/layout；dtype 改变必须携带 quantization，channel padding
必须携带 slice policy。

每笔物理工作还必须带按资源命名的 accounting ID。partition validator 在 `cpu_memory`、
`ps_pl_load`、`ps_pl_store` 和 `bridge_pack_unpack` 各域检查全局唯一性。`stage_dma` boundary 禁止
持有 `ps_pl_load/store` ID；VTA operator 的 total physical DMA 已包含 fixed-schedule spill，因而
不存在第二个 spill accounting domain。

## Shape-Aware Service Surface

schema v3 禁止 publication profile 使用单个 `primitive + threads -> GOP/s` 常数。CPU/VTA fused
unit 和 CPU memory 参数必须保存为冻结 schedule 下的数值 support points，并声明 feature names、
支持域和插值方法。当前实现使用 `normalized_inverse_distance_v1`：只在 support box 内插值，越界
直接拒绝，不静默外推。VTA load/compute/store overlap 不由 Stage 4 单 kernel 猜测；只有 Stage 5
受控 pipeline 验证后的 `system_overlap_models` 才能进入最终 stage service。
