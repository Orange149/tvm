# P7R114 USMP 式生命周期图可行性审计

## 结论

状态：`insufficient_evidence_fail_closed`。现有冻结证据不足以构建跨 graph storage、pipeline slot、
command queue、FINISH 与 replay 的统一 birth-death conflict graph。因此本轮按预注册规则
fail closed：不生成 lifetime packing 数字，也不把聚合高水位改名为 naive sum。

现有数据能证明的是：帧/阶段时间区间、边界 slot 的地址/代际/大小、每进程一个被观测队列的
submit 大小峰值、静态 storage/alias/reader/writer 关系，以及若干阶段性的聚合内存高水位。
它们尚未由统一 resource_id、物理 byte range 和 device completion 事件串起来。

## 可保留的局部观测

| topology | frames | queue records | observed queue IDs | max insn B | max UOP B |
|---|---:|---:|---:|---:|---:|
| A | 32 | 510 | 1 | 5312 | 2792 |
| B | 32 | 476 | 1 | 5312 | 2792 |
| C | 32 | 510 | 1 | 5312 | 2792 |
| D | 32 | 510 | 1 | 5312 | 2792 |

这些峰值只是已有单进程 queue trace 的 component maxima；不是对齐后的容量，不代表多个队列，
也不是 data/control 统一 lifetime packing 结果。

## 三种比较的状态

- naive sum：`not_computable`。snapshot 只有 aggregate high-water/count，没有逐资源 ledger；
  high-water 不是 logical-size naive sum。
- per-queue max：`partially_supported_existing_scope_only`。A/B/C/D 各只观测到一个 queue_id，
  可以复核该队列的 insn/UOP submit 峰值，但无法同 graph/slot/replay 资源统一比较。
- lifetime packing：`not_computable`。缺少完整 birth/death/access/device-done 与
  static-to-runtime physical mapping。

## 阻塞缺口

- `A:allocation_and_free_event_clock`
- `A:device_completion_event_separate_from_blocking_duration`
- `A:finish_decoded_and_observed_per_submit`
- `A:internal_tensor_access_events`
- `A:per_allocation_identity_address_size`
- `A:queue_backing_resource_and_physical_range`
- `A:queue_to_frame_stage_invocation_link`
- `A:replay_retained_resource_lifetime`
- `A:static_to_runtime_resource_mapping`
- `B:allocation_and_free_event_clock`
- `B:device_completion_event_separate_from_blocking_duration`
- `B:finish_decoded_and_observed_per_submit`
- `B:internal_tensor_access_events`
- `B:per_allocation_identity_address_size`
- `B:queue_backing_resource_and_physical_range`
- `B:queue_to_frame_stage_invocation_link`
- `B:replay_retained_resource_lifetime`
- `B:static_to_runtime_resource_mapping`
- `C:allocation_and_free_event_clock`
- `C:device_completion_event_separate_from_blocking_duration`
- `C:finish_decoded_and_observed_per_submit`
- `C:internal_tensor_access_events`
- `C:per_allocation_identity_address_size`
- `C:queue_backing_resource_and_physical_range`
- `C:queue_to_frame_stage_invocation_link`
- `C:replay_retained_resource_lifetime`
- `C:static_to_runtime_resource_mapping`
- `D:allocation_and_free_event_clock`
- `D:device_completion_event_separate_from_blocking_duration`
- `D:finish_decoded_and_observed_per_submit`
- `D:internal_tensor_access_events`
- `D:per_allocation_identity_address_size`
- `D:queue_backing_resource_and_physical_range`
- `D:queue_to_frame_stage_invocation_link`
- `D:replay_retained_resource_lifetime`
- `D:static_to_runtime_resource_mapping`
- `profile:bounded_tail_capture`
- `profile:resource_physical_range`
- `profile:unified_correlation_fields`
- `replay:execution_and_retained_resource_lifetime`

## FINISH 与 replay

- FINISH：runtime 源码确实在正常提交前追加并检查最后一条 FINISH，已有证书也据此推导数量；
  但冻结的 `[VTA_QUEUE]` JSON 没有从每批实际提交字节流解码得到 FINISH 数量/末位置，故不能作为
  生命周期图的逐批观测证据。
- replay：P7R107/P7R108 对同一 manifest 证明 `replay_policy=disabled`，capture/replay 调用均被
  拒绝。这只支持该 exact manifest 的 fail-closed 安全主张；没有 replay execution 或 retained
  resource lifetime 证据。

## 下一步最小埋点

完整字段、事件类型、失败条件与建议埋点位置见 `instrumentation_contract.json`。最小资格运行先只做
一个冻结 topology/manifest：warmup 后覆盖至少 `pipeline_depth + 2` 帧及完整 drain，trace 不限长，
要求事件序号无缺口、零 dropped event。任何字段缺失、FINISH 未逐批解码或 device_done 未独立记录，
都继续 fail closed。

## 主张边界

offline feasibility audit of named frozen artifacts only; no board contact, no new memory measurement, no USMP implementation, no unified peak-memory reduction, and no performance claim。
