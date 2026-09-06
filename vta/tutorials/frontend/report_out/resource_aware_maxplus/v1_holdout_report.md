# V1-P2 Holdout Report

- Native performance only; RPC performance used: `false`.
- Stage cases: `7`; DDR cases: `12`; candidate throughput evaluations: `0`.
- Independent MXNet reference top1: `282`.
- Exact top1 matches: `1/6`; remaining outputs use the preregistered ImageNet cat-equivalent policy `[281, 282, 283, 284, 285]`.
- This is a semantic/determinism gate, not numerical tensor equivalence; P3 shortlist must pass the native numerical reference.

## Gate

- `independent_reference_class_equivalent`: `true`
- `deterministic_output_hash`: `true`
- `serial_pipeline_equivalence`: `true`
- `native_vta_profiler_valid`: `true`
- `cpu_thread_control_effective`: `true`
- `serial_profile_masks_match_current_policy`: `true`
- `memory_correctness`: `true`
- `historical_affinity_holdout_residual_within_20_percent`: `true`
- `historical_affinity_holdout_bottleneck_preserved`: `true`

## Grouped Holdout

`three_stage_b` fit-only max-load prediction: `227.916 ms`; measured pipeline cycle: `222.580 ms`; residual: `-2.34%`.

This holdout used `serial_reuse_or_pipeline_contiguous_disjoint_v1` and matches the current overlapping-mask policy: `false`. Its residual is a historical-policy diagnostic and is not evidence for current-policy CPU contention. The frozen additive logical-op/core-work model prices all 172 reachable segments; P3 must still compile every shortlisted segment before board execution.
