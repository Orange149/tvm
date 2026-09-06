# Stage 4b Native Calibration Report

> **Legacy diagnostic only.** This ResNet-shaped matrix is retained for reproduction.
> It is not a schema-v2 portable hardware profile and cannot pass the publication gate.

## Status

- Formal sessions: 5 independent board boots.
- Protocol SHA256: `f8c7bb0d90fa468d1cba0872d2b0516cb7c580339c9e5c96e9012bc00c609a94`.
- Native measured samples: `19300`.
- RPC performance measurements: none.
- Fallback or `estimated_from`: none.
- Family case counts: `{"bridge": 15, "cpu_memory": 36, "cpu_operator": 80, "dma": 36, "submit_sync": 6, "vta_operator": 20}`.
- Cases exceeding 10% CV or relative CI half-width: `21`.
- Cases failing deterministic output-signature correctness: `6`.

## Publication Gate

Stage 4b has complete process logs, but its correctness/precision gates are not satisfied.
The current model must not be used as the final publication model.
See `calibration_uncertainty.json` and `stage4b_precision_diagnostics.csv`.
The pre-registered five-session precision extension is exhausted. Do not add more
sessions or relax the threshold without defining a new diagnostic protocol first.

## Precision Diagnostics

| Family | Case | Metric | CV | Relative CI Half-Width | Session Medians |
|---|---|---|---:|---:|---|
| `bridge` | `bridge_cpu_to_vta_1048576` | `run_ms` | 14.03% | 18.07% | `[6.5109200000000005, 8.113626, 7.056910500000001, 9.366259, 7.859674]` |
| `bridge` | `bridge_cpu_to_vta_16384` | `run_ms` | 15.04% | 19.23% | `[0.0126005, 0.012105000000000001, 0.0138, 0.017145, 0.012584999999999999]` |
| `bridge` | `bridge_cpu_to_vta_262144` | `run_ms` | 15.26% | 19.69% | `[1.4437795, 1.198407, 1.0489905, 1.0517954999999999, 1.0250949999999999]` |
| `bridge` | `bridge_vta_to_cpu_1048576` | `run_ms` | 11.94% | 12.84% | `[2.7087125, 3.3201330000000002, 2.7229475, 3.4303395, 2.7535925]` |
| `cpu_operator` | `cpu_add_relu_tail_elementwise_p0_t2` | `run_ms` | 19.81% | 24.27% | `[0.304758, 0.455074, 0.306693, 0.303933, 0.309768]` |
| `cpu_operator` | `cpu_add_relu_tail_elementwise_p0_t3` | `run_ms` | 18.94% | 17.03% | `[0.366709, 0.38776900000000003, 0.2644225, 0.2588425, 0.368974]` |
| `cpu_operator` | `cpu_add_relu_tail_elementwise_p0_t4` | `run_ms` | 15.18% | 13.52% | `[0.332823, 0.32226299999999997, 0.3271085, 0.247187, 0.2448325]` |
| `cpu_operator` | `cpu_add_relu_tail_elementwise_p1_t3` | `run_ms` | 23.20% | 27.48% | `[0.12334600000000001, 0.193742, 0.12804100000000002, 0.12237200000000001, 0.1809015]` |
| `cpu_operator` | `cpu_add_relu_tail_elementwise_p2_t3` | `run_ms` | 29.11% | 25.69% | `[0.09315100000000001, 0.103801, 0.054991, 0.055636000000000005, 0.095791]` |
| `cpu_operator` | `cpu_add_relu_tail_elementwise_p2_t4` | `run_ms` | 18.23% | 20.54% | `[0.08568100000000001, 0.091501, 0.055830500000000005, 0.086146, 0.0915455]` |
| `cpu_operator` | `cpu_add_relu_tail_elementwise_p3_t3` | `run_ms` | 40.00% | 54.76% | `[0.04575, 0.096541, 0.0461255, 0.0466055, 0.0463505]` |
| `cpu_operator` | `cpu_add_relu_tail_elementwise_p3_t4` | `run_ms` | 24.64% | 27.71% | `[0.0778355, 0.070966, 0.038475999999999996, 0.078871, 0.072196]` |
| `cpu_operator` | `cpu_head_pool_dense_p0_t3` | `run_ms` | 20.88% | 25.99% | `[0.12838650000000001, 0.195332, 0.128746, 0.129901, 0.128701]` |
| `cpu_operator` | `cpu_head_pool_dense_p0_t4` | `run_ms` | 14.47% | 17.26% | `[0.172067, 0.171452, 0.171767, 0.1222215, 0.181952]` |
| `cpu_operator` | `cpu_head_pool_dense_p1_t3` | `run_ms` | 12.36% | 11.25% | `[0.4527795, 0.46291899999999997, 0.3609635, 0.3626585, 0.451925]` |
| `cpu_operator` | `cpu_head_pool_dense_p1_t4` | `run_ms` | 9.02% | 10.44% | `[0.4352895, 0.423289, 0.433474, 0.346623, 0.42367900000000003]` |
| `cpu_operator` | `cpu_skip_proj_1x1_conv_p0_t4` | `run_ms` | 34.54% | 45.79% | `[2.97972, 3.0018149999999997, 2.9542795, 5.690097, 2.9604749999999997]` |
| `cpu_operator` | `cpu_skip_proj_1x1_conv_p2_t4` | `run_ms` | 38.38% | 48.87% | `[1.449435, 2.8505085, 1.4479045, 1.446209, 2.8648040000000004]` |
| `cpu_operator` | `cpu_skip_proj_1x1_conv_p3_t4` | `run_ms` | 32.27% | 30.20% | `[2.6575615, 2.1854465, 2.675547, 1.3545284999999998, 1.3560729999999999]` |
| `vta_operator` | `vta_conv1x1_p0` | `run_ms` | 17.44% | 14.46% | `[5.1391865, 5.0890260000000005, 3.679132, 3.678082, 5.1494465]` |
| `vta_operator` | `vta_conv3x3_c_small_p0` | `run_ms` | 16.73% | 17.46% | `[4.161072, 4.1607865, 5.600651, 5.608931, 4.151546]` |

## Correctness Diagnostics

- `vta_operator` / `vta_conv1x1_p0`: 41 distinct output signatures.
- `vta_operator` / `vta_conv1x1_p1`: 27 distinct output signatures.
- `vta_operator` / `vta_conv1x1_p2`: 63 distinct output signatures.
- `vta_operator` / `vta_skip_proj_p0`: 61 distinct output signatures.
- `vta_operator` / `vta_skip_proj_p1`: 22 distinct output signatures.
- `vta_operator` / `vta_skip_proj_p2`: 26 distinct output signatures.

## Accounting Boundary

The model preserves CPU process time, VTA stage phases, DMA bytes/calls, bridge adapter time,
and submit/sync observations as separate fields. `effective_stage_gops` is not used, and DMA
is not added a second time to a black-box VTA runtime.
