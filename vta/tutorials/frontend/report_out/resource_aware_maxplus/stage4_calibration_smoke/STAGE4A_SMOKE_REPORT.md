# RAMPS Stage 4a Native Calibration Smoke

## Result

Stage 4a used static native packages on `root@192.168.1.105`. RPC was not used for any
performance measurement. The board was `operating`, `udmabuf0` was `201326592` bytes, and all
raw outputs were finite and stable.

Two native checks completed:

1. Existing ResNet18 `three_stage_e` package: serial and pipeline both completed for 8/8 inputs,
   top1 was consistently 285, and pipeline throughput was 10.063 FPS after skipping three frames.
2. Synthetic residual-3x3 bucket package: one CPU stage and one VTA stage completed 8/8 native
   serial runs and exposed set/run/get, process CPU time, and VTA DMA profiler counters.

## Measured Service Data

| Bucket | Device | set ms | run ms | get ms | wall ms | process CPU ms | effective GOP/s |
|---|---|---:|---:|---:|---:|---:|---:|
| `residual_3x3_conv` | CPU | 0.354 | 46.508 | 0.886 | 47.715 | 47.704 | 4.971 |
| `conv3x3_c_small` | VTA | 2.427 | 9.462 | 2.910 | 14.802 | 9.265 | 24.435 |

The one-thread CPU bucket consumed 47.704 core-ms over 47.715 wall-ms, validating the process CPU
clock instrumentation. In `three_stage_e`, the three-thread CPU prefix consumed 184.284 core-ms
over 80.022 wall-ms, or 2.303 average cores. The old approximation `wall_ms * requested_threads`
would report 240.066 core-ms and overestimate demand by about 30.3%.

The VTA bucket generated 747,264 load bytes in 126 calls and 200,704 store bytes in 14 calls per
frame. Total VTA device wait was 6.306 ms per frame.

## DMA Identification

The runtime profiler records only one `device_run_wait_us` for a complete instruction stream, so a
single conv cannot identify independent load/store bandwidth. A dedicated native packed-function
runner and a 24-case identification matrix were added. The matrix independently varies load bytes,
store bytes, load calls, store calls, tensor size, stride and padding. All 24 cases passed output
correctness.

Under a busy-poll isolation protocol, the contiguous design matrix has rank `5/5`, condition number
`19.584`, constrained fit `R2=0.997`, and MAE `0.657 us`. The fitted diagnostic bandwidths are
`1.425 GB/s` load and `0.859 GB/s` store. Strided and padded accesses add median device time of
`6.10 us` and `7.83 us` relative to the contiguous model for the measured cases.

The default driver sleep experiment does not fit a simple additive model (`R2=0.404`). This is an
expected mechanism: host synchronization behaves approximately as
`max(device_completion, first_poll_time)`, rather than `device_time + fixed_sleep`. RAMPS must keep
device DMA service and host poll/sync policy as separate event constraints.

The DMA smoke matrix is ready, but Stage 4 still does **not** pass the publication calibration gate.
The measured values are one-session diagnostics; all CPU/VTA/DMA/bridge buckets still require three
independent sessions and uncertainty aggregation.

## Evidence Status

- Native static execution: passed.
- Correctness and raw stability: passed.
- CPU core-demand timing: passed.
- VTA DMA byte/call accounting: passed.
- Independent contiguous load/store identification: passed in busy-poll isolation.
- Strided/padded profiler coverage: passed.
- Three-session uncertainty and complete bucket coverage: pending.
- Formal `service_model.json`: not generated.

Detailed artifacts:

- [native bucket report](native_bucket_residual3x3/NATIVE_BUCKET_SMOKE_REPORT.md)
- [native bucket JSON](native_bucket_residual3x3/native_bucket_smoke_summary.json)
- [ResNet18 instrumentation report](native_three_stage_e/NATIVE_SMOKE_REPORT.md)
- [hardware fingerprint](hardware_fingerprint_192.168.1.105.json)
- [native DMA identification report](native_vta_dma_identification_busy_poll/DMA_IDENTIFICATION_REPORT.md)
- [native DMA identification JSON](native_vta_dma_identification_busy_poll/dma_identification_summary.json)
