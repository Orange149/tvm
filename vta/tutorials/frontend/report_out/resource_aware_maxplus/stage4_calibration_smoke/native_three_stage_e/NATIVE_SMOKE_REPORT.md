# RAMPS Native Instrumentation Smoke

- Performance path: native static package and board-side C++ runner.
- RPC role: persisted all-VTA correctness reference only.
- Pipeline throughput: `10.0629 fps` (`99.375 ms` cycle).
- Serial/pipeline top1 match: `True`; top1: `[285]`.

| Stage | Device | Serial wall ms | Process CPU ms | Average cores | Pipeline wall ms |
|---|---|---:|---:|---:|---:|
| `stage0_cpu` | `cpu` | 80.022 | 184.284 | 2.303 | 103.885 |
| `stage1_vta` | `vta` | 71.337 | 19.278 | 0.270 | 72.276 |
| `stage2_cpu` | `cpu` | 23.831 | 76.295 | 3.202 | 39.595 |

The CPU values demonstrate why `wall_time * requested_threads` is invalid. The
measured average utilization is about 2.30/3 cores for stage0 and 3.20/4 cores
for stage2 rather than exactly the requested thread count.

This smoke validates instrumentation only. Formal calibration still requires
isolated native buckets, three sessions, warmup 5, 20 measured runs and no
fallback schedule or estimated field.
