# Native Residual-3x3 Bucket Smoke

- Board: `root@192.168.1.105`
- Performance path: native static package; RPC performance measurement: `false`.
- Runs: `8`, skipped: `3`.
- Raw output stable/finite: `True/True`.

| Bucket | Device | set ms | run ms | get ms | wall ms | process CPU ms | GOP/s |
|---|---|---:|---:|---:|---:|---:|---:|
| `residual_3x3_conv` | CPU | 0.354 | 46.508 | 0.886 | 47.715 | 47.704 | 4.971 |
| `conv3x3_c_small` | VTA | 2.427 | 9.462 | 2.910 | 14.802 | 9.265 | 24.435 |

Per VTA frame: load `747264` B / `126` calls; store `200704` B / `14` calls; average `6771.2` B/call; total device wait `6.306` ms.

## Gate

Instrumentation passed, but the formal calibration gate remains **blocked**.
VTA profiler exposes total device wait and DMA bytes/calls but not separate load/store device service time; formal PS-PL bandwidth cannot divide both byte counters by the same compute-inclusive wait.
A DMA-only native microbenchmark or additional device-side DMA timestamps are required
before the three-session no-fallback calibration can begin.
