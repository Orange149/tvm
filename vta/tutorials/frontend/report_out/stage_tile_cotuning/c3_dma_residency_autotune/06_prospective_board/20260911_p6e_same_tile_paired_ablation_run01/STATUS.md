# C3-P6e same-tile paired ablation result

Status: **passed; residency mechanisms provide local fixed-configuration speedups**.

All 11 unique binaries passed all three exact board correctness seeds (33/33 checks). The run then completed all 360 planned samples: 30 control and 30 mechanism samples for each of six pairs. Every pair appeared in every sequence position five times, each side ran first 15 times, and every round showed the mechanism faster than its same-tile control.

| Workload/config | Mechanism | Control median ms | Mechanism median ms | Improvement | Frozen decision |
|---|---|---:|---:|---:|---|
| W00 / 252 | input-stationary | 6.923124 | 6.480230 | 6.40% | improved |
| W00 / 251 | weight-barrier | 7.062296 | 6.756443 | 4.33% | indeterminate |
| W02 / 177 | input-stationary | 7.576501 | 6.389389 | 15.67% | improved |
| W02 / 177 | weight-barrier | 7.568941 | 7.007755 | 7.41% | improved |
| W09 / 63 | input-stationary | 1.816428 | 1.126991 | 37.96% | improved |
| W09 / 82 | weight-barrier | 5.781612 | 2.264393 | 60.84% | improved |

The runtime counters give a coherent mechanism explanation. At fixed ConfigEntity, all six mechanisms reduced total LOAD bytes and LOAD calls. Examples:

- W02/config177 input-stationary: LOAD bytes 1,077,248 → 711,680; calls 256 → 64; median latency improves 15.7%.
- W09/config82 weight-barrier: LOAD bytes 964,096 → 177,664; calls 448 → 226; median latency improves 60.8%.
- W00/config251 weight-barrier: LOAD bytes 796,672 → 538,624 and all 30 paired rounds are faster, but its 4.33% median benefit remains below the preregistered 5% improvement threshold.

## Joint interpretation with P6d

P6d and P6e are not contradictory:

- P6e shows that changing only the residency mode at a fixed tile/thread configuration can reduce runtime DMA and speed up the operator.
- P6d shows that choosing configurations primarily for DMA reduction can still lose to the protected TopHub incumbent because the chosen tile/thread configuration may have worse compute utilization or scheduling overhead.

Therefore the useful thesis direction is not “minimize DMA bytes.” It is **hardware- and mechanism-aware joint autotuning**: expose residency as a scheduling choice, use SRAM/DMA/request/synchronization features to prune or rank candidates, retain compute-parallelism features and the TopHub incumbent, and validate the final choice by board latency.

## Claim boundary

This is a one-boot, three-workload, operator-level causal ablation. It supports real-hardware correctness, request reduction, and local same-configuration latency effects. It does not yet establish generic B7 search-efficiency, grouped holdout generalization, multi-boot stability, stage speedup, or end-to-end FPS gain.
