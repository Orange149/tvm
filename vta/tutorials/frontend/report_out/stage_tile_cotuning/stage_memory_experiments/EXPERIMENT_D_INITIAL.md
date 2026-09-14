# Experiment D: initial shared-DDR topology table

This table re-analyzes the four frozen Top-20 topology representatives. It is a one-boot
mechanism study, not a fitted contention model or a cross-boot significance claim.
Per-stage values are medians over frames, while cycle uses the first/last completion span;
their maximum is a diagnostic proxy and is not treated as a strict finite-sample lower bound.

| topology | islands | FPS | cycle | max CPU stage | VTA run sum | framework copy | VTA DMA | mutex wait |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 1 | 10.872 | 91.977 ms | 91.292 ms | 70.042 ms | 0.903 MB | 13.255 MB | 0.001 ms |
| B | 1 | 10.310 | 96.990 ms | 102.134 ms | 68.788 ms | 1.104 MB | 13.010 MB | 0.001 ms |
| C | 1 | 10.994 | 90.962 ms | 91.797 ms | 70.329 ms | 1.004 MB | 13.255 MB | 0.001 ms |
| D | 2 | 10.501 | 95.233 ms | 95.441 ms | 69.811 ms | 1.305 MB | 16.445 MB | 20.337 ms |

## Controlled contrasts

- A and C have exactly equal VTA LOAD calls, LOAD bytes, and STORE bytes. C nevertheless
  adds +100352 framework-copy bytes and changes cycle by -1.015 ms. Their different CPU tail
  placement means this pair proves DMA invariance, not a per-byte latency coefficient.
- D versus B adds one VTA island: framework-copy bytes change +18.18%, LOAD calls +4.05%,
  LOAD bytes +29.30%, and weight LOAD bytes +49.61%. Mutex wait rises +20.336 ms, while
  measured cycle changes -1.757 ms. The improved CPU partition can hide or outweigh higher
  memory demand, so a scalar 'more bytes = lower FPS' rule is invalid.

## Model decision

The outer search must use a resource maximum, not add every cost to the critical path.
Per-workload DMA is statically aggregated after tile selection. Boundary bytes/copy mode,
VTA re-entry, single-VTA service, maximum CPU-stage service, and DMA calls/weight payload
remain separate features. No coefficients are fit to these four confounded samples.
Cross-boot repetitions and a larger calibration/holdout set are still required before M2
may claim better ranking accuracy.
