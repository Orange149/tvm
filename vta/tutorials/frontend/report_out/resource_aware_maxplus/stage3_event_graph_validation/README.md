# RAMPS Stage 3 Event-Graph Validation

This report validates deterministic mechanisms before board calibration. It does not
constitute hardware accuracy evidence.

| Case | Expected | Analytical | Karp | Critical resources | Pass |
|---|---:|---:|---:|---|---|
| `balanced_cpu_vta_cpu` | 10.000 | 10.000 | 10.000 | `fifo_backpressure, finite_buffers, stage_worker, vta_mutex` | yes |
| `two_vta_islands_mutex` | 16.000 | 16.000 | 16.000 | `vta_mutex` | yes |
| `independent_cpu_vta` | 11.000 | 11.000 | 11.000 | `stage_worker, vta_mutex` | yes |
| `boundary_communication_bottleneck` | 14.000 | 14.000 | 14.000 | `bridge` | yes |
| `ps_pl_load_co_critical` | 12.000 | 12.000 | 12.000 | `finite_buffers, ps_pl_load, stage_worker, vta_mutex` | yes |

## FIFO Interpretation

Reverse free-slot tokens encode producer backpressure. In this deterministic linear case, queue depth changes buffering but not the 20 ms physical bottleneck.

The marked graph contains forward zero-token data dependencies and reverse
free-slot channels carrying `queue_depth` tokens. Queue depth is frozen in the
current paper instance; it is not claimed as a searched optimization variable.

## Evidence Boundary

These synthetic checks establish implementation semantics and analytical/Karp
consistency. CPU contention magnitudes, VTA overlap, DMA bandwidth and boundary
service accuracy still require the no-fallback hardware calibration in stage 4.
