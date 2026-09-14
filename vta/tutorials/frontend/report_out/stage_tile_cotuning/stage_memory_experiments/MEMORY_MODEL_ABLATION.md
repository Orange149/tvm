# M0/M1/M2 stage-memory search ablation

**2026-09-08 closeout qualification (numeric results unchanged):** M1/M2 use the
original set/get-copy boundary-service costs, not recalibrated shared-K2 costs.
An inactive modeled DDR bandwidth lower bound does not establish that physical
DDR contention is absent. DMA descriptors alone are not implemented safe pruning.

The complete search covers 972,528 execution configurations and 4,623 topologies. No
complete-topology FPS is used to fit a coefficient. M2 reuses the qualified component
bandwidth and replaces the older GOP traffic proxy with exact incumbent-tile TIR DMA.

| model | measured A/B/C/D predicted order | Spearman | regret@1 | evals to 95% oracle | Top-20 topologies |
|---|---|---:|---:|---:|---:|
| M0 | `B>C>A>D` | -0.400 | 7.39% | 2 | 1 |
| M1 | `A>B>C>D` | 0.400 | 0.00% | 1 | 4 |
| M2 | `A>B>C>D` | 0.400 | 0.00% | 1 | 4 |

Actual third-boot median order is `A>C>D>B`. M1/M2 Top-20 overlap is 20/20; identical: `True`.
M2 becomes the predicted bottleneck for 0/972,528 configurations.
The closest case reaches 17.27% of its M1 resource bound, so the qualified DDR lower
bound is not merely absent from Top-20; it is non-critical throughout this search space.
Per-workload reuse reduces the same 8-trial neighborhood from a hypothetical 312 stage-
placement trials to 80 unique-workload trials (74.36% reduction; 232 avoided).

## Decision

- M2 does not receive a fitted DDR coefficient and does not double-count VTA LOAD/STORE.
- If M2 leaves the natural Top-20 unchanged, this is a valid negative ranking result:
  the modeled bandwidth lower bound is non-critical; actual DDR contention remains unproven.
- Exact DMA remains useful before Top-20: it exposes endpoint tradeoffs, supplies a shared
  resource lower bound, and describes fragmentation/repeated-transfer tradeoffs; it does not alone justify rejecting a tile.
- The measured validation contains only four frozen topology representatives. Spearman and
  regret are reported descriptively, not as a fitted generalization claim.
