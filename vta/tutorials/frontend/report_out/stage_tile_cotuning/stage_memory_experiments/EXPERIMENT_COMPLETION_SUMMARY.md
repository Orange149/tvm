# Stage--tile memory experiment completion summary

Date: 2026-09-07. The core experiment phase is complete. Remaining work is paper
presentation and final artifact freezing, not another open-ended AutoTVM or board campaign.

## Completed evidence chain

| experiment | scope | result | decision |
|---|---|---|---|
| TopHub recovery | 10 VTA workloads | 10/10 incumbent hit, 0 fallback; rebuilt topology B recovered the historical binary/performance class | TopHub is the immutable inner-loop incumbent |
| bounded tile neighborhood | 80 board trials | 32 correct, 41 compile failures, 7 wrong answers, 0 safe >=2% replacements | no claim of beating AutoTVM; retain DMA laws and failures |
| static DMA extraction | 10 workloads | eight direct templates match runtime exactly; projection convolution traffic also matches, with only graph/ACC residual | tile-fixed logical DMA is available before full-topology deployment |
| experiment A | 4 topologies, 5 VTA stages, 39 workload-stage placements | every repeated workload selects the same TopHub config | tune 10 unique workloads once instead of every placement |
| experiment B | 5 complete stages | input, weight, and STORE bytes add exactly; LOAD-byte error is -0.12% to -0.16% | aggregate workload DMA and keep the small graph-level residual explicit |
| endpoint delta | VTA 03..15/16/17 | 15->16 adds 217600 B static conv LOAD and removes 100352 B boundary; 16->17 adds no VTA DMA and removes another 100352 B | distinguish DMA--boundary tradeoff from boundary-only dominance |
| experiment C | one executor / materialized split / shared split, 30 interleaved trials | 20.327/22.929/22.065 ms; shared saves 1.290 ms versus materialized; all VTA DMA counters identical | the tested block cut adds framework handoff, not DMA fragmentation |
| experiment D | four frozen topologies, three independent boots | output hashes and per-frame DMA are invariant; fine sub-FPS ordering varies | use memory mechanisms and coarse selection, not a fitted topology DDR coefficient |
| M0/M1/M2 | 972528 configurations, 4623 topologies | Spearman -0.400/0.400/0.400; regret@1 7.39%/0/0; M1/M2 Top-20 identical | use M1 for final ranking; retain M2 exact DMA for deltas and pruning evidence |

## Final ablation interpretation

M0 contains compute-only CPU, serialized-VTA, and four-core CPU resource bounds. M1 adds
CPU--VTA boundary ownership, boundary host work, and VTA-mutex boundary work. On the four
frozen representatives, M0 predicts `B>C>A>D`; M1 predicts `A>B>C>D` and selects the actual
third-boot median Top-1 A. M0's Top-20 contains one topology, while M1 contains four.

M2 replaces the earlier GOP traffic proxy with incumbent-tile TIR DMA for all 87 legal VTA
segments and uses independently qualified component bandwidth as a parallel shared-DDR lower
bound. It does not fit the four topology outcomes and does not add DMA time to VTA service.
M2 is the maximum resource for 0/972528 candidates; its closest case reaches only 17.27% of
the M1 bound. Consequently M2 cannot improve the natural Top-20, and the pre-registered
fallback to M1 is taken. This is a negative ranking result, not a failed experiment.

## Search-budget result

The four frozen topologies contain 39 workload-stage placements. Repeating the same eight
tile trials per placement would require 312 board measurements. Workload/config invariance
reduces this to 10 unique workloads and 80 measurements: 232 trials avoided, or 74.36%.

## Claims supported

- Stage partitioning changes shared-memory live tensors, boundary materialization, VTA-island
  re-entry, serialized device pressure, and CPU/VTA resource balance even when tuned workload
  DMA is nearly additive.
- Incumbent-tile DMA request count, payload, fragmentation, and repeated input/weight loads can
  be extracted statically and fed back to endpoint comparison without profiling every topology.
- Boundary-aware M1 materially improves Top-1 selection on the frozen four-topology validation;
  exact DMA M2 is off the critical path on this ResNet18/AXU5EVB instance.
- The final method is a bounded hierarchical stage--tile search, not a claim to outperform
  TopHub or jointly enumerate the AutoTVM space for every stage.

## Claims not supported

- Runtime DMA requests are not physical AXI bursts.
- The current hardware does not expose compute-stall cycles, so LOAD/STORE wait cycles are not
  claimed.
- No universal scalar DDR penalty or statistically significant sub-1-FPS topology advantage is
  claimed from four mixed topologies.
- Cross-operator SRAM residency, hardware prefetch, and hardware pre-eviction were not
  implemented and are outside the one-month completion boundary.
