# P7 grouped-holdout protocol addendum

Frozen on 2026-09-11 before any P7 board correctness or latency label was observed.

## Scope decisions

- The current-boot G2 recovery gate passed: TopHub correctness was 10/10 and the three-repeat topology-B median was 10.4511 FPS, within 5% of 10.724 FPS.
- The formal grouped holdouts remain W01/W04/W07/W08. Their complete common qualified pools contain 19/19/21/21 candidates, 80 unique candidates total. The pool will be measured completely before a pool-oracle claim.
- E00/E01/E02 do not enter this run because no frozen generation/lower/FSim/AXU qualification exists. They are not replaced after observing grouped-holdout labels.
- Mode 4 (`weight_stationary_barrier`) is admitted only as a separately identified experimental candidate family. Its old “not production” warning remains valid; admission to measurement is not production promotion.
- B1 is the statically selected `paper_inspired_hybrid` candidate and is not called an exact 2026 reproduction.

## Fair comparison

All B0--B8 policies use the same 80-candidate physical label pool. Duplicate choices share one measurement; failures still consume their positions in gross dispatch budgets. Random uses 1,000 deterministic hash-key permutations per workload. The XGB policy is a sequential ConfigEntity-only pool tuner with four deterministic random warmup trials; its executable implementation must be frozen before timing starts.

The proposed B8 rule gives equal one-third weight to three groups: DMA volume/request shape, command footprint, and compute structure. Compute structure minimizes outer tile waves while maximizing inner tile work and virtual-thread product. Each field is converted to a normalized distinct-value rank before group averaging, so byte-scale fields cannot numerically dominate count or compute fields. B0 is always protected at position one.

Because B0 protection makes “reach B0 within 2%” trivially one dispatch for protected methods, the final analysis must also report pool-oracle Regret@budget, time to the best achieved by B2/B3, and incremental exploration dispatches after the shared B0. Gross dispatch remains the primary accounting denominator.

## Atomic board order

1. Freeze and hash the full contract and all policy orders.
2. Run one correctness-only workload per immutable run, in order W01, W04, W07, W08; all three frozen seeds must match exactly.
3. Stop on the first wrong answer, timeout, RPC/device error, boot change, or new storage error; do not replace the candidate.
4. Only after all 80 candidates pass may a timing runner be frozen and executed with three warmups and five complete randomized blocks per workload. No timing is collected by correctness runs.

This addendum cannot be edited in response to P7 results. A later correction must be a new dated addendum and invalidates affected confirmatory claims.
