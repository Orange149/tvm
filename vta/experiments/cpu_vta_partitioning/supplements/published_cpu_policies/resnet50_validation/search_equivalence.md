# Exact search contracts

This implementation optimizes the existing integer static score. It does not
claim that the score is a bound on Linux execution or measured throughput.

## Future-equivalent labels

A state fixes the covered position, next device, used VTA-island count, resource
context, gate, verified position-specific frontier/layout interface, and (in
serial mode) the set of CPU groups already used. The group set is explicit:
zero-service groups still constrain future compatibility.

The complete-model label stores CPU demand on each core, total host CPU demand,
VTA service sum, DDR proxy sum, CPU-stage service maximum, per-core team service,
and service sums for every exact serial group. The baseline separately stores
physical single-core wall-service sums, VTA sum and DDR sum. No complete-model
CPU capacity is silently included in the baseline.

For identical states and identical integer resource labels, every legal common
suffix has identical legality, resource updates and final score. Alternating
device order and island count fix the number of prefix stages. Thus canonical
prefix order is preserved under appending a common suffix. Keeping the first K
distinct prefixes per exact label is sufficient for the global canonical Top-K.
Different resource labels with the same current scalar score are not merged.

## Safe lower bounds

All accumulated resources are nonnegative. Relaxed suffix minima are computed
independently for VTA service, host work, DDR, CPU-stage maximum and total CPU
work. Different minima may come from different suffixes; this weakens the bound
and cannot overestimate a feasible completion. The full frozen score is a
monotone max of the relevant resource expressions.

A state or transition is removed only when its bound is **strictly greater**
than the current Kth complete feasible score. Equal-score states remain. There
is no single-prefix Pareto deletion, score-only K-state beam, resource bucket,
or restricted resource family.

## Algebraic hot-loop simplification for the four-on model

Let `R_host = max_c(D[c] + H * 1[c in M])`, where all quantities are integers
and nonnegative. Every subset capacity is at most `R_host`:

- If `M` is not contained in subset `S`, its capacity is the ceiling of the
  average of `D[c]` over `S`, at most `max_c D[c] <= R_host`.
- If `M` is contained in `S`, then
  `sum(D[S]) + H <= sum(D[c] + H * 1[c in M] for c in S)`, because `M` is
  nonempty. Its ceiling average is again at most the integer `R_host`.

`R_team = max_c sum_{i:c in A_i} s_i` dominates the largest CPU-stage service
and each exact serial-group service sum: choose any core of the relevant
nonempty stage/group. Therefore the final complete/four-on score is exactly
`max(VTA_sum, DDR_sum, R_host, R_team)`.

The label still retains the original resource fields. This simplification is
not used for any model with guards disabled. Original Python scalar scoring
and the independent C++ verifier still calculate the original full expression.
Random nonnegative workloads spanning all nine contexts test the identity.

## Deterministic feasible-bound initialization

Before the full DP, all qualified one-VTA-island three-stage paths are scored
using the **current isolated inputs**, across the same requested contexts and
execution modes. Their best K complete paths provide a valid initial upper
threshold. Every path remains subject to the subsequent full-DAG certificate.
This initialization does not restrict the domain, use historical rankings or
FPS, or introduce simulated annealing. Its evaluated count and wall time are
reported separately and included in the search budget.

The ResNet18 cold acceptance freezes the new search result before opening the
historical static lists for comparison. The earlier no-initializer budget
failure and historical-perfect-upper-bound run remain separate development
records, not evidence that unseen ResNet50 search will take the same time.

## Verification boundary

Small full domains are checked against the existing independent C++ exhaustive
oracle. ResNet18 is re-executed by the new DP against the original full-domain
ordered Top-20. Selected paths are also checked by an independent integer C++
scorer, including largest-remainder CPU allocation using 128-bit products.
ResNet50 selected-path checks are not a second exhaustive global Top-K proof.

Engineering-budget interruptions produce an explicitly incomplete result and
streamed, integrity-checked JSON checkpoint. Partial rankings never authorize
the pipeline measurement campaign.
