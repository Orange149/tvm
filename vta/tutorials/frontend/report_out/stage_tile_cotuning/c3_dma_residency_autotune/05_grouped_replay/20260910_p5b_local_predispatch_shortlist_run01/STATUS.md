# C3 P5b local pre-dispatch shortlist status

Status: **frozen_label_free_shortlist**. This run used no SSH, RPC, board, latency label, correctness label, or regret calculation.

The frozen P4b pool contains 250 unique candidate IDs across 10 workloads. Static lowering succeeded for 165 candidates; those candidates form the only board-eligible set. The other 85 records are retained in `filtered_failures.json` and excluded from every shortlist: 25 `lower:dma_pad_innermost`, 24 `lower:allocation_capacity`, 22 `lower:dma_2d_pattern`, and 14 `lower:dma_compact_buffer`. This separates local compile/screening cost from future board-dispatch cost; the latter remains zero in this run.

B4--B7 are frozen at requested budgets 4, 8, and 16, counted as unique candidate IDs within each workload/strategy pair. Every prefix preserves the successful protected original incumbent. B7 budget 4 contains one candidate from each of `original`, `input_stationary`, `weight_stationary`, and `paper_inspired_hybrid` for all 10 workloads. This diversity is an exploratory shortlist policy, not evidence that weight or hybrid mode improves performance. Because some workloads have fewer than 16 lower-successful candidates, effective budget 16 is 12 for W02, 15 for W00/W01/W03--W06, and 16 for W07--W09.

The feature contract is `c3_static_request_shape_v1`, and the frozen feature-set SHA-256 is `ee48faf94f166074777dfae0e22f157062ef0c7d34efd1876972c394e3cbf0c7`. The request-shape ranking includes static bytes/calls, small/strided/padded counts, reload ratios, per-memory maximum request sizes, and exact request histograms (with derived size diversity/minimum).

This is a local pre-dispatch planning artifact only. A successful lower is not a board correctness or latency result. This run does not evaluate strategy effectiveness, regret, or pass G5.
