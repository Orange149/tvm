# C3 P5-local historical replay status

Status: **complete as a retrospective development-data replay**. This run did not use SSH, RPC, or a board.

The frozen 80-point pool joined successfully: 10 workloads, 8 positions per workload, 32 passing measurements, 7 wrong answers, and 41 compile failures. All TopHub incumbents remain in every B2/B4/B5/B6 pool. Budgets are 4, 8, and 16; because each pool contains eight candidates, budget 16 has effective budget 8. B2 uses 1,000 deterministic random permutations per workload and budget. `results.jsonl` contains 120 aggregate records.

At budget 4, B2 finds a within-2% candidate in 42.29% of all workload/permutation cases (the denominator includes the two unscorable workloads). **B2 is the only evaluable baseline in this run.** B4 is retained only as a `diagnostic_oracle_upper_bound_leaky`: it puts candidates known after dispatch to be correct ahead of wrong answers. B5 and B6 are `not_evaluable_missing_prefeatures`: their DMA/request features are runtime observations available only for passing candidates. Their zero-regret diagnostic numbers are therefore forbidden as baseline-effectiveness results. The identical B4/B5/B6 aggregates demonstrate no incremental value from DMA or request-shape ordering.

## Available-case downgrade

- Full ConfigEntity and historical outcome: 80/80.
- Latency plus runtime DMA counters: 32/80, only archived `correct=true` artifacts.
- SRAM working-set estimate, lowered-TIR hash, exact request histogram, and maximum request size: 0/80.
- W03 and W06 have no passing candidate, so they have no latency oracle.
- B6 is a partial Pareto over available byte/call/small/strided/padded/reload counters, not the complete P4 feature contract.

This run is historical/development evidence only. It is neither grouped holdout nor prospective board validation and cannot by itself pass G5. Before any B4/B5/B6 effectiveness claim, their features must be generated without board outcomes for every candidate before dispatch.
