# P7R130 tile—DMA—qualification factor map

> Descriptive analysis of immutable local/board artifacts; no new FPGA candidate was dispatched.

## Coverage

| workload | identities | static pass | FSim pass | FPGA observed/pass | timed |
|---|---:|---:|---:|---:|---:|
| Y00 | 24 | 18 | 11 | 0/0 | 0 |
| Y01 | 24 | 24 | 24 | 23/0 | 0 |
| Y02 | 9 | 8 | 8 | 6/2 | 2 |

## What is now frozen for the search experiment

- Hard legality/correctness gates are separated from performance ranking.
- Ranking uses a simple label-free shared-memory Pareto rule; no complex model is retained without evidence.
- Same-tile mode diversity is preserved before repeatedly sampling one mode.
- TopHub is not exposed to the selector; the primary endpoint is trials/regret to a completed pool oracle.

## Claim boundary

The CSV files expose exact per-identity and paired deltas. One-knob pairs are controlled associations inside the sampled pool, not universal causal laws. Current-board W05 health failed before candidate dispatch, so this run adds no latency or FPGA-performance claim.
