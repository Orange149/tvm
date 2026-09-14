# Latency-unseen Y01 recovery confirmation

> Candidate latency and service-proxy weights were frozen before this recovery run; historical correctness was already exposed.

## Pool and protocol

- Workloads: 1
- Candidates excluding sealed references: 6
- Measured-ok: 3
- Lower-invalid: 0
- Compile-invalid: 0
- FPGA-invalid: 3
- Policies: 10 (including paper-mode and cross-tile shared-memory ablations)
- Seeds: 20; gross budgets: [1, 2, 4, 6]
- Total workload-policy-seed runs: 200

## Simple regret versus the non-reference full-pool oracle

| policy | regret@1 median [IQR] % | regret@2 median [IQR] % | regret@4 median [IQR] % | regret@6 median [IQR] % |
|---|---:|---:|---:|---:|
| Random | 44.825151 [44.825151] | 0.000000 [44.825151] | 0.000000 [0.000000] | 0.000000 [0.000000] |
| stock-knob XGB | 291.498033 [493.345764] | 44.825151 [538.170915] | 0.000000 [44.825151] | 0.000000 [0.000000] |
| rules-only | n/a | n/a | 0.000000 [0.000000] | 0.000000 [0.000000] |
| paper minimum-access mode only | n/a | n/a | 22.412575 [168.161592] | 0.000000 [0.000000] |
| shared-memory lexicographic | n/a | n/a | 0.000000 [0.000000] | 0.000000 [0.000000] |
| shared-memory service proxy | n/a | 0.000000 [0.000000] | 0.000000 [0.000000] | 0.000000 [0.000000] |
| validity V | 44.825151 [403.628186] | 44.825151 [44.825151] | 0.000000 [0.000000] | 0.000000 [0.000000] |
| V + ΔT bytes/calls | 0.000000 [0.000000] | 0.000000 [0.000000] | 0.000000 [0.000000] | 0.000000 [0.000000] |
| V + ΔT +request | 0.000000 [0.000000] | 0.000000 [0.000000] | 0.000000 [0.000000] | 0.000000 [0.000000] |
| V + ΔT +command | 0.000000 [0.000000] | 0.000000 [0.000000] | 0.000000 [0.000000] | 0.000000 [0.000000] |

## Pool-oracle quality at budget 6

| policy | success@2% | success@5% | median trials-to-2% | median trials-to-5% |
|---|---:|---:|---:|---:|
| Random | 1.000 | 1.000 | 3.000 | 3.000 |
| stock-knob XGB | 1.000 | 1.000 | 3.500 | 3.500 |
| rules-only | 1.000 | 1.000 | 3.000 | 3.000 |
| paper minimum-access mode only | 1.000 | 1.000 | 4.500 | 4.500 |
| shared-memory lexicographic | 1.000 | 1.000 | 3.000 | 3.000 |
| shared-memory service proxy | 1.000 | 1.000 | 2.000 | 2.000 |
| validity V | 1.000 | 1.000 | 3.500 | 3.500 |
| V + ΔT bytes/calls | 1.000 | 1.000 | 1.000 | 1.000 |
| V + ΔT +request | 1.000 | 1.000 | 1.000 | 1.000 |
| V + ΔT +command | 1.000 | 1.000 | 1.000 | 1.000 |

## Sealed-reference success at budget 6

| policy | success@2% | success@5% | eventual trials-to-2% success | eventual trials-to-5% success |
|---|---:|---:|---:|---:|
| Random | 1.000 | 1.000 | 1.000 | 1.000 |
| stock-knob XGB | 1.000 | 1.000 | 1.000 | 1.000 |
| rules-only | 1.000 | 1.000 | 1.000 | 1.000 |
| paper minimum-access mode only | 1.000 | 1.000 | 1.000 | 1.000 |
| shared-memory lexicographic | 1.000 | 1.000 | 1.000 | 1.000 |
| shared-memory service proxy | 1.000 | 1.000 | 1.000 | 1.000 |
| validity V | 1.000 | 1.000 | 1.000 | 1.000 |
| V + ΔT bytes/calls | 1.000 | 1.000 | 1.000 | 1.000 |
| V + ΔT +request | 1.000 | 1.000 | 1.000 | 1.000 |
| V + ΔT +command | 1.000 | 1.000 | 1.000 | 1.000 |

The sealed reference is the completed FPGA-correct Y01 pool oracle; TopHub is not used.

## Search resource cost at budget 6

| policy | complete rate | logical LOAD MiB | logical STORE MiB | DMA calls | FPGA invocations | wall ms |
|---|---:|---:|---:|---:|---:|---:|
| Random | 1.000 | 295.412 | 0.743 | 98064.0 | 36.0 | 20743.463 |
| stock-knob XGB | 1.000 | 295.412 | 0.743 | 98064.0 | 36.0 | 20743.463 |
| rules-only | 1.000 | 295.412 | 0.743 | 98064.0 | 36.0 | 20743.463 |
| paper minimum-access mode only | 1.000 | 295.412 | 0.743 | 98064.0 | 36.0 | 20743.463 |
| shared-memory lexicographic | 1.000 | 295.412 | 0.743 | 98064.0 | 36.0 | 20743.463 |
| shared-memory service proxy | 1.000 | 295.412 | 0.743 | 98064.0 | 36.0 | 20743.463 |
| validity V | 1.000 | 295.412 | 0.743 | 98064.0 | 36.0 | 20743.463 |
| V + ΔT bytes/calls | 1.000 | 295.412 | 0.743 | 98064.0 | 36.0 | 20743.463 |
| V + ΔT +request | 1.000 | 295.412 | 0.743 | 98064.0 | 36.0 | 20743.463 |
| V + ΔT +command | 1.000 | 295.412 | 0.743 | 98064.0 | 36.0 | 20743.463 |

## Limitations

- Logical VTA LOAD/STORE bytes and calls come from runtime profiles; they are not physical AXI burst counts.
- TopHub is a sealed post-order metric reference only. It is absent from every dispatch and training label.
- Workload×seed median/IQR and per-workload 20-seed summaries are stored in `summary.json`.
- Full dispatch histories and revealed phase outcomes are stored in `runs.jsonl`.
