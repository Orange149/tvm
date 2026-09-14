# Prospective Y00 equal-budget search confirmation

> Prospective board labels collected only after the P7R132 candidate pool and search rules were frozen.

## Pool and protocol

- Workloads: 1
- Candidates excluding sealed references: 11
- Measured-ok: 1
- Lower-invalid: 0
- Compile-invalid: 0
- FPGA-invalid: 10
- Policies: 9 (including paper-mode and cross-tile shared-memory ablations)
- Seeds: 20; gross budgets: [1, 2, 4, 8]
- Total workload-policy-seed runs: 180

## Simple regret versus the non-reference full-pool oracle

| policy | regret@1 median [IQR] % | regret@2 median [IQR] % | regret@4 median [IQR] % | regret@8 median [IQR] % |
|---|---:|---:|---:|---:|
| Random | 0.000000 [0.000000] | 0.000000 [0.000000] | 0.000000 [0.000000] | 0.000000 [0.000000] |
| stock-knob XGB | 0.000000 [0.000000] | 0.000000 [0.000000] | 0.000000 [0.000000] | 0.000000 [0.000000] |
| rules-only | n/a | n/a | n/a | 0.000000 [0.000000] |
| paper minimum-access mode only | n/a | n/a | n/a | 0.000000 [0.000000] |
| shared-memory lexicographic | n/a | n/a | n/a | 0.000000 [0.000000] |
| validity V | 0.000000 [0.000000] | 0.000000 [0.000000] | 0.000000 [0.000000] | 0.000000 [0.000000] |
| V + ΔT bytes/calls | n/a | n/a | n/a | 0.000000 [0.000000] |
| V + ΔT +request | n/a | n/a | n/a | 0.000000 [0.000000] |
| V + ΔT +command | n/a | n/a | n/a | 0.000000 [0.000000] |

## Pool-oracle quality at budget 8

| policy | success@2% | success@5% | median trials-to-2% | median trials-to-5% |
|---|---:|---:|---:|---:|
| Random | 0.850 | 0.850 | 6.000 | 6.000 |
| stock-knob XGB | 0.750 | 0.750 | 5.000 | 5.000 |
| rules-only | 1.000 | 1.000 | 8.000 | 8.000 |
| paper minimum-access mode only | 0.200 | 0.200 | 9.000 | 9.000 |
| shared-memory lexicographic | 1.000 | 1.000 | 8.000 | 8.000 |
| validity V | 0.850 | 0.850 | 4.500 | 4.500 |
| V + ΔT bytes/calls | 1.000 | 1.000 | 8.000 | 8.000 |
| V + ΔT +request | 1.000 | 1.000 | 8.000 | 8.000 |
| V + ΔT +command | 1.000 | 1.000 | 8.000 | 8.000 |

## Sealed-reference success at budget 8

| policy | success@2% | success@5% | eventual trials-to-2% success | eventual trials-to-5% success |
|---|---:|---:|---:|---:|
| Random | 0.850 | 0.850 | 1.000 | 1.000 |
| stock-knob XGB | 0.750 | 0.750 | 1.000 | 1.000 |
| rules-only | 1.000 | 1.000 | 1.000 | 1.000 |
| paper minimum-access mode only | 0.200 | 0.200 | 1.000 | 1.000 |
| shared-memory lexicographic | 1.000 | 1.000 | 1.000 | 1.000 |
| validity V | 0.850 | 0.850 | 1.000 | 1.000 |
| V + ΔT bytes/calls | 1.000 | 1.000 | 1.000 | 1.000 |
| V + ΔT +request | 1.000 | 1.000 | 1.000 | 1.000 |
| V + ΔT +command | 1.000 | 1.000 | 1.000 | 1.000 |

The sealed reference is the completed FPGA-correct Y00 pool oracle; TopHub is not used.

## Search resource cost at budget 8

| policy | complete rate | logical LOAD MiB | logical STORE MiB | DMA calls | FPGA invocations | wall ms |
|---|---:|---:|---:|---:|---:|---:|
| Random | 1.000 | 54.271 | 15.844 | 26396.0 | 12.0 | 19031.997 |
| stock-knob XGB | 1.000 | 52.627 | 15.844 | 26448.0 | 12.0 | 19054.726 |
| rules-only | 1.000 | 40.280 | 15.844 | 13032.0 | 12.0 | 16452.375 |
| paper minimum-access mode only | 1.000 | 33.368 | 10.562 | 20624.0 | 8.0 | 16971.886 |
| shared-memory lexicographic | 1.000 | 40.280 | 15.844 | 13032.0 | 12.0 | 16452.375 |
| validity V | 1.000 | 53.688 | 15.844 | 28168.0 | 12.0 | 19556.240 |
| V + ΔT bytes/calls | 1.000 | 40.280 | 15.844 | 13032.0 | 12.0 | 16452.375 |
| V + ΔT +request | 1.000 | 40.280 | 15.844 | 13032.0 | 12.0 | 16452.375 |
| V + ΔT +command | 1.000 | 40.280 | 15.844 | 13032.0 | 12.0 | 16452.375 |

## Limitations

- Logical VTA LOAD/STORE bytes and calls come from runtime profiles; they are not physical AXI burst counts.
- TopHub is a sealed post-order metric reference only. It is absent from every dispatch and training label.
- Workload×seed median/IQR and per-workload 20-seed summaries are stored in `summary.json`.
- Full dispatch histories and revealed phase outcomes are stored in `runs.jsonl`.
