# Prospective Y00 equal-budget search confirmation

> Prospective board labels collected only after the P7R132 candidate pool and search rules were frozen.

## Pool and protocol

- Workloads: 1
- Candidates excluding sealed references: 11
- Measured-ok: 11
- Lower-invalid: 0
- Compile-invalid: 0
- FPGA-invalid: 0
- Policies: 9 (including paper-mode and cross-tile shared-memory ablations)
- Seeds: 20; gross budgets: [1, 2, 4, 8]
- Total workload-policy-seed runs: 180

## Simple regret versus the non-reference full-pool oracle

| policy | regret@1 median [IQR] % | regret@2 median [IQR] % | regret@4 median [IQR] % | regret@8 median [IQR] % |
|---|---:|---:|---:|---:|
| Random | 96.060737 [341.745382] | 32.320454 [24.765988] | 4.327443 [10.403235] | 0.000000 [0.000000] |
| stock-knob XGB | 82.420396 [120.696316] | 15.648284 [50.899518] | 8.654885 [15.648284] | 0.000000 [0.000000] |
| rules-only | 0.000000 [0.000000] | 0.000000 [0.000000] | 0.000000 [0.000000] | 0.000000 [0.000000] |
| paper minimum-access mode only | 96.060737 [358.417552] | 15.648284 [60.800767] | 0.000000 [15.648284] | 0.000000 [0.000000] |
| shared-memory lexicographic | 0.000000 [0.000000] | 0.000000 [0.000000] | 0.000000 [0.000000] | 0.000000 [0.000000] |
| validity V | 72.554090 [503.964260] | 32.320454 [40.392559] | 8.654885 [15.648284] | 0.000000 [8.654885] |
| V + ΔT bytes/calls | 82.420396 [0.000000] | 49.047444 [0.000000] | 8.654885 [0.000000] | 0.000000 [0.000000] |
| V + ΔT +request | 82.420396 [0.000000] | 49.047444 [0.000000] | 8.654885 [0.000000] | 0.000000 [0.000000] |
| V + ΔT +command | 82.420396 [0.000000] | 49.047444 [0.000000] | 8.654885 [0.000000] | 0.000000 [0.000000] |

## Pool-oracle quality at budget 8

| policy | success@2% | success@5% | median trials-to-2% | median trials-to-5% |
|---|---:|---:|---:|---:|
| Random | 0.900 | 0.900 | 4.500 | 4.500 |
| stock-knob XGB | 0.800 | 0.800 | 5.000 | 5.000 |
| rules-only | 1.000 | 1.000 | 1.000 | 1.000 |
| paper minimum-access mode only | 1.000 | 1.000 | 3.000 | 3.000 |
| shared-memory lexicographic | 1.000 | 1.000 | 1.000 | 1.000 |
| validity V | 0.600 | 0.600 | 7.000 | 7.000 |
| V + ΔT bytes/calls | 1.000 | 1.000 | 5.000 | 5.000 |
| V + ΔT +request | 1.000 | 1.000 | 5.000 | 5.000 |
| V + ΔT +command | 1.000 | 1.000 | 5.000 | 5.000 |

## Sealed-reference success at budget 8

| policy | success@2% | success@5% | eventual trials-to-2% success | eventual trials-to-5% success |
|---|---:|---:|---:|---:|
| Random | 0.900 | 0.900 | 1.000 | 1.000 |
| stock-knob XGB | 0.800 | 0.800 | 1.000 | 1.000 |
| rules-only | 1.000 | 1.000 | 1.000 | 1.000 |
| paper minimum-access mode only | 1.000 | 1.000 | 1.000 | 1.000 |
| shared-memory lexicographic | 1.000 | 1.000 | 1.000 | 1.000 |
| validity V | 0.600 | 0.600 | 1.000 | 1.000 |
| V + ΔT bytes/calls | 1.000 | 1.000 | 1.000 | 1.000 |
| V + ΔT +request | 1.000 | 1.000 | 1.000 | 1.000 |
| V + ΔT +command | 1.000 | 1.000 | 1.000 | 1.000 |

The sealed reference is the completed FPGA-correct Y00 pool oracle; TopHub is not used.

## Search resource cost at budget 8

| policy | complete rate | logical LOAD MiB | logical STORE MiB | DMA calls | FPGA invocations | wall ms |
|---|---:|---:|---:|---:|---:|---:|
| Random | 1.000 | 178.609 | 52.812 | 107020.0 | 40.0 | 29803.914 |
| stock-knob XGB | 1.000 | 168.437 | 52.812 | 100520.0 | 40.0 | 28967.350 |
| rules-only | 1.000 | 104.152 | 52.812 | 40200.0 | 40.0 | 26733.471 |
| paper minimum-access mode only | 1.000 | 166.839 | 52.812 | 103120.0 | 40.0 | 29133.579 |
| shared-memory lexicographic | 1.000 | 104.152 | 52.812 | 40200.0 | 40.0 | 26733.471 |
| validity V | 1.000 | 221.080 | 52.812 | 161360.0 | 40.0 | 32384.577 |
| V + ΔT bytes/calls | 1.000 | 104.152 | 52.812 | 40200.0 | 40.0 | 26733.471 |
| V + ΔT +request | 1.000 | 104.152 | 52.812 | 40200.0 | 40.0 | 26733.471 |
| V + ΔT +command | 1.000 | 104.152 | 52.812 | 40200.0 | 40.0 | 26733.471 |

## Limitations

- Logical VTA LOAD/STORE bytes and calls come from runtime profiles; they are not physical AXI burst counts.
- TopHub is a sealed post-order metric reference only. It is absent from every dispatch and training label.
- Workload×seed median/IQR and per-workload 20-seed summaries are stored in `summary.json`.
- Full dispatch histories and revealed phase outcomes are stored in `runs.jsonl`.
