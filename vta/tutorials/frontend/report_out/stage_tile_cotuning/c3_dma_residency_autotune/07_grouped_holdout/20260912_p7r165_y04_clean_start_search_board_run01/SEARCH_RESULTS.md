# Prospective Y04 equal-budget search confirmation

> Prospective board labels collected only after the candidate pool and search rules were frozen.

## Pool and protocol

- Workloads: 1
- Candidates excluding sealed references: 24
- Measured-ok: 24
- Lower-invalid: 0
- Compile-invalid: 0
- FPGA-invalid: 0
- Policies: 9 (including paper-mode and cross-tile shared-memory ablations)
- Seeds: 20; gross budgets: [1, 2, 4, 8]
- Total workload-policy-seed runs: 180

## Simple regret versus the non-reference full-pool oracle

| policy | regret@1 median [IQR] % | regret@2 median [IQR] % | regret@4 median [IQR] % | regret@8 median [IQR] % |
|---|---:|---:|---:|---:|
| Random | 75.350005 [237.863412] | 53.737006 [48.183859] | 30.680095 [34.255450] | 16.624300 [15.874805] |
| stock-knob XGB | 232.369791 [400.408120] | 44.417943 [222.792767] | 34.318115 [37.711555] | 0.000000 [20.138249] |
| rules-only | 75.350005 [131.927979] | 37.956135 [37.393870] | 37.956135 [0.000000] | 37.956135 [0.000000] |
| paper minimum-access mode only | 222.073935 [284.766134] | 169.884114 [188.552840] | 75.350005 [94.534109] | 37.956135 [0.000000] |
| shared-memory lexicographic | 37.956135 [0.000000] | 37.956135 [0.000000] | 37.956135 [0.000000] | 37.956135 [0.000000] |
| validity V | 314.598999 [456.340430] | 96.437060 [252.142982] | 50.879750 [32.263378] | 16.624300 [37.956135] |
| V + ΔT bytes/calls | 53.583052 [0.000000] | 53.583052 [0.000000] | 53.583052 [0.000000] | 53.583052 [0.000000] |
| V + ΔT +request | 53.583052 [0.000000] | 53.583052 [0.000000] | 53.583052 [0.000000] | 53.583052 [0.000000] |
| V + ΔT +command | 53.583052 [0.000000] | 53.583052 [0.000000] | 53.583052 [0.000000] | 53.583052 [0.000000] |

## Pool-oracle quality at budget 8

| policy | success@2% | success@5% | median trials-to-2% | median trials-to-5% |
|---|---:|---:|---:|---:|
| Random | 0.200 | 0.200 | 18.000 | 18.000 |
| stock-knob XGB | 0.600 | 0.600 | 7.000 | 7.000 |
| rules-only | 0.000 | 0.000 | 12.000 | 12.000 |
| paper minimum-access mode only | 0.000 | 0.000 | 15.500 | 15.500 |
| shared-memory lexicographic | 0.000 | 0.000 | 9.000 | 9.000 |
| validity V | 0.400 | 0.400 | 9.500 | 9.500 |
| V + ΔT bytes/calls | 0.000 | 0.000 | 13.000 | 13.000 |
| V + ΔT +request | 0.000 | 0.000 | 13.000 | 13.000 |
| V + ΔT +command | 0.000 | 0.000 | 13.000 | 13.000 |

## Sealed-reference success at budget 8

| policy | success@2% | success@5% | eventual trials-to-2% success | eventual trials-to-5% success |
|---|---:|---:|---:|---:|
| Random | 0.200 | 0.200 | 1.000 | 1.000 |
| stock-knob XGB | 0.600 | 0.600 | 1.000 | 1.000 |
| rules-only | 0.000 | 0.000 | 1.000 | 1.000 |
| paper minimum-access mode only | 0.000 | 0.000 | 1.000 | 1.000 |
| shared-memory lexicographic | 0.000 | 0.000 | 1.000 | 1.000 |
| validity V | 0.400 | 0.400 | 1.000 | 1.000 |
| V + ΔT bytes/calls | 0.000 | 0.000 | 1.000 | 1.000 |
| V + ΔT +request | 0.000 | 0.000 | 1.000 | 1.000 |
| V + ΔT +command | 0.000 | 0.000 | 1.000 | 1.000 |

The sealed reference is the completed FPGA-correct Y04 pool oracle; TopHub is not used.

## Search resource cost at budget 8

| policy | complete rate | logical LOAD MiB | logical STORE MiB | DMA calls | FPGA invocations | wall ms |
|---|---:|---:|---:|---:|---:|---:|
| Random | 1.000 | 40.345 | 0.825 | 143720.0 | 130.0 | 10061.366 |
| stock-knob XGB | 1.000 | 41.508 | 0.825 | 57335.0 | 100.0 | 6163.640 |
| rules-only | 1.000 | 9.915 | 0.825 | 143210.0 | 250.0 | 10026.284 |
| paper minimum-access mode only | 1.000 | 9.915 | 0.825 | 143210.0 | 250.0 | 10026.284 |
| shared-memory lexicographic | 1.000 | 9.915 | 0.825 | 143210.0 | 250.0 | 10026.284 |
| validity V | 1.000 | 43.364 | 0.825 | 105765.0 | 110.0 | 8375.149 |
| V + ΔT bytes/calls | 1.000 | 43.683 | 0.825 | 148540.0 | 120.0 | 10134.549 |
| V + ΔT +request | 1.000 | 43.683 | 0.825 | 148540.0 | 120.0 | 10134.549 |
| V + ΔT +command | 1.000 | 43.683 | 0.825 | 148540.0 | 120.0 | 10134.549 |

## Limitations

- Logical VTA LOAD/STORE bytes and calls come from runtime profiles; they are not physical AXI burst counts.
- TopHub is a sealed post-order metric reference only. It is absent from every dispatch and training label.
- Workload×seed median/IQR and per-workload 20-seed summaries are stored in `summary.json`.
- Full dispatch histories and revealed phase outcomes are stored in `runs.jsonl`.
