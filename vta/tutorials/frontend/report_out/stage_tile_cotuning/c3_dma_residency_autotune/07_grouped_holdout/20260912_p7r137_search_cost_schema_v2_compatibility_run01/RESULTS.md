# Equal-budget P7Q development replay

> Development-only replay on previously exposed P7Q labels; not prospective confirmation.

## Pool and protocol

- Workloads: 4
- Candidates excluding sealed references: 76
- Measured-ok: 75
- Lower-invalid: 0
- Compile-invalid: 0
- FPGA-invalid: 1
- Policies: 9 (including paper-mode and cross-tile shared-memory ablations)
- Seeds: 20; gross budgets: [1, 2, 4, 8, 12]
- Total workload-policy-seed runs: 720

## Simple regret versus the non-reference full-pool oracle

| policy | regret@1 median [IQR] % | regret@2 median [IQR] % | regret@4 median [IQR] % | regret@8 median [IQR] % | regret@12 median [IQR] % |
|---|---:|---:|---:|---:|---:|
| Random | 14.079066 [33.084300] | 7.986456 [15.426300] | 0.433750 [9.979437] | 0.077686 [1.127316] | 0.000000 [0.297771] |
| stock-knob XGB | 15.962715 [41.926171] | 10.057122 [17.619694] | 0.433750 [10.030672] | 0.000000 [0.000000] | 0.000000 [0.000000] |
| rules-only | 0.130363 [0.443779] | 0.038843 [0.385478] | 0.000000 [0.000000] | 0.000000 [0.000000] | 0.000000 [0.000000] |
| paper minimum-access mode only | 13.406371 [260.710199] | 0.385478 [11.080448] | 0.077686 [0.385478] | 0.000000 [0.077686] | 0.000000 [0.077686] |
| shared-memory lexicographic | 0.000000 [0.168851] | 0.000000 [0.000000] | 0.000000 [0.000000] | 0.000000 [0.000000] | 0.000000 [0.000000] |
| validity V | 266.717342 [253.580099] | 151.672709 [254.017459] | 134.448415 [236.631174] | 0.000000 [0.077686] | 0.000000 [0.077686] |
| V + ΔT bytes/calls | 133.835580 [235.317040] | 132.257676 [237.683896] | 0.427631 [2.945334] | 0.094890 [0.914253] | 0.038843 [0.893501] |
| V + ΔT +request | 133.835580 [236.965663] | 132.257676 [237.683896] | 1.736574 [67.253384] | 0.189780 [1.061026] | 0.038843 [0.893501] |
| V + ΔT +command | 133.835580 [236.965663] | 132.257676 [248.961836] | 1.736574 [67.253384] | 0.189780 [1.061026] | 0.000000 [0.077686] |

## Pool-oracle quality at budget 12

| policy | success@2% | success@5% | median trials-to-2% | median trials-to-5% |
|---|---:|---:|---:|---:|
| Random | 0.838 | 0.912 | 4.000 | 3.000 |
| stock-knob XGB | 1.000 | 1.000 | 4.000 | 3.000 |
| rules-only | 1.000 | 1.000 | 1.000 | 1.000 |
| paper minimum-access mode only | 1.000 | 1.000 | 2.000 | 2.000 |
| shared-memory lexicographic | 1.000 | 1.000 | 1.000 | 1.000 |
| validity V | 0.887 | 0.975 | 5.000 | 5.000 |
| V + ΔT bytes/calls | 0.750 | 1.000 | 3.500 | 3.000 |
| V + ΔT +request | 0.750 | 1.000 | 4.000 | 3.000 |
| V + ΔT +command | 0.887 | 1.000 | 4.000 | 3.000 |

## Sealed-reference success at budget 12

| policy | success@2% | success@5% | eventual trials-to-2% success | eventual trials-to-5% success |
|---|---:|---:|---:|---:|
| Random | 0.000 | 0.000 | 0.000 | 0.000 |
| stock-knob XGB | 0.000 | 0.000 | 0.000 | 0.000 |
| rules-only | 0.000 | 0.000 | 0.000 | 0.000 |
| paper minimum-access mode only | 0.000 | 0.000 | 0.000 | 0.000 |
| shared-memory lexicographic | 0.000 | 0.000 | 0.000 | 0.000 |
| validity V | 0.000 | 0.000 | 0.000 | 0.000 |
| V + ΔT bytes/calls | 0.000 | 0.000 | 0.000 | 0.000 |
| V + ΔT +request | 0.000 | 0.000 | 0.000 | 0.000 |
| V + ΔT +command | 0.000 | 0.000 | 0.000 | 0.000 |

No P7Q non-reference candidate enters either sealed TopHub equivalence band. This is an exposed-data property, not a prospective search conclusion.

## Search resource cost at budget 12

| policy | complete rate | logical LOAD MiB | logical STORE MiB | DMA calls | FPGA invocations | wall ms |
|---|---:|---:|---:|---:|---:|---:|
| Random | 0.000 | 0.000 | 0.000 | 0.0 | 0.0 | 0.000 |
| stock-knob XGB | 0.000 | 0.000 | 0.000 | 0.0 | 0.0 | 0.000 |
| rules-only | 0.000 | 0.000 | 0.000 | 0.0 | 0.0 | 0.000 |
| paper minimum-access mode only | 0.000 | 0.000 | 0.000 | 0.0 | 0.0 | 0.000 |
| shared-memory lexicographic | 0.000 | 0.000 | 0.000 | 0.0 | 0.0 | 0.000 |
| validity V | 0.000 | 0.000 | 0.000 | 0.0 | 0.0 | 0.000 |
| V + ΔT bytes/calls | 0.000 | 0.000 | 0.000 | 0.0 | 0.0 | 0.000 |
| V + ΔT +request | 0.000 | 0.000 | 0.000 | 0.0 | 0.0 | 0.000 |
| V + ΔT +command | 0.000 | 0.000 | 0.000 | 0.0 | 0.0 | 0.000 |

## Limitations

- P7Q contains 75 measured candidates and only one FPGA-invalid candidate; it has no lower- or compile-invalid examples for a balanced validity test.
- P7Q phase wall-clock and resource costs are unavailable. Known-sum zero must not be interpreted as zero tuning cost.
- Logical VTA LOAD/STORE bytes and calls come from runtime profiles; they are not physical AXI burst counts.
- TopHub is a sealed post-order metric reference only. It is absent from every dispatch and training label.
- Workload×seed median/IQR and per-workload 20-seed summaries are stored in `summary.json`.
- Full dispatch histories and revealed phase outcomes are stored in `runs.jsonl`.
