# P7R116 equal-budget P7Q development replay

> Development-only replay on previously exposed P7Q labels; not prospective confirmation.

## Pool and protocol

- Workloads: 4
- Candidates excluding sealed references: 76
- Measured-ok: 75
- Lower-invalid: 0
- Compile-invalid: 0
- FPGA-invalid: 1
- Policies: 7 (five required families; three cumulative ΔT feature boundaries)
- Seeds: 20; gross budgets: [4, 8, 12]
- Total workload-policy-seed runs: 560

## Simple regret versus the non-reference full-pool oracle

| policy | regret@4 median [IQR] % | regret@8 median [IQR] % | regret@12 median [IQR] % |
|---|---:|---:|---:|
| Random | 0.433750 [9.979437] | 0.077686 [1.127316] | 0.000000 [0.297771] |
| stock-knob XGB | 0.433750 [10.030672] | 0.000000 [0.000000] | 0.000000 [0.000000] |
| rules-only | 0.000000 [0.000000] | 0.000000 [0.000000] | 0.000000 [0.000000] |
| validity V | 134.448415 [236.631174] | 0.000000 [0.077686] | 0.000000 [0.077686] |
| V + ΔT bytes/calls | 0.427631 [2.945334] | 0.094890 [0.914253] | 0.038843 [0.893501] |
| V + ΔT +request | 1.736574 [67.253384] | 0.189780 [1.061026] | 0.038843 [0.893501] |
| V + ΔT +command | 1.736574 [67.253384] | 0.189780 [1.061026] | 0.000000 [0.077686] |

## Sealed-reference success at budget 12

| policy | success@2% | success@5% | eventual trials-to-2% success | eventual trials-to-5% success |
|---|---:|---:|---:|---:|
| Random | 0.000 | 0.000 | 0.000 | 0.000 |
| stock-knob XGB | 0.000 | 0.000 | 0.000 | 0.000 |
| rules-only | 0.000 | 0.000 | 0.000 | 0.000 |
| validity V | 0.000 | 0.000 | 0.000 | 0.000 |
| V + ΔT bytes/calls | 0.000 | 0.000 | 0.000 | 0.000 |
| V + ΔT +request | 0.000 | 0.000 | 0.000 | 0.000 |
| V + ΔT +command | 0.000 | 0.000 | 0.000 | 0.000 |

No P7Q non-reference candidate enters either sealed TopHub equivalence band. This is an exposed-data property, not a prospective search conclusion.

## Limitations

- P7Q contains 75 measured candidates and only one FPGA-invalid candidate; it has no lower- or compile-invalid examples for a balanced validity test.
- P7Q phase wall-clock costs are unavailable. They remain null, so wall-clock completeness is zero; known-sum zero must not be interpreted as zero tuning cost.
- TopHub is a sealed post-order metric reference only. It is absent from every dispatch and training label.
- Workload×seed median/IQR and per-workload 20-seed summaries are stored in `summary.json`.
- Full dispatch histories and revealed phase outcomes are stored in `runs.jsonl`.
