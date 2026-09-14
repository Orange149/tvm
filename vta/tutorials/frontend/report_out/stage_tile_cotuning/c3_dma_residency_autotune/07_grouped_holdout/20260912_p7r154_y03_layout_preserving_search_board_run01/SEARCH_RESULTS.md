# Prospective Y03 equal-budget search confirmation

> Prospective board labels collected only after the candidate pool and search rules were frozen.

## Pool and protocol

- Workloads: 1
- Candidates excluding sealed references: 13
- Measured-ok: 13
- Lower-invalid: 0
- Compile-invalid: 0
- FPGA-invalid: 0
- Policies: 9 (including paper-mode and cross-tile shared-memory ablations)
- Seeds: 20; gross budgets: [1, 2, 4, 8]
- Total workload-policy-seed runs: 180

## Simple regret versus the non-reference full-pool oracle

| policy | regret@1 median [IQR] % | regret@2 median [IQR] % | regret@4 median [IQR] % | regret@8 median [IQR] % |
|---|---:|---:|---:|---:|
| Random | 1466.056454 [894.669587] | 711.255906 [1299.354521] | 153.966750 [127.623355] | 0.000000 [39.525859] |
| stock-knob XGB | 1080.166058 [1779.474238] | 175.059864 [671.730047] | 96.746305 [159.238564] | 0.000000 [0.000000] |
| rules-only | 0.000000 [0.000000] | 0.000000 [0.000000] | 0.000000 [0.000000] | 0.000000 [0.000000] |
| paper minimum-access mode only | 288.730068 [1306.817890] | 153.966750 [125.640309] | 39.525859 [153.966750] | 0.000000 [0.000000] |
| shared-memory lexicographic | 0.000000 [0.000000] | 0.000000 [0.000000] | 0.000000 [0.000000] | 0.000000 [0.000000] |
| validity V | 1536.311745 [3855.219212] | 175.059864 [681.611511] | 96.746305 [159.238564] | 0.000000 [39.525859] |
| V + ΔT bytes/calls | 1506.290401 [0.000000] | 711.255906 [0.000000] | 159.238564 [0.000000] | 39.525859 [0.000000] |
| V + ΔT +request | 1506.290401 [0.000000] | 711.255906 [0.000000] | 159.238564 [0.000000] | 39.525859 [0.000000] |
| V + ΔT +command | 1506.290401 [0.000000] | 711.255906 [0.000000] | 159.238564 [0.000000] | 39.525859 [0.000000] |

## Pool-oracle quality at budget 8

| policy | success@2% | success@5% | median trials-to-2% | median trials-to-5% |
|---|---:|---:|---:|---:|
| Random | 0.550 | 0.550 | 8.000 | 8.000 |
| stock-knob XGB | 0.900 | 0.900 | 5.000 | 5.000 |
| rules-only | 1.000 | 1.000 | 1.000 | 1.000 |
| paper minimum-access mode only | 1.000 | 1.000 | 5.000 | 5.000 |
| shared-memory lexicographic | 1.000 | 1.000 | 1.000 | 1.000 |
| validity V | 0.650 | 0.650 | 6.500 | 6.500 |
| V + ΔT bytes/calls | 0.000 | 0.000 | 9.000 | 9.000 |
| V + ΔT +request | 0.000 | 0.000 | 9.000 | 9.000 |
| V + ΔT +command | 0.000 | 0.000 | 9.000 | 9.000 |

## Sealed-reference success at budget 8

| policy | success@2% | success@5% | eventual trials-to-2% success | eventual trials-to-5% success |
|---|---:|---:|---:|---:|
| Random | 0.550 | 0.550 | 1.000 | 1.000 |
| stock-knob XGB | 0.900 | 0.900 | 1.000 | 1.000 |
| rules-only | 1.000 | 1.000 | 1.000 | 1.000 |
| paper minimum-access mode only | 1.000 | 1.000 | 1.000 | 1.000 |
| shared-memory lexicographic | 1.000 | 1.000 | 1.000 | 1.000 |
| validity V | 0.650 | 0.650 | 1.000 | 1.000 |
| V + ΔT bytes/calls | 0.000 | 0.000 | 1.000 | 1.000 |
| V + ΔT +request | 0.000 | 0.000 | 1.000 | 1.000 |
| V + ΔT +command | 0.000 | 0.000 | 1.000 | 1.000 |

The sealed reference is the completed FPGA-correct Y03 pool oracle; TopHub is not used.

## Search resource cost at budget 8

| policy | complete rate | logical LOAD MiB | logical STORE MiB | DMA calls | FPGA invocations | wall ms |
|---|---:|---:|---:|---:|---:|---:|
| Random | 1.000 | 2017.556 | 6.602 | 615320.0 | 60.0 | 61856.556 |
| stock-knob XGB | 1.000 | 1174.937 | 6.602 | 276330.0 | 40.0 | 31188.118 |
| rules-only | 1.000 | 673.198 | 6.602 | 151100.0 | 60.0 | 24197.150 |
| paper minimum-access mode only | 1.000 | 1533.628 | 6.602 | 219480.0 | 60.0 | 28035.120 |
| shared-memory lexicographic | 1.000 | 673.198 | 6.602 | 144340.0 | 60.0 | 24104.678 |
| validity V | 1.000 | 2035.969 | 6.602 | 651310.0 | 60.0 | 64339.059 |
| V + ΔT bytes/calls | 1.000 | 2094.800 | 6.602 | 729420.0 | 40.0 | 68225.221 |
| V + ΔT +request | 1.000 | 2094.800 | 6.602 | 729420.0 | 40.0 | 68225.221 |
| V + ΔT +command | 1.000 | 2094.800 | 6.602 | 729420.0 | 40.0 | 68225.221 |

## Limitations

- Logical VTA LOAD/STORE bytes and calls come from runtime profiles; they are not physical AXI burst counts.
- TopHub is a sealed post-order metric reference only. It is absent from every dispatch and training label.
- Workload×seed median/IQR and per-workload 20-seed summaries are stored in `summary.json`.
- Full dispatch histories and revealed phase outcomes are stored in `runs.jsonl`.
