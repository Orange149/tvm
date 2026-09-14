# P7R real-FPGA legality differential

This is a label-preserving diagnostic; it collects no latency.

| workload/config | FPGA | tile h×w×ci×co | output tile vectors | cthread tile vectors | co groups | max UOP pushes | store x/y/stride |
|---|---:|---:|---:|---:|---:|---:|---|
| W04/330 | fpga_fail | 7×7×1×1 | 49 | 98 | 8 | 21 | 7/7/14 |
| W04/455 | fpga_pass | 14×2×1×4 | 112 | 224 | 2 | 168 | 2/56/14 |
| W04/461 | fpga_pass | 2×14×1×4 | 112 | 224 | 2 | 24 | 28/4/196 |
| E00/1061 | fpga_fail | 14×7×1×1 | 98 | 196 | 5 | 42 | 7/14/42 |
| E01/951 | fpga_fail | 21×3×12×5 | 315 | 630 | 2 | 105 | 3/105/21 |
| E01/755 | fpga_fail | 21×1×12×2 | 42 | 84 | 5 | 21 | 1/42/21 |
| E02/1163 | fpga_pass | 6×3×1×6 | 108 | 216 | 2 | 108 | 3/36/6 |
| E02/1167 | fpga_pass | 6×6×1×6 | 216 | 432 | 2 | 108 | 216/1/216 |

Interpretation is deliberately deferred until the full TIR/DMA feature table is compared.
A feature that only separates these eight observations is a hypothesis, not yet a hardware rule.
