# P7R117 tile—shared-memory analysis

> Local no-latency pilot. Values are min/median/max over candidates that passed the corresponding gate.

| workload | mode | static | FSim | data bytes | DMA calls | insn peak B | uop peak B | aligned command backing B |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Y00 | original | 3/3 | 1 | 6482944/7481216/8159360 | 1248/2496/4992 | 53424/53424/53424 | 848/848/848 | 61440/61440/61440 |
| Y00 | input_stationary | 3/3 | 2 | 4412928/5391296/6688832 | 832/1664/3328 | 30128/75056/119984 | 68/874/1680 | 36864/81920/126976 |
| Y00 | weight_stationary | 1/3 | 0 | 6482944/6482944/6482944 | 1664/1664/1664 | -- | -- | -- |
| Y00 | paper_inspired_hybrid | 1/3 | 0 | 4412928/4412928/4412928 | 1248/1248/1248 | -- | -- | -- |
| Y01 | original | 3/3 | 3 | 3797248/4835584/4835584 | 78/936/13416 | 3504/30128/329648 | 20/68/68 | 8192/36864/335872 |
| Y01 | input_stationary | 3/3 | 3 | 3624192/3624192/3624192 | 52/208/1768 | 2048/5376/42816 | 20/68/68 | 8192/12288/49152 |
| Y01 | weight_stationary | 3/3 | 3 | 3797248/4835584/4835584 | 78/936/13416 | 3504/30128/329648 | 20/68/68 | 8192/36864/335872 |
| Y01 | paper_inspired_hybrid | 3/3 | 3 | 3624192/4143360/4143360 | 52/520/6760 | 2048/15984/165744 | 20/68/68 | 8192/20480/172032 |
| Y02 | original | 0/3 | 0 | -- | -- | -- | -- | -- |
| Y02 | input_stationary | 0/3 | 0 | -- | -- | -- | -- | -- |
| Y02 | weight_stationary | 0/3 | 0 | -- | -- | -- | -- | -- |
| Y02 | paper_inspired_hybrid | 0/3 | 0 | -- | -- | -- | -- | -- |

## Result

- Gross/static/FSim-pass: 36/20/15.
- Per-candidate 4-KiB-aligned instruction+UOP backing spans 8192--335872 B (median 36864 B).
- The old 8 KiB W05 result is therefore an exact-allowlist result, not a universal VTA constant.
- Mode-2 weight-stationary has no realized weight-DMA reduction and stays a negative control. The next formal pool must use the explicit barrier weight-residency mechanism.
- No latency or FPGA claim is made from this analysis.
