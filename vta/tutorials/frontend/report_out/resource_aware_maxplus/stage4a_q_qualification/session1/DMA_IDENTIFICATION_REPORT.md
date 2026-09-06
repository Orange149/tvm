# Native VTA DMA Identification Smoke

- Execution: native static packed functions; RPC performance: `false`.
- Cases/success/failure: `36/36/0`.
- Design rank: `5/5`.
- Condition number: `19.584`.
- Linear diagnostic fit: `R2=0.997`, `MAE=0.737 us`.
- Contiguous load/store identified: `True`.
- Stage 4a-R matched design ready: `True`; matched pairs: `8`.
- Compute slope: `5.751675652173916` us/additional ALU repeat.
- Compute DMA fixed: `True`.
- Leave-one-size-out median/P95 APE: `2.963842098961392` / `6.677543204432136` percent.
- Sample total wall range valid: `True`; observed `[25.92915, 89.7725]` ms.
- Protocol SHA256 valid: `True`.
- Stage 4a-Q qualification passed: `True`.
- Formal calibration ready: `False`.

| case | access | load B | store B | load calls | store calls | device us | wall us |
|---|---|---:|---:|---:|---:|---:|---:|
| `acc1_2x16_out1` | contiguous | 2048 | 512 | 1 | 1 | 13.716 | 25.929 |
| `acc1_2x16_out2` | contiguous | 2048 | 1024 | 1 | 2 | 14.070 | 27.769 |
| `acc2_2x16_out1` | contiguous | 4096 | 512 | 2 | 1 | 13.835 | 27.694 |
| `acc2_2x16_out2` | contiguous | 4096 | 1024 | 2 | 2 | 15.877 | 30.967 |
| `acc1_8x16_out1` | contiguous | 8192 | 2048 | 1 | 1 | 18.652 | 31.068 |
| `acc1_8x16_out2` | contiguous | 8192 | 4096 | 1 | 2 | 20.867 | 34.669 |
| `acc2_8x16_out1` | contiguous | 16384 | 2048 | 2 | 1 | 25.001 | 38.982 |
| `acc2_8x16_out2` | contiguous | 16384 | 4096 | 2 | 2 | 27.028 | 42.305 |
| `acc1_16x16_out1` | contiguous | 16384 | 4096 | 1 | 1 | 27.989 | 40.344 |
| `acc1_16x16_out2` | contiguous | 16384 | 8192 | 1 | 2 | 31.051 | 44.892 |
| `acc2_16x16_out1` | contiguous | 32768 | 4096 | 2 | 1 | 38.274 | 52.164 |
| `acc2_16x16_out2` | contiguous | 32768 | 8192 | 2 | 2 | 41.925 | 57.180 |
| `acc1_16x32_out1` | contiguous | 32768 | 8192 | 1 | 1 | 45.743 | 58.201 |
| `acc1_16x32_out2` | contiguous | 32768 | 16384 | 1 | 2 | 52.369 | 66.221 |
| `acc2_16x32_out1` | contiguous | 65536 | 8192 | 2 | 1 | 66.930 | 80.998 |
| `acc2_16x32_out2` | contiguous | 65536 | 16384 | 2 | 2 | 73.551 | 88.777 |
| `strided_8x16_stride20_out1_control` | contiguous | 8192 | 2048 | 1 | 1 | 18.508 | 30.900 |
| `strided_8x16_stride20_out1_treatment` | strided | 8192 | 2048 | 1 | 1 | 22.544 | 35.249 |
| `strided_8x16_stride20_out2_control` | contiguous | 8192 | 4096 | 1 | 2 | 20.792 | 34.689 |
| `strided_8x16_stride20_out2_treatment` | strided | 8192 | 4096 | 1 | 2 | 24.589 | 38.625 |
| `strided_16x16_stride24_out1_control` | contiguous | 16384 | 4096 | 1 | 1 | 27.952 | 40.256 |
| `strided_16x16_stride24_out1_treatment` | strided | 16384 | 4096 | 1 | 1 | 36.043 | 48.827 |
| `strided_16x16_stride24_out2_control` | contiguous | 16384 | 8192 | 1 | 2 | 31.701 | 45.347 |
| `strided_16x16_stride24_out2_treatment` | strided | 16384 | 8192 | 1 | 2 | 40.068 | 54.392 |
| `padded_8x16_p1_out1_control` | contiguous | 11520 | 2880 | 1 | 1 | 21.863 | 34.295 |
| `padded_8x16_p1_out1_treatment` | padded | 8192 | 2880 | 1 | 1 | 24.925 | 37.225 |
| `padded_8x16_p1_out2_control` | contiguous | 11520 | 5760 | 1 | 2 | 25.357 | 38.978 |
| `padded_8x16_p1_out2_treatment` | padded | 8192 | 5760 | 1 | 2 | 27.104 | 40.878 |
| `padded_16x16_p2_out1_control` | contiguous | 25600 | 6400 | 1 | 1 | 37.813 | 50.001 |
| `padded_16x16_p2_out1_treatment` | padded | 16384 | 6400 | 1 | 1 | 42.175 | 54.374 |
| `padded_16x16_p2_out2_control` | contiguous | 25600 | 12800 | 1 | 2 | 42.912 | 56.697 |
| `padded_16x16_p2_out2_treatment` | padded | 16384 | 12800 | 1 | 2 | 46.903 | 60.543 |
| `compute_alu_repeat1_16x16` | contiguous | 16384 | 4096 | 1 | 1 | 27.869 | 40.427 |
| `compute_alu_repeat2_16x16` | contiguous | 16384 | 4096 | 1 | 1 | 33.737 | 47.344 |
| `compute_alu_repeat4_16x16` | contiguous | 16384 | 4096 | 1 | 1 | 44.479 | 60.608 |
| `compute_alu_repeat8_16x16` | contiguous | 16384 | 4096 | 1 | 1 | 68.216 | 89.773 |

## Matched pairs

- `padded_16x16_p2_out1`: observed `4.363 us`, byte-adjusted `10.879 us`.
- `padded_16x16_p2_out2`: observed `3.992 us`, byte-adjusted `10.508 us`.
- `padded_8x16_p1_out1`: observed `3.062 us`, byte-adjusted `5.415 us`.
- `padded_8x16_p1_out2`: observed `1.747 us`, byte-adjusted `4.100 us`.
- `strided_16x16_stride24_out1`: observed `8.091 us`, byte-adjusted `8.091 us`.
- `strided_16x16_stride24_out2`: observed `8.366 us`, byte-adjusted `8.366 us`.
- `strided_8x16_stride20_out1`: observed `4.036 us`, byte-adjusted `4.036 us`.
- `strided_8x16_stride20_out2`: observed `3.798 us`, byte-adjusted `3.798 us`.

## Gate

Stage 4a-Q passed; Stage 4b still requires a separately frozen three-session protocol.
