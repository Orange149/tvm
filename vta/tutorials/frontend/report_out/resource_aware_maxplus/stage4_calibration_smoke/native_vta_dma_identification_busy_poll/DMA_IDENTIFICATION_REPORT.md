# Native VTA DMA Identification Smoke

- Execution: native static packed functions; RPC performance: `false`.
- Cases/correct: `24/24`.
- Design rank: `5/5`.
- Condition number: `19.584`.
- Linear diagnostic fit: `R2=0.997`, `MAE=0.657 us`.
- Contiguous load/store identified: `True`.
- DMA bucket matrix ready: `True`; access kinds: `contiguous,padded,strided`.
- Fitted contiguous bandwidth: load `1.425 GB/s`, store `0.859 GB/s`.
- Median extra device time: strided `6.098 us`, padded `7.828 us`.
- Formal calibration ready: `False`.

| case | access | load B | store B | load calls | store calls | device us | wall us |
|---|---|---:|---:|---:|---:|---:|---:|
| `acc1_2x16_out1` | contiguous | 2048 | 512 | 1 | 1 | 13.693 | 25.823 |
| `acc1_2x16_out2` | contiguous | 2048 | 1024 | 1 | 2 | 14.067 | 27.483 |
| `acc2_2x16_out1` | contiguous | 4096 | 512 | 2 | 1 | 14.368 | 28.046 |
| `acc2_2x16_out2` | contiguous | 4096 | 1024 | 2 | 2 | 15.835 | 30.983 |
| `acc1_8x16_out1` | contiguous | 8192 | 2048 | 1 | 1 | 19.763 | 32.138 |
| `acc1_8x16_out2` | contiguous | 8192 | 4096 | 1 | 2 | 22.430 | 36.460 |
| `acc2_8x16_out1` | contiguous | 16384 | 2048 | 2 | 1 | 25.718 | 39.445 |
| `acc2_8x16_out2` | contiguous | 16384 | 4096 | 2 | 2 | 28.484 | 43.924 |
| `acc1_16x16_out1` | contiguous | 16384 | 4096 | 1 | 1 | 28.697 | 41.943 |
| `acc1_16x16_out2` | contiguous | 16384 | 8192 | 1 | 2 | 32.210 | 47.248 |
| `acc2_16x16_out1` | contiguous | 32768 | 4096 | 2 | 1 | 39.389 | 54.372 |
| `acc2_16x16_out2` | contiguous | 32768 | 8192 | 2 | 2 | 43.834 | 59.854 |
| `acc1_16x32_out1` | contiguous | 32768 | 8192 | 1 | 1 | 46.954 | 59.212 |
| `acc1_16x32_out2` | contiguous | 32768 | 16384 | 1 | 2 | 54.795 | 69.397 |
| `acc2_16x32_out1` | contiguous | 65536 | 8192 | 2 | 1 | 68.086 | 81.981 |
| `acc2_16x32_out2` | contiguous | 65536 | 16384 | 2 | 2 | 74.832 | 91.049 |
| `strided_8x20_to_8x16_out1` | strided | 8192 | 2048 | 1 | 1 | 23.824 | 36.652 |
| `strided_8x20_to_8x16_out2` | strided | 8192 | 4096 | 1 | 2 | 25.979 | 40.380 |
| `strided_16x24_to_16x16_out1` | strided | 16384 | 4096 | 1 | 1 | 38.578 | 51.416 |
| `strided_16x24_to_16x16_out2` | strided | 16384 | 8192 | 1 | 2 | 41.209 | 56.039 |
| `padded_8x16_p1_out1` | padded | 8192 | 2880 | 1 | 1 | 26.614 | 38.739 |
| `padded_8x16_p1_out2` | padded | 8192 | 5760 | 1 | 2 | 29.061 | 42.632 |
| `padded_16x16_p2_out1` | padded | 16384 | 6400 | 1 | 1 | 43.525 | 55.712 |
| `padded_16x16_p2_out2` | padded | 16384 | 12800 | 1 | 2 | 48.211 | 61.820 |

## Gate

The native DMA bucket matrix is ready, but publication calibration still requires three independent sessions and uncertainty aggregation.
