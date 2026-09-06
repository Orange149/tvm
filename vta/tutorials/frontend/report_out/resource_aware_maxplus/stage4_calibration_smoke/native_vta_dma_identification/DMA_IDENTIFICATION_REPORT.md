# Native VTA DMA Identification Smoke

- Execution: native static packed functions; RPC performance: `false`.
- Cases/correct: `16/16`.
- Design rank: `5/5`.
- Condition number: `19.584`.
- Linear diagnostic fit: `R2=0.404`, `MAE=7.607 us`.
- Contiguous load/store identified: `False`.
- Fitted contiguous bandwidth: load `4.643 GB/s`, store `1.290 GB/s`.
- Formal calibration ready: `False`.

| case | load B | store B | load calls | store calls | device us | wall us |
|---|---:|---:|---:|---:|---:|---:|
| `acc1_2x16_out1` | 2048 | 512 | 1 | 1 | 68.168 | 80.026 |
| `acc1_2x16_out2` | 2048 | 1024 | 1 | 2 | 67.436 | 80.533 |
| `acc2_2x16_out1` | 4096 | 512 | 2 | 1 | 67.354 | 81.430 |
| `acc2_2x16_out2` | 4096 | 1024 | 2 | 2 | 67.528 | 82.597 |
| `acc1_8x16_out1` | 8192 | 2048 | 1 | 1 | 67.362 | 79.058 |
| `acc1_8x16_out2` | 8192 | 4096 | 1 | 2 | 67.983 | 81.617 |
| `acc2_8x16_out1` | 16384 | 2048 | 2 | 1 | 68.237 | 81.585 |
| `acc2_8x16_out2` | 16384 | 4096 | 2 | 2 | 67.905 | 82.692 |
| `acc1_16x16_out1` | 16384 | 4096 | 1 | 1 | 67.957 | 80.390 |
| `acc1_16x16_out2` | 16384 | 8192 | 1 | 2 | 67.732 | 81.316 |
| `acc2_16x16_out1` | 32768 | 4096 | 2 | 1 | 66.976 | 80.688 |
| `acc2_16x16_out2` | 32768 | 8192 | 2 | 2 | 67.496 | 82.487 |
| `acc1_16x32_out1` | 32768 | 8192 | 1 | 1 | 66.913 | 78.638 |
| `acc1_16x32_out2` | 32768 | 16384 | 1 | 2 | 67.531 | 80.838 |
| `acc2_16x32_out1` | 65536 | 8192 | 2 | 1 | 68.271 | 81.976 |
| `acc2_16x32_out2` | 65536 | 16384 | 2 | 2 | 125.467 | 140.302 |

## Gate

The matrix is algebraically full-rank, but the constrained service-time fit does not pass the R2>=0.90 identification gate. Default driver sleep masks small transfers; use the busy-poll isolation protocol or larger kernels.
