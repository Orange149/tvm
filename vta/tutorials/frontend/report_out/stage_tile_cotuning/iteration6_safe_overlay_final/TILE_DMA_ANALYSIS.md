# Bounded AutoTVM tile/DMA analysis

Measured 80 configurations: {'pass': 32, 'wrong_answer': 7, 'compile_host': 41}.

| changed axis | pass | compile fail | wrong answer | median time ratio | median LOAD-call ratio | median LOAD-byte ratio |
|---|---:|---:|---:|---:|---:|---:|
| h_nthread | 1 | 8 | 1 | 1.236 | 1.000 | 1.000 |
| oc_nthread | 7 | 2 | 1 | 1.302 | 1.000 | 1.000 |
| tile_ci | 0 | 10 | 0 | -- | -- | -- |
| tile_co | 5 | 10 | 1 | 1.078 | 2.000 | 1.254 |
| tile_h | 7 | 6 | 1 | 1.068 | 2.000 | 1.486 |
| tile_w | 4 | 5 | 1 | 3.374 | 4.500 | 4.173 |

Observed pruning signals:

- Reducing spatial tile width fragmented DMA most strongly among passing pairs: the median LOAD request count rose 4.5x and runtime rose 3.374x.
- Reducing tile height raised median LOAD requests 2x and runtime 1.068x; the worst passing pair reached about 7x requests and 5.446x runtime.
- Reducing output-channel tile raised median LOAD requests 2x and runtime 1.078x.
- Changing virtual-thread axes did not reduce DMA traffic and increased median runtime (oc_nthread 1.302x; the one comparable h_nthread pair 1.236x).
- All ten tile_ci neighbours failed compilation, so this direction can be pruned for the current incumbent neighbourhood and hardware fingerprint.
- Two projection incumbents failed the isolated correctness runner although complete TopHub Relay stages passed after FPGA reset; they must retain TopHub unless a candidate is validated at stage level.

Limits:

- Ratios use only pairs where both the TopHub incumbent and neighbour passed the same direct runner.
- This bounded one-axis neighbourhood does not prove a global AutoTVM optimum.
- DMA counters include VTA DDR-to-SRAM and SRAM-to-DDR operations, not CPU-GraphExecutor boundary copies.
