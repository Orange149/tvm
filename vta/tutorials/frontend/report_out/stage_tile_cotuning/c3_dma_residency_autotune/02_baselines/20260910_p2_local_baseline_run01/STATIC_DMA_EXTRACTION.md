# Static VTA DMA extraction

The selected TopHub config is instantiated and lowered to static-shape TIR. Loop variables are enumerated at compile time and every `VTALoadBuffer2D`/`VTAStoreBuffer2D` descriptor is counted.

| workload | config | LOAD calls | avg LOAD B | small LOAD | R_input | R_weight | R_store | runtime source |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| h56_ci64_co64_k3s1 | 828 | 112 | 6656.0 | 0 | 2.43 | 7.00 | 1.00 | direct_template |
| h56_ci64_co128_k3s2 | 681 | 64 | 11216.0 | 0 | 2.11 | 4.00 | 1.00 | direct_template |
| h28_ci128_co128_k3s1 | 1330 | 128 | 5664.0 | 0 | 4.29 | 2.00 | 1.00 | direct_template |
| h56_ci64_co128_k1s2 | 1113 | 64 | 6232.0 | 32 | 1.82 | 4.00 | 1.00 | isolated_relay_unit |
| h28_ci128_co256_k3s2 | 463 | 64 | 10880.0 | 0 | 4.00 | 1.00 | 1.00 | direct_template |
| h14_ci256_co256_k3s1 | 575 | 128 | 6176.0 | 64 | 4.00 | 1.00 | 1.00 | direct_template |
| h28_ci128_co256_k1s2 | 526 | 64 | 3832.0 | 32 | 1.79 | 2.00 | 1.00 | isolated_relay_unit |
| h14_ci256_co512_k3s2 | 203 | 64 | 20000.0 | 32 | 2.00 | 1.00 | 1.00 | direct_template |
| h7_ci512_co512_k3s1 | 243 | 128 | 18824.0 | 64 | 2.00 | 1.00 | 1.00 | direct_template |
| h14_ci256_co512_k1s2 | 203 | 64 | 3400.0 | 32 | 1.72 | 1.00 | 1.00 | direct_template |

Validation:

- All six compared DMA fields are bit-exact for 8/8 correct direct-template profiles.
- For the two projection Relay-unit profiles, static convolution input, weight, and STORE fields are exact; runtime adds ACC/graph-level requests outside the isolated convolution TIR.
- Output STORE redundancy is 1.00 for all ten workloads; input redundancy ranges from 1.72x to 4.29x and weight redundancy from 1.00x to 7.00x under the transferred TopHub tiles.
- These are logical DMA requests after schedule selection, not physical AXI transactions or measured transfer time.
