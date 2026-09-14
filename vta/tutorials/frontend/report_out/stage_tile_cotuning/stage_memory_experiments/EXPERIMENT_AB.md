# Stage memory experiment A/B

## A. Workload/config invariance

Topologies: 4; unique workloads: 10; all invariant: **True**.

| workload | stage placements | distinct tile configs | dispatch/config match | invariant |
|---|---:|---:|---:|---:|
| h14_ci256_co256_k3s1 | 4 | 1 | True | True |
| h14_ci256_co512_k1s2 | 3 | 1 | True | True |
| h14_ci256_co512_k3s2 | 4 | 1 | True | True |
| h28_ci128_co128_k3s1 | 4 | 1 | True | True |
| h28_ci128_co256_k1s2 | 4 | 1 | True | True |
| h28_ci128_co256_k3s2 | 4 | 1 | True | True |
| h56_ci64_co128_k1s2 | 4 | 1 | True | True |
| h56_ci64_co128_k3s2 | 4 | 1 | True | True |
| h56_ci64_co64_k3s1 | 4 | 1 | True | True |
| h7_ci512_co512_k3s1 | 4 | 1 | True | True |

Result: stage identity did not change the TopHub tile for any of the ten workloads. AutoTVM can therefore be reused per unique workload in this frozen ResNet18 set; this does not imply that stage boundary or shared-DDR costs vanish.

## B. DMA additivity

| stage tail | conv occurrences | covered | full coverage | LOAD calls error | LOAD bytes error | WGT bytes error | STORE bytes error |
|---|---:|---:|---:|---:|---:|---:|---:|
| layer4_block0_add_relu_tail | 15 | 15 | True | -5.13% | -0.16% | +0.00% | +0.00% |
| layer4_block0_main_preadd | 14 | 14 | True | -5.22% | -0.14% | +0.00% | +0.00% |
| layer4_block0_skip_proj | 15 | 15 | True | -5.13% | -0.16% | +0.00% | +0.00% |
| layer3_block0_add_relu_tail | 10 | 10 | True | -6.34% | -0.16% | +0.00% | +0.00% |
| layer4_block1_add_relu_tail | 5 | 5 | True | -1.92% | -0.12% | +0.00% | +0.00% |

The controlled 03..15 -> 03..16 extension adds the layer4 1x1 projection. Observed minus isolated residuals:

| metric | observed delta | isolated op | residual |
|---|---:|---:|---:|
| load_buffer_2d_calls | 66 | 64 | 2 |
| load_buffer_2d_bytes | 219648 | 217600 | 2048 |
| load_buffer_2d_inp_bytes | 86528 | 86528 | 0 |
| load_buffer_2d_wgt_bytes | 131072 | 131072 | 0 |
| load_buffer_2d_acc_calls | 2 | 0 | 2 |
| load_buffer_2d_acc_bytes | 2048 | 0 | 2048 |
| store_buffer_2d_calls | 2 | 2 | 0 |
| store_buffer_2d_bytes | 25088 | 25088 | 0 |
| driver_run_insns | 127 | 121 | 6 |
| push_alu_op_calls | 8 | 6 | 2 |

This is an initial additivity result from same-incumbent evidence. All five stages have workload-occurrence coverage; the two projections that failed the bare template checker use independently correct isolated Relay-unit profiles. Input bytes, weight bytes, and STORE bytes are exactly additive in all five stages. The remaining LOAD-call/instruction residual includes ACC/ALU and graph-level work omitted by a conv-only workload sum. Runtime DMA calls are logical runtime requests, not physical AXI bursts or compute-stall cycles.
