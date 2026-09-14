# Experiment C: controlled VTA stage boundary

Date: 2026-09-07. Board: `192.168.1.247:9091`. The fixed bitstream SHA-256 is
`7bf1ac95b1182c670cd25241111f33725b4ea37ed6d66f2df37b0b2e7df528d6`.

## Question and controls

The operator sequence is fixed to all five layer4 units. The control compiles
them into one VTA GraphExecutor. The two experimental variants split after
`layer4_block0_add_relu_tail` and compile block0 and block1 as separate VTA
GraphExecutors. All three builds query the same three TopHub workloads with
the same config sequence (`203, 243, 203`) and have zero fallback queries.

The boundary is `data0`, shape `(1, 512, 7, 7)`, dtype `float32`, or 100,352 B.

- `materialized`: producer output is copied once to a distinct `ext_dev` DDR
  NDArray, which is zero-copy bound to the consumer input.
- `shared_zero_copy`: producer output and consumer input are bound to the same
  `ext_dev` DDR NDArray, so the framework performs no boundary copy.

Both are controlled VTA-to-VTA DDR paths. They deliberately exclude the RPC
CPU-device conversion and are not a substitute for the existing native
CPU-VTA dual-view u-dma-buf experiment.

## Correctness

The monolithic VTA result exactly matches its monolithic quantized LLVM
reference. Both split variants exactly match the independently split and
quantized LLVM reference, and materialized and shared-zero-copy outputs are
bit-identical.

Inserting the stage boundary changes 30 of 25,088 final elements (0.120%)
relative to the monolithic graph, with maximum absolute difference 16. This is
not a zero-copy error: both split paths have the same SHA-256 and each matches
its own LLVM reference. It records a stage-local compilation/quantization
effect that must be distinguished from handoff correctness.

## Timing

The primary repeat uses 3 warmups followed by 30 rotating-order observations
per variant on one boot.

| variant | median | P25 | P75 | min | max |
|---|---:|---:|---:|---:|---:|
| monolithic | 20.327 ms | 20.181 ms | 20.727 ms | 19.687 ms | 22.448 ms |
| materialized | 22.929 ms | 22.440 ms | 23.462 ms | 21.624 ms | 25.203 ms |
| shared zero-copy | 22.065 ms | 21.427 ms | 22.562 ms | 20.699 ms | 23.960 ms |

Paired by repeat, `materialized - shared` has median 1.290 ms, mean 1.025 ms,
IQR `[0.523, 1.552]` ms, and is positive in 27/30 observations. The shared
path remains slower than monolithic by a paired median 1.407 ms (28/30), while
materialized is slower by 2.422 ms (30/30). A prior 12-repeat run produced the
same ordering (20.624, 23.555, and 21.788 ms respectively).

## Runtime memory profile

| metric | monolithic | materialized | shared zero-copy |
|---|---:|---:|---:|
| VTA LOAD calls | 522 | 522 | 522 |
| VTA LOAD payload | 8,736,256 B | 8,736,256 B | 8,736,256 B |
| VTA STORE calls | 10 | 10 | 10 |
| VTA STORE payload | 125,440 B | 125,440 B | 125,440 B |
| driver instructions | 939 | 939 | 939 |
| profiled boundary-copy calls | 0 | 1 | 0 |
| profiled boundary-copy bytes | 0 | 100,352 B | 0 |
| profiled boundary-copy service | 0 | 0.288 ms | 0 |
| device run wait | 17.903 ms | 17.904 ms | 17.862 ms |

The profiler labels the controlled `ext_dev` copy in its generic
`mem_copy_from_host` bucket; the allocation/device audit shows that source,
destination, producer interface, and consumer interface are all remote
`ext_dev(0)`. Therefore the robust claim is one logical framework copy of
100,352 B, not a measured physical AXI direction.

## Result for the memory-aware model

This legal block-aligned cut does **not** fragment VTA DMA and requires no
`D_stage-cut` correction to LOAD/STORE calls or payload. Its cost is instead:

1. one removable framework materialization when distinct buffers are used;
2. an Executor/launch handoff cost that remains after zero-copy; and
3. a small stage-local numerical change caused by independent compilation and
   quantization.

Thus workload DMA is additive for this boundary, but stage is not irrelevant.
The outer partition model should keep boundary bytes/copy mode and Executor or
VTA-island service terms separate from the per-workload DMA aggregate. This
runtime profile records logical VTA DMA requests and host wait, not physical
DDR transactions or compute-stall cycles. Multi-frame FPS and cross-boot
confidence belong to Experiment D.

Raw artifacts: `experiment_c_boundary_repeat30.json` (primary) and
`experiment_c_boundary.json` (12-repeat replication). Reproduction script:
`profile_vta_stage_boundary.py`.
