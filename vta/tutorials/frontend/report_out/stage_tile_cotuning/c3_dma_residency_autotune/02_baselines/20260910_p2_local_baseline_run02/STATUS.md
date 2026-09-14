# C3 P2 local baseline run02

- Status: `completed_local_post_refactor`
- Local gate: `P2_LOCAL_BASELINE_PASS`
- Scope: original schedule only; local static analysis, FSim, tests, and AXU5EVB cross compilation.
- Board access: none. No SSH or RPC board connection was made, and no board FPS/latency claim is made.

## Results

- Static DMA: 10 workloads reproduced; run02 `static_dma.json` is byte-for-byte identical and JSON-semantically equal to the frozen artifact. Both SHA-256 values are `ff154ced1154c54fdd168597a1ab23433854c06652004538b821bd96fb48a3a5`.
- Correctness: 10 workloads × seeds `[0, 20250901, 20260910]` passed local FSim against the independent NumPy reference, 30/30 elementwise-equal.
- Compile-only: 10/10 incumbent configurations built and exported with `aarch64-xilinx-linux-g++ 9.2.0` for the frozen AXU5EVB target.
- Regression tests: `test_stage_tile_cotuning.py` reported 19 passed; `test_c3_candidate_identity.py` reported 7 passed.
- TIR: all 10 run02 IR-JSON hashes equal the corresponding run01 post-change hashes. W00/W02/W09 script hashes also retain the frozen values recorded by the P3 original-equivalence test.

## Source guard

The root agent disclosed a three-comment clarification in `vta_conv2d.py` stating that the safe weight mode does not reduce DMA. The run-start hash already reflected that comment-only state: `b6b79a3569a7de1ab3f8ea290d34390e30588e392b4c1eb3ff5c93fe925715b0`. It remained identical at run end. `vta_conv2d_residency.py` likewise remained at `5adec7be746ecf7912b36f078970521115a9bd730707dfd4216d044114bf629f`.

The unchanged run-start/end hashes, byte-identical static DMA, 10/10 TIR-IR equality, 30/30 FSim correctness, and 10/10 cross compilation jointly show that the disclosed edit did not alter executable original-path semantics. It is not classified as an unknown concurrent source change.

## Limits

This completes the requested post-refactor P2 local baseline. It does not establish board execution, physical AXI traffic, latency, throughput, or full G2. Those remain unevaluable while board access is prohibited.
