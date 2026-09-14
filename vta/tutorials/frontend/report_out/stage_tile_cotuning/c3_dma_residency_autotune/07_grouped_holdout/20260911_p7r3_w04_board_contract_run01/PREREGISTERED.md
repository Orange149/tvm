# P7R3 W04 board correctness contract

Frozen before any P7R board label on 2026-09-11.

- This is a development-only correctness probe, not a confirmatory workload and not a timing run.
- Candidate selection used local lowering and logical DMA only; all three candidates passed the fixed three-seed FSim gate.
- Execution is paired with the exact same ConfigEntity under the original schedule and bracketed by protected TopHub config463 sentinels.
- Config330 tests the label-free minimum-load point. Config455/461 form a horizontal/vertical geometry pair.
- Input-stationary config463 is not dispatched because the compiler already rejected its accumulator allocation.
- Any wrong answer, new SD error, changed boot/FPGA/u-dma-buf state, or external timeout stops the run. The board is never rebooted or powered off by the experiment.
- No latency, FPS, or simulator-time claim is permitted from this run.

Preflight amendment before any candidate was built, uploaded, or executed: the first runner
invocation rejected the valid tmpfs RPC working directory because it hard-coded `/tmp`. The
actual frozen path is `/var/volatile/vta_c3_ram/runtime`; the contract and runner were corrected,
and the failed preflight directory is retained as evidence. Candidate selection and order did not change.
