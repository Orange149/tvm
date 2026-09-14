# P7R local joint-space certificate

- Status: `completed`
- Scope: local VTA lowering and logical DMA only; no board/latency/FPS
- Workloads: E03
- Joint candidates attempted: 864
- Locally eligible (both paths lower and input bytes decrease): 74
- Reject reasons: `{"analytic_acc_capacity": 120, "analytic_no_cross_context_reuse": 144, "same_tile_original_rejected": 526}`

Eligibility is not a speed or correctness claim. Next gate is three-seed FSim, followed by
cross-compilation and real-FPGA correctness before any timing.
