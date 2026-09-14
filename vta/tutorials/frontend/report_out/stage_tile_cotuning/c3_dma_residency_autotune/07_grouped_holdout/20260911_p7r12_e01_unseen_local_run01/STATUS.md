# P7R local joint-space certificate

- Status: `completed`
- Scope: local VTA lowering and logical DMA only; no board/latency/FPS
- Workloads: E01
- Joint candidates attempted: 576
- Locally eligible (both paths lower and input bytes decrease): 108
- Reject reasons: `{"analytic_acc_capacity": 72, "analytic_no_cross_context_reuse": 192, "same_tile_original_rejected": 180, "sram_capacity": 24}`

Eligibility is not a speed or correctness claim. Next gate is three-seed FSim, followed by
cross-compilation and real-FPGA correctness before any timing.
