# P7R local joint-space certificate

- Status: `completed`
- Scope: local VTA lowering and logical DMA only; no board/latency/FPS
- Workloads: E00
- Joint candidates attempted: 1024
- Locally eligible (both paths lower and input bytes decrease): 45
- Reject reasons: `{"analytic_acc_capacity": 96, "analytic_input_capacity": 18, "analytic_no_cross_context_reuse": 512, "same_tile_original_rejected": 346, "sram_capacity": 7}`

Eligibility is not a speed or correctness claim. Next gate is three-seed FSim, followed by
cross-compilation and real-FPGA correctness before any timing.
