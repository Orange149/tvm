# P7R local joint-space certificate

- Status: `completed`
- Scope: local VTA lowering and logical DMA only; no board/latency/FPS
- Workloads: E02
- Joint candidates attempted: 768
- Locally eligible (both paths lower and input bytes decrease): 56
- Reject reasons: `{"analytic_input_capacity": 36, "analytic_no_cross_context_reuse": 192, "analytic_weight_capacity": 74, "same_tile_original_rejected": 410}`

Eligibility is not a speed or correctness claim. Next gate is three-seed FSim, followed by
cross-compilation and real-FPGA correctness before any timing.
