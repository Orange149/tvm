# P7R local joint-space certificate

- Status: `completed`
- Scope: local VTA lowering and logical DMA only; no board/latency/FPS
- Workloads: W05
- Joint candidates attempted: 400
- Locally eligible (both paths lower and input bytes decrease): 19
- Reject reasons: `{"analytic_acc_capacity": 15, "analytic_input_capacity": 15, "analytic_no_cross_context_reuse": 160, "analytic_weight_capacity": 12, "other_lower": 14, "same_tile_original_rejected": 165}`

Eligibility is not a speed or correctness claim. Next gate is three-seed FSim, followed by
cross-compilation and real-FPGA correctness before any timing.
