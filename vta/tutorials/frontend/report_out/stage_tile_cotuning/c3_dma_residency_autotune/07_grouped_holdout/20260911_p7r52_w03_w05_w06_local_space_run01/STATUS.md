# P7R local joint-space certificate

- Status: `completed`
- Scope: local VTA lowering and logical DMA only; no board/latency/FPS
- Workloads: W03, W05, W06
- Joint candidates attempted: 1152
- Locally eligible (both paths lower and input bytes decrease): 227
- Reject reasons: `{"analytic_acc_capacity": 63, "analytic_input_capacity": 25, "analytic_no_cross_context_reuse": 252, "analytic_weight_capacity": 13, "same_tile_original_rejected": 572}`

Eligibility is not a speed or correctness claim. Next gate is three-seed FSim, followed by
cross-compilation and real-FPGA correctness before any timing.
