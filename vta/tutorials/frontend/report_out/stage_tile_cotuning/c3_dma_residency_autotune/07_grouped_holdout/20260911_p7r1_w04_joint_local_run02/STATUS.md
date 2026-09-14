# P7R local joint-space certificate

- Status: `completed`
- Scope: local VTA lowering and logical DMA only; no board/latency/FPS
- Workloads: W04
- Joint candidates attempted: 320
- Locally eligible (both paths lower and input bytes decrease): 27
- Reject reasons: `{"no_input_dma_reduction": 9, "same_tile_original_rejected": 275, "sram_capacity": 9}`

Eligibility is not a speed or correctness claim. Next gate is three-seed FSim, followed by
cross-compilation and real-FPGA correctness before any timing.
