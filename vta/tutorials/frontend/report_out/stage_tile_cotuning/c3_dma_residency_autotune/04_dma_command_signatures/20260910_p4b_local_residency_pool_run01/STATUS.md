# C3-P4b local residency pool run 01

- Status: `completed`
- Scope: local TIR lowering and logical DMA extraction only
- Board/RPC/SSH: not used
- FSim timing: not used
- Historical sources: 80
- Unique forced tile entities: 60
- Deduplicated sources: 20
- Protected original incumbents: 10
- Experimental residency candidates: 180

Legality and DMA-reduction distributions are in `summary.json`. Static DMA is a compiler feature, not a latency/FPS result.
- input_stationary: lower 32/60, target-DMA positive 25/32
- weight_stationary: lower 43/60, target-DMA positive 0/43
- paper_inspired_hybrid: lower 35/60, target-DMA positive 28/35
