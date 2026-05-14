# YOLOv3-tiny Packed Graph Top20 Search

This run splits the already-correct all-VTA graph JSON instead of rebuilding stages from Relay.
Boundary tensors keep physical shape, dtype, layout, padding, and quantization domain.

- Candidates: 1
- Buildable: 1
- Pipeline OK: 0
- Baseline RPC all-VTA FPS: 3.7243
- Baseline packed smoke FPS: 6.5215
