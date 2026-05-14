# YOLOv3-tiny Packed Graph Top20 Search

This run splits the already-correct all-VTA graph JSON instead of rebuilding stages from Relay.
Boundary tensors keep physical shape, dtype, layout, padding, and quantization domain.

- Candidates: 3
- Buildable: 3
- Pipeline OK: 3
- Baseline RPC all-VTA FPS: 3.7243
- Baseline packed smoke FPS: 6.5215
- Best: yolo_hetero_cpu5_vta25_35_heads FPS 3.9771

## Top 10
1. `yolo_hetero_cpu5_vta25_35_heads` 3.9771 fps correctness=True
2. `yolo_hetero_cpu5_vta_heads` 3.8383 fps correctness=True
3. `yolo_hetero_cpu5_vta25_heads` 3.8322 fps correctness=True
