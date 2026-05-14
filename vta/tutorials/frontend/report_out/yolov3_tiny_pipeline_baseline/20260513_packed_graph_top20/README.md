# YOLOv3-tiny Packed Graph Top20 Search

This run splits the already-correct all-VTA graph JSON instead of rebuilding stages from Relay.
Boundary tensors keep physical shape, dtype, layout, padding, and quantization domain.

- Candidates: 20
- Buildable: 20
- Pipeline OK: 20
- Baseline RPC all-VTA FPS: 3.7243
- Baseline packed smoke FPS: 6.5215
- Best: yolo_packed_n25_35_to_heads FPS 3.2837

## Top 10
1. `yolo_packed_n25_35_to_heads` 3.2837 fps correctness=True
2. `yolo_packed_n25_to_heads` 3.1987 fps correctness=True
3. `yolo_packed_n25_40_60_to_heads` 3.0366 fps correctness=True
4. `yolo_packed_n25_40_63_to_heads` 3.0253 fps correctness=True
5. `yolo_packed_n25__n25_40__heads` 2.8648 fps correctness=True
6. `yolo_packed_n25_40_48_to_heads` 2.7823 fps correctness=True
7. `yolo_packed_n25_40_to_heads` 2.7407 fps correctness=True
8. `yolo_packed_n25_43_to_heads` 2.7370 fps correctness=True
9. `yolo_packed_n25_44_to_heads` 2.7365 fps correctness=True
10. `yolo_packed_n25__n25_63__heads` 2.7355 fps correctness=True
