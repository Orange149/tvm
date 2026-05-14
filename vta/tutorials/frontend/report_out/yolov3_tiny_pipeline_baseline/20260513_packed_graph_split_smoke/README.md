# YOLOv3-tiny Packed Graph Split Smoke

This run splits the already-correct all-VTA graph JSON instead of rebuilding stages from Relay.
The boundary tensors keep their physical shape, dtype, layout, padding, and quantization domain.

- Candidate: yolo_packed_route25_shared40_to_heads
- Serial correctness: True
- Pipeline OK: 1
- Pipeline FPS: 6.5215
