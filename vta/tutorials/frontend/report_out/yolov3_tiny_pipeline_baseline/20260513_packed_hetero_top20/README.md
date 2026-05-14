# YOLOv3-tiny Packed Heterogeneous CPU/VTA Top20 Search

This run is the first true CPU/VTA heterogeneous YOLO split run:

- CPU prefix is compiled separately with the CPU target and produces boundary node 5.
- VTA suffix stages are cut from the already-correct all-VTA packed graph JSON.
- VTA-side boundary tensors keep physical shape, dtype, layout, padding, and quantization domain.
- Candidates that use CPU tail from the all-VTA graph are not included, because those fragments still carry VTA device constraints.

- Candidates: 20
- Buildable: 20
- Serial raw gate passed: 20
- Pipeline OK: 20
- Baseline RPC all-VTA FPS: 3.7243
- Previous packed-graph smoke FPS: 6.5215
- Best: yolo_hetero_cpu5_vta25_51_60_heads FPS 4.2922

The 4.2922 fps best result should be compared as a true CPU/VTA split against RPC all-VTA 3.7243 fps.
The earlier 6.5215 fps packed smoke was not the same experiment class: it split an all-VTA native package and did not pay the CPU prefix boundary/work cost.

## Top 10
1. `yolo_hetero_cpu5_vta25_51_60_heads` 4.2922 fps correctness=True
2. `yolo_hetero_cpu5_vta25_35_heads` 4.1146 fps correctness=True
3. `yolo_hetero_cpu5_vta25_43_56_heads` 3.8880 fps correctness=True
4. `yolo_hetero_cpu5_vta25_40_63_heads` 3.8516 fps correctness=True
5. `yolo_hetero_cpu5_vta25_40_60_heads` 3.8442 fps correctness=True
6. `yolo_hetero_cpu5_vta25_48_60_heads` 3.8307 fps correctness=True
7. `yolo_hetero_cpu5_vta25_heads` 3.8223 fps correctness=True
8. `yolo_hetero_cpu5_vta25_44_56_heads` 3.8173 fps correctness=True
9. `yolo_hetero_cpu5_vta_heads` 3.8115 fps correctness=True
10. `yolo_hetero_cpu5_vta25_40_51_heads` 3.3673 fps correctness=True
