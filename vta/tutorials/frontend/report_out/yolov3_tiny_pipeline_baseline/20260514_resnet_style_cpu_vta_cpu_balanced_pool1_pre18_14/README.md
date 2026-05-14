# YOLOv3-tiny ResNet-Style CPU/VTA/CPU Smoke

- Candidates: 1
- Buildable: 1
- Pipeline OK: 1
- Best: yolo_pool1_to_dual_pre18_14 fps=3.2233
- Correctness policy: detection_gate

Baselines: RPC all-VTA 3.724 fps; native single-stage pipeline 2.962 fps; packed graph smoke 6.5215 fps; packed hetero best 4.2922 fps.

## Best Details

- Candidate: `yolo_pool1_to_dual_pre18_14`
- Split: `stage0_cpu -> stage1_vta -> stage2_cpu`
- Stage mean ms: [184.9074986, 185.0419927, 305.1964688]
- Raw gate passed: False
- Detection gate passed: True
- Raw mismatch max mean delta: 0.5200830395587006
