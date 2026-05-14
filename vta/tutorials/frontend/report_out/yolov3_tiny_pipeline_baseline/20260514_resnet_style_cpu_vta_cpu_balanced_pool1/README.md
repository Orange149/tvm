# YOLOv3-tiny ResNet-Style CPU/VTA/CPU Smoke

- Candidates: 1
- Buildable: 1
- Pipeline OK: 1
- Best: yolo_pool1_to_dual_pre_logits fps=4.7758
- Correctness policy: detection_gate

Baselines: RPC all-VTA 3.724 fps; native single-stage pipeline 2.962 fps; packed graph smoke 6.5215 fps; packed hetero best 4.2922 fps.

## Best Details

- Candidate: `yolo_pool1_to_dual_pre_logits`
- Split: `stage0_cpu -> stage1_vta -> stage2_cpu`
- Stage mean ms: [148.3120825, 210.3110777, 84.4344149]
- Raw gate passed: False
- Detection gate passed: True
- Raw mismatch max mean delta: 0.5200288105587005
