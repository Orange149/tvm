# YOLOv3-tiny ResNet-Style CPU/VTA/CPU Smoke

- Candidates: 1
- Buildable: 1
- Pipeline OK: 1
- Best: `yolo_pool2_to_dual_pre_logits` fps=4.8715
- Correctness policy: detection_gate

Baselines: RPC all-VTA 3.724 fps; native single-stage pipeline 2.962 fps; packed graph smoke 6.5215 fps; packed hetero best 4.2922 fps.

## Best Details

- Split: `stage0_cpu -> stage1_vta -> stage2_cpu`
- Stage mean ms: [205.845902, 160.9632974, 82.3332854]
- Raw gate passed: False
- Detection gate passed: True
- Raw mismatch max mean delta: 0.5199897765587006
