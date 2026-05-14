# YOLOv3-tiny ResNet-Style CPU/VTA/CPU Smoke

- Candidates: 100
- Buildable: 88
- Pipeline OK: 88
- Best: yolo_pool2_to_dual_pre_logits__rt_s04_s22_q2_poll1000_post1000 fps=5.1131
- Correctness policy: detection_gate

Baselines: RPC all-VTA 3.724 fps; native single-stage pipeline 2.962 fps; packed graph smoke 6.5215 fps; packed hetero best 4.2922 fps.

## Best Details

- Candidate: `yolo_pool2_to_dual_pre_logits__rt_s04_s22_q2_poll1000_post1000`
- Split: `stage0_cpu -> stage1_vta -> stage2_cpu`
- Stage mean ms: [195.9976552, 165.1157442, 137.6137912]
- Raw gate passed: False
- Detection gate passed: True
- Raw mismatch max mean delta: 0.5199897765587006
