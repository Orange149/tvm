# YOLOv3-tiny Multi-Island Relay Split Search

- Candidates: 100
- Buildable: 81
- Pipeline OK: 81
- Max VTA islands: 3
- Fine search policy: adaptive_balance
- Best: yolo_multi_pool2_logits fps=5.4402
- Correctness policy: detection_gate

Baselines: RPC all-VTA 3.724 fps; native single-stage pipeline 2.962 fps; packed graph smoke 6.5215 fps; packed hetero best 4.2922 fps.

## Best Details

- Candidate: `yolo_multi_pool2_logits`
- Split: `cpu/vta/cpu`
- Stage mean ms: [169.3982, 176.03178215, 5.16300205]
- Raw gate passed: False
- Raw sanity passed: True
- Detection gate passed: True
- Raw mismatch max mean delta: 0.5208248835587006

## Pattern Summary

- `cpu/vta/cpu` count=28 best=5.4402 mean=3.0663
- `cpu/vta/cpu/vta/cpu` count=35 best=5.2789 mean=4.2023
- `cpu/vta/cpu/vta/cpu/vta/cpu` count=18 best=4.8520 mean=4.3146

## Top 10

1. `yolo_multi_pool2_logits` 5.4402 fps split=`cpu/vta/cpu` raw_sanity=True detection=True
2. `yolo_multi_pool2_shared13__small_pre18_logits` 5.2789 fps split=`cpu/vta/cpu/vta/cpu` raw_sanity=True detection=True
3. `yolo_multi_pool2_trunk12__small_pre18_logits` 5.1381 fps split=`cpu/vta/cpu/vta/cpu` raw_sanity=True detection=True
4. `yolo_multi_pool2_dual_pre_logits` 4.9278 fps split=`cpu/vta/cpu` raw_sanity=True detection=True
5. `yolo_multi_pool1_trunk12__small_pre18_dual_pre_logits` 4.9060 fps split=`cpu/vta/cpu/vta/cpu` raw_sanity=True detection=True
6. `yolo_multi_pool3_logits` 4.8877 fps split=`cpu/vta/cpu` raw_sanity=True detection=True
7. `yolo_multi_pool0_pool1__pool3_small_pre18__dual_pre18_14_logits` 4.8520 fps split=`cpu/vta/cpu/vta/cpu/vta/cpu` raw_sanity=True detection=True
8. `yolo_multi_pool2_shared13__small_pre18_dual_pre_logits` 4.7821 fps split=`cpu/vta/cpu/vta/cpu` raw_sanity=True detection=True
9. `yolo_multi_pool0_pool1__pool3_dual_pre_logits` 4.7728 fps split=`cpu/vta/cpu/vta/cpu` raw_sanity=True detection=True
10. `yolo_multi_pool1_pool2__pool3_dual_pre_logits` 4.6340 fps split=`cpu/vta/cpu/vta/cpu` raw_sanity=True detection=True
