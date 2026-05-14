# YOLOv3-tiny CPU/VTA/CPU Status - 2026-05-13

## What Now Works

- The native runner supports explicit non-adjacent stage input sources via
  `--stageN-input-sources input:0|stageM:K[,stageM:K]`.
- This enables YOLO route/skip tensors to bypass a VTA stage instead of being
  forced through the VTA graph as pass-through outputs.
- The Relay search path now builds real `cpu/vta/cpu` packages.
- Summary rows now include measured stage timing fields:
  `stage_mean_ms`, `stage_set_mean_ms`, `stage_run_mean_ms`, `stage_get_mean_ms`,
  `bottleneck_stage`, and `stage_imbalance_ms`.
- Candidate ranking now prefers balanced CPU/VTA/CPU resource time and penalizes
  zero-tail/logits-only cuts.

## Smoke Results

Output roots:

- `20260513_cpu_vta_cpu_bypass_smoke`
- `20260513_cpu_vta_cpu_prelogits_smoke`
- `20260513_cpu_vta_cpu_float_tail_smoke`

All smoke candidates built and ran native serial on board, but none passed raw
correctness yet.

Observed pattern:

- 13x13 YOLO head is correct or near-correct.
- 26x26 YOLO head is consistently wrong.
- The failure is not a timeout or runner crash; it is a real raw-output mismatch.

Example:

- Candidate: `yolo_pool2_to_dual_pre_logits`
- Stage mean ms: about `[208.6, 151.7, 55.1]`
- Serial gate: failed
- Failed output: index `0`, shape `[1,255,26,26]`
- Matching output: 13x13 head remains correct

## Current Diagnosis

The framework is now past static validation. The remaining issue is semantic:
the small 26x26 YOLO head does not match the all-VTA baseline when its tail is
compiled for CPU.

The likely problem is in the small-head quant/layout boundary around:

- route/upsample/concat
- `small_pre26` (`[1,256,26,26]`)
- final 255-channel conv

The next useful fix is to inspect the exact lowered Relay/TIR around the 26x26
head and add an explicit adapter or choose a boundary where the CPU tail sees
the same logical tensor domain as the all-VTA graph.

## Adapter Attempts

### 1. Route bypass adapter

Implemented generic runner support:

```text
--stageN-input-sources input:0|stageM:K[,stageM:K]
```

This lets CPU tail consume a route tensor directly from stage0 instead of
forcing it through stage1/VTA as a pass-through output.

Result:

- Build: passed
- Native serial: ran
- Correctness: failed
- Remaining mismatch: output index 0, `[1,255,26,26]`

### 2. Balanced Relay CPU/VTA/CPU split

Candidate:

```text
yolo_pool2_to_dual_pre_logits
```

Measured serial stage times:

```text
[208.6, 151.7, 55.1] ms
```

This is a more balanced split, but correctness still fails on the 26x26 head.

### 3. Packed-logits decode tail

Candidate:

```text
yolo_hetero_cpu5_vta25_51_63_cpu_heads
```

This tries to let VTA produce packed logits (`node51`, `node63`) and make CPU
tail decode them. It failed at runtime because the all-VTA packed decode
function still has an ext-dev constraint:

```text
device_type has an unsatisfied constraint: 12 == ...
```

Conclusion: the CPU tail adapter cannot simply reuse the all-VTA packed graph
library. It must recompile an equivalent CPU decode adapter from Relay/TIR or
implement an explicit packed-logits-to-YOLO-output adapter.

## Known Good Baseline

The true CPU-prefix/VTA-suffix path remains correct:

- Best: `yolo_hetero_cpu5_vta25_51_60_heads`
- Pipeline throughput: `4.2922 fps`
- Correctness: passed

That result is not yet balanced `cpu/vta/cpu`; it is a correct CPU/VTA baseline
to beat.
