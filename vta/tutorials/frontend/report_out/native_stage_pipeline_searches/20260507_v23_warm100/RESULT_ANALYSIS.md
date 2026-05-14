# V2.3 Warm100 Native Search Result Analysis

## Summary

本轮共实测 100 个 warm/prewarmed native pipeline 候选，分 5 批完成，每批 20 个。所有候选均通过 native serial/pipeline 一致性检查和 RPC `all_vta` baseline correctness gate，失败数为 0。

全局最佳为 `islands_3_9__10_19`：

- batch: `batch003`
- pipeline throughput: `10.532301854209882 fps`
- stage layout: `cpu / vta / vta / cpu`
- VTA islands: `[3, 9]` 和 `[10, 19]`
- bottleneck: `stage0_cpu`
- stage ms: `82.48 / 46.07 / 42.46 / 2.93`
- PS-PL total bandwidth: `2.1016 GB/s`
- DMA fragmentation score: `1.3047`
- SRAM peak utilization: `38.28125%`
- correctness: `cat_equivalent`, baseline top1 `282`, native top1 `285`

相比 batch001 的 `three_stage_e` (`9.310548 fps`)，最佳候选提升约 `13.1%`。

## Why These 100 Candidates Were Tested

搜索不是随机抽样，而是按 V2.3 流程逐步收缩：

1. 全量枚举约 `13,457` 个 raw island 组合，得到约 `7,374` 个 native candidates。
2. 使用分桶 calibrated static model 计算解释性分数，保留 `static-shortlist-n=256`。
3. 对 shortlist 前 `128` 个做 buildability/package-only 筛选，并复用持久 build cache。
4. 按 batch 每次测 20 个，并用 `measured_ids_cumulative.txt` 排除已测候选，最终覆盖前 100 个可测候选。

这 100 个候选中，`6` 个来自 known/manual baseline，`94` 个来自 calibrated score top。也就是说，本轮主要验证的是“静态模型认为最有希望”的前 100 个 buildable native 候选，而不是全空间均匀采样。

## Key Results

每批最佳：

| Batch | Best candidate | FPS | Correctness |
|---|---:|---:|---|
| batch001 | `three_stage_e` | `9.310548` | pass |
| batch002 | `islands_1_4__5_12__13_17` | `8.568619` | pass |
| batch003 | `islands_3_9__10_19` | `10.532302` | pass |
| batch004 | `islands_3_8__10_19` | `10.241719` | pass |
| batch005 | `islands_1_4__5_10__13_14` | `8.994690` | pass |

全局 Top 10：

| Rank | Candidate | Batch | FPS | Notes |
|---:|---|---|---:|---|
| 1 | `islands_3_9__10_19` | batch003 | `10.532302` | best |
| 2 | `islands_3_9__10_18` | batch003 | `10.524254` | nearly tied |
| 3 | `islands_3_8__10_19` | batch004 | `10.241719` | 5-stage variant |
| 4 | `islands_1_7__8_13` | batch003 | `9.719342` | keeps late CPU tail large |
| 5 | `islands_1_6__8_14` | batch004 | `9.493757` | 5-stage variant |
| 6 | `islands_1_7__8_14` | batch003 | `9.449696` | 4-stage variant |
| 7 | `islands_3_8__10_18` | batch004 | `9.432142` | close to top family |
| 8 | `three_stage_e` | batch001 | `9.310548` | best manual baseline |
| 9 | `islands_1_6__8_13` | batch004 | `9.163935` | 5-stage variant |
| 10 | `islands_3_9__10_16` | batch004 | `9.030448` | shorter late VTA island |

## Patterns

### 1. More stages are not automatically better

Measured throughput by stage count:

| Stage count | Count | Mean FPS | Median FPS | Max FPS |
|---:|---:|---:|---:|---:|
| 3 | 5 | `6.540` | `5.946` | `9.311` |
| 4 | 11 | `8.707` | `8.859` | `10.532` |
| 5 | 35 | `8.143` | `8.315` | `10.242` |
| 6 | 36 | `8.227` | `8.177` | `8.995` |
| 7 | 13 | `7.924` | `7.928` | `8.224` |

结论：不要机械追求 stage count。`cpu/vta/vta/cpu` 从 stage count 看是 4-stage，但由于两个 VTA stage 相邻，且 runner 对 VTA `set_input/run/get_output` 使用全局 mutex，它在资源占用上更接近 `cpu / vta-block / cpu`。因此更合理的判断单位是“有效设备段”和“单设备资源占用”，而不是 stage 数本身。

按有效设备段压缩后：

| Effective pattern | Count | Mean FPS | Median FPS | Max FPS |
|---|---:|---:|---:|---:|
| `cpu/vta/cpu` | 38 | `8.112` | `8.307` | `10.532` |
| `cpu/vta/cpu/vta/cpu` | 49 | `8.192` | `8.175` | `10.242` |
| `cpu/vta/cpu/vta/cpu/vta/cpu` | 13 | `7.924` | `7.928` | `8.224` |

最高分来自有效 `cpu/vta/cpu`，但不是因为“stage 少”本身，而是因为它让 CPU early stage、VTA total occupancy、CPU tail 三者最接近吞吐最优平衡。

### 2. 两个大 VTA island 比三个小 VTA island 更容易出高分

Measured throughput by VTA island count:

| VTA island count | Count | Mean FPS | Median FPS | Max FPS |
|---:|---:|---:|---:|---:|
| 1 | 5 | `6.540` | `5.946` | `9.311` |
| 2 | 24 | `8.376` | `8.257` | `10.532` |
| 3 | 71 | `8.154` | `8.171` | `8.995` |

结论：两个连续大 island 是更好的默认结构。三个 island 的上限明显低，主要因为中间会多出 CPU micro-stage 或 VTA boundary，pipeline stage 变多但有效重叠不够。

### 3. 最好的 family 是 `3_9 + 10_18/19`

Top 3 都属于同一类：

- `islands_3_9__10_19`: `10.532302 fps`
- `islands_3_9__10_18`: `10.524254 fps`
- `islands_3_8__10_19`: `10.241719 fps`

这类方案保留最前面的 `stem + layer1_block0` 在 CPU，避免 VTA 处理早期大分辨率输入。随后把中后段主干卷积分成两个连续 VTA island，VTA stage 的实测时间约 `40-46 ms`，比 CPU stage0 短，pipeline bottleneck 留在 CPU stage0。

### 4. CPU 适合两端，VTA 适合中后段主干卷积

从 Top candidates 看，较优分配是：

- CPU: `stem`、`layer1_block0_main_preadd`、`layer1_block0_add_relu_tail`
- VTA: `layer1_block1` 到 `layer2_block1`
- VTA: `layer3_block0` 到 `layer4_block1`
- CPU: `head`

原因：

- 早期 `stem` 和 `layer1_block0` 输入分辨率大，VTA 侧 DMA/pack/set_input 成本容易抬高，CPU 3 线程反而更稳定。
- 中后段 residual 3x3、stride2 downsample、skip projection 是 VTA 的优势区，实测 VTA achieved GOPS 通常在 `30-40 GOPS`。
- `head` 计算量小，放 CPU 可避免为了小尾巴额外引入 VTA stage 和边界。
- late layer4 如果全部留给 CPU 会形成 `stage3_cpu/stage4_cpu/stage5_cpu` 瓶颈；但如果 VTA island 切得过碎，也会损失同步和边界开销。

### 5. 当前瓶颈多在 CPU，不在 VTA

100 个候选的 stage bottleneck 分布：

- `stage0_cpu`: 57
- `stage5_cpu`: 16
- `stage4_cpu`: 12
- `stage1_vta`: 10
- 其他 VTA/CPU stage: 5

最优候选 `islands_3_9__10_19` 的实测 stage ms 为：

- `stage0_cpu`: `82.48 ms`
- `stage1_vta`: `46.07 ms`
- `stage2_vta`: `42.46 ms`
- `stage3_cpu`: `2.93 ms`

这说明当前最佳不是让 VTA 单独跑得最快，而是让 VTA 足够快，同时让 CPU 端不留下过重尾部。stage0 CPU 是主要上限。

### 6. 静态模型能缩小搜索空间，但还不能精确排序

这 100 个候选主要来自 calibrated static score top，因此静态模型已经能把搜索集中到可行区域。但 Top 10 里仍有明显重排，说明以下实测因素还没有被静态模型完全捕捉：

- CPU stage0 实测波动和线程调度。
- VTA stage set_input/run/get_output 的组合开销。
- 多 stage native pipeline 中的边界同步和 file/cache 状态。
- `cat_equivalent` correctness 下的 relaxed gate，需要显式保留标记。

处理办法不是放弃静态模型，而是把它改成“先验模型 + 实测残差校正”的两层排序器：

1. 第一层继续用 calibrated static model 缩小搜索空间。
   - 仍然全量枚举 native candidates。
   - 用 CPU/VTA/DMA/SRAM/boundary 静态估算生成 shortlist。
   - 这一层目标是高召回，不要求精确排序。

2. 第二层用 warm native measurements 回灌校正。
   - 每跑完一个 batch，把实测 `stage*_ms`、`stage*_run_ms`、`stage*_set/get_ms`、`ps_pl_total_bw_gbps`、`dma_fragmentation_score`、`stage_balance_bottleneck` 写回一个 learned residual model。
   - 对同一 candidate family 及其邻域，例如 `3_9__10_19` 周围的 `3_8/9__10_18/19`，用实测 residual 修正静态分。
   - 下一批排序使用 `static_score_ms + learned_residual_ms`，而不是只看原始 static score。

3. 对 CPU stage0 单独建模。
   - 当前最佳候选的瓶颈仍是 `stage0_cpu`，且 CPU stage0 对线程调度、cache 状态和 early high-resolution input 更敏感。
   - 静态模型应为 `stage0_cpu` 使用实测分桶残差，例如按 `unit_names`、threads、输入 shape、是否包含 `stem/layer1_block0` 建立 correction。
   - 对 stage0 估算偏低的候选加 penalty，避免把实际 CPU bottleneck 排得过高。

4. 对 VTA 提交路径建模为全局资源。
   - 因为 native runner 对 VTA `set_input/run/get_output` 使用全局 mutex，多个 VTA stage 不能按完全并行 stage 评分。
   - 静态模型应显式计算：
     - `total_vta_set_input_ms`
     - `total_vta_run_ms`
     - `total_vta_get_output_ms`
     - `total_vta_occupied_ms`
   - 排序时用 `max(max_cpu_stage_ms, total_vta_occupied_ms)` 作为 pipeline cycle 的核心项。

5. 把 boundary/file-cache 状态从噪声变成受控变量。
   - 正式 ranking 只使用 `file_cache_state=prewarmed` 的 rows。
   - cold/prewarm 行不参与排序，只用于估算 deploy/cache overhead。
   - 对每个 candidate 记录 boundary bytes、boundary count、CPU/VTA crossing count，用实测 residual 学习 boundary penalty。

6. correctness 不参与吞吐排序，但影响候选可信度。
   - `passes_correctness_gate=false` 的候选直接 rank ineligible。
   - `cat_equivalent` relaxed 的候选允许进入排名，但必须在报告里显式标记。
   - 如果 Top candidates 中 relaxed 过多，下一轮应对 Top 5 做 exact correctness verification 或更多图片验证。

下一轮推荐流程：

```text
batch N measured rows
  -> update family residuals
  -> update CPU stage0 residual
  -> update VTA occupied-time residual
  -> update DMA/boundary residual
  -> re-rank remaining shortlist
  -> choose next 20 by corrected predicted fps + diversity guard
```

这样静态模型的角色会从“最终裁判”变成“高召回候选生成器”，最终排序逐步由 warm native 实测校正。

## Heuristic Search Algorithm

基于这 100 个实测结果，下一轮启发式不能只按切图数量或 island 形状排序。核心目标应该是直接优化 warm native throughput：

```text
predicted_cycle_ms =
  max(
    max_cpu_stage_ms,
    total_vta_occupied_ms,
    max_effective_boundary_ms
  )
  + launch_sync_overhead_ms
  + dma_fragmentation_penalty_ms
  + sram_spill_penalty_ms

predicted_fps = 1000 / predicted_cycle_ms
```

这里的 `total_vta_occupied_ms` 比单个 VTA stage ms 更重要，因为当前 native runner 对 VTA 提交路径加了全局 mutex，同一个候选内多个 VTA stage 不能当作独立并行资源。相邻 VTA stage 应该合并为一个 VTA resource block 来估算吞吐。

### Metrics to use

1. 固定首尾原则：
   - 默认让 `stem` 和 `layer1_block0` 留在 CPU。
   - 默认让 `head` 留在 CPU。
   - 除非校准显示 CPU stage0 明显改善，否则不要把 index `0-2` 放进 VTA。

2. CPU/VTA compute model：
   - CPU stage ms 用分桶 CPU GOPS 估算，线程数固定按当前实测策略 `stage0=3, tail=4, middle CPU=1`。
   - VTA stage ms 用分桶 VTA GOPS 估算，但同一候选内所有 VTA stage 要累加成 `total_vta_occupied_ms`。
   - 对相邻 VTA stage，允许作为 pipeline stage 分开保存结果，但评分时按一个 VTA resource block 处理。

3. PS-PL 和 DMA model：
   - 估算 `ps_pl_total_dma_bytes / ps_pl_total_bw_gbps` 得到 DMA ms。
   - 对 small/strided/padded call ratio 加 fragmentation penalty。
   - 本轮 Top 10 的 PS-PL total bandwidth median 约 `1.764 GB/s`，高于其余候选 median `1.524 GB/s`；带宽不是唯一决定因素，但低带宽候选应降权。
   - Top candidates 的 DMA fragmentation score 约 `1.20-1.31`，没有出现严重碎片化；若 score 明显升高，直接降低优先级。

4. SRAM/on-chip memory model：
   - 当前 100 个候选 peak SRAM utilization 基本都在 `38.28125%`，没有触发 spill。
   - 继续保留 SRAM gate：若 INP/WGT/ACC/OUT 任一 peak utilization 接近硬上限或出现 tile spill 风险，则直接拒绝或加大 penalty。
   - 在当前 ResNet18 搜索范围内，SRAM 不是主要排序因子，但必须作为 correctness/buildability 前置约束。

5. Stage balance model：
   - 评分时看 `max_cpu_stage_ms`、`total_vta_occupied_ms`、CPU tail ms 三者是否接近。
   - 最优 `islands_3_9__10_19` 的实测约为：
     - CPU early: `82.48 ms`
     - VTA block total: `46.07 + 42.46 = 88.54 ms`
     - CPU head: `2.93 ms`
     - throughput cycle: `94.95 ms`
   - 这说明优秀方案不是让每个 pipeline stage 都一样长，而是让“最重 CPU stage”和“全局 VTA 总占用”接近，并避免留下一个 `70-80 ms` 的 CPU tail。

### Candidate generation heuristic

1. 优先生成有效 `cpu/vta/cpu` 或最多 `cpu/vta/cpu/vta/cpu` 的候选。
   - `cpu/vta/cpu` 是首选，因为边界少，且 Top 2 都属于这个有效形态。
   - `cpu/vta/cpu/vta/cpu` 只在它能显著降低 CPU tail 或 VTA total occupancy 时进入候选。
   - `cpu/vta/cpu/vta/cpu/vta/cpu` 默认低优先级，因为本轮 max 只有 `8.224 fps`。

2. 优先让 early high-resolution blocks 留 CPU。
   - 如果 VTA island 从 `1` 开始，必须确认它没有显著增加 VTA set/input 和 DMA 压力。
   - 当前最佳 family 从 `3` 开始，说明 `stem + layer1_block0` 留 CPU 更合适。

3. 优先让 VTA 覆盖中后段大卷积。
   - 第一优先族：
     - `3_9__10_19`
     - `3_9__10_18`
     - `3_8__10_19`
     - `3_8__10_18`
   - 这些候选让 CPU early 和 VTA total occupancy 都落在约 `80-95 ms` 的吞吐有效区间。

4. 控制 CPU tail。
   - 如果 `layer4` 大量留给 CPU，tail stage 经常成为 `stage4_cpu/stage5_cpu` bottleneck。
   - 如果 CPU tail 估计超过 `75-80 ms`，除非 early CPU stage 明显更低，否则降权。

5. 控制 VTA total occupancy。
   - 如果多个 VTA island 的 `sum(vta_stage_ms)` 超过 `95-100 ms`，即使单个 VTA stage 不慢，也应降权。
   - 如果 VTA stage 被切得很碎且每段低于 `20-25 ms`，通常说明新增边界和调度开销不划算。

6. 最终 score：

```text
score_ms =
  max(max_cpu_stage_ms, total_vta_occupied_ms)
  + boundary_ms
  + dma_ms
  + dma_fragmentation_penalty_ms
  + sram_spill_penalty_ms
  + vta_stage_count_penalty_ms
  + cpu_tail_penalty_ms
```

推荐 penalty：

- `vta_stage_count_penalty_ms = max(0, vta_stage_count - 2) * 2.0`
- `cpu_tail_penalty_ms = max(0, cpu_tail_ms - 75) * 0.5`
- `dma_fragmentation_penalty_ms` 使用 calibrated DMA bucket 估算，不用固定常数。
- `sram_spill_penalty_ms` 只在 utilization 接近上限或出现 tile spill 风险时启用。

### Next candidates to prioritize

```text
core family:
  3_9__10_19
  3_9__10_18
  3_8__10_19
  3_8__10_18

local variants:
  3_9__10_17
  3_9__10_16
  3_8__10_17
  3_7__10_19
  3_7__10_18

early-start alternatives:
  1_7__8_13
  1_7__8_14
  1_6__8_13
  1_6__8_14

limited three-island controls:
  1_4__5_10__13_14
  1_4__5_8__10_13
  1_4__5_12__13_13
```

这些不是因为它们 stage count 特定，而是因为它们在实测中更可能满足：

- CPU early stage 不超过约 `85-95 ms`
- VTA total occupancy 不超过约 `90-100 ms`
- CPU tail 不超过约 `75-80 ms`
- PS-PL total bandwidth 不低于约 `1.5 GB/s`
- DMA fragmentation score 不显著高于 `1.3`
- SRAM peak utilization 不触发 spill

## Practical Takeaways

- 当前最佳切图不是更多 stage，而是让 CPU early、全局 VTA 占用、CPU tail 在吞吐上平衡。
- VTA 适合连续 residual/stride2/skip-proj 主干块，尤其 index `3-19` 中间段。
- CPU 适合 early high-resolution stem/layer1_block0 和 tiny head。
- 有效 `cpu/vta/cpu` 是最值得优先搜索的结构；`cpu/vta/cpu/vta/cpu` 是补充；更多有效设备段目前只适合少量探索。
- 后续若要提升超过 `10.5 fps`，优先优化 `stage0_cpu`，或者尝试专门降低 early CPU stage 的成本，而不是继续增加 VTA 切图数量。
