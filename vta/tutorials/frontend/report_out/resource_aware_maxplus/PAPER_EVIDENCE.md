# RAMPS Paper Evidence Ledger

## Purpose

This file is the canonical source for paper-ready RAMPS evidence. Add future
CPU contention, FIFO, zero-shot, energy, and hardware-sensitivity results here
after their corresponding acceptance gates pass. Dated experiment directories
remain the source of raw data and reproducibility artifacts; conclusions should
not be copied into multiple independent reports.

Evidence status has three levels:

- **Established**: supported by a completed, reproducible statistical test.
- **Pilot only**: useful diagnostic data that is excluded from model fitting and
  publication main results.
- **Not established**: a planned claim whose required experiment has not passed.

## Evidence Status

| Claim | Status | Current evidence |
|---|---|---|
| CPU/VTA boundary communication must be modeled | **Established** | 200 ResNet18 native pipeline profiles |
| More VTA islands increase direct boundary-copy cost | **Established on the current ResNet18/VTA system** | Direct runtime counters grouped by 1/2/3 islands |
| The current CPU core-contention model is accurate | **Pilot only** | Atomic CPU wall/core-ms plus nested-prefix capacity reproduces the order of five measured candidates, but cycle MAPE is 14.66% and no new unseen rank-1 has completed board validation |
| The current FIFO/queue-depth model is accurate | **Not established** | No controlled queue-depth identification yet |
| Full RAMPS B3 outperforms its ablations | **Not established** | Direct communication term passed; combined event graph has not passed |
| ResNet-trained RAMPS ranks an unseen DNN zero-shot | **Not established** | Direct ResNet price transfer fails; after six YOLO serial-component profiles, the historical oracle topology is static rank 6, but there is no exact thread match or prospective board result |
| A portable schema-v3 hardware profile is calibrated | **Not established** | Two-layer code contract exists; legacy Stage 4b fails correctness, precision and service-fit gates |
| RAMPS reduces board-search cost after amortizing calibration | **Not established** | No frozen support-subset cost ledger or prospective Top-K experiment yet |
| Default-static M0/M1 reduce ResNet measured-pool board evaluations | **Rejected in current form** | Need 60/67 evaluations to 95% pool oracle versus random median 33 |
| The ResNet V1 k-best implementation matches its static objective oracle | **Established** | Top-20 candidate IDs and scores exactly match streaming enumeration over 972,528 legal execution configurations |
| Per-stage TVM threads affect pipeline throughput | **Established on one ResNet18 topology** | First-stage `threads=1/2/3/4` achieve 5.413/6.237/9.312/10.449 FPS under the frozen runtime |
| Six target-component profiles recover a strong YOLO topology shortlist | **Pilot only** | The historical 5.440 FPS oracle topology appears at static rank 6 among 105,696 configurations; evidence is retrospective and topology-level |
| The current static model predicts absolute FPS | **Not established** | The raw score is a resource lower-bound reciprocal; the roughly 1.42 correction comes from complete-candidate outcomes and is excluded from the formal model |
| DDR/PS-PL/cache and synchronization residuals are physically identified | **Not established** | P7A--P7D now require component-only matched controls before absolute FPS is reported |
| A systematic missing cost transfers across the two ResNet batches | **Established as a diagnostic** | Additive offsets trained in opposite directions are 36.459/33.764 ms and achieve 5.535%/6.046% held-out MAPE; physical ownership is not yet identified |

## Publication Decision Rule

The main claim is candidate triage, not exact per-candidate latency regression. Publication evidence must
therefore prioritize `regret@K`, Top-K recall, board evaluations required to reach 95% of oracle throughput,
and end-to-end calibration/build/measurement cost. MAE/MAPE remain service-model diagnostics.

Protocol v4 defines a 212-case semantic pool, but neither full-pool compilation nor full-pool measurement is
part of the current plan. H2 audits 20 builder-equivalence templates. H3 freezes at most 80 board cases before
target-DNN throughput labels are read: at most 64 fit cases and at least 16 grouped holdout cases, selected by
identifiability, observed lowering-signature coverage and grouped D-optimal information.

**2026-09-01 scope update:** the complete H2/H3 hardware-profile branch is conditional rather than the next
execution stage. The current next step is a leakage audit and low-budget regret curve for M0/M1 on the
existing measured candidate pool. The 200-candidate direct-profile ablation remains mechanism evidence only:
its candidate-specific stage/copy timings cannot be used as zero-feedback ranking inputs.

**2026-09-01 P1 result:** the leakage gate passes for the historical M0/M1 scorer because candidate outcomes
are used only as retrospective labels. However, all 200 records use `default_static_estimate_no_calibration_host`.
M0 and M1 require 60 and 67 evaluations to reach 95% of the measured-pool oracle, compared with a uniform
random median of 33. The apparent M2 result of 11 uses the rejected `wall_ms * requested_threads` core-demand
proxy and is not publication eligible. The next branch is a 16-case minimum service qualification, not full H2.

For the `m`th DNN sharing one HardwareProfile, report both:

```text
C_RAMPS(m) = C_profile / m + C_static + K * C_board
C_exhaustive = N * C_board
```

If full RAMPS does not improve prospective regret/board evaluations over stage balance and calibrated-DMA,
or its amortized total cost is not lower, the full model remains a negative result or ablation rather than an
established contribution.

## Stage 4b Legacy Calibration Review

The five-session, 19,300-sample Stage 4b archive remains useful for diagnosing
runtime behavior, but it is not a portable service model:

- six VTA operator cases have inconsistent output hashes across sessions;
- 21/193 points exceed the precision gate when CPU/VTA operator service uses
  the declared `run_ms` accounting boundary;
- the operator matrix uses ResNet-shaped semantic buckets;
- no grouped leave-one-shape-out service function has passed validation.

The archived `service_model.json` is therefore marked
`legacy_diagnostic_only=true`, and all publication gates remain false. The
replacement requires a model-independent `hardware_profile.json` and static
`PartitionWorkload` extraction.

## Established Result E1: Direct Communication Awareness

### Paper-ready claim

On the evaluated Zynq UltraScale+ MPSoC with the VTA native pipeline runtime,
explicitly modeling measured CPU--VTA boundary-copy service time significantly
improves steady-state cycle prediction over a shared-compute-only model.

This is a mechanism-level oracle-service result. It establishes the need for a
communication term, not yet a fully static or cross-model predictor.

### Measurement

All 200 measured ResNet18 candidates contain VTA runtime profiler outputs:

```text
profile/serial/single_run_status.json
profile/serial/single_run_events.json
profile/pipeline/benchmark_totals_status.json
profile/pipeline/benchmark_totals_events.json
```

The relevant counters are measured inside copy, cache-maintenance, DMA enqueue,
and wait functions:

- `mem_copy_from_host_bytes/us`
- `mem_copy_to_host_bytes/us`
- `flush_cache_bytes/us`
- `invalidate_cache_bytes/us`
- `load_buffer_2d_bytes/calls/enqueue_us`
- `store_buffer_2d_bytes/calls/enqueue_us`
- `synchronize_load_bytes/store_bytes`
- `device_run_wait_us`

These counters are more specific than whole GraphExecutor `set_input_ms` and
`get_output_ms`, which also include synchronization, dispatch, allocation, and
output-handle work.

### Ablation

For each candidate, the ablation compares:

```text
T_shared_compute = max(max CPU stage run, sum VTA stage run)

T_comm_aware = max(max CPU stage run,
                   sum VTA stage run
                   + direct Host/VTA boundary copy
                   + explicit coherence)
```

Multiple VTA stages are summed because the current runner serializes VTA
`set_input/run/get_output` with a global mutex. VTA-internal load/store DMA is
not added a second time because it is already contained in measured VTA
`run_ms`.

### Main result

| Model | MAE | RMSE | Spearman | Kendall | Top-10 regret |
|---|---:|---:|---:|---:|---:|
| Shared compute without direct boundary copy | 17.3617 ms | 20.1302 ms | 0.7700 | 0.6255 | 0.0000 |
| Shared compute with direct boundary copy | 12.8220 ms | 13.6655 ms | 0.9310 | 0.7911 | 0.0000 |

- MAE improvement: **4.5397 ms**, or **26.15%** relative to the baseline MAE.
- Grouping unit: VTA outer span, 23 groups.
- Grouped bootstrap repetitions: 5,000.
- 95% confidence interval: **[1.8054, 7.6966] ms**.
- One-sided bootstrap p-value: **0.000200**.
- Decision: **communication mechanism supported**.

### Scaling with partition granularity

Serial single-run direct Host/VTA copy:

| VTA islands | N | Median bytes | Median copy | Mean copy | P95 copy |
|---:|---:|---:|---:|---:|---:|
| 1 | 13 | 1,003,520 B | 3.431 ms | 3.324 ms | 3.879 ms |
| 2 | 78 | 1,856,512 B | 6.212 ms | 6.479 ms | 8.820 ms |
| 3 | 109 | 3,311,616 B | 10.959 ms | 10.788 ms | 14.286 ms |

Pipeline totals normalized per inference:

| VTA islands | Median copy | Mean copy | P95 copy |
|---:|---:|---:|---:|
| 1 | 3.017 ms | 4.220 ms | 9.775 ms |
| 2 | 5.831 ms | 6.113 ms | 8.634 ms |
| 3 | 11.366 ms | 10.856 ms | 15.342 ms |

Across all 200 serial profiles:

- Median Host-to-VTA copy: 4.974 ms.
- Median VTA-to-Host copy: 3.264 ms.
- Median total direct copy: 7.883 ms.
- P95 total direct copy: 13.831 ms.
- Maximum total direct copy: 15.392 ms.
- Median DMA load bytes: 10,573,824 B.
- Median DMA store bytes: 1,179,136 B.
- Median DMA load/store calls: 1,668.
- Explicit flush/invalidate time is zero on these coherent-path profiles.

### Contention example

`three_stage_f` requires 2.875 ms Host-to-VTA plus 0.298 ms VTA-to-Host
copy in a serial run. Under pipeline execution, the normalized values increase
to 18.569 ms plus 0.303 ms per inference. Thus copy service time is not a
constant `bytes / peak_bandwidth`; CPU/memory contention can change effective
communication bandwidth.

Concrete serial profiles:

| Candidate | H2D | D2H | VTA load bytes | VTA store bytes | Device wait |
|---|---:|---:|---:|---:|---:|
| `three_stage_e` | 802,816 B / 2.902 ms | 25,088 B / 0.084 ms | 12,025,856 B | 1,229,312 B | 60.193 ms |
| `three_stage_f` | 802,816 B / 2.875 ms | 100,352 B / 0.298 ms | 13,520,384 B | 1,630,720 B | 71.362 ms |

### Earlier HP/HPC evidence

Separate HP/HPC experiments established that communication is also sensitive
to coherence topology and access behavior:

- HP-only six-case mean explicit coherence overhead: 79.712 ms.
- HP-only maximum explicit coherence overhead: 226.452 ms.
- HPC/coherent explicit coherence overhead: 0 ms in matched measurements.
- HP and HPC effective DMA throughput: approximately 0.162 GB/s.
- Four HP ports increased effective throughput to 0.222 GB/s but did not remove
  non-coherent flush/invalidate cost.
- A 903,168-byte HP boundary incurred 5.258 ms flush cost.
- A 1,843,200-byte HP boundary incurred 57.571 ms flush cost.

The historical field `total_bw_gbps` is numerically decimal GB/s because its
implementation computes `bytes / us / 1000`; the field name is misleading.

## Why Full B3 Is Not Yet Established

The complete proposed predictor contains several mechanisms:

```text
B3 = hardware service model
   + shared-resource timed event graph
   + finite FIFO/buffer constraints
   + monotonic residual and uncertainty
```

E1 validates only one part: measured Host/VTA boundary communication improves
an oracle-service prediction when measured stage `run_ms` is already known.
The remaining claims require different evidence.

### CPU contention

The number of threads requested by a CPU stage is not equal to the number of
cores occupied for the entire `run_ms`. The archived pilot approximated CPU
demand as `threads * run_ms / 4`, which overpredicted cycle time for several
candidates. A valid model requires controlled thread/core experiments or CPU
occupancy/performance-counter measurements.

### FIFO and queue depth

Queue depth is the number of in-flight frame tokens allowed between adjacent
stages. A producer blocks when its output FIFO is full; a consumer blocks when
the FIFO is empty. The current closed-form FIFO penalties have not been derived
from an instantiated timed event graph or validated with controlled
`queue_depth = 1, 2, 4` experiments and measured blocking events.

### Full B3

Even if communication is individually useful, combining it with an inaccurate
CPU or FIFO term can make the full prediction worse. Full B3 is established
only when the combined, frozen model beats compute-only, communication-only,
and shared-resource ablations on held-out candidates.

### Cross-model zero-shot prediction

The current E1 calculation reads measured stage runtime and measured copy time
from the same ResNet18 candidate being evaluated. It therefore explains an
observed execution but cannot select an unmeasured candidate. Zero-shot means:

1. calibrate hardware and fit/freeze all parameters using ResNet18 only;
2. derive YOLO or SqueezeNet features from graph structure, operations, tensor
   bytes, layout, DMA calls, SRAM use, and candidate stage plan;
3. predict and rank those candidates before reading any board throughput label;
4. evaluate the frozen ranking once on the unseen model.

Until that protocol succeeds, the paper may claim that communication awareness
is necessary on the evaluated platform, but not that RAMPS already generalizes
across DNNs.

## Portability Contract

RAMPS must separate invariant algorithms from hardware-specific calibration and
DNN-specific static extraction.

### Invariant across DNNs and boards

- boundary-contract and correctness rules;
- physical service equations and units;
- shared-resource timed-event-graph construction;
- maximum-cycle-mean solver;
- uncertainty-aware candidate ranking and pruning;
- statistical evaluation protocol.

No model name, layer name, candidate ID, or measured candidate throughput may
be a prediction feature.

### Calibrated once per hardware fingerprint

- CPU wall performance and actual core-time demand by generic operator kind,
  thread count, shape and arithmetic intensity;
- shared CPU/DDR saturation curves;
- VTA compute/load/store service and overlap;
- DMA bandwidth/latency by size, stride, padding and transaction count;
- bridge, launch, submit, synchronization and cache-maintenance overhead;
- SRAM capacities, legal tile rules, runtime mutex and available resources.

The hardware fingerprint includes CPU/DDR/PL clocks, bitstream and VTA config,
runtime/compiler revision, target flags, memory-port topology and OS scheduling
configuration. A fingerprint change triggers recalibration, not algorithm
rewriting.

### Extracted from each new DNN without board throughput labels

- compiler-fused operator DAG and dependencies;
- logical/physical shapes, OP counts and tensor bytes;
- layout, padding, quantization and boundary adapters;
- estimated DMA calls, SRAM/tile traffic and legal device assignments;
- candidate cuts, CPU thread choices and FIFO memory requirements.

The expected workflow is therefore:

```text
new board -> automated hardware calibration -> frozen resource model
new DNN   -> static graph extraction -> safe candidate enumeration
          -> max-plus prediction/ranking -> build and measure only a small Top-K
```

ResNet18, YOLOv3-tiny and a future SqueezeNet evaluation are validation models,
not sources of hard-coded partition rules. Portability is established only if
the same algorithm and feature schema work after replacing the hardware
calibration JSON and DNN graph, without adding model-specific scoring constants.

## P8 Shared-Slot Zero-Copy Qualification

A frozen seven-stage ResNet18 stress case contains three VTA islands, six
CPU--VTA boundaries, and 9,031,680 bytes of framework materialization per
frame. In one boot, all output-equivalence, slot-generation, address-range,
and VTA-traffic gates passed. Two ordinary-copy baselines were measured and
must remain separate:

| Ordinary-copy baseline | B0 boundary API | B1 zero-copy API | B0/B1 latency | Observed reduction |
|---|---:|---:|---:|---:|
| Runner default safe copy, for historical-profile consistency | 16.324 ms | 0.111 ms | 178.855/163.093 ms | 15.762 ms (8.81%) |
| Fastest correct `memcpy`, for the primary ablation | 4.484 ms | 0.111 ms | 167.399/162.662 ms | 4.737 ms (2.83%) |

The default-runner result is close to the same partition's historical direct
copy observation of 17.239 ms. The previously reported 11.366 ms is the median
over 109 different three-island partitions, not a constant cost for every
three-island partition. VTA internal LOAD/STORE bytes are unchanged between B0
and B1, so they are not counted as eliminated traffic.

These are single-boot qualification observations, not publication-level
speedup claims. The optimized `memcpy` comparison is the primary zero-copy
ablation. Its 2.83% end-to-end latency reduction remains a single-boot stress-case
observation unless two additional boots produce a paired boot-level 95%
confidence interval excluding zero. The runner-default comparison exists only
to preserve comparability with the historical 200-partition dataset; its
diagnostic purpose is complete and it will not consume additional boots.

The cross-boot P8C result establishes the key distinction. Zero-copy reduced
boundary API service by 97.11% and framework materialization by 100%, but the
three-boot B0--B2 initiation-interval difference had a 95% confidence interval
of [-2.477, 1.363] ms. Thus zero-copy reduces serial latency when exposed copy
lies on the single-frame critical path, whereas steady-state throughput changes
only when that copy exceeds available pipeline slack or contributes to the
critical shared-resource demand. No steady-state FPS improvement is claimed.

Reproducibility artifacts are under
`v1_p8_latency_island_scaling/three_island_sessions/` and
`v1_p8_latency_island_scaling/three_island_sessions_runner_default/`.

## Archived Pilot

The incomplete 37/72 `set_input + get_output` board experiment is retained as
pilot-only raw data. It is excluded from communication inference, RAMPS fitting,
and publication main results because its timing proxy mixes multiple runtime
operations and its CPU/FIFO formulas were not identified.

Useful pilot observations are limited to:

- 37/37 completed processes passed correctness;
- 24 candidates completed session 1 and 13 completed session 2;
- repeated-candidate median inter-session cycle difference was 5.31%, maximum
  8.47%;
- failures were SSH/network failures before runner launch.

## Reproducibility Sources

| Artifact | Purpose |
|---|---|
| `20260828_communication_preexperiment/historical/direct_profile_metrics.json` | Exact E1 metrics and bootstrap result |
| `20260828_communication_preexperiment/historical/direct_profile_records.csv` | Per-candidate measured/predicted cycles and direct copy time |
| `20260828_communication_preexperiment/DIRECT_COMMUNICATION_GO_NO_GO.json` | Machine-readable E1 gate |
| `20260828_communication_preexperiment/PREEXPERIMENT_REPORT.md` | Generated concise E1 report |
| `20260828_communication_preexperiment/board/PILOT_ONLY.json` | Machine-readable exclusion of the 37-process pilot |
| `20260828_communication_preexperiment/board/README.md` | Allowed and prohibited uses of pilot data |
| `RAMPS_SYSTEM_MODEL.md` | Canonical theory, mathematical model, portability contract and references |

## Safe Paper Wording

Supported wording:

> Adding directly profiled CPU--VTA boundary-copy service time reduces cycle
> prediction MAE by 26.15% over a shared-compute-only oracle model across 200
> ResNet18 partitions (4.54 ms absolute improvement; outer-span grouped 95%
> CI 1.81--7.70 ms; one-sided p=0.0002), while increasing rank correlation
> from 0.770 to 0.931.

Unsupported wording at the current stage:

> The complete RAMPS model accurately predicts arbitrary DNN partition
> throughput without board measurements.
