# C3-P6d balanced board timing result

Status: **completed; current selected mechanisms do not improve latency**.

All 270 planned samples were collected: 30 samples for each of 9 candidates. Every workload permutation and every within-workload candidate permutation occurred exactly five times. The boot ID, FPGA state, u-dma-buf size, RAM RPC process, and SD read-only state remained valid through the final check; no new storage error appeared.

| Workload | Candidate | Median ms | IQR ms | Relative to incumbent | Frozen decision |
|---|---|---:|---:|---:|---|
| W00 | original | 5.844164 | 5.836311–5.870826 | reference | reference |
| W00 | input-stationary | 6.460745 | 6.450912–6.469775 | -10.55% | regressed |
| W00 | weight-barrier | 6.746918 | 6.733275–6.767482 | -15.45% | regressed |
| W02 | original | 5.376428 | 5.361788–5.400031 | reference | reference |
| W02 | input-stationary | 6.384604 | 6.377104–6.406324 | -18.75% | regressed |
| W02 | weight-barrier | 7.022020 | 6.988157–7.043478 | -30.61% | regressed |
| W09 | original | 1.103096 | 1.099549–1.108841 | reference | reference |
| W09 | input-stationary | 1.121171 | 1.106074–1.125641 | -1.64% | equivalent |
| W09 | weight-barrier | 2.244652 | 2.237955–2.254522 | -103.49% | regressed |

The result separates three propositions that must not be conflated:

1. The mechanisms are real: board counters show that they change and often reduce DMA traffic.
2. The mechanisms are correct: P6c passed all 27 exact checks.
3. The selected mechanisms are not faster than the protected incumbent: five regress and one is equivalent under the frozen thresholds.

The clearest counterexample is W09 weight-barrier. Total LOAD bytes fall from 217,600 to 177,664 per inference, but LOAD calls rise from 64 to 226 and synchronization instructions from 121 to about 590; latency consequently doubles. W00 input-stationary also cuts LOAD bytes by 32.7% and LOAD calls by half yet loses 10.6%, showing that tiling/parallelism and accelerator utilization can dominate byte count.

## Interpretation boundary

P6d compares each selected candidate with the strongest original incumbent, as preregistered. The candidates do not always share the incumbent's tile/thread configuration—for example, W00 changes `oc_nthread` from 2 to 1. Therefore P6d rejects these **composite candidate configurations**, but does not isolate the causal cost/benefit of residency alone. A same-tile mode-0 paired ablation is the next required diagnostic.

The board runtime reported that `ext_dev` has no specialized timer and uses the default timer. This is the established VTA measurement path and the balanced within-boot comparison is tight, but no cross-boot or absolute-cycle claim is made.

`samples.jsonl` is the untouched observed output. TVM documents that `time_evaluator` invokes the function `1 + number × repeat` times, so with `number=1, repeat=1` the profiler counters cover two invocations. `samples_counter_normalized.jsonl` preserves raw counters and divides them by two; latency values are unchanged. The collection script was corrected after this finding so future runs emit both raw and normalized fields directly.
