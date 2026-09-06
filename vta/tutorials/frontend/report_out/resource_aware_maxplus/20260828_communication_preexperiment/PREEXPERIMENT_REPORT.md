# RAMPS Communication Go/No-Go Pre-Experiment

This is an oracle-service mechanism test. It does not establish zero-shot static prediction.
The canonical paper-ready evidence is maintained in `../PAPER_EVIDENCE.md`.

## Direct Runtime-Profiler Communication Ablation

This analysis uses the saved Host-to-VTA and VTA-to-Host copy timers at the runtime boundary. It does not infer communication from whole-stage `set/get` time.

- Records with direct profiles: `200`.
- Shared-compute MAE: `17.3617 ms`.
- With direct boundary copy MAE: `12.8220 ms`.
- MAE improvement: `4.5397 ms`, grouped 95% CI `[1.8054, 7.6966]`, one-sided p `0.000200`.
- Spearman: `0.7700` without communication, `0.9310` with communication.
- Oracle-service regret@1: `0.1709` without communication, `0.0000` with communication; evaluations to 95% oracle: `4` versus `1`.
- These low-budget values consume candidate-specific stage/copy timings. They are an upper-bound mechanism result, not a zero-feedback search result.
- Mechanism decision: **communication_mechanism_supported**.
- This stops the old `set+get` board protocol; it does not claim that full B3 has already been validated.

## Decision

- Status: **communication_supported_old_protocol_stopped**

- [x] `all_200_historical_candidates_have_direct_profiles`
- [x] `direct_copy_mae_improvement_ci_positive`
- [x] `direct_copy_one_sided_p_below_0_05`
- [x] `communication_aware_rank_correlation_not_worse`

This decision permits continued RAMPS identification with direct communication features. It does not validate the complete B3 event graph or cross-model zero-shot prediction.
