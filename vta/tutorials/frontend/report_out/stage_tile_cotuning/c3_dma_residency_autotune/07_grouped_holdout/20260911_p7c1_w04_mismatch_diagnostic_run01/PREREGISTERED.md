# P7c1 W04 mismatch diagnostic

Frozen after the P7c failure and before this diagnostic execution.

- Diagnostic runner SHA-256: `5ec21a27639f287f72304fb1957c434938fe015e75045fce52e59c2b429a5067`.
- Target: the failed W04 `input_stationary` ConfigSpace-index-139 candidate `906c1bdc...`, repeated three times.
- Control: the already-passed W04 `original` ConfigSpace-index-139 candidate `056aba50...`, once immediately before and once after the target.
- Each execution uses the same three frozen seeds. Output is prefilled with alternating nonzero values to expose missing stores.
- The diagnostic collects correctness only. It must not produce a performance label or replace the failed candidate.
