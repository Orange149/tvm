# P7R10 E00 local scan attempt

- Status: `aborted_no_result`
- Board/RPC: not used
- Cause: the legacy exact DMA extractor expanded every surrounding loop and did not reach the first
  100 candidates within two minutes on the larger E00 geometry.
- Recovery: add a semantics-equivalent compact extractor that folds descriptor-invariant loops into
  multiplicity; validate its totals against the legacy extractor before starting a new run directory.
