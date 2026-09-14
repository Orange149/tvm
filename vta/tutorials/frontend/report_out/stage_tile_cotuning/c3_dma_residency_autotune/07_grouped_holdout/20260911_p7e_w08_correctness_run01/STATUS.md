# Status

Infrastructure-stopped after 18/21 candidates had passed all three exact checks (54/54). Candidate 19 was never executed: allocation of its weight NDArray failed because the original runner repeatedly allocated board buffers until the fixed u-dma-buf pool was exhausted.

This is not a candidate wrong answer. No performance measurement was collected. The immutable 18-candidate prefix is retained; positions 19--21 will be run exactly once with a versioned resume runner that reuses one board-buffer set.
