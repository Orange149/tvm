# Selected full-graph command capacity

- Status: `passed`
- Candidate: `dc939aafdfbb9e7e84172f7ce12b651551b75e95753ca7e33c30967750f7abb2` (`input_stationary`)
- Observed peak: instruction 1065280 B, UOP 1320 B
- Page-aligned backing: instruction 1069056 B, UOP 4096 B
- Positive replay: three inputs and all outputs match the peak run exactly
- Negative replay: one page removed from `insn` and rejected before the violating device submission
- Default RPC restored; one exact post-restore health inference passed
