# P7R25 E00 discriminating correctness contract

- Frozen before executing E00 config 1081 on the real FPGA.
- Purpose: distinguish a multi-cycle `cthread` slot-reuse hazard from the
  short-row request-shape hypothesis.
- Config 1081 has five output-channel groups after two-way virtual threading,
  but a contiguous width-oriented DMA geometry.
- Three exact-output seeds; no latency or FPS measurement.

