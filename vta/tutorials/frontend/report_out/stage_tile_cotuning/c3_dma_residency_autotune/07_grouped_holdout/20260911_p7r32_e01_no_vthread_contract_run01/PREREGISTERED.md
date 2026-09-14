# P7R32 E01 no-vthread causal contract

- Frozen before executing config 301 on FPGA.
- Config 301 matches failing config 877 in `h3-w21-ci1-co5` and differs only
  in `oc_nthread=1` versus `oc_nthread=2`.
- Purpose: distinguish a virtual-thread hardware hazard from a 2-D store-shape
  hazard. Three exact-output seeds, no latency or FPS.

