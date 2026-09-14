# P7R22 E01 correctness-only adaptation

Frozen after E01 config951 original failed correctness and before any E01 latency label.

The adaptation uses only correctness and hardware structure, not performance:

- Replace non-power-of-two `tile_co=5` with `tile_co=2`, for which the output-channel outer extent
  remains divisible by two virtual contexts.
- Request-shape candidate: config755, `(h,w,ci,co)=(21,1,12,2)`, candidate
  `e86a77d2789b6771e38340ad1205bb9581745a88c12719e3176619ddc94d54ba`.
- Contiguous control: config764, `(1,21,12,2)`, candidate
  `df49003918e585a0408125c64ee8f7232a936652adf3a94d4ddd2f62d473e96b`.
- Both first receive the fixed three-seed FSim gate. Only if both pass may a new real-FPGA
  correctness contract be frozen; timing remains forbidden until original and transformed modules
  are exact on all board seeds.
- E02's negative latency result is not used to choose between these two predeclared orientation
  strata; it only prevents claiming the simple request-shape rule as already validated.
