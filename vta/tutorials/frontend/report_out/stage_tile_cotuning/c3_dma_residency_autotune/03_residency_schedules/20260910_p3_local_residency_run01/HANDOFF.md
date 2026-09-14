# Handoff

This run is closed as `partial_completed_input_only`.

Use `results.json` for the accepted local DMA matrix and `outer_weight_failure.log` for the rejected design. The fuller TIR hashes and original-schedule equivalence record remain one directory above in `p3_local_results.json`.

Any successor must start a new run directory. It must not restore outer-cache promotion unless it independently proves VTA dependency legality, DMA compact/2D-pattern legality, SRAM capacity, FSim correctness, and actual weight-LOAD reduction. Until then, the thesis-safe claim is limited to the input-stationary mechanism and the discovered weight-lifetime legality boundary.
