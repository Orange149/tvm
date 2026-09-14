# Status

Complete. W04 input-stationary tile_h=14 with tile_w=1 and 2 produced wrong answers for all three seeds; tile_w=7 reused the earlier failure and tile_w=14 was rejected statically because 3136 accumulator vectors exceed the 2048-vector capacity. Together with the passing tile_h=7, tile_w=14 control, this indicates a schedule/geometry-dependent real-FPGA legality hazard rather than a simple raw-capacity threshold. The precise hardware cause remains unresolved. No timing was collected.
