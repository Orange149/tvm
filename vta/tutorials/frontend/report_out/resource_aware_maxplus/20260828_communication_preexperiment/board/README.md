# Archived Set/Get Pilot

## Status

The 37 completed native processes under `session001` and `session002` are an
archived instrumentation pilot. Do not resume the planned 72-process protocol.

The raw runs remain useful for:

- native runner and correctness stability checks;
- completion-interval parser validation;
- inter-process variability for the 13 candidates measured twice;
- diagnosing why the assumed CPU core-pool constraint overpredicts cycle time.

Observed pilot facts that remain valid within this limited scope:

- 37/37 completed processes passed correctness;
- all 24 candidates completed session 1 and 13 completed session 2;
- for those 13 repeated candidates, the median inter-session cycle difference
  was 5.31% and the maximum was 8.47%;
- the recorded failures were SSH/network failures before runner launch, not
  candidate runtime failures.

They must not be used for:

- communication-cost significance tests;
- RAMPS parameter fitting or model selection;
- B0/B1/B2/B3 publication comparisons;
- completing or reporting the registered 72-process experiment.

The reason is methodological: stage `set_input_ms + get_output_ms` combines
copy, synchronization, allocation, output-handle work and runtime dispatch. It
is not a direct communication service measurement. The original B2/B3 pilot
also approximated CPU demand as `threads * run_ms` and FIFO constraints with
closed-form penalties; neither quantity was identified from controlled data or
constructed as a max-plus timed event graph.

The valid communication-value result is based on direct VTA runtime copy
counters from all 200 historical ResNet18 candidates. See
`../PREEXPERIMENT_REPORT.md` and
`../DIRECT_COMMUNICATION_GO_NO_GO.json`.
