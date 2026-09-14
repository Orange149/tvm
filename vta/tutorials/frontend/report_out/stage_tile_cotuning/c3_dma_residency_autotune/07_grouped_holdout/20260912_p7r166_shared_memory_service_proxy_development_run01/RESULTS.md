# P7R166 shared-memory service-cost proxy development freeze

> This is exposed-label development, not confirmation. Y04 is part of the fit; Y01 latency is the frozen next holdout.

The earlier bytes-first rule failed on Y04 because it rewarded fewer bytes even when a candidate issued more DMA requests and submissions. The frozen replacement is:

`score = DMA bytes + 65,536 * DMA calls + 131,072 * max(submissions - 1, 0)`

The coefficients are byte-equivalent search penalties, not physical AXI bytes or measured hardware constants.

| workload | bytes-only regret | legacy lexicographic regret | service-proxy regret | proxy-selected mode |
|---|---:|---:|---:|---|
| W01 | 0.000% | 0.000% | 0.000% | weight_stationary_barrier |
| W04 | 3.591% | 0.000% | 0.000% | weight_stationary_barrier |
| W07 | 0.385% | 0.385% | 0.385% | paper_inspired_hybrid |
| W08 | 0.078% | 0.078% | 0.078% | original |
| Y00 | 0.000% | 0.000% | 0.000% | input_stationary |
| Y03 | 0.000% | 0.000% | 0.000% | original |
| Y04 | 75.350% | 37.956% | 0.000% | input_stationary |

Across the seven exposed workloads, the frozen proxy has maximum first-dispatch regret 0.385478% and mean 0.066166%.
It must now be tested unchanged on Y01; these seven workloads cannot establish generalization.
