# P7R113 explicit analytic ΔT proxy

This is a post-hoc, fully offline leave-one-workload-out analysis of 56 P7Q same-tile pairs.
The fitted models are calibrated linear proxies with non-negative coefficients on physical cost deltas; they are not theoretical upper/lower bounds.
Because the paired ConfigEntity is identical, MAC count, tile shape, and the compute lower bound cancel from ΔT.

| Model | MAE (ms) | Spearman | nDCG@4 | regret@1 (ms) | regret@2 (ms) | regret@4 (ms) |
|---|---:|---:|---:|---:|---:|---:|
| constant mean | 0.410725 | -0.0958 | 0.4406 | 0.895892 | 0.601604 | 0.475752 |
| mode mean | 0.303263 | 0.4922 | 0.6164 | 0.417852 | 0.399492 | 0.224073 |
| mode + bytes/calls | 0.119648 | 0.8356 | 0.9487 | 0.327558 | 0.000915 | 0.000000 |
| + request shape | 0.155524 | 0.8075 | 0.9593 | 0.000000 | 0.000000 | 0.000000 |
| + command | 0.191115 | 0.7663 | 0.9616 | 0.000915 | 0.000000 | 0.000000 |

Ranking is within each held-out workload; lower predicted ΔT is better. Regret@k is the best observed ΔT among the first k predictions minus the held-out workload's ΔT oracle. nDCG uses linear gain `worst ΔT - candidate ΔT`. Exact prediction ties use stable candidate ID.

Claim boundary: all P7Q labels were already exposed, all measurements come from one boot, and only four ResNet18 workload groups are present. These results are model-development evidence, not prospective cross-workload or cross-boot validation.
