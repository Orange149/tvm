# ResNet50-v2 five-policy ablation

This independent package reuses the frozen exact resource-label DP, modules,
runtime, native runner, 2,751 component records and 299 DMA records. It does not
modify the original `resnet50_validation` package or its sealed experiments.

The first four policies form CPU refinement × host CPU demand. The fifth is the
original four-on model. CPU refinement combines existing whole-session thread
shares and thread-team service, without separately attributing those two factors.
All five use the same 58,267,668,578-configuration domain. The singleton/full-mask
host shortcuts used by the DP are algebraically equivalent to all 15 subset
capacities in the independent scalar and C++ verifiers. No old full-pipeline FPS
is used by ranking. Old static paths are rescored as feasible upper thresholds;
they are not forced into the result.

From the published_cpu_policies directory, with the vta-resnet Python environment:

```
python run_resnet50_ablation.py audit
python run_resnet50_ablation.py validate --verify-search
python run_resnet50_ablation.py rank
python run_resnet50_ablation.py prepare
python run_resnet50_ablation.py validate
python run_resnet50_ablation.py board run --execute-board
python run_resnet50_ablation.py status
python run_resnet50_ablation.py analyze
```

`rank --model KEY --resume` resumes only an identical incomplete checkpoint.
The per-policy cumulative limit remains two hours and 8 GiB; exhausted searches
are uncertified, and no board measurements begin before all lists are frozen.

The five strict Top-20 lists are deduplicated with three fixed baseline bridges
(previously selected at budgets 1/5/20, original ranks 1/5/11). These bridges are
not a new measurement of the baseline's full Top-20. At most 103 configurations,
206 native sessions and 5,047 complete-network inferences are authorized. Each
configuration gets a one-frame qualification and 8+16+24 performance frames.
No new profile, automatic reboot, anchor, drift rejection or extra measurement.

All results are single-input/single-performance-boot observations. Shares include
warmup and initialization probes; same-size core sets use the frozen homogeneous
core approximation. These policies are planning proxies, not rigorous latency
bounds. Effects describe candidate selection, not a claim that a scoring switch
directly speeds up an unchanged executable configuration.

## Completed cohort and replay requirements

`artifacts/resnet50_v2_ablation_01/` and `runs/resnet50_v2_ablation_01/` contain
the completed 101-configuration union: 98 internal-policy configurations and
three baseline bridges, 202 sessions and 4,949 complete-network inferences.
The five policies' budget-1/5/20 FPS in this cohort are:

| Policy | Budget 1 | Budget 5 | Budget 20 |
|---|---:|---:|---:|
| Coarse CPU, no host demand | 3.061646 | 3.209041 | 3.422307 |
| Coarse CPU, host demand | 3.491608 | 3.552564 | 3.560153 |
| Fine CPU, no host demand | 3.539314 | 3.580395 | 3.580395 |
| Fine CPU, host demand | 3.580395 | 3.580395 | 3.583897 |
| Frozen complete model | 4.154139 | 4.154139 | 4.220706 |

The original forty-configuration comparison is a different cohort; its FPS must
not be substituted into this table. The three bridges do not establish a new
baseline Top-20 optimum. Read `analysis/RESULTS.json`, `analysis/VERIFICATION.json`
and `ranking/FREEZE.json` for exact values and evidence identities. All 98 internal
configurations allow partially overlapping CPU groups; this scoring ablation is
not an isolated comparison of overlapping and disjoint resource domains.

Result replay needs the original calibration, forty-configuration performance
and ablation artifacts/runs, not just this source tree. The cross-network analysis
also reads these two historical tables, which are deliberately outside this code
commit:

- `paper/my_paper/analysis/20261109_staged_ablation/selection_budgets.csv`
- `paper/my_paper/analysis/20261109_host_work_ablation/selection_budgets.csv`

Restore those original tables for the full `analyze` command; do not synthesize
them or combine different batches into one gain. All raw evidence and sealed
scientific sources remain unchanged. Local test commands and TVM prerequisites
are documented in `../resnet50_validation/README.md`.
