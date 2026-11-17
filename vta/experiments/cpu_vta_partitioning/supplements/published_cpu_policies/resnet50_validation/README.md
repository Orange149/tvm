# ResNet50-v2 frozen-method validation

This batch preserves the four-on complete model (isolated session thread shares,
thread-team service, VTA host CPU demand, host concentration) and the distinct
Tarnawski single-physical-core adaptation. It does not add measured-FPS feedback,
annealing, a new runtime, or a new kernel autotuning campaign.

The completed calibration controller identity is `resnet50_v2_validation_native_history_02`.
Its modules and profile capsule are reused byte-for-byte from
`resnet50_v2_validation_native_history_01`. That deployment-only preflight
reached an inherited 600-second hash deadline before any new native measurement.
That controller gives the same full hash verification 1,800 seconds; it
does not relax checks or change native execution.
It restores the frozen ResNet18 compiler's native `ApplyHistoryBest` dispatch:
exact workloads, with model then target-key matching. An earlier extra target
string equality test wrongly rejected reusable records. That preparation batch
is preserved at `resnet50_v2_validation_01`, stopped after 27 isolated sessions
(270 stage calls, no complete-network inference). Its slow default-schedule
costs are not transfer-validation results. Read-only audit:

```bash
python run_resnet50_validation.py audit --archived
```

Unchanged assets and isolated records can be reused only after byte/placement/
input/window checks, retaining original task, boot and receipt identities.
The repair introduces neither a scoring change nor new kernel tuning.

The Gluon pretrained model is quantized once with scale 8 and first-convolution
exclusion. The typed DAG has 38 structural units; 741 interval recompositions
are checked. CPU/VTA search modules number 314/299. The 2,751 component settings
and 299 DMA records are local-stage measurements, not complete DNN pipelines.
All variants use their own real costs. The 58,267,668,578 complete combinations
are a domain cardinality, not a required DP state or board-measurement count.

## Local preparation

Use the `vta-resnet` Python environment from the `published_cpu_policies`
directory. Preparation never implicitly connects to the board.

```bash
python run_resnet50_validation.py audit
python run_resnet50_validation.py validate --verify-search
python run_resnet50_validation.py prepare --stage all
python run_resnet50_validation.py validate --seal
python run_resnet50_validation.py status
```

The independent ARM full graph and complete-frontier probe are compiled and
executed locally under QEMU. QEMU validates loading and numerical output, not
hardware DMA or real-kernel scheduling. Failed preflight identities are kept.

`validate --verify-search` actually executes adversarial small-domain independent
C++ enumeration and the new DP on frozen ResNet18 costs. Acceptance initializes
feasible bounds from the current isolated costs, without historical rankings,
then certifies the full domain. A historical rank file alone is not search acceptance.

## Exact state search

State boundaries carry covered position, next device, VTA-island count, frontier
identity and serial-group compatibility. Labels retain exact integer CPU-core,
host, VTA, DDR, stage-maximum, team and group demands. Only identical state and
label prefixes can merge, keeping the first 20 distinct canonical paths. Lower
bounds can prune only above the twentieth complete static score; equality is
retained. There is no scalar beam, approximate bucket or single-prefix Pareto
truncation. Source/input-bound resumable checkpoints and actual state/RSS counts
are recorded. The default cap is 7,200 seconds and 8 GiB. Incomplete search does
not authorize pipeline measurement.

ResNet50 final paths are independently rescored in C++. Small domains have full
independent enumeration; ResNet50 is not claimed to have a second independent
global brute-force Top-20 proof.

## Board campaign

After all local checks and the controller source seal have passed:

```bash
python run_resnet50_validation.py board run --execute-board
python run_resnet50_validation.py board resume --execute-board
python run_resnet50_validation.py status
python run_resnet50_validation.py analyze
```

The controller sequences 39 pilot components plus 3 DMA records, remaining
components and traffic, host ranking, immutable selection, and the shared
pipeline union. Pilot records belong to the full table and are not remeasured.
The component protocol is 2 warmup + 8 measured frames, with per-output checking
and a startup affinity probe in the same process. Source, graph, program,
runtime, placement, window and boot identities are retained.

The sealed controller has a known first-ranking limitation: it requests checkpoint
resumption even when no checkpoint exists. Its completed native measurements
remain valid. The historical `operations/continue_after_first_rank.py` and its
receipts stay with the original artifacts; that one-run helper is not a required
source dependency. After the controller has exited with exactly
`ValueError: no checkpoint to resume`, and only when there is no checkpoint,
the equivalent recovery uses the public commands:

```bash
python run_resnet50_validation.py rank
python run_resnet50_validation.py board resume --execute-board
```

If an identity-matched incomplete checkpoint already exists, use
`rank --resume` instead. Unrelated errors, a changed boot or an exhausted search
budget require inspection; these commands do not authorize new attempts or a
larger budget. Do not rerun ranking merely to read completed results. Accepted
measurements are replayed, not repeated. The original scientific Python sources
are retained byte-for-byte so their completed controller seals remain valid.

Both methods obtain strict Top-20. Their union is at most 40 configurations,
80 native sessions and 1,960 complete board inferences: one qualification frame
and 8 warmup + 16 timestamps + 24 trailing frames per configuration. Sixteen
timestamps give 15 intervals. Initializer/probe calls, reference inference and
component calls are separate accounts. No anchors, automatic frame extensions,
retests, hardware retuning or reboot are performed. Incorrect outputs, affinity,
hashes, frame identities or early stage exhaustion stop acceptance.

## Independent status monitoring

`monitor_resnet50_status.py` is outside the sealed scientific sources. When run,
it checks host-side progress and process liveness once per minute. Every ten
minutes it attempts a lightweight, read-only board resource snapshot. A busy
native-session lock defers that snapshot; a detected race discards its metrics.
Formal full-pipeline timing receives no additional resource SSH probes.

Reports are under the active artifact directory's `operations/health/`:
`latest.json`, `history.jsonl`, `alerts.jsonl` and `monitor.log`. Notices include
twenty minutes without accepted progress, exited processes, changed boot
identity, available memory below 256 MiB, and available storage below 512 MiB.
These are monitoring notices, not performance rejection or retry criteria.
The monitor cannot launch experiments, change parameters, clean files, reboot,
or automatically send chat messages. Resource snapshots are not peak-memory
proofs. The existing experimental acceptance rules remain unchanged.

From the `published_cpu_policies` directory, a one-time host-only check is:

```bash
python monitor_resnet50_status.py --once
```

The singleton lock rejects another continuous instance for the same batch.
No monitor is started by importing the code or by reading this document. Both
reported performance cohorts have completed; inspect their state before deciding
whether monitoring is needed.

The completed calibration artifacts and runs live under
`artifacts/resnet50_v2_validation_native_history_02/` and
`runs/resnet50_v2_validation_native_history_02/`. Old packages and results are not modified.
The generated report separates Top-1 selection, best-of-5/20 selection, numerical
prediction error, state-search cost and measurement cost. Results are one-input,
one-boot finite-window observations, not significant or universal performance
claims. No automatic paper changes, Git commit or push are part of this batch.

## Frozen selections measured after an external restart

The original calibration boot ended before any full-pipeline measurement. Its
component costs, DMA records and exact Top-20 selections remain frozen. The
separate `resnet50_v2_top20_board_20261117_01` execution cohort references those
original costs and records their original boot identities; all forty selected
pipelines are measured together in the new boot. This is a cross-boot profile
reference, not a same-boot isolated/pipeline comparison.

`run_resnet50_top20_board.py` reuses the existing runner, qualification, task
identity, measurement, recovery and acceptance functions. It adds no compile,
calibration, ranking or execution mechanism. The two methods' ordered Top-20
lists, their union and the measurement order are unchanged. Its separate run
identity rejects a further boot change. Outputs and state live under the new
cohort's `artifacts/` and `runs/` directories.

```bash
python run_resnet50_top20_board.py status
python run_resnet50_top20_board.py resume --execute-board
python run_resnet50_top20_board.py analyze
```

The performance cohort has its own read-only health monitor, started with
`monitor_resnet50_status.py --batch resnet50_v2_top20_board_20261117_01 --execute-board`.
During its formal `PIPELINE` phase that monitor reads only host-side state.

## Local tests and retained data

From `published_cpu_policies`, with the existing `vta-resnet` environment:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -c pyproject.toml -q tests/test_resnet50*.py
```

The pytest configuration supplies both this directory and the shared project
`src` directory; individual test files do not rely on another test changing
`sys.path`. Actual-graph checks additionally require the matching host TVM library
and repository Python modules. These tests load the graph, without board access.

The source commit does not contain weights, SDKs, local configuration, compiled
modules or raw results. Restore the original `artifacts/` and `runs/` directories
for result replay, including the frozen ResNet18 parent runtime capsule. Preserve
their manifests and original source paths; do not regenerate evidence to satisfy
a hash check. See `PREPARATION.md` for preparation prerequisites. Source seals,
module hashes and measurement records are deliberately not rewritten by cleanup.
