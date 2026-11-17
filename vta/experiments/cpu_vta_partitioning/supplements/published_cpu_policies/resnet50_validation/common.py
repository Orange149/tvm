"""Frozen ResNet50 protocol, without TVM, transport, or performance imports."""
from pathlib import Path

from cpu_vta_partitioning.common import digest, read, sha256, write

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
ARCHIVED_OUTPUT = ROOT / 'artifacts/resnet50_v2_validation_01'
ARCHIVED_RUN = ROOT / 'runs/resnet50_v2_validation_01'
DEFAULT_OUTPUT = ROOT / 'artifacts/resnet50_v2_validation_native_history_02'
DEFAULT_RUN = ROOT / 'runs/resnet50_v2_validation_native_history_02'
LOCAL_CONFIG = PROJECT / 'configs/local.json'
SOURCE_CAPSULE = ROOT / 'artifacts/run_scope_tail_comparison_20261112_01/resnet18/capsule'
HARDWARE = ROOT / 'artifacts/frame_stream_top20_static_validation/ddr.json'
METHODS = ('tarnawski', 'complete')
LABELS = {'tarnawski': 'Tarnawski platform adaptation', 'complete': 'Frozen complete model'}
EXPECTED = {'units': 38, 'cpu_modules': 314, 'vta_modules': 299,
            'components': 2751, 'dma': 299, 'max_candidates': 40}


def protocol():
    return dict(experiment='resnet50_v2_validation_native_history_02', revision=3,
        network='resnet50_v2', input_shape=[1, 3, 224, 224], input_index=0,
        methods=list(METHODS), max_vta_islands=3, seed=20261116, top_k=20,
        calibration=dict(warmup=2, scored=8, tail=0),
        qualification=dict(warmup=0, scored=1, tail=0),
        performance=dict(warmup=8, scored=16, tail=24), queue_depth=2,
        rounds=1, max_candidates=40, max_pipeline_sessions=80,
        max_pipeline_DNN_inferences=1960,
        environment=dict(cores=4, frequency_khz=1066666, poll_sleep_ns=1000,
                         post_start_sleep_ns=1000, spin_count=0, exclude_worker0=1, bind_threads=1),
        model_switches=dict(thread_weights=True, host_cpu_demand=True,
                            team_guard=True, host_concentration=True),
        compilation_policy='native AutoTVM ApplyHistoryBest, as frozen ResNet18: model then target-key with exact workload; no new tuning',
        deployment_hash_timeout_seconds=1800,
        candidate_domain='all qualified intervals, nine contexts, all masks, open and compatible serial groups',
        cpu_shares='whole isolated native session, including warmup/startup probe; uniform fallback if unobservable',
        core_permutation='same-cardinality homogeneous-core approximation, explicitly not all masks measured',
        primary_scope='input tensor ready through all network outputs; no image preprocessing',
        search=dict(budget_seconds=7200, memory_bytes=8 * 1024**3,
                    merge='exact state and integer resource labels; canonical first20 distinct prefixes',
                    pruning='strict static lower bound above twentieth complete score only',
                    independent_global_enumeration=False),
        anchors=False, drift_rejection=False, automatic_remeasurement=False,
        automatic_reboot=False, runtime_changes=False, pipeline_FPS_for_ranking=False,
        unmodeled_work='producer, collector, output bookkeeping, and scheduling are not zero; not completely profiled',
        traffic_scope='requested VTA DMA and CPU frontier proxy, not measured total DDR traffic')


def frozen(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError('immutable artifact differs: ' + str(path))
    else:
        write(path, value)
    return value


def controllers():
    files = sorted(Path(__file__).parent.glob('*.py'))
    files += [ROOT / 'run_resnet50_validation.py']
    for name in ('yolo_transfer/profile.py', 'yolo_transfer/traffic.py', 'yolo_transfer/ranking.py',
                 'common_top20/runtime.py', 'cpu_policies/open_resources/runtime.py',
                 'cpu_policies/open_resources/domain.py', 'cpu_policies/open_resources/costs.py',
                 'cpu_policies/open_resources/ddr.py', 'cpu_policies/open_resources/common.py',
                 'staged_ablation/model.py', 'staged_ablation/common.py',
                 'model_ablation/model.py', 'model_ablation/common.py',
                 'resource_envelope/model.py'):
        files.append(ROOT / name)
    files += [PROJECT / 'src/cpu_vta_partitioning' / p for p in
              ('common.py', 'transport/__init__.py', 'transport/scripts.py',
               'measurement/__init__.py', 'measurement/tasks.py', 'prepare/payload.py', 'prepare/build.py')]
    return {str(p.resolve()): sha256(p) for p in files}


def freeze_controllers(output):
    return frozen(Path(output) / 'CONTROLLERS.json',
                  dict(sources=controllers(), protocol=protocol()))


def assert_controllers(output):
    doc = read(Path(output) / 'CONTROLLERS.json')
    if doc != dict(sources=controllers(), protocol=protocol()):
        raise ValueError('controller changed: preserve old batch and create a new identity')
    return doc


def timed_deploy(ssh, bundle, run_dir):
    """Record actual deployment/checking wall time separately from measurement."""
    import time
    from cpu_vta_partitioning.common import now
    bundle, run_dir = Path(bundle), Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    directory = run_dir / 'deployments'
    directory.mkdir(parents=True, exist_ok=True)
    serial = len(list(directory.glob('*.json')))
    started_at, began = now(), time.monotonic()
    try:
        from .deployment_reuse import deploy
        source = ARCHIVED_OUTPUT / 'profile_capsule'
        prepared = bundle.parent / 'profile_capsule'
        if prepared != bundle and (prepared / 'manifest.json').is_file():
            if read(bundle / 'manifest.json').get('source_profile_bundle_id') == read(prepared / 'manifest.json')['bundle_id']:
                source = prepared
        remote = deploy(ssh, bundle, run_dir, source)
    except Exception as error:
        write(directory / f'{serial:04d}.json', dict(status='FAILED', started_at=started_at,
            elapsed_seconds=time.monotonic()-began, error_type=type(error).__name__,
            error=str(error), manifest_sha256=sha256(bundle / 'manifest.json')))
        raise
    write(directory / f'{serial:04d}.json', dict(status='COMPLETE', started_at=started_at,
        elapsed_seconds=time.monotonic()-began, remote=remote,
        manifest_sha256=sha256(bundle / 'manifest.json'),
        scope='deploy or verify existing payload; includes transfer, archive, and hash checks'))
    return remote
