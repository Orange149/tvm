"""Protocol and identities, independent of archived transfer experiments."""
from pathlib import Path
from cpu_vta_partitioning.common import digest, read, sha256, write

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
ASSETS = ROOT / 'artifacts/yolo_transfer_01'
PROFILE = ASSETS / 'profile_bundle_bridge02'
OLD_RUN = ROOT / 'runs/yolov3_tiny416_component_profile_02'
DEFAULT_OUTPUT = ROOT / 'artifacts/yolo_validation_01'
DEFAULT_RUN = ROOT / 'runs/yolov3_tiny416_validation_01'
LOCAL_CONFIG = PROJECT / 'configs/local.json'
HARDWARE = ROOT / 'artifacts/frame_stream_top20_static_validation/ddr.json'
METHODS = ('tarnawski', 'coarse_host', 'fine_host')
LABELS = dict(tarnawski='Tarnawski单线程物理核适配', coarse_host='粗粒度CPU模型', fine_host='本文主模型')

def protocol():
    return dict(experiment='yolov3_tiny416_validation', revision=1, seed=20261109,
        methods=list(METHODS), unit_count=14, max_vta_islands=3, input_index=0,
        calibration=dict(warmup=2, scored=8, tail=0),
        qualification=dict(warmup=0, scored=1, tail=0),
        performance=dict(warmup=8, scored=16, tail=4),
        confirmation=dict(warmup=8, scored=64, tail=32), queue_depth=2,
        max_candidates=60, max_confirmation_candidates=9,
        anchors=False, drift_rejection=False, automatic_remeasurement=False,
        automatic_reboot=False, runtime_changes=False, host_concentration=False,
        new_fusion_search=False, pipeline_FPS_for_ranking=False,
        old_performance_reused=False, single_input_single_boot_pipeline=True,
        primary_scope='input tensor ready through all 8 graph outputs; excludes image loading/preprocessing/box parsing/NMS',
        cpu_shares='isolated whole-session thread CPU proportions including startup probe and warmup; uniform fallback if unobservable',
        core_permutation='homogeneous-core approximation, not every mask measured',
        unmodeled_work='producer/collector, output bookkeeping and scheduling are not zero; unmodeled separately',
        traffic_scope='measured VTA requested DMA bytes; CPU logical frontier proxy; neither is total DDR/AXI traffic')

def frozen(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError('immutable artifact differs: ' + str(path))
    else:
        write(path, value)

def source_identity():
    files = [ROOT / 'run_yolo_validation.py']
    for directory in ('yolo_validation', 'yolo_transfer', 'staged_ablation',
                      'cpu_policies/open_resources', 'resource_envelope'):
        files.extend(sorted(p for p in (ROOT / directory).rglob('*')
                            if p.suffix in ('.py', '.cc', '.h')))
    return {str(p.relative_to(ROOT)): sha256(p) for p in files}

def assert_sources(output):
    manifest = read(Path(output) / 'PREPARATION.json')
    if manifest['sources'] != source_identity() or manifest['protocol'] != protocol():
        raise ValueError('controller or frozen method changed; create a new batch')
    if sha256(Path(manifest['profile_bundle'])/'manifest.json')!=manifest['profile_manifest_sha256']:
        raise ValueError('profile capsule manifest changed')
    return manifest
