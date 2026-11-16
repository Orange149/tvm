"""Independent identities; no edits to archived controllers or measurements."""
from pathlib import Path
from cpu_vta_partitioning.common import read, sha256
from yolo_validation.common import ROOT, LOCAL_CONFIG, frozen

SOURCE = ROOT / 'artifacts/yolo_validation_01'
OUTPUT = ROOT / 'artifacts/yolo_cpu_policy_check_01'
RUN = ROOT / 'runs/yolov3_tiny416_cpu_policy_check_01'


def protocol():
    return dict(experiment='yolov3_tiny416_cpu_policy_check_01', revision=1,
        seed=20261109, input_index=0, unit_count=14, max_vta_islands=3,
        qualification=dict(warmup=0, scored=1, tail=0),
        performance=dict(warmup=8, scored=64, tail=32), queue_depth=2,
        max_candidates=10, rounds=1, anchors=False, drift_rejection=False,
        automatic_remeasurement=False, automatic_reboot=False,
        model='fine_host', host_concentration=False,
        fixed_family='resource_group; >=2 distinct pairwise disjoint CPU groups',
        ranking_reads_pipeline_FPS=False, old_performance_reused=False,
        scope='single input, single boot; all eight graph outputs; excludes preprocessing and NMS')


def source_identity():
    from yolo_validation.common import source_identity as old_sources
    import cpu_vta_partitioning
    result = old_sources()
    files = [ROOT / 'run_yolo_cpu_policy_check.py', ROOT / 'run_yolo_validation_gate_compat.py',
             ROOT / 'run_yolo_validation_resume.py', *sorted(Path(__file__).parent.glob('*.py'))]
    result.update({str(p.relative_to(ROOT)): sha256(p) for p in files})
    base = Path(cpu_vta_partitioning.__file__).parent
    result.update({'@base/' + str(p.relative_to(base)): sha256(p) for p in sorted(base.rglob('*.py'))})
    return result


def assert_prepared(output):
    doc = read(Path(output) / 'PREPARATION.json')
    if doc['sources'] != source_identity() or doc['protocol'] != protocol():
        raise ValueError('controller identity or protocol changed; use a new batch')
    for path, value in doc['source_files'].items():
        if sha256(path) != value:
            raise ValueError('frozen source input changed: ' + path)
    return doc
