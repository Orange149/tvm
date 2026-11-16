"""Protocol and execution identities, without performance-based selection."""
from copy import deepcopy
from pathlib import Path
import random

from cpu_vta_partitioning.common import digest, read, sha256
from yolo_validation.common import ROOT, LOCAL_CONFIG, frozen

PARENT = ROOT / 'artifacts/yolo_wait_diagnosis_01/capsule'
OUTPUT = ROOT / 'artifacts/yolo_affinity_check_01'
RUN = ROOT / 'runs/yolov3_tiny416_affinity_check_01'
ENV = dict(cores=4, frequency_khz=1066666, poll_sleep_ns=1000,
           post_start_sleep_ns=1000, spin_count=0, exclude_worker0=1, bind_threads=1)


def protocol():
    return dict(experiment='yolov3_tiny416_affinity_check_01', revision=1,
        seed=20261110, input_index=0, rounds=2, queue_depth=2,
        qualification=dict(warmup=0, scored=1, tail=0),
        component=dict(warmup=2, scored=8, tail=0),
        performance=dict(warmup=8, scored=64, tail=32), environment=deepcopy(ENV),
        wait_policy='original_without_adapter', anchors=False, drift_rejection=False,
        automatic_remeasurement=False, automatic_reboot=False,
        initialization_affinity_probe_every_process=True,
        detailed_timing_probe=False, maximum_native_sessions=20,
        complete_DNN_inferences=836, isolated_CPU_stage_calls=44,
        no_scores_DP_or_Top20_changes=True, old_performance_reused=False,
        primary_scope='tensor ready through all 8 graph outputs; excludes preprocessing and NMS',
        migration_scope='permission to migrate, not a measured migration trajectory')


def identified(value):
    value = deepcopy(value)
    value.pop('id', None)
    value['id'] = 'cfg_' + digest(value)[:24]
    return value


def configurations():
    parent = read(PARENT / 'plans/configurations.json')
    result = dict(pipelines={}, components={})
    for allocation in ('overlap4', 'split3'):
        for mode in ('pinned', 'group'):
            cfg = deepcopy(parent['pipelines'][allocation])
            cfg['affinity_mode'] = mode
            for stage in cfg['stages']:
                if stage['device'] == 'cpu':
                    stage['worker_core_map'] = list(stage['mask']) if mode == 'pinned' else []
                    stage['caller_core'] = stage['mask'][0] if mode == 'pinned' else None
            name = allocation + '_' + mode
            result['pipelines'][name] = identified(cfg)
            component = deepcopy(cfg)
            component['stages'] = [deepcopy(cfg['stages'][0])]
            result['components'][name] = identified(component)
    return result


def validate_config(value, programs, *, component=False):
    if value != identified(value):
        raise ValueError('affinity configuration identity differs')
    choices = configurations()['components' if component else 'pipelines']
    if value not in choices.values():
        raise ValueError('configuration outside the four frozen matched cases')
    for stage in value['stages']:
        program = programs[stage['program']]
        if any(program[k] != stage[k] for k in ('device', 'begin', 'end')):
            raise ValueError('program interval differs')
        if stage['device'] == 'cpu' and stage['threads'] != len(stage['mask']):
            raise ValueError('thread budget differs from the frozen group')
    return value


def task_order():
    names = sorted(configurations()['pipelines'])
    random.Random(protocol()['seed']).shuffle(names)
    components = [dict(scope='component', case=n, phase=p, round=0)
                  for n in names for p in ('qualify', 'measure')]
    qualifications = [dict(scope='pipeline', case=n, phase='qualify', round=0) for n in names]
    rounds = [dict(scope='pipeline', case=n, phase='measure', round=r)
              for r, order in ((1, names), (2, list(reversed(names)))) for n in order]
    return components + qualifications + rounds


def sources():
    import cpu_vta_partitioning
    from yolo_validation.common import source_identity
    files = source_identity()
    paths = [ROOT / 'run_yolo_affinity_check.py',
             *sorted(Path(__file__).parent.glob('*.py')),
             ROOT / 'tests/test_yolo_affinity_check.py']
    files.update({str(p.relative_to(ROOT)): sha256(p) for p in paths})
    base = Path(cpu_vta_partitioning.__file__).parent
    files.update({'@base/' + str(p.relative_to(base)): sha256(p) for p in sorted(base.rglob('*.py'))})
    return files


def assert_prepared(output):
    prepared = read(Path(output) / 'PREPARATION.json')
    if prepared['sources'] != sources() or prepared['protocol'] != protocol():
        raise ValueError('frozen affinity controller changed; use a new batch')
    for path, expected in prepared['source_files'].items():
        if sha256(path) != expected:
            raise ValueError('archived source input changed: ' + path)
    return prepared
