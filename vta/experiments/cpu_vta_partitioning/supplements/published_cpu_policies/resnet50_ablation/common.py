"""Independent protocol, source identities and immutable output helpers."""
from pathlib import Path

from cpu_vta_partitioning.common import digest, read, sha256, write
from resnet50_validation.common import protocol as original_protocol

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'artifacts/resnet50_v2_validation_native_history_02'
PERFORMANCE = ROOT / 'artifacts/resnet50_v2_top20_board_20261117_01'
OUTPUT = ROOT / 'artifacts/resnet50_v2_ablation_01'
RUN = ROOT / 'runs/resnet50_v2_ablation_01'
LOCAL_CONFIG = ROOT.parents[1] / 'configs/local.json'
MODELS = ('coarse_no_host', 'coarse_host', 'fine_no_host', 'fine_host', 'complete')
LABELS = ('Coarse CPU, no host demand', 'Coarse CPU, host demand',
          'Fine CPU, no host demand', 'Fine CPU, host demand', 'Frozen complete model')


def settings(key):
    if key not in MODELS:
        raise ValueError('unknown ResNet50 ablation model: ' + str(key))
    fine = key.startswith('fine') or key == 'complete'
    return dict(key=key, label=LABELS[MODELS.index(key)], thread_weights=fine,
                team_guard=fine, host_cpu_demand=key in ('coarse_host', 'fine_host', 'complete'),
                host_guard=key == 'complete',
                distribution='session_weights' if fine else 'balanced',
                scalar_key='concentrated' if key == 'complete' else key)


def protocol():
    result = original_protocol()
    result.update(experiment='resnet50_v2_ablation_01', revision=1, seed=20261117,
                  methods=list(MODELS), models=[settings(k) for k in MODELS],
                  max_candidates=103, max_pipeline_sessions=206,
                  max_pipeline_DNN_inferences=5047, new_component_profiles=0,
                  original_pipeline_performance_reused=False,
                  bridge_scope='three previously selected baseline configurations, not a new baseline Top20',
                  model_switches=None)
    return result


def frozen(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError('immutable ablation record differs: ' + str(path))
    else:
        write(path, value)
    return value


def scoring_sources():
    from resnet50_validation import search
    result = search._implementation()
    for name in ('common.py', 'model.py', 'search.py'):
        path = Path(__file__).with_name(name)
        result['resnet50_ablation/' + name] = sha256(path)
    return result


def controller_sources():
    from resnet50_validation.common import controllers
    result = controllers()
    for path in [*sorted(Path(__file__).parent.glob('*.py')), ROOT / 'run_resnet50_ablation.py']:
        result[str(path.resolve())] = sha256(path)
    return result


def assert_controller(output):
    expected = dict(sources=controller_sources(), protocol=protocol())
    if read(Path(output) / 'CONTROLLERS.json') != expected:
        raise ValueError('ablation controller changed after sealing')
    return expected
