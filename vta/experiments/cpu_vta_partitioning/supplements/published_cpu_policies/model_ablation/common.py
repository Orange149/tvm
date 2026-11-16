"""Protocol and immutable capsule identity; no TVM or board imports."""
from hashlib import sha256
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path('/home/orange/code/tvm/vta/experiments/cpu_vta_partitioning')
CAPSULE = ROOT.parent.parent
PARENT = ROOT / 'artifacts/bundles/8a942b2047c031339168f9d2'
SOURCE = ROOT / 'artifacts/frame_stream_top20_static_validation'
DEFAULT_OUTPUT = ROOT / 'artifacts/model_ablation_01'
DEFAULT_RUN_ID = 'resnet18_model_ablation_01'
DEFAULT_LOCAL_CONFIG = PROJECT / 'configs/local.json'

MODELS = (
    ('uniform', '基础容量模型', False, False, False),
    ('weights', '仅线程份额', True, False, False),
    ('host', '仅主机约束', False, True, False),
    ('team', '仅线程组约束', False, False, True),
    ('weights_host', '线程份额＋主机约束', True, True, False),
    ('weights_team', '线程份额＋线程组约束', True, False, True),
    ('host_team', '主机约束＋线程组约束', False, True, True),
    ('complete', '完整模型', True, True, True),
)

def settings(key):
    for name, label, weights, host, team in MODELS:
        if key == name:
            return dict(key=name, label=label, thread_weights=weights, host_guard=host, team_guard=team)
    raise ValueError('unknown ablation model: ' + str(key))

def protocol():
    return dict(experiment='resnet18_model_ablation', revision=1, seed=20261108,
        input_index=0, queue_depth=2, performance=dict(warmup=8, scored=16, tail=4),
        environment=dict(cores=4, frequency_khz=1066666, poll_sleep_ns=1000,
            post_start_sleep_ns=1000, spin_count=0, exclude_worker0=1, bind_threads=1),
        models=[settings(row[0]) for row in MODELS], top_k=20, max_pool=160,
        boot_policy='one_boot_no_automatic_reboot', anchors=False,
        automatic_remeasurement=False, performance_rejection_threshold=None,
        new_component_profiles=0, runtime_changes=False,
        single_boot_observation_only=True, board_reboot_forbidden=True)

def api(bundle=None):
    from cpu_policies.core_model.common import api as legacy
    return legacy(bundle or (CAPSULE if (CAPSULE/'manifest.json').is_file() else PARENT))

def identity():
    files = [ROOT/'run_model_ablation.py', ROOT/'pyproject.toml']
    for directory in ('model_ablation', 'cpu_policies', 'resource_envelope'):
        files += sorted(p for p in (ROOT/directory).rglob('*') if p.suffix in ('.py', '.h', '.cc', '.json'))
    files += sorted((ROOT/'configs').rglob('*.json'))
    values = {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in files}
    return sha256(json.dumps(values, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
