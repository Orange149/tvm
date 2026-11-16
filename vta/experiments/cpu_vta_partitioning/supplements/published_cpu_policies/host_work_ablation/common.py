"""Independent protocol and identity; existing campaigns are not modified."""
from hashlib import sha256
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path('/home/orange/code/tvm/vta/experiments/cpu_vta_partitioning')
CAPSULE = ROOT.parent.parent
PARENT = ROOT / 'artifacts/bundles/8a942b2047c031339168f9d2'
SOURCE = ROOT / 'artifacts/frame_stream_top20_static_validation'
LEGACY_SEARCH = ROOT / 'artifacts/model_ablation_01'
DEFAULT_OUTPUT = ROOT / 'artifacts/host_work_ablation_01'
DEFAULT_RUN_ID = 'resnet18_host_work_ablation_01'
DEFAULT_LOCAL_CONFIG = PROJECT / 'configs/local.json'

# Last three fields preserve the established weights/host-guard/team contract.
MODELS = (
    ('ignored', '忽略主机CPU需求', True, False, True),
    ('included', '计入主机CPU需求', True, False, True),
    ('complete', '完整模型', True, True, True),
)
LEGACY_MODELS = {'included': 'weights_team', 'complete': 'complete'}

def settings(key):
    for name, label, weights, host, team in MODELS:
        if key == name:
            return dict(key=name, label=label, thread_weights=weights,
                        host_cpu_demand=key != 'ignored', host_guard=host, team_guard=team)
    raise ValueError('unknown host-work model: ' + str(key))

def protocol():
    return dict(experiment='resnet18_host_work_ablation', revision=1, seed=20261109,
        input_index=0, queue_depth=2, performance=dict(warmup=8, scored=16, tail=4),
        environment=dict(cores=4, frequency_khz=1066666, poll_sleep_ns=1000,
            post_start_sleep_ns=1000, spin_count=0, exclude_worker0=1, bind_threads=1),
        models=[settings(row[0]) for row in MODELS], top_k=20, max_pool=60,
        boot_policy='one_boot_no_automatic_reboot', anchors=False,
        automatic_remeasurement=False, performance_rejection_threshold=None,
        new_component_profiles=0, runtime_changes=False,
        single_boot_observation_only=True, board_reboot_forbidden=True,
        host_cpu_definition='sum of isolated VTA run CPU work and input/output CPU work',
        ignored_mode='exclude VTA host CPU work from CPU capacity only; retain VTA wall service and DDR',
        component_costs_read_only=True, old_pipeline_performance_reused=False)

def api(bundle=None):
    from cpu_policies.core_model.common import api as legacy
    return legacy(bundle or (CAPSULE if (CAPSULE/'manifest.json').is_file() else PARENT))

def identity():
    paths = [ROOT/'run_host_work_ablation.py', ROOT/'pyproject.toml']
    for directory in ('host_work_ablation', 'model_ablation', 'cpu_policies', 'resource_envelope'):
        paths += sorted(p for p in (ROOT/directory).rglob('*') if p.suffix in ('.py', '.h', '.cc', '.json'))
    paths += sorted((ROOT/'configs').rglob('*.json'))
    items = {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in paths}
    return sha256(json.dumps(items, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
