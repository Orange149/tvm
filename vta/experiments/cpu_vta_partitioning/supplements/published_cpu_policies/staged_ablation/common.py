"""Independent protocol; importing this module cannot connect to a board."""
from hashlib import sha256
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path('/home/orange/code/tvm/vta/experiments/cpu_vta_partitioning')
CAPSULE = ROOT.parent.parent
PARENT = ROOT/'artifacts/bundles/8a942b2047c031339168f9d2'
SOURCE = ROOT/'artifacts/frame_stream_top20_static_validation'
LEGACY_SEARCH = ROOT/'artifacts/model_ablation_01'
HOST_SEARCH = ROOT/'artifacts/host_work_ablation_01'
DEFAULT_OUTPUT = ROOT/'artifacts/staged_ablation_01'
DEFAULT_RUN_ID = 'resnet18_staged_ablation_01'
DEFAULT_LOCAL_CONFIG = PROJECT/'configs/local.json'

# key, readable label, worker shares, host CPU demand, team guard, host guard, budget
MODELS = (
    ('tarnawski', 'Tarnawski设备负载模型适配', False, False, False, False, 20),
    ('coarse_no_host', '粗粒度CPU／不计主机需求', False, False, False, False, 20),
    ('coarse_host', '粗粒度CPU／计入主机需求', False, True, False, False, 20),
    ('fine_no_host', '细化CPU／不计主机需求', True, False, True, False, 20),
    ('fine_host', '本文主模型：细化CPU／计入主机需求', True, True, True, False, 20),
    ('weights_only', '仅细化线程份额／计入主机需求', True, True, False, False, 5),
    ('team_only', '仅加入线程组约束／计入主机需求', False, True, True, False, 5),
    ('concentrated', '主模型＋主机集中约束', True, True, True, True, 5),
)

def settings(key):
    for name,label,weights,host,team,guard,budget in MODELS:
        if key == name:
            return dict(key=name,label=label,thread_weights=weights,host_cpu_demand=host,
                        team_guard=team,host_guard=guard,measurement_budget=budget,
                        domain='singleton_physical_CPU' if key=='tarnawski' else 'full_CPU_resource_domain')
    raise ValueError('unknown staged-ablation model: '+str(key))

def protocol():
    return dict(experiment='resnet18_staged_ablation',revision=1,seed=20261110,
        input_index=0,queue_depth=2,performance=dict(warmup=8,scored=16,tail=4),
        environment=dict(cores=4,frequency_khz=1066666,poll_sleep_ns=1000,
            post_start_sleep_ns=1000,spin_count=0,exclude_worker0=1,bind_threads=1),
        models=[settings(r[0]) for r in MODELS],top_k=20,max_selection_pool=115,max_pool=121,
        boot_policy='one_boot_no_automatic_reboot',anchors=False,
        automatic_remeasurement=False,performance_rejection_threshold=None,
        new_component_profiles=0,runtime_changes=False,single_boot_observation_only=True,
        board_reboot_forbidden=True,component_costs_read_only=True,
        old_pipeline_performance_reused=False,
        excluded_work=['YOLO','Bayesian optimization','2023 TVM pipeline baseline'],
        host_cpu_definition='all isolated VTA run CPU work plus input/output CPU work',
        mechanisms='audit historical matched evidence first; at most six missing configurations only')

def api(bundle=None):
    from cpu_policies.core_model.common import api as legacy
    return legacy(bundle or (CAPSULE if (CAPSULE/'manifest.json').is_file() else PARENT))

def identity():
    paths=[ROOT/'run_staged_ablation.py',ROOT/'pyproject.toml']
    for directory in ('staged_ablation','host_work_ablation','model_ablation','cpu_policies','resource_envelope'):
        paths+=sorted(p for p in (ROOT/directory).rglob('*') if p.suffix in ('.py','.h','.cc','.json'))
    paths+=sorted((ROOT/'configs').rglob('*.json'))
    items={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in paths}
    return sha256(json.dumps(items,sort_keys=True,separators=(',',':')).encode()).hexdigest()
