"""Frozen definitions for the open CPU allocation experiment.

This package deliberately lives below cpu_policies: the original pilot's
non-recursive controller identity remains unchanged.
"""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
_SNAPSHOT_BUNDLE = ROOT.parent.parent
_PROJECT = Path('/home/orange/code/tvm/vta/experiments/cpu_vta_partitioning')
DEFAULT_PARENT = (_SNAPSHOT_BUNDLE if (_SNAPSHOT_BUNDLE / 'plans/programs.json').is_file()
                  else ROOT / 'artifacts/bundles/be7a3c600c2ed47f245dfb9f')
DEFAULT_SOURCE_RUN = ROOT / 'runs/resnet18_published_policy_dp_pilot_01'
DEFAULT_RUN_ID = 'resnet18_open_cpu_resource_02'
DEFAULT_LOCAL_CONFIG = _PROJECT / 'configs/local.json'


def protocol():
    return json.loads((ROOT / 'configs/open_resources/protocol.json').read_text())


def identity():
    files = [ROOT / 'run_open_resources.py',
             *sorted((ROOT / 'cpu_policies/open_resources').rglob('*.py')),
             *sorted((ROOT / 'cpu_policies/open_resources').rglob('*.cc')),
             *sorted((ROOT / 'cpu_policies/open_resources').rglob('*.h')),
             *sorted((ROOT / 'configs/open_resources').glob('*.json'))]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    return hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def load_legacy(bundle=None):
    bundle = Path(bundle or DEFAULT_PARENT).resolve()
    paths = [bundle / 'sources/base_controller/src', bundle / 'sources/cpu_strategy']
    if any(not p.is_dir() for p in paths):
        raise ValueError('frozen controller primitives missing from bundle')
    for p in reversed(paths):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    return bundle


def api(bundle=None):
    load_legacy(bundle)
    from cpu_vta_partitioning import common
    return common
