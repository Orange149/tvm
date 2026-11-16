"""Independent identities and access to the frozen controller primitives."""
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
PROJECT = Path('/home/orange/code/tvm/vta/experiments/cpu_vta_partitioning')
_SNAPSHOT_BUNDLE = ROOT.parent.parent
DEFAULT_PARENT = (_SNAPSHOT_BUNDLE if (_SNAPSHOT_BUNDLE / 'plans/programs.json').is_file()
                  else ROOT / 'artifacts/bundles/8a942b2047c031339168f9d2')
DEFAULT_SOURCE_RUN = ROOT / 'runs/resnet18_open_cpu_resource_02'
DEFAULT_RUN_ID = 'resnet18_prediction_gap_01'
DEFAULT_LOCAL_CONFIG = PROJECT / 'configs/local.json'


def protocol():
    return json.loads((ROOT / 'configs/diagnosis/protocol.json').read_text())


def identity():
    paths = [ROOT / 'run_diagnosis.py',
             *sorted((ROOT / 'cpu_policies/diagnosis').glob('*.py')),
             *sorted((ROOT / 'cpu_policies/open_resources').glob('*.py')),
             *sorted((ROOT / 'configs/diagnosis').glob('*.json'))]
    items = {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in paths}
    return sha256(json.dumps(items, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def load_legacy(bundle=None):
    bundle = Path(bundle or DEFAULT_PARENT).resolve()
    paths = [bundle / 'sources/base_controller/src', bundle / 'sources/cpu_strategy']
    if any(not p.is_dir() for p in paths):
        raise ValueError('frozen controller primitives missing from diagnostic bundle')
    for p in reversed(paths):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    return bundle


def api(bundle=None):
    load_legacy(bundle)
    from cpu_vta_partitioning import common
    return common
