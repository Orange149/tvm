"""Independent identity for the per-core cost and caller-placement validation."""
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
PROJECT = Path('/home/orange/code/tvm/vta/experiments/cpu_vta_partitioning')
REPO = PROJECT.parents[2]
_SNAPSHOT_BUNDLE = ROOT.parent.parent
DEFAULT_PARENT = (_SNAPSHOT_BUNDLE if (_SNAPSHOT_BUNDLE / 'plans/programs.json').is_file()
                  else ROOT / 'artifacts/bundles/abbcccc62af230ee844745ac')
DEFAULT_SOURCE_RUN = ROOT / 'runs/resnet18_prediction_gap_02'
DEFAULT_RUN_ID = 'resnet18_core_model_validation_01'
DEFAULT_LOCAL_CONFIG = PROJECT / 'configs/local.json'


def protocol():
    return json.loads((ROOT / 'configs/core_model/protocol.json').read_text())


def identity():
    paths = [ROOT / 'run_core_model.py']
    for folder in ('core_model', 'run_scope', 'diagnosis', 'open_resources'):
        for pattern in (('*.py', '*.cc', '*.h') if folder in ('core_model', 'run_scope') else ('*.py',)):
            paths.extend(sorted((ROOT / 'cpu_policies' / folder).glob(pattern)))
        paths.extend(sorted((ROOT / 'configs' / folder).glob('*.json')))
    items = {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in paths}
    return sha256(json.dumps(items, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def load_legacy(bundle=None):
    bundle = Path(bundle or DEFAULT_PARENT).resolve()
    paths = [bundle / 'sources/base_controller/src', bundle / 'sources/cpu_strategy']
    if any(not p.is_dir() for p in paths):
        raise ValueError('frozen controller primitives missing from core-model capsule')
    for path in reversed(paths):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return bundle


def api(bundle=None):
    load_legacy(bundle)
    from cpu_vta_partitioning import common
    return common
