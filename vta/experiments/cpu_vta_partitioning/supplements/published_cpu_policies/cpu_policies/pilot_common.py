"""Frozen, board-independent definitions for the single-boot pilot."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
_SNAPSHOT_BUNDLE = ROOT.parent.parent
DEFAULT_BUNDLE = (_SNAPSHOT_BUNDLE if (_SNAPSHOT_BUNDLE / 'plans/programs.json').is_file()
                  else Path('/home/orange/code/tvm/vta/experiments/cpu_vta_partitioning/supplements/cpu_strategy/artifacts/bundles/79381dea6edbd338ca267cb9'))
DEFAULT_LOCAL_CONFIG = Path('/home/orange/code/tvm/vta/experiments/cpu_vta_partitioning/configs/local.json')
DEFAULT_RUN_ID = 'resnet18_published_policy_dp_pilot_01'


def protocol():
    return json.loads((ROOT / 'configs/pilot.json').read_text())


def identity():
    files = [ROOT / 'run.py', *sorted((ROOT / 'cpu_policies').glob('*.py')),
             *sorted((ROOT / 'configs').glob('*.json'))]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    return hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def load_legacy(bundle=None):
    """Use byte-frozen algorithm/measurement primitives, never its campaign."""
    bundle = Path(bundle or DEFAULT_BUNDLE).resolve()
    paths = [bundle / 'sources/base_controller/src', bundle / 'sources/cpu_strategy']
    if any(not p.is_dir() for p in paths):
        raise ValueError('frozen controller sources missing from bundle')
    for p in reversed(paths):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    return bundle
