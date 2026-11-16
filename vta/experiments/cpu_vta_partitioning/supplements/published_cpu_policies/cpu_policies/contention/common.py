"""Capsule identity without changing any preceding experiment identity."""
from hashlib import sha256
import json
from pathlib import Path
from ..core_model.common import PROJECT, REPO, api, load_legacy

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PARENT = ROOT / 'artifacts/bundles/2353f0c03ea0f3a6cecdb8a0'
FULL_PARENT = ROOT / 'artifacts/bundles/8a942b2047c031339168f9d2'
DEFAULT_RUN_ID = 'resnet18_controlled_contention_02'
DEFAULT_LOCAL_CONFIG = PROJECT / 'configs/local.json'

def protocol():
    return json.loads((ROOT / 'configs/contention/protocol.json').read_text())

def identity():
    paths = [ROOT / 'run_contention.py', ROOT / 'configs/contention/protocol.json']
    paths += [ROOT / 'run_contention_compat.py']
    paths += sorted(p for p in (ROOT / 'cpu_policies').rglob('*')
                    if p.is_file() and p.suffix in ('.py', '.cc', '.h'))
    items = {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in paths}
    return sha256(json.dumps(items, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
