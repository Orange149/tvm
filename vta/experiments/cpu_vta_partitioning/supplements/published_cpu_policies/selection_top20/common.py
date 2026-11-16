from hashlib import sha256
import json
from pathlib import Path
from cpu_policies.core_model.common import api as legacy_api, PROJECT, REPO

ROOT = Path(__file__).resolve().parents[1]
_CAPSULE = ROOT.parent.parent
FULL_PARENT = ROOT / 'artifacts/bundles/8a942b2047c031339168f9d2'
STATIC_PARENT = ROOT / 'artifacts/bundles/e62a7e586f5e816bbf7d2db9'
STATIC_SOURCE = ROOT / 'artifacts/frame_stream_top20_static_validation'
DEFAULT_RUN_ID = 'resnet18_autotvm_diverse_top20_01'
DEFAULT_LOCAL_CONFIG = PROJECT / 'configs/local.json'

def api(bundle=None):
    return legacy_api(bundle or (_CAPSULE if (_CAPSULE/'manifest.json').is_file() else FULL_PARENT))

def protocol():
    return json.loads(Path(__file__).with_name('protocol.json').read_text())

def identity():
    paths = [ROOT/'run_diverse_top20.py', ROOT/'pyproject.toml']
    for directory in ('selection_top20', 'cpu_policies'):
        paths += sorted(p for p in (ROOT/directory).rglob('*') if p.suffix in ('.py', '.json', '.h', '.cc'))
    paths += sorted((ROOT/'configs').rglob('*.json'))
    paths += [ROOT/'tests/test_selection_top20.py']
    values = {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in paths}
    return sha256(json.dumps(values, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
