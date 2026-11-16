from hashlib import sha256
import json
from pathlib import Path
from cpu_policies.core_model.common import api as legacy_api, PROJECT, REPO

ROOT = Path(__file__).resolve().parents[1]
_CAPSULE = ROOT.parent.parent
FULL_PARENT = ROOT / 'artifacts/bundles/8a942b2047c031339168f9d2'
FRAME_PARENT = ROOT / 'artifacts/bundles/452424b19638b5b8bcdb8456'
STATIC_SOURCE = ROOT / 'artifacts/frame_stream_top20_static_validation'
DEFAULT_RUN_ID = 'resnet18_dp_frame_stream_top20_01'
DEFAULT_LOCAL_CONFIG = PROJECT / 'configs/local.json'

def api(bundle=None):
    return legacy_api(bundle or (_CAPSULE if (_CAPSULE/'manifest.json').is_file() else FULL_PARENT))

def protocol():
    return json.loads((ROOT/'configs/frame_stream_top20/protocol.json').read_text())

def identity():
    paths=[ROOT/'run_frame_stream_top20.py',ROOT/'configs/frame_stream_top20/protocol.json']
    for directory in ('frame_stream_top20','frame_stream','cpu_policies'):
        paths+=sorted(p for p in (ROOT/directory).rglob('*') if p.suffix in ('.py','.h','.cc'))
    paths+=sorted((ROOT/'configs').rglob('*.json'))
    values={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in paths}
    return sha256(json.dumps(values,sort_keys=True,separators=(',',':')).encode()).hexdigest()
