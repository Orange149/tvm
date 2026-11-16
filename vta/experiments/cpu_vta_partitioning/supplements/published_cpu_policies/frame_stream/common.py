from hashlib import sha256
import json
from pathlib import Path
from cpu_policies.core_model.common import api as legacy_api, PROJECT, REPO

ROOT = Path(__file__).resolve().parents[1]
_BUNDLE = ROOT.parent.parent
DEFAULT_PARENT = (_BUNDLE if (_BUNDLE/'plans/programs.json').is_file()
                  else ROOT/'artifacts/bundles/a32daf05cfb604dcf12cff24')
DEFAULT_SOURCE_RUN = ROOT/'runs/resnet18_controlled_contention_02'
DEFAULT_RUN_ID = 'resnet18_cpu_frame_stream_01'
DEFAULT_LOCAL_CONFIG = PROJECT/'configs/local.json'

def api(bundle=None):
    return legacy_api(bundle or DEFAULT_PARENT)

def protocol():
    return json.loads((ROOT/'configs/frame_stream/protocol.json').read_text())

def identity():
    files=[ROOT/'run_frame_stream.py',ROOT/'configs/frame_stream/protocol.json']
    files+=sorted(p for p in (ROOT/'frame_stream').rglob('*') if p.suffix in ('.py','.h','.cc'))
    # Shared helper code is authenticated too, without changing its own identity scope.
    files+=sorted(p for p in (ROOT/'cpu_policies').rglob('*') if p.suffix in ('.py','.h','.cc'))
    values={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in files}
    return sha256(json.dumps(values,sort_keys=True,separators=(',',':')).encode()).hexdigest()
