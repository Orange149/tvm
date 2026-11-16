"""Local paths, immutable outputs, and a guard against remote execution."""
from contextlib import contextmanager
from pathlib import Path
import os
import socket
import subprocess

from cpu_vta_partitioning.common import digest, read, sha256, write

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'artifacts/dependency_audit_01'
WAIT_RUN = ROOT / 'runs/yolov3_tiny416_wait_diagnosis_01'
WAIT_BUNDLE = ROOT / 'artifacts/yolo_wait_diagnosis_01/capsule'
AFFINITY_RUN = ROOT / 'runs/yolov3_tiny416_affinity_check_01'
REPORT = ROOT.parents[4] / 'paper/my_paper/analysis/20261110_yolo_dependency_audit'


def protocol():
    return dict(experiment='yolov3_tiny416_dependency_audit_01', revision=1,
        model_source='isolated CPU0-2, original waiting, observed callbacks, 3/4 threads',
        target_program='cpu_00_02', kernels_per_frame=12,
        isolated_window=dict(warmup=2, scored=8, tail=0),
        diagnostic_window=dict(warmup=8, scored=64, tail=32),
        new_native_sessions=0, new_DNN_inferences=0, board_access=False,
        full_pipeline_prediction=False, score_or_DP_changes=False,
        stage_wall_already_contains_completion_dependencies=True,
        reference='Daydream ATC2020 section 4; dependency representation only',
        model_scope='ideal single-stage callback work and observed inter-launch caller CPU; not Linux scheduling')


def sources():
    paths = [ROOT / 'run_dependency_audit.py', *sorted(Path(__file__).parent.glob('*.py')),
             ROOT / 'tests/test_dependency_audit.py']
    return {str(p.relative_to(ROOT)): sha256(p) for p in paths}


def freeze(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError('immutable offline artifact differs: ' + str(path))
    else:
        write(path, value)
    return value


@contextmanager
def offline_guard():
    """Deny connections and process launches, except local font discovery."""
    observed = dict(network_attempts=0, forbidden_process_attempts=0, local_font_commands=[])
    originals = dict(connect=socket.socket.connect, connect_ex=socket.socket.connect_ex,
        create_connection=socket.create_connection, getaddrinfo=socket.getaddrinfo,
        Popen=subprocess.Popen, system=os.system)
    def deny_network(*args, **kwargs):
        observed['network_attempts'] += 1
        raise RuntimeError('dependency audit is offline; network access forbidden')
    def deny_system(*args, **kwargs):
        observed['forbidden_process_attempts'] += 1
        raise RuntimeError('dependency audit forbids shell/process execution')
    def popen(args, *positional, **kwargs):
        if (isinstance(args, (list, tuple)) and args and not kwargs.get('shell')
                and Path(str(args[0])).name in ('fc-list', 'fc-match')):
            observed['local_font_commands'].append(list(args))
            return originals['Popen'](args, *positional, **kwargs)
        return deny_system()
    socket.socket.connect = socket.socket.connect_ex = deny_network
    socket.create_connection = socket.getaddrinfo = deny_network
    subprocess.Popen, os.system = popen, deny_system
    try:
        yield observed
    finally:
        socket.socket.connect, socket.socket.connect_ex = originals['connect'], originals['connect_ex']
        socket.create_connection, socket.getaddrinfo = originals['create_connection'], originals['getaddrinfo']
        subprocess.Popen, os.system = originals['Popen'], originals['system']


def require_stage(output, stage):
    output = Path(output)
    freeze_data = read(output / 'INPUT_FREEZE.json')
    if freeze_data['sources'] != sources() or freeze_data['protocol'] != protocol():
        raise ValueError('offline code or protocol changed; use another output batch')
    for name, expected in freeze_data['files'].items():
        if sha256(output / name) != expected:
            raise ValueError('frozen input changed: ' + name)
    if stage == 'replay':
        seal = read(output / 'REPLAY_FREEZE.json')
        if seal['input_freeze_sha256'] != sha256(output / 'INPUT_FREEZE.json'):
            raise ValueError('replay uses a different input freeze')
        for name, expected in seal['files'].items():
            if sha256(output / name) != expected:
                raise ValueError('replay artifact changed: ' + name)
    return freeze_data
