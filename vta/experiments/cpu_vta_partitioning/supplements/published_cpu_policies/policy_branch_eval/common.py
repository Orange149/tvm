"""Offline boundaries and immutable experiment artifacts."""
from contextlib import contextmanager
from pathlib import Path
import builtins
import io
import os
import socket
import subprocess

from cpu_vta_partitioning.common import digest, read, sha256, write

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'artifacts/policy_branch_eval_01'
REPORT = ROOT.parents[4] / 'paper/my_paper/analysis/20261110_policy_branch_eval'
NETWORKS = ('resnet18', 'yolov3_tiny416')


def protocol():
    return dict(experiment='policy_branch_eval_01', revision=1,
        networks=list(NETWORKS), resource_score='fine_host, unchanged',
        fixed_family='resource_group gate; at least two distinct disjoint compute groups; identical groups may be reused',
        original_branch='complete original domain, including open and serial execution',
        mixed_selection='alternate original and fixed, original first; seek 10 unseen per branch; skip duplicates; fill from remaining branch',
        per_branch_quota=10, top_k=20, tie_order='original integer score and canonical execution tuple',
        new_component_profiles=0, new_board_sessions=0, board_access=False,
        historical_FPS_used_in_selection=False, assessment='known-data developmental replay; not independent validation',
        source='Pipe-it disjoint compute groups; project adaptation, not complete reproduction')


def freeze(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError('immutable artifact differs: ' + str(path))
    else:
        write(path, value)
    return value


def sources():
    paths = [ROOT / 'run_policy_branch_eval.py', *sorted(Path(__file__).parent.glob('*.py')),
             ROOT / 'tests/test_policy_branch_eval.py',
             ROOT / 'yolo_cpu_policy_check/ranking.py', ROOT / 'staged_ablation/model.py',
             ROOT / 'cpu_policies/open_resources/costs.py', ROOT / 'cpu_policies/open_resources/domain.py']
    return {str(p.relative_to(ROOT)): sha256(p) for p in paths}


@contextmanager
def offline_guard(output):
    """No network or shells; only local C++ oracle compilation/execution and fonts."""
    output = Path(output).resolve()
    ledger = dict(network_attempts=0, forbidden_process_attempts=0, local_commands=[])
    saved = (socket.socket.connect, socket.socket.connect_ex, socket.create_connection,
             socket.getaddrinfo, subprocess.Popen, os.system)

    def deny_network(*args, **kwargs):
        ledger['network_attempts'] += 1
        raise RuntimeError('offline branch evaluation forbids network access')

    def deny_process(*args, **kwargs):
        ledger['forbidden_process_attempts'] += 1
        raise RuntimeError('offline branch evaluation forbids this process')

    def popen(args, *pos, **kw):
        if isinstance(args, (list, tuple)) and args and not kw.get('shell'):
            name = Path(str(args[0])).name
            local_oracle = name.startswith('scalar_oracle_') and Path(args[0]).resolve().is_relative_to(output)
            compiler = name in ('c++', 'g++') and '-o' in args and Path(args[args.index('-o')+1]).resolve().is_relative_to(output)
            if local_oracle or compiler or name in ('fc-list', 'fc-match'):
                ledger['local_commands'].append(list(map(str, args)))
                return saved[4](args, *pos, **kw)
        return deny_process()

    socket.socket.connect = socket.socket.connect_ex = deny_network
    socket.create_connection = socket.getaddrinfo = deny_network
    subprocess.Popen, os.system = popen, deny_process
    try:
        yield ledger
    finally:
        socket.socket.connect, socket.socket.connect_ex = saved[:2]
        socket.create_connection, socket.getaddrinfo = saved[2:4]
        subprocess.Popen, os.system = saved[4:]


@contextmanager
def isolated_inputs_only():
    """Deny ordinary file reads of historical performance throughout ranking."""
    saved = builtins.open, io.open, Path.open
    forbidden = {'RESULTS.json', 'analysis.json', 'configurations.csv', 'candidates.csv',
                 'matched_comparisons.csv'}

    def wrap(original):
        def guarded(path, mode='r', *args, **kwargs):
            if isinstance(path, (str, os.PathLike)) and ('r' in mode or '+' in mode):
                p = Path(path).resolve()
                if 'runs' in p.parts or ('paper' in p.parts and 'analysis' in p.parts) or p.name in forbidden:
                    raise RuntimeError('ranking may not read pipeline performance: ' + str(p))
            return original(path, mode, *args, **kwargs)
        return guarded
    builtins.open, io.open = wrap(saved[0]), wrap(saved[1])
    # Python 3.10 pathlib retains a separate io.open reference in its accessor.
    Path.open = wrap(saved[2])
    try:
        yield
    finally:
        builtins.open, io.open, Path.open = saved


def require_ranked(output):
    output = Path(output)
    seal = read(output / 'RANK_FREEZE.json')
    if seal['sources'] != sources() or seal['protocol'] != protocol():
        raise ValueError('code or protocol changed; use a new offline output')
    for name, expected in seal['files'].items():
        if sha256(output / name) != expected:
            raise ValueError('frozen ranking artifact changed: ' + name)
    return seal
