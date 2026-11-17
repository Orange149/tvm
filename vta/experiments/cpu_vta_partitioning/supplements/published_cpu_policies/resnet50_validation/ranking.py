"""Isolated-only inputs, exact DP ranking and immutable two-method selection."""
from pathlib import Path
import random
import time

from cpu_vta_partitioning.common import digest, now, read, sha256, write
from cpu_vta_partitioning.prepare.build import copy_checked
from cpu_policies.open_resources.costs import Costs
from cpu_policies.open_resources.domain import canonical_tuple, configuration, validate_candidate
from staged_ablation.model import score as baseline_score
from model_ablation.model import score as complete_score
from yolo_transfer.ranking import ddr_table, replay_traffic
from .common import METHODS, HARDWARE, assert_controllers, frozen, protocol
from .components import profile_bundle, replay


def scalar_score(cfg, costs, method):
    if method == 'tarnawski':
        return baseline_score(cfg, costs, 'tarnawski')
    if method == 'complete':
        return complete_score(cfg, costs, 'complete')
    raise ValueError('unfrozen method')


def prepare_inputs(output, run_dir):
    output, run_dir = Path(output), Path(run_dir)
    assert_controllers(output)
    bundle = profile_bundle(output)
    records = replay(output, run_dir / 'components')
    traffic = replay_traffic(bundle, run_dir / 'traffic')
    root = output / 'ranking/inputs'
    frozen(root / 'isolated_costs.json', dict(status='COMPLETE', costs=records, pipeline_FPS_used=False))
    sources = {'programs.json': bundle / 'plans/programs.json', 'traffic.json': run_dir / 'traffic/traffic.json',
               'hardware.json': HARDWARE, 'profile_manifest.json': bundle / 'manifest.json'}
    for name, path in sources.items():
        copy_checked(path, root / name)
    programs = read(root / 'programs.json')
    frozen(root / 'ddr.json', ddr_table(programs, traffic, read(HARDWARE)['provenance']))
    frozen(root / 'INPUTS.json', {name: dict(source=str(path.resolve()), sha256=sha256(path))
                                 for name, path in sources.items()})
    return root


def load_inputs(output):
    """The search cannot open a run or FPS file; its allowlist is this immutable projection."""
    root = Path(output) / 'ranking/inputs'
    programs = read(root / 'programs.json')
    if any(p.get('kind', 'interval') != 'interval' for p in programs.values()):
        raise ValueError('reference programs must not become search candidates')
    records = read(root / 'isolated_costs.json')
    if records.get('pipeline_FPS_used') is not False or records['status'] != 'COMPLETE':
        raise ValueError('only complete isolated costs are admissible')
    ddr = read(root / 'ddr.json')
    hashes = {name: sha256(root / name) for name in
              ('programs.json', 'isolated_costs.json', 'traffic.json', 'hardware.json', 'profile_manifest.json', 'ddr.json')}
    return programs, records['costs'], ddr, hashes


def verify_search_acceptance(output):
    path = Path(output) / 'SEARCH_ACCEPTANCE.json'
    if not path.exists():
        raise ValueError('new DP must pass small-domain C++ and exact ResNet18 replay before new board work')
    doc = read(path)
    if (doc.get('passed') is not True or doc.get('small_independent_Cpp_enumeration') is not True
            or doc.get('ResNet18_fresh_DP_replay') != 'passed' or doc.get('pipeline_FPS_used') is not False
            or not doc.get('sources') or not doc.get('files')):
        raise ValueError('DP offline acceptance has not passed')
    for p, expected in doc.get('sources', {}).items():
        if sha256(p) != expected:
            raise ValueError('DP changed after offline acceptance: ' + p)
    for p, expected in doc.get('files', {}).items():
        if sha256(p) != expected:
            raise ValueError('DP acceptance evidence changed: ' + p)
    return doc


def freeze_result(output, result, programs, records, ddr, hashes):
    from .search_validation import verify_selected_cpp
    if result['status'] != 'COMPLETE':
        raise ValueError('incomplete search cannot freeze a Top-20')
    lists, pool, details = {}, {}, {}
    manifest = read(profile_bundle(output) / 'manifest.json')
    unit_count = manifest['unit_count']
    for method in METHODS:
        rows = result['methods'][method]['rankings']
        if len(rows) != 20 or len({r['id'] for r in rows}) != 20:
            raise ValueError('each method must select exactly twenty distinct execution configurations')
        costs = Costs(records, programs=programs, ddr=ddr,
                      distribution='balanced' if method == 'tarnawski' else 'session_weights')
        keys = []
        details[method] = {}
        for rank, row in enumerate(rows, 1):
            validate_candidate(row, programs, unit_count=unit_count, max_vta_islands=3)
            cfg = {**configuration(row), 'id': row['id']}
            if cfg['id'] != 'cfg_' + digest(configuration(cfg))[:24]:
                raise ValueError('configuration identity changed')
            value = scalar_score(cfg, costs, method)
            if value['score_ns'] != row['score_ns']:
                raise ValueError('scalar and DP scores differ')
            if cfg['id'] in pool and pool[cfg['id']] != cfg:
                raise ValueError('execution identity collision')
            pool[cfg['id']] = cfg
            details[method][cfg['id']] = value
            keys.append((row['score_ns'], canonical_tuple(cfg)))
            if row.get('static_rank', rank) != rank:
                raise ValueError('noncanonical static rank')
        if keys != sorted(keys):
            raise ValueError('scores or canonical ties are not sorted')
        lists[method] = rows
    if len(pool) > 40:
        raise ValueError('pipeline union exceeds the approved scope')
    cpp = verify_selected_cpp(result['methods'],
        Costs(records, programs=programs, ddr=ddr, distribution='session_weights'),
        Path(output) / 'ranking/selected_cpp', unit_count=unit_count)
    order = sorted(pool)
    random.Random(protocol()['seed']).shuffle(order)
    target = Path(output) / 'ranking'
    docs = {'rankings.json': dict(lists=lists), 'pool.json': dict(sorted(pool.items())),
            'scores.json': details, 'execution_order.json': order, 'protocol.json': protocol(),
            'proof.json': dict(input_hashes=hashes, counts=result['counts'], statistics=result['statistics'],
                exact_DP=True, small_domain_and_ResNet18_acceptance=verify_search_acceptance(output),
                selected_path_independent_Cpp_scoring=cpp,
                independent_global_enumeration=False, pipeline_FPS_used=False)}
    for name, doc in docs.items():
        frozen(target / name, doc)
    seal = dict(files={n: sha256(target / n) for n in docs}, input_hashes=hashes,
                candidate_count=len(pool), pipeline_FPS_used=False)
    if (target / 'FREEZE.json').exists():
        if any(read(target / 'FREEZE.json').get(k) != v for k, v in seal.items()):
            raise ValueError('frozen selection changed')
    else:
        write(target / 'FREEZE.json', {**seal, 'frozen_at': now()})
    return dict(status='COMPLETE', candidates=len(pool), methods=list(lists),
                freeze=str(target / 'FREEZE.json'))


def rank(output, run_dir, *, resume=False):
    from .search import rank as exact_rank
    output, run_dir = Path(output), Path(run_dir)
    assert_controllers(output)
    verify_search_acceptance(output)
    if not (output / 'ranking/inputs/INPUTS.json').exists():
        prepare_inputs(output, run_dir)
    programs, records, ddr, hashes = load_inputs(output)
    manifest = read(profile_bundle(output) / 'manifest.json')
    costs = Costs(records, programs=programs, ddr=ddr, distribution='session_weights')
    started = time.monotonic()
    result = exact_rank(programs, costs, unit_count=manifest['unit_count'], max_vta_islands=3,
        k=20, methods=METHODS, output_dir=output / 'ranking/search',
        budget_seconds=protocol()['search']['budget_seconds'],
        memory_bytes=protocol()['search']['memory_bytes'], resume=resume)
    write(output / 'ranking/SEARCH_STATUS.json', dict(status=result['status'],
        elapsed_seconds=time.monotonic()-started, statistics=result.get('statistics', {}),
        counts=result.get('counts'), checkpoint=result.get('checkpoint')), replace=True)
    if result['status'] != 'COMPLETE':
        return {k: result.get(k) for k in ('status', 'counts', 'statistics', 'checkpoint')}
    return freeze_result(output, result, programs, records, ddr, hashes)
