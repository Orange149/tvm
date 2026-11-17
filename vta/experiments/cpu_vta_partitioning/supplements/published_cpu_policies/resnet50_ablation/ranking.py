"""Static-only complete-domain ranking, provenance and five-list freezing."""
from pathlib import Path
import random
import subprocess
import sys
import time

from cpu_vta_partitioning.common import digest, read, sha256, write
from cpu_vta_partitioning.prepare.build import copy_checked
from cpu_policies.open_resources.domain import canonical_tuple, configuration, validate_candidate
from resnet50_validation.common import assert_controllers
from .common import ROOT, SOURCE, MODELS, settings, protocol, frozen, scoring_sources
from .model import costs_for, score


INPUT_NAMES = ('programs.json', 'isolated_costs.json', 'traffic.json', 'hardware.json',
               'profile_manifest.json', 'ddr.json')


def inputs(output):
    assert_controllers(SOURCE)
    seal = read(SOURCE / 'ranking/FREEZE.json')
    if seal['pipeline_FPS_used'] is not False:
        raise ValueError('original static inputs used pipeline feedback')
    target = Path(output) / 'ranking/inputs'
    hashes = {}
    for name in INPUT_NAMES:
        source = SOURCE / 'ranking/inputs' / name
        expected = seal['input_hashes'][name]
        if sha256(source) != expected:
            raise ValueError('source isolated input changed: ' + name)
        if not (target / name).exists():
            copy_checked(source, target / name, expected)
        if sha256(target / name) != expected:
            raise ValueError('new isolated projection changed: ' + name)
        hashes[name] = expected
    data = read(target / 'isolated_costs.json')
    if data['pipeline_FPS_used'] is not False or data['status'] != 'COMPLETE':
        raise ValueError('incomplete or non-isolated costs')
    return read(target / 'programs.json'), data['costs'], read(target / 'ddr.json'), hashes


def acceptance(output):
    from .verification import small_domains, resnet18_replay
    output = Path(output)
    for name, fn in (('small', small_domains), ('resnet18', resnet18_replay)):
        marker = output / 'acceptance' / name / 'ACCEPTANCE.json'
        if marker.exists():
            doc = read(marker)
            if not doc.get('passed') or doc['scoring_sources'] != scoring_sources():
                raise ValueError('acceptance implementation changed')
        else:
            fn(marker.parent)
    result = dict(passed=True, scoring_sources=scoring_sources(),
                  files={n: sha256(output / 'acceptance' / n / 'ACCEPTANCE.json')
                         for n in ('small', 'resnet18')}, pipeline_FPS_used=False)
    frozen(output / 'ACCEPTANCE.json', result)
    return result


def require_acceptance(output):
    output = Path(output); doc = read(output / 'ACCEPTANCE.json')
    if not doc['passed'] or doc['scoring_sources'] != scoring_sources():
        raise ValueError('five-policy offline acceptance is missing or changed')
    for name, h in doc['files'].items():
        if sha256(output / 'acceptance' / name / 'ACCEPTANCE.json') != h:
            raise ValueError('acceptance evidence changed')
    return doc


def source_lists():
    seal = read(SOURCE / 'ranking/FREEZE.json')
    for name, h in seal['files'].items():
        if sha256(SOURCE / 'ranking' / name) != h:
            raise ValueError('source frozen lists changed')
    return read(SOURCE / 'ranking/rankings.json')['lists']


def run_one(output, key, resume=False):
    from .search import rank
    from .verification import selected_cpp
    output = Path(output); require_acceptance(output)
    programs, records, ddr, hashes = inputs(output)
    target = output / 'ranking/models' / key
    identity = dict(model=settings(key), input_hashes=hashes, scoring_sources=scoring_sources())
    frozen(target / 'IDENTITY.json', identity)
    if (target / 'RESULTS.json').exists():
        doc = read(target / 'RESULTS.json')
        if doc['identity'] != identity or doc['status'] != 'COMPLETE':
            raise ValueError('ranked result identity differs')
        return dict(model=key, status='COMPLETE', reused_verified=True)
    costs = costs_for(key, records, programs, ddr)
    old = source_lists(); started = time.monotonic()
    if key == 'complete':
        source = read(SOURCE / 'ranking/search/search_result.json')
        if not source['exact_complete_domain'] or source['counts']['complete'] != 58267668578:
            raise ValueError('source full-model exact proof is incomplete')
        rows = old['complete']
        proof = dict(source_result_sha256=sha256(SOURCE / 'ranking/search/search_result.json'),
                     source_freeze_sha256=sha256(SOURCE / 'ranking/FREEZE.json'),
                     original_search_seconds=source['elapsed_seconds'],
                     original_statistics=source['statistics'],
                     exact_complete_domain=True, counts=source['counts'], ranking_repeated=False)
    else:
        checkpoint = target / 'search/search_checkpoint.jsonl.gz'
        result = rank(programs, costs, key=key, unit_count=38,
            output_dir=checkpoint.parent, resume=resume,
            incumbents={'complete': old['complete']},
            budget_seconds=protocol()['search']['budget_seconds'],
            memory_bytes=protocol()['search']['memory_bytes'])
        if result['status'] != 'COMPLETE':
            write(target / 'INCOMPLETE.json', result, replace=True)
            return dict(model=key, status=result['status'], stop_reason=result['stop_reason'],
                        checkpoint=result['checkpoint'])
        if result['counts']['complete'] != 58267668578:
            raise ValueError('ResNet50 complete domain changed')
        rows = result['methods']['complete']['rankings']; proof = result
    scalar = selected_cpp({key: rows}, programs, records, ddr, target / 'cpp')
    for row in rows:
        validate_candidate(row, programs, unit_count=38, max_vta_islands=3)
        if score(row, costs, key)['score_ns'] != row['score_ns']:
            raise ValueError('DP and scalar policy differ')
    frozen(target / 'RESULTS.json', dict(status='COMPLETE', identity=identity, rankings=rows,
        proof=proof, selected_Cpp=scalar, preparation_seconds=time.monotonic()-started,
        pipeline_FPS_used=False))
    return dict(model=key, status='COMPLETE', first=rows[0]['id'],
                first_score_ms=rows[0]['score_ns']/1e6, elapsed_seconds=time.monotonic()-started)


def freeze(output):
    output = Path(output); acc = require_acceptance(output)
    programs, records, ddr, hashes = inputs(output)
    lists, pool, details, proofs = {}, {}, {}, {}
    for key in MODELS:
        path = output / 'ranking/models' / key / 'RESULTS.json'; doc = read(path)
        if (doc['status'] != 'COMPLETE' or doc['identity']['input_hashes'] != hashes
                or doc['identity']['scoring_sources'] != scoring_sources()):
            raise ValueError('uncertified policy result')
        rows = doc['rankings']; costs = costs_for(key, records, programs, ddr)
        if len(rows) != 20 or len({r['id'] for r in rows}) != 20:
            raise ValueError('each policy requires exactly twenty configurations')
        if [(r['score_ns'], canonical_tuple(r)) for r in rows] != sorted(
                (r['score_ns'], canonical_tuple(r)) for r in rows):
            raise ValueError('noncanonical score/tie order')
        lists[key] = rows; details[key] = {}; proofs[key] = sha256(path)
        for row in rows:
            cfg = configuration(row); name = row['id']
            if name != 'cfg_' + digest(cfg)[:24]:
                raise ValueError('execution identity depends on scoring policy')
            cfg = {**cfg, 'id': name}
            if name in pool and pool[name] != cfg:
                raise ValueError('configuration identity collision')
            pool[name] = cfg
            details[key][name] = score(cfg, costs, key)
    baseline = source_lists()['tarnawski']; bridges = []
    for budget, rank in ((1,1), (5,5), (20,11)):
        row = baseline[rank-1]; cfg = {**configuration(row), 'id': row['id']}
        pool[row['id']] = cfg
        bridges.append(dict(previous_budget=budget, original_static_rank=rank,
                            configuration=cfg, not_a_new_baseline_Top20=True))
    if len(pool) > 103:
        raise ValueError('approved candidate budget exceeded')
    order = sorted(pool); random.Random(protocol()['seed']).shuffle(order)
    target = output / 'ranking'
    docs = {'rankings.json': dict(lists=lists), 'pool.json': dict(sorted(pool.items())),
            'scores.json': details, 'bridges.json': bridges, 'protocol.json': protocol(),
            'execution_order.json': order,
            'proof.json': dict(exact_DP=True, pipeline_FPS_used=False,
                independent_global_enumeration=False, small_domain_and_ResNet18_acceptance=acc,
                model_result_sha256=proofs, input_hashes=hashes)}
    for name, doc in docs.items():
        frozen(target / name, doc)
    frozen(target / 'FREEZE.json', dict(files={n: sha256(target/n) for n in docs},
        candidate_count=len(pool), input_hashes=hashes, pipeline_FPS_used=False))
    return dict(status='RANKINGS_FROZEN', candidates=len(pool), native_sessions=2*len(pool),
                complete_DNN_inferences=49*len(pool), estimated_measurement_minutes=31.6667234289*len(pool)/60)


def rank_all(output, resume=False):
    output = Path(output)
    for key in ('complete', *MODELS[:-1]):
        command = [sys.executable, str(ROOT/'run_resnet50_ablation.py'), 'rank', '--model', key,
                   '--output', str(output)]
        if resume and (output/'ranking/models'/key/'search/search_checkpoint.jsonl.gz').exists():
            command.append('--resume')
        process = subprocess.run(command)
        if process.returncode:
            raise RuntimeError('policy search stopped; no pipeline measurements: ' + key)
    return freeze(output)
