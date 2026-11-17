"""Independent exhaustive small domains and selected-path C++ verification."""
from pathlib import Path
import json
import shutil
import subprocess
import time

from cpu_vta_partitioning.common import read, sha256, write
from cpu_policies.open_resources import domain
from cpu_policies.open_resources.costs import integer_weights, quantize_ns, ddr_stage_service
from resnet50_validation.search_validation import _small
from staged_ablation.search import engine as exhaustive_engine
from .common import ROOT, MODELS, scoring_sources, settings
from .model import costs_for, score
from . import search


def selected_cpp(lists, programs, records, ddr, output, unit_count=38):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    original = ROOT / 'resnet50_validation/search_scalar.cc'
    text = original.read_text()
    replacements = {
        '  I answer = std::max(vta, ddr);':
        '  if (!cfg.at("host_cpu_demand").get<bool>()) host = 0;\n  I answer = std::max(vta, ddr);',
        '  answer = std::max(answer, *std::max_element(team.begin(), team.end()));':
        '  if (cfg.at("team_guard").get<bool>()) answer = std::max(answer, *std::max_element(team.begin(), team.end()));',
        '  for (int core = 0; core < 4; ++core)\n    answer = std::max(answer, demand[core] + ((host_set & (1 << core)) ? host : 0));':
        '  if (cfg.at("host_guard").get<bool>()) for (int core = 0; core < 4; ++core)\n    answer = std::max(answer, demand[core] + ((host_set & (1 << core)) ? host : 0));',
    }
    for before, after in replacements.items():
        if text.count(before) != 1:
            raise ValueError('independent scalar source changed')
        text = text.replace(before, after)
    source = output / 'scalar.cc'
    if source.exists() and source.read_text() != text:
        raise ValueError('scalar implementation differs')
    source.write_text(text)
    compiler = shutil.which('g++')
    if not compiler:
        raise RuntimeError('independent C++ compiler is unavailable')
    header = ROOT / 'cpu_policies/open_resources/picojson.h'
    exe = output / ('scalar_' + sha256(source)[:20])
    if not exe.exists():
        subprocess.run([compiler, '-O2', '-std=c++17', '-I', str(header.parent), str(source), '-o', str(exe)],
                       check=True, capture_output=True, text=True, timeout=120)
    payload, expected = [], []
    for key, rows in lists.items():
        cfg_settings = settings(key)
        costs = costs_for(key, records, programs, ddr)
        for cfg in rows:
            stages = []
            for row in cfg['stages']:
                c = costs.get(row, cfg)
                stages.append(dict(device=row['device'], begin=row['begin'], end=row['end'], mask=row['mask'],
                    service_ns=quantize_ns(c['service_ms']), run_ns=quantize_ns(c['run_core_ms']),
                    boundary_ns=quantize_ns(c['boundary_core_ms']),
                    ddr_ns=quantize_ns(ddr_stage_service(ddr, row)),
                    weights=list(integer_weights(c['cpu_core_weights'], threads=row['threads']))))
            payload.append(dict(method='complete', gate=cfg['execution_mode'], host_mask=cfg['host_mask'],
                compute_pool=cfg['compute_pool'], unit_count=unit_count, max_vta_islands=3,
                host_cpu_demand=cfg_settings['host_cpu_demand'], team_guard=cfg_settings['team_guard'],
                host_guard=cfg_settings['host_guard'], stages=stages))
            expected.append(score(cfg, costs, key)['score_ns'])
            if cfg.get('score_ns', expected[-1]) != expected[-1]:
                raise ValueError('scalar and frozen DP score differ')
    result = subprocess.run([str(exe.resolve())], input=json.dumps(dict(configurations=payload)),
                            capture_output=True, text=True, check=True, timeout=120)
    actual = json.loads(result.stdout)['scores_ns']
    if actual != expected:
        raise ValueError('C++ selected-path scores differ')
    report = dict(passed=True, selected_paths=len(actual), original_source_sha256=sha256(original),
                  scalar_source_sha256=sha256(source), scores_ns=actual,
                  independent_global_enumeration=False, pipeline_FPS_used=False)
    write(output / 'RESULTS.json', report, replace=True)
    return report


def keys(rows):
    return [(r['score_ns'], domain.canonical_tuple(r)) for r in rows]


def small_domains(output):
    output = Path(output); results = []; started = time.monotonic()
    contexts = [domain.context(k) for k in ('shared4', 'host_shared_2', 'host_reserved_3')]
    for kind in ('unequal_work', 'service_ties', 'all_zero_ties', 'four_CPU_stages'):
        n, programs, records, ddr = _small(kind)
        for key in MODELS:
            costs = costs_for(key, records, programs, ddr)
            dest = output / kind / key
            new = search.rank(programs, costs, key=key, unit_count=n, contexts=contexts,
                              output_dir=dest / 'new')
            engine = exhaustive_engine(dest / 'oracle_source', settings(key)['scalar_key'])
            oracle = engine.rank_all(programs, costs, ddr=ddr, unit_count=n, contexts=contexts,
                                     output_dir=dest / 'oracle', verify=True)
            rows = new['methods']['complete']['rankings']
            if (new['status'] != 'COMPLETE' or not oracle['verification']['passed']
                    or keys(rows) != keys(oracle['methods']['enhanced']['rankings'])):
                raise ValueError('independent full enumeration mismatch: ' + kind + '/' + key)
            selected_cpp({key: rows}, programs, records, ddr, dest / 'selected', unit_count=n)
            results.append(dict(case=kind, model=key, ordered_Top20_identical=True,
                                counts=new['counts'], statistics=new['statistics']))
            print('small domain passed', kind, key, flush=True)
    result = dict(passed=True, cases=results, scoring_sources=scoring_sources(),
                  elapsed_seconds=time.monotonic()-started, pipeline_FPS_used=False)
    write(output / 'ACCEPTANCE.json', result, replace=True)
    return result


def resnet18_replay(output):
    """Fresh exact DP against the original, independently verified five lists."""
    from types import FunctionType
    from staged_ablation import search as legacy
    output = Path(output); results = []; began = time.monotonic()
    programs, records, ddr, hashes = legacy.inputs()
    for key in MODELS:
        old_key = settings(key)['scalar_key']
        # The legacy verifier creates an engine directory. Redirect that single
        # side effect into the NEW acceptance directory; old artifacts stay read-only.
        namespace = {**legacy.verify_one.__globals__,
            'engine': lambda unused, k: legacy.engine(output / k / 'legacy_source', k)}
        verify = FunctionType(legacy.verify_one.__code__, namespace)
        expected, proof = verify(ROOT / 'artifacts/staged_ablation_01' / old_key, old_key)
        costs = costs_for(key, records, programs, ddr)
        rows = expected['methods']['enhanced']['rankings']
        result = search.rank(programs, costs, key=key, unit_count=21,
                             incumbents={'complete': rows}, output_dir=output / key / 'fresh')
        if (result['status'] != 'COMPLETE'
                or keys(result['methods']['complete']['rankings']) != keys(rows)
                or result['counts']['complete'] != 966240306):
            raise ValueError('ResNet18 full-domain replay differs: ' + key)
        results.append(dict(model=key, input_hashes=hashes,
                            old_proof_sha256=sha256(ROOT / 'artifacts/staged_ablation_01' / old_key / 'proof.json'),
                            ordered_Top20_identical=True, statistics=result['statistics'],
                            elapsed_seconds=result['elapsed_seconds']))
        print('ResNet18 exact replay passed', key, flush=True)
    value = dict(passed=True, models=results, scoring_sources=scoring_sources(),
                 elapsed_seconds=time.monotonic()-began, pipeline_FPS_used=False)
    write(output / 'ACCEPTANCE.json', value, replace=True)
    return value
