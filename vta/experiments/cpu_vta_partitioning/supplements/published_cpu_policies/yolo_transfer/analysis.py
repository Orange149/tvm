"""Post-freeze analysis only; never imported by the ranking/scoring modules."""
import csv
from datetime import datetime
from itertools import combinations
import math
from pathlib import Path
import statistics

from cpu_vta_partitioning.common import read, sha256, verify_inventory, write

LABELS = {'uniform_strict20': '原均分核时模型严格排序20项',
          'weighted_strict20': '逐线程核时与资源约束严格排序20项',
          'weighted_behavior20': '逐线程核时与资源约束按行为筛选20项'}


def evaluate(lists, measurements):
    if set(lists) != set(LABELS) or any(len(v) != 20 or len(set(v)) != 20 for v in lists.values()):
        raise ValueError('three distinct twenty-entry lists are required')
    union = {name for values in lists.values() for name in values}
    if set(measurements) != union:
        raise ValueError('all methods require the same complete measurement union')
    if any(type(v.get('ii_ms')) not in (int, float) or not math.isfinite(v['ii_ms']) or v['ii_ms'] <= 0
           for v in measurements.values()):
        raise ValueError('invalid observed completion interval')
    best = min(union, key=lambda name: (measurements[name]['ii_ms'], name))
    pool_ii = measurements[best]['ii_ms']
    result = {}
    for method, names in lists.items():
        result[method] = {}
        for budget in (1, 5, 20):
            selected = min(names[:budget], key=lambda name: (measurements[name]['ii_ms'], name))
            ii = measurements[selected]['ii_ms']
            result[method][str(budget)] = {'selected': selected, 'ii_ms': ii, 'fps': 1000 / ii,
                'pool_relative_II_regret_percent': (ii / pool_ii - 1) * 100}
    effects = {method: {budget: {'FPS_change_percent':
                    (result['uniform_strict20'][budget]['ii_ms'] / result[method][budget]['ii_ms'] - 1) * 100,
                'II_reduction_percent':
                    (1 - result[method][budget]['ii_ms'] / result['uniform_strict20'][budget]['ii_ms']) * 100}
                for budget in ('1', '5', '20')}
               for method in ('weighted_strict20', 'weighted_behavior20')}
    return {'methods': result, 'effects_vs_original': effects, 'pool_best': best,
            'pool_best_fps': 1000 / pool_ii, 'single_boot_observation_only': True,
            'statistical_significance_or_equivalence_claimed': False,
            'candidate_pool_exhausts_real_performance_domain': False}


def ranking_metrics(predictions, measurements):
    if not predictions or set(predictions) != set(measurements):
        raise ValueError('prediction and observation identities differ')
    names = sorted(predictions)
    p = [predictions[n] / 1e6 for n in names]
    actual = [measurements[n]['ii_ms'] for n in names]
    differences = [a - b for a, b in zip(p, actual)]
    ties = comparable = discordant = observed_ties = 0
    for i, j in combinations(range(len(names)), 2):
        if predictions[names[i]] == predictions[names[j]]:
            ties += 1
        elif actual[i] == actual[j]:
            observed_ties += 1
        else:
            comparable += 1
            discordant += (p[i] < p[j]) != (actual[i] < actual[j])
    rho = None
    if len(set(predictions.values())) > 1 and len(set(actual)) > 1:
        from scipy.stats import spearmanr
        rho = float(spearmanr(p, actual).statistic)
    return {'count': len(names), 'MAE_ms': statistics.mean(abs(v) for v in differences),
            'bias_ms': statistics.mean(differences), 'predicted_tied_pairs': ties,
            'observed_tied_pairs_when_prediction_differs': observed_ties,
            'comparable_pairs': comparable, 'discordant_pairs': int(discordant),
            'spearman_rho': rho, 'correlation_undefined_when_prediction_constant': len(set(predictions.values())) == 1}


def write_csv(path, rows):
    if not rows:
        raise ValueError('empty analysis table')
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def configuration_text(cfg):
    partition = ' → '.join(('CPU' if s['device'] == 'cpu' else 'VTA') +
                str(s['begin']) + '—' + str(s['end'] - 1) for s in cfg['stages'])
    cpu = [s for s in cfg['stages'] if s['device'] == 'cpu']
    return {'partition': partition, 'CPU_threads': '+'.join(str(s['threads']) for s in cpu),
            'CPU_core_sets': str([s['mask'] for s in cpu]), 'caller_cores': str([s['caller_core'] for s in cpu]),
            'VTA_host_cores': str(cfg['host_mask']), 'control_cores': str(cfg['control_mask']),
            'same_group_serial': cfg['execution_mode'] == 'resource_group'}


def analyze(run_dir):
    from cpu_policies.open_resources.costs import Costs
    from cpu_policies.open_resources.domain import contexts
    from . import pipeline
    run_dir = Path(run_dir).resolve()
    state = read(run_dir / 'state.json')
    if state['status'] != 'COMPLETE' or not state['post_deployment_hashes_passed']:
        raise ValueError('complete validated pipeline campaign required before analysis')
    bundle = Path(state['bundle'])
    manifest = pipeline.validate(bundle)
    if manifest['bundle_id'] != state['bundle_id']:
        raise ValueError('run and pipeline capsule differ')
    freeze = read(bundle / 'plans/ranking_freeze.json')
    if read(run_dir / 'selection_freeze.json') != freeze:
        raise ValueError('the measured lists differ from the premeasurement freeze')
    pool = read(bundle / 'plans/pool.json')
    ranked = read(bundle / 'plans/rankings.json')['lists']
    order = read(bundle / 'plans/execution_order.json')
    if set(order) != set(pool) or len(order) != len(pool):
        raise ValueError('frozen execution order differs')
    measurements = {}; receipts = []
    for qualification in (True, False):
        for name in order:
            document = pipeline.make_task(bundle, name, qualification)
            receipt = read(run_dir / 'accepted' / state['boot_id'] / (document['task_id'] + '.json'))
            if receipt['task'] != document or receipt['boot_id'] != state['boot_id']:
                raise ValueError('pipeline result task or boot differs')
            if datetime.fromisoformat(receipt['measurement_started_at']) < datetime.fromisoformat(freeze['frozen_at']):
                raise ValueError('measurement preceded the candidate freeze')
            verify_inventory(receipt['raw'], receipt['files'])
            result = pipeline.summary(receipt['raw'], document, bundle)
            if result != receipt['summary']:
                raise ValueError('pipeline summary changed on raw evidence replay')
            if not qualification:
                measurements[name] = result
            receipts.append(receipt)
    lists = {method: [row['id'] for row in ranked[method]] for method in LABELS}
    decision = evaluate(lists, measurements)
    output = run_dir / 'analysis'; output.mkdir(exist_ok=True)
    positions = {method: {name: i + 1 for i, name in enumerate(names)} for method, names in lists.items()}
    contexts_by_id = {ctx['id']: ctx for ctx in contexts()}
    rows = []
    for name, cfg in sorted(pool.items()):
        result = measurements[name]
        rows.append({'id': name, **configuration_text(cfg), 'context': contexts_by_id[cfg['context_id']]['label'],
            'original_prediction_ms': cfg['original_score_ns'] / 1e6,
            'new_prediction_ms': cfg['selected_score_ns'] / 1e6, 'measured_II_ms': result['ii_ms'],
            'FPS': result['fps'], 'process_core_ms_per_completion': result['process_core_ms_per_completion'],
            'average_CPU_cores': result['average_cpu_cores'],
            **{method + '_position': positions[method].get(name, '') for method in LABELS}})
    write_csv(output / 'candidates.csv', rows)
    budgets = []
    for method, values in decision['methods'].items():
        for budget, row in values.items():
            budgets.append({'method': LABELS[method], 'budget': int(budget), **row,
                            **configuration_text(pool[row['selected']])})
    write_csv(output / 'selection_budgets.csv', budgets)
    common = {'original': ranking_metrics({n: c['original_score_ns'] for n, c in pool.items()}, measurements),
              'weighted_guards': ranking_metrics({n: c['selected_score_ns'] for n, c in pool.items()}, measurements)}
    method_rankings = {}
    for method, names in lists.items():
        scores = {r['id']: r['score_ns'] for r in ranked[method]}
        method_rankings[method] = ranking_metrics(scores, {n: measurements[n] for n in names})
    write(output / 'prediction_metrics.json', {'same_common_pool': common, 'selected_sets': method_rankings}, replace=True)
    profiles = read(bundle / 'provenance/static_inputs/isolated_costs.json')['costs']
    programs = read(bundle / 'plans/programs.json')
    costs = Costs(profiles, programs=programs, distribution='session_weights')
    stages = []
    for name, cfg in pool.items():
        for index, stage in enumerate(cfg['stages']):
            value = costs.get(stage, cfg)
            observed = measurements[name]['stage_service_ms'][index]
            stages.append({'configuration': name, 'stage_index': index, 'program': stage['program'],
                'device': stage['device'], 'threads': stage['threads'], 'cores': str(stage['mask']),
                'isolated_service_ms': value['service_ms'], 'pipeline_service_ms': observed,
                'service_ratio': observed / value['service_ms'] if value['service_ms'] else None,
                'isolated_run_core_ms': value['run_core_ms'], 'isolated_boundary_core_ms': value['boundary_core_ms'],
                'core_weights': str(value.get('cpu_core_weights', [])),
                'thread_share_scope': 'isolated_whole_session_proxy_not_exact_scored_worker_computation'})
    write_csv(output / 'stage_profile_comparison.csv', stages)
    curves = []
    for record in profiles.values():
        stage = record['configuration']['stages'][0]
        if stage['device'] == 'cpu' and stage['end'] == stage['begin'] + 1:
            cost = record['cost']
            curves.append({'unit': stage['begin'], 'threads': stage['threads'], 'cores': str(stage['mask']),
                'service_ms': cost['service_ms'], 'run_ms': cost['run_ms'],
                'run_core_ms': cost['run_core_ms'], 'boundary_core_ms': cost['boundary_core_ms'],
                'whole_session_worker_shares': str(record.get('session_thread_weights', []))})
    curves.sort(key=lambda row: (row['unit'], row['threads']))
    if {(r['unit'], r['threads']) for r in curves} != {(i, n) for i in range(14) for n in range(1, 5)}:
        raise ValueError('all fourteen unit thread curves must be present')
    write_csv(output / 'unit_thread_curves.csv', curves)
    raw_files = sum(len(r['files']) for r in receipts)
    raw_bytes = sum(info['bytes'] for r in receipts for info in r['files'].values())
    cost = {'pipeline_configurations': len(pool), 'accepted_pipeline_sessions': len(receipts),
            'complete_DNN_inferences': len(pool) * 29,
            'accepted_native_and_collection_seconds': sum(r['measurement_and_collection_seconds'] for r in receipts),
            'accepted_raw_files': raw_files, 'accepted_raw_bytes': raw_bytes,
            'pipeline_preparation': read(bundle / 'plans/preparation_cost.json'),
            'ranking_and_profile_sources': read(bundle / 'provenance/static_inputs/INPUTS.json'),
            'component_stage_calls_are_not_complete_DNN_inferences': True}
    result = {'selection': decision, 'prediction': common, 'ranking_by_selected_set': method_rankings,
              'cost': cost, 'boot_id': state['boot_id'], 'bundle_id': state['bundle_id'],
              'scope': 'one input, one boot, one short window per candidate; model transferred without YOLO pipeline fitting',
              'hashes_replayed': True, 'stage_core_times_summed_in_pipeline': False,
              'waiting_cache_and_control_work_fully_explained': False}
    write(output / 'analysis.json', result, replace=True)
    lines = ['YOLOv3-tiny416：冻结静态候选的单启动验证', '',
        '三种方法使用同批数据；每项为8帧预热、16个完成时间戳、4帧尾部，16个时间戳对应15个间隔。',
        '以下为观察值，不将帧当独立实验，也不声称覆盖候选池之外的最优方案。', '']
    for method, values in decision['methods'].items():
        lines.append(LABELS[method] + '：预算1／5／20 FPS = ' +
                     '／'.join(f"{values[b]['fps']:.3f}" for b in ('1', '5', '20')))
    lines += ['', '同一候选并集上的预测平均绝对误差：原模型' + f"{common['original']['MAE_ms']:.3f}" +
              ' ms，修正模型' + f"{common['weighted_guards']['MAE_ms']:.3f}" + ' ms。',
              '逐线程份额取自隔离会话的线程计数，包含预热和统计工作；并非精确的计入窗口纯计算份额。',
              '生产、收集、缓存与同步开销仍存在近似，阶段隔离时间和流水时间不要求相等。',
              f"完整流水共{len(pool)}项、{len(receipts)}个原生会话、{len(pool) * 29}次完整推理。", '',
              '详见 selection_budgets.csv、candidates.csv、stage_profile_comparison.csv、unit_thread_curves.csv 和 analysis.json。']
    (output / 'REPORT_zh.txt').write_text('\n'.join(lines) + '\n')
    return {'analysis': str(output / 'analysis.json'), 'candidates': len(pool),
            'pool_best_fps': decision['pool_best_fps'], 'single_boot_only': True}
