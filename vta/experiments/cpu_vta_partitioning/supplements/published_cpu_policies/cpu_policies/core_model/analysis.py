"""Evidence-bound comparison of per-core costs and caller placement.

All methods use one newly measured isolated table. Pipeline observations are
used only to evaluate the rankings frozen before those observations existed.
"""
import json
import math
from pathlib import Path
import statistics

from ..diagnosis.analysis import _bound_record, _csv, _sha


METHOD_LABELS = {
    'balanced_fixed': '核时均分／调用线程固定最小核心',
    'weighted_fixed': '隔离逐线程核时／调用线程固定最小核心',
    'balanced_movable': '核时均分／允许选择调用线程',
    'weighted_movable': '隔离逐线程核时／允许选择调用线程',
}
WARNINGS = [
    '输入固定为0，只有一次启动、每个候选一次短窗口，不代表跨启动稳定性或统计显著性。',
    '16个完成时间戳对应15个间隔；帧和间隔不当作独立板子启动。',
    '所有方法使用同一新隔离成本表；逐线程权重从隔离样本取得，流水FPS不是成本或选点输入。',
    '非代表核心集合和worker顺序采用同构核置换近似，并未逐个核心排列重新标定。',
    '逐线程CPU增量包含同步等在核活动，读取包络和Linux记账误差仍然存在。',
    '本轮只搜索固定切图下的225／1024项资源配置，不是完整切图DP或全worker排列搜索。',
    'VTA和DDR约束维持共同口径；生产、收集和结果处理没有独立成本项，不将其解释为零。',
    '性能接近或方向不一致时只报告观察值，不作统计等效判断，也不追加测到显著。',
]


def _read(path):
    return json.loads(Path(path).read_text())


def average_ranks(values):
    """One-based average ranks, preserving ties rather than hash order."""
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.] * len(values)
    begin = 0
    while begin < len(order):
        end = begin + 1
        while end < len(order) and values[order[end]] == values[order[begin]]:
            end += 1
        rank = (begin + 1 + end) / 2
        for i in order[begin:end]:
            result[i] = rank
        begin = end
    return result


def prediction_metrics(predictions, measured):
    if (not predictions or len(predictions) != len(measured)
            or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in predictions + measured)):
        raise ValueError('positive matched prediction/measurement vectors required')
    absolute = [abs(p - m) for p, m in zip(predictions, measured)]
    relative = [error / m * 100 for error, m in zip(absolute, measured)]
    a, b = average_ranks(predictions), average_ranks(measured)
    aa, bb = statistics.mean(a), statistics.mean(b)
    denominator = (sum((v - aa) ** 2 for v in a) * sum((v - bb) ** 2 for v in b)) ** .5
    rho = sum((x - aa) * (y - bb) for x, y in zip(a, b)) / denominator if denominator else None
    return {'configurations': len(measured), 'MAE_ms': statistics.mean(absolute),
            'MAPE_percent': statistics.mean(relative), 'median_APE_percent': statistics.median(relative),
            'max_APE_percent': max(relative), 'spearman_on_measured_pool_only': rho}


def placement_pairs(rows):
    """Four matched 4/4 placements, changing ordered maps on the same sets."""
    grid = {(r['front_caller'], r['middle_caller']): r for r in rows
            if r['front_threads'] == r['middle_threads'] == 4
            and r['front_mask'] == r['middle_mask'] == '0,1,2,3'
            and r['front_caller'] in (0, 3) and r['middle_caller'] in (0, 3)}
    result = []
    for original, changed, label in [((0, 0), (0, 3), '前段调用线程固定0，中段由0改3'),
                                      ((3, 3), (3, 0), '前段调用线程固定3，中段由3改0'),
                                      ((0, 0), (3, 0), '中段调用线程固定0，前段由0改3'),
                                      ((3, 3), (0, 3), '中段调用线程固定3，前段由3改0')]:
        if original not in grid or changed not in grid:
            continue
        before, after = grid[original], grid[changed]
        result.append({'comparison': label, 'before_configuration': before['configuration'],
                       'after_configuration': after['configuration'], 'before_fps': before['fps'],
                       'after_fps': after['fps'], 'fps_change_percent': (after['fps'] / before['fps'] - 1) * 100,
                       'before_ii_ms': before['ii_ms'], 'after_ii_ms': after['ii_ms'],
                       'scope': 'single_boot_observation_not_significance_or_equivalence'})
    return result


def _placement(cfg):
    front, middle = cfg['stages'][0], cfg['stages'][2]
    return {'front_threads': front['threads'], 'front_mask': ','.join(map(str, front['mask'])),
            'front_worker_map': ','.join(map(str, front['worker_core_map'])),
            'front_caller': front['caller_core'], 'middle_threads': middle['threads'],
            'middle_mask': ','.join(map(str, middle['mask'])),
            'middle_worker_map': ','.join(map(str, middle['worker_core_map'])),
            'middle_caller': middle['caller_core']}


def analyze(run_dir, bundle_dir=None):
    from .campaign import validate_state, validated_records
    from .common import api
    from .model import Costs, score_configuration
    from .domain import baseline_candidate
    root = Path(run_dir).resolve()
    state = validate_state(root, validate_payload=False)
    validated_records(root, state)
    bundle = Path(bundle_dir or state['bundle']).resolve()
    common = api(bundle)
    if str(bundle) != str(Path(state['bundle']).resolve()):
        raise ValueError('analysis bundle differs from the bound experiment')
    programs = _read(bundle / 'plans/programs.json')
    cost_doc = _read(root / 'costs.json')
    rankings, pool, screen = (_read(root / leaf) for leaf in ('rankings.json', 'search_pool.json', 'screening.json'))
    ddr = _read(bundle / 'plans/baseline_ddr.json')
    if (set(rankings['methods']) != set(METHOD_LABELS) or set(pool) != set(screen['measurements'])
            or screen['boot_id'] != state['boot_id'] or screen['rounds'] != 1):
        raise ValueError('analysis pool, methods or screening coverage differs')
    models = {kind: Costs(cost_doc, programs=programs, ddr=ddr, distribution=kind)
              for kind in ('balanced', 'weighted')}
    bound_records, source_files = [], {}
    seen = set()
    for path in sorted((root / 'records' / str(state['boot_id'])).rglob('*.json')):
        record = _read(path)
        doc = record.get('document', {})
        if 'diagnosis_case' not in doc:
            continue
        if doc['task_id'] in seen or record['boot_id'] != state['boot_id']:
            raise ValueError('duplicate diagnostic task or wrong boot in analysis')
        seen.add(doc['task_id'])
        frames, _, _, files = _bound_record(root, record)
        source_files.update(files)
        bound_records.append((record, frames))
    measurements, stages = [], []
    bridge = None
    for record, frames in bound_records:
        doc, summary = record['document'], record['summary']
        if doc['kind'] != 'performance':
            continue
        cfg = doc['configuration']
        kept = frames[doc['warmup']:doc['warmup'] + doc['scored']]
        ii = (kept[-1]['completion_ms'] - kept[0]['completion_ms']) / (len(kept) - 1)
        if abs(ii - summary['ii_ms']) > 1e-6:
            raise ValueError('performance summary differs from raw completion window')
        if doc['diagnosis_block'] == 'old_bridge':
            if bridge is not None or cfg != baseline_candidate():
                raise ValueError('legacy bridge coverage or fixed configuration differs')
            bridge = {'configuration': cfg['id'], 'ii_ms': ii, 'fps': 1000 / ii,
                      'executor_sha256': doc['executor_sha256'], 'task_id': doc['task_id']}
            continue
        key = cfg['id']
        if key not in pool or pool[key] != cfg or doc.get('run_probe') is not False:
            raise ValueError('formal pipeline differs from frozen pool or has counters enabled')
        selected = screen['measurements'][key]
        if selected['document'] != doc or selected['boot_id'] != state['boot_id']:
            raise ValueError('screened performance document differs from accepted raw result')
        values = {name: score_configuration(cfg, model) for name, model in models.items()}
        row = {'configuration': key, **_placement(cfg), 'ii_ms': ii, 'fps': 1000 / ii,
               'completion_intervals': len(kept) - 1,
               'process_core_ms_per_completion': summary.get('process_core_ms_per_completion')}
        for name, value in values.items():
            row[name + '_prediction_ms'] = value['score_ms']
            row[name + '_signed_error_percent'] = (value['score_ms'] / ii - 1) * 100
            row[name + '_absolute_error_percent'] = abs(value['score_ms'] / ii - 1) * 100
            row[name + '_core_demand_ms'] = ','.join(map(str, value['cpu_core_demand_ms']))
        measurements.append(row)
        for index, stage in enumerate(cfg['stages']):
            stages.append({'configuration': key, 'stage_index': index, 'program': stage['program'],
                           'device': stage['device'], 'threads': stage['threads'],
                           'worker_core_map': ','.join(map(str, stage['worker_core_map'])),
                           **{part + '_ms': statistics.mean(frame[f'stage{index}_' + ('ms' if part == 'service' else part + '_ms')] for frame in kept)
                              for part in ('set', 'run', 'get', 'service')}})
    if len(measurements) != len(pool) or bridge is None:
        raise ValueError('missing formal measurements or legacy bridge')
    by_id = {row['configuration']: row for row in measurements}
    best = min(row['ii_ms'] for row in measurements)
    methods = []
    for name, entries in rankings['methods'].items():
        top = entries[0]
        cfg = top['configuration']
        measured = by_id[cfg['id']]
        kind = 'weighted' if name.startswith('weighted') else 'balanced'
        fresh = score_configuration(cfg, models[kind])
        if fresh['score_ns'] != top['score_ns']:
            raise ValueError('frozen method top prediction differs from common costs')
        methods.append({'method': name, 'label': METHOD_LABELS[name],
                        'configuration': cfg['id'], **_placement(cfg), 'prediction_ms': fresh['score_ms'],
                        'ii_ms': measured['ii_ms'], 'fps': measured['fps'],
                        'pool_best_regret_percent': (measured['ii_ms'] / best - 1) * 100})
    metrics = {kind: prediction_metrics([row[kind + '_prediction_ms'] for row in measurements],
                                       [row['ii_ms'] for row in measurements])
               for kind in ('balanced', 'weighted')}
    pairs = placement_pairs(measurements)
    baseline = by_id[baseline_candidate()['id']]
    bridge.update(new_program_ii_ms=baseline['ii_ms'], new_program_fps=baseline['fps'],
                  ii_change_percent=(baseline['ii_ms'] / bridge['ii_ms'] - 1) * 100)
    components = []
    for key, value in sorted(cost_doc['costs'].items()):
        cfg, cost = value['configuration'], value['cost']
        stage = cfg['stages'][0]
        components.append({'component_key': key, 'program': stage['program'], 'threads': stage['threads'],
                           'worker_core_map': ','.join(map(str, stage['worker_core_map'])),
                           **{field: cost.get(field) for field in ('service_ms', 'set_ms', 'run_ms', 'get_ms', 'run_core_ms', 'boundary_core_ms')},
                           'worker_run_core_ms': ','.join(map(str, cost.get('worker_run_core_ms', []))),
                           'cpu_core_weights': ','.join(map(str, cost.get('cpu_core_weights', [])))})
    report = {'experiment': state['experiment'], 'bundle_id': state['bundle_id'], 'boot_id': state['boot_id'],
              'accepted_tasks': len(bound_records), 'components': components, 'pipeline_count': len(measurements),
              'measurements': measurements, 'method_top1': methods, 'prediction_metrics': metrics,
              'caller_placement_pairs': pairs, 'legacy_bridge': bridge,
              'rank_verification': rankings.get('verification'), 'selection': rankings.get('selection'),
              'costs_sha256': _sha(root / 'costs.json'), 'rankings_sha256': _sha(root / 'rankings.json'),
              'pool_sha256': _sha(root / 'search_pool.json'), 'source_raw_results': source_files,
              'scope_warnings': WARNINGS, 'pipeline_feedback_used_for_costs': False,
              'full_DP_changed': False, 'automatic_additional_measurements': False}
    destination = root / 'analysis'; destination.mkdir(exist_ok=True)
    for leaf, rows in [('measurements.csv', measurements), ('stage_times.csv', stages),
                        ('method_top1.csv', methods), ('caller_placement_pairs.csv', pairs),
                        ('components.csv', components)]:
        _csv(destination / leaf, rows)
    (destination / 'analysis.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    _write_report(destination, report)
    _plot(destination, measurements)
    return {'report': str(destination / 'README.md'), 'analysis': str(destination / 'analysis.json'),
            'accepted_tasks': len(bound_records), 'pipeline_count': len(measurements),
            'method_top1_count': len(methods)}


def _write_report(destination, report):
    lines = ['# 逐核负载与调用线程放置：固定切图验证', '',
             f"本轮启动`{report['boot_id']}`，接受{report['accepted_tasks']}项任务，"
             f"取得{len(report['components'])}项共同组件成本和{report['pipeline_count']}项流水结果。",
             '切图固定为CPU0–2→VTA3–4→CPU5–7→VTA8–19→CPU20。末段单线程核心0，主机／控制共享四核，CPU门控关闭。', '',
             '## 四种方法各自的静态第一名', '',
             '各方法在流水计时之前冻结，同一配置共用实测。以下完成间隔均来自16个时间戳的15个间隔。', '',
             '|方法|前段线程／映射|中段线程／映射|预测ms|实测ms|FPS|', '|---|---|---|---:|---:|---:|']
    for row in report['method_top1']:
        lines.append(f"|{row['label']}|{row['front_threads']}／{row['front_worker_map']}|{row['middle_threads']}／{row['middle_worker_map']}|"
                     f"{row['prediction_ms']:.3f}|{row['ii_ms']:.3f}|{row['fps']:.3f}|")
    lines += ['', '## 同一实测池上的预测误差', '',
              '|核时分配|MAE ms|平均绝对百分比误差|中位绝对百分比误差|Spearman|', '|---|---:|---:|---:|---:|']
    for kind, label in [('balanced', '均分'), ('weighted', '隔离逐线程实测')]:
        row = report['prediction_metrics'][kind]
        rho = '不可定义（存在常量序列）' if row['spearman_on_measured_pool_only'] is None else f"{row['spearman_on_measured_pool_only']:.3f}"
        lines.append(f"|{label}|{row['MAE_ms']:.3f}|{row['MAPE_percent']:.2f}%|{row['median_APE_percent']:.2f}%|{rho}|")
    a, b = report['prediction_metrics']['balanced'], report['prediction_metrics']['weighted']
    lines += ['', '隔离逐线程评分在本池中的平均绝对百分比误差' + ('较小' if b['MAPE_percent'] < a['MAPE_percent'] else '未降低') +
              '。这是固定、少量实测配置的观察，不能外推为完整切图域的预测精度。', '',
              '## 固定四线程／核心集合，只改变调用线程位置', '',
              '|匹配变化|前FPS|后FPS|变化|', '|---|---:|---:|---:|']
    for row in report['caller_placement_pairs']:
        lines.append(f"|{row['comparison']}|{row['before_fps']:.3f}|{row['after_fps']:.3f}|{row['fps_change_percent']:+.2f}%|")
    bridge = report['legacy_bridge']
    lines += ['', '## 新旧执行器桥接', '',
              f"相同固定基准：旧程序{bridge['fps']:.3f} FPS，新程序{bridge['new_program_fps']:.3f} FPS，"
              f"完成间隔变化{bridge['ii_change_percent']:+.2f}%。单次测量不能证明统计等效，也不将差异全部归因于编译。", '',
              '## 解释范围', '']
    lines.extend('- ' + text for text in WARNINGS)
    lines += ['', '逐核成本修正和调用线程选择应分别看误差与实际选点；改善小或方向不一致时保留未获支持的结论。'
              '本批不自动启动全量profile、完整DP或追加确认；后续采用哪些机制以本次证据另行冻结。', '',
              '## 产物', '', 'measurements.csv给出每项配置的线程数、映射、两个预测及实测结果；'
              'method_top1.csv列四个冻结选点；caller_placement_pairs.csv给出匹配干预；'
              'stage_times.csv给出全部阶段耗时；components.csv和analysis.json保留成本、范围和原始凭据哈希。', '']
    (destination / 'README.md').write_text('\n'.join(lines))


def _plot(destination, rows):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, label, color in [('balanced', 'Uniform work', '#377eb8'), ('weighted', 'Isolated per-worker work', '#e68613')]:
        axes[0].scatter([row['ii_ms'] for row in rows], [row[name + '_prediction_ms'] for row in rows], label=label, color=color)
    minimum = min(min(row['balanced_prediction_ms'], row['weighted_prediction_ms'], row['ii_ms']) for row in rows)
    maximum = max(max(row['balanced_prediction_ms'], row['weighted_prediction_ms'], row['ii_ms']) for row in rows)
    axes[0].plot([minimum, maximum], [minimum, maximum], '--', color='gray')
    axes[0].set(xlabel='Measured completion interval (ms)', ylabel='Predicted completion interval (ms)', title='Frozen predictions versus one-boot observations')
    axes[0].legend(fontsize=8)
    labels = [str(i + 1) for i in range(len(rows))]
    axes[1].bar(labels, [row['fps'] for row in rows], color='#2878b5')
    axes[1].set(xlabel='Configuration row in measurements.csv', ylabel='FPS', title='Fixed-cut pipeline measurements')
    for ax in axes: ax.grid(axis='y', alpha=.2)
    fig.tight_layout(); fig.savefig(destination / 'prediction_comparison.png', dpi=150)
    fig.savefig(destination / 'prediction_comparison.svg'); plt.close(fig)
