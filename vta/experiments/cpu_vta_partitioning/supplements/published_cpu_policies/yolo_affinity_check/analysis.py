"""Post-measurement replay; average intervals before taking their reciprocal."""
from pathlib import Path
import csv
import json
import statistics

from cpu_vta_partitioning.common import digest, read, sha256, verify_inventory, write
from .common import assert_prepared, configurations
from .prepare import validate_bundle
from .runtime import make_task, summary

LABELS = dict(overlap4='四线程重叠共享', split3='三线程计算分组', pinned='逐线程固定', group='核组内迁移')


def table(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                         for k, v in row.items()} for row in rows)


def aggregate(isolated, pipeline):
    cases, comparisons = [], []
    for name in sorted(pipeline):
        rounds = pipeline[name]
        if sorted(rounds) != [1, 2]:
            raise ValueError('two distinct rounds are required')
        ii = statistics.mean(rounds[r]['ii_ms'] for r in (1, 2))
        service = statistics.mean(rounds[r]['stage_service_ms'][0] for r in (1, 2))
        iso = isolated[name]['cost']['service_ms']
        cases.append(dict(case=name, allocation=name.rsplit('_', 1)[0], affinity_mode=name.rsplit('_', 1)[1],
            round1_FPS=rounds[1]['fps'], round2_FPS=rounds[2]['fps'],
            round1_II_ms=rounds[1]['ii_ms'], round2_II_ms=rounds[2]['ii_ms'],
            mean_II_ms=ii, aggregate_FPS=1000 / ii,
            isolated_front_service_ms=iso, pipeline_front_service_ms=service,
            front_slowdown_ratio=service / iso,
            process_core_ms_per_completion=statistics.mean(rounds[r]['process_core_ms_per_completion'] for r in (1, 2)),
            stage_services_ms=[statistics.mean(rounds[r]['stage_service_ms'][i] for r in (1, 2))
                               for i in range(len(rounds[1]['stage_service_ms']))]))
    indexed = {row['case']: row for row in cases}
    for allocation in ('overlap4', 'split3'):
        fixed, group = [indexed[allocation + '_' + mode] for mode in ('pinned', 'group')]
        gains = [100 * (fixed[f'round{r}_II_ms'] / group[f'round{r}_II_ms'] - 1) for r in (1, 2)]
        comparisons.append(dict(allocation=allocation, pinned_FPS=fixed['aggregate_FPS'], group_FPS=group['aggregate_FPS'],
            group_FPS_change_percent=100 * (fixed['mean_II_ms'] / group['mean_II_ms'] - 1),
            round_gain_percent=gains, both_rounds_faster=all(v > 0 for v in gains),
            both_rounds_slower=all(v < 0 for v in gains), direction_mixed=not(all(v > 0 for v in gains) or all(v < 0 for v in gains)),
            pinned_front_service_ms=fixed['pipeline_front_service_ms'], group_front_service_ms=group['pipeline_front_service_ms'],
            pinned_front_slowdown=fixed['front_slowdown_ratio'], group_front_slowdown=group['front_slowdown_ratio'],
            front_slowdown_reduced=group['front_slowdown_ratio'] < fixed['front_slowdown_ratio'],
            pinned_process_core_ms=fixed['process_core_ms_per_completion'], group_process_core_ms=group['process_core_ms_per_completion']))
    return cases, comparisons


def analyze(output, run_dir):
    output, run_dir = Path(output).resolve(), Path(run_dir).resolve()
    assert_prepared(output)
    bundle = output / 'capsule'
    manifest = validate_bundle(bundle)
    state, identity = read(run_dir / 'state.json'), read(run_dir / 'run.json')
    if state['status'] != 'COMPLETE' or identity['bundle_id'] != manifest['bundle_id']:
        raise ValueError('complete, matching measurement batch required')
    isolated, pipeline, sessions, stages, threads, receipts = {}, {}, [], [], [], {}
    for spec in read(bundle / 'plans/task_order.json'):
        doc = make_task(bundle, spec)
        receipt = run_dir / 'accepted' / identity['boot_id'] / (doc['task_id'] + '.json')
        value = read(receipt)
        if value['task'] != doc or value['boot_id'] != identity['boot_id']:
            raise ValueError('receipt task or boot differs')
        raw = Path(value['raw'])
        if raw.resolve() != (run_dir/'raw'/identity['boot_id']/doc['task_id']).resolve():
            raise ValueError('raw data escaped the expected task directory')
        verify_inventory(raw, value['files'])
        if ((raw/'boot_id').read_text().strip() != identity['boot_id'] or
                (raw/'task_hash').read_text().strip() != digest(doc)):
            raise ValueError('raw boot or task hash differs')
        result = summary(raw, doc, bundle)
        if result != value['summary']:
            raise ValueError('accepted summary did not replay')
        receipts[str(receipt)] = sha256(receipt)
        sessions.append(dict(task_id=doc['task_id'], **spec, correct=result['correct'],
            ii_ms=result.get('ii_ms'), fps=result.get('fps'),
            stage_service_ms=result.get('stage_service_ms'),
            process_core_ms_per_completion=result.get('process_core_ms_per_completion'),
            measurement_and_collection_seconds=value['measurement_and_collection_seconds']))
        for item in result['thread_ledger']:
            threads.append(dict(task_id=doc['task_id'], **spec, **item))
        if spec['phase'] == 'measure':
            if spec['scope'] == 'component':
                isolated[spec['case']] = result
            else:
                pipeline.setdefault(spec['case'], {})[spec['round']] = result
                for i, stage in enumerate(doc['stages']):
                    stages.append(dict(task_id=doc['task_id'], case=spec['case'], round=spec['round'],
                        stage=i, program=stage['program'], device=stage['device'], threads=stage['threads'],
                        allowed_cores=stage['mask'], worker_core_map=stage['worker_core_map'],
                        pipeline_service_ms=result['stage_service_ms'][i],
                        window_inventory=result['window_work']['stages'][i]))
    if len(sessions) != 20 or len(isolated) != 4 or len(pipeline) != 4:
        raise ValueError('frozen experiment coverage incomplete')
    cases, comparisons = aggregate(isolated, pipeline)
    target = run_dir / 'analysis'
    target.mkdir(parents=True, exist_ok=True)
    results = dict(cases=cases, comparisons=comparisons, isolated=isolated, pipeline=pipeline,
        boot_id=identity['boot_id'], bundle_id=manifest['bundle_id'], receipts=receipts,
        migration_trajectories_measured=False, single_boot=True, statistical_significance_claimed=False,
        DP_scores_and_old_profiles_unchanged=True)
    write(target / 'RESULTS.json', results, replace=True)
    costs = dict(native_sessions=20, complete_DNN_inferences=836, isolated_CPU_stage_calls=44,
        startup_probes_not_full_DNNs=True, local_QEMU_CPU_stage_calls=4,
        accepted_measurement_and_collection_seconds=sum(s['measurement_and_collection_seconds'] for s in sessions),
        preparation=read(bundle / 'plans/preparation_cost.json'))
    write(target / 'COSTS.json', costs, replace=True)
    for name, rows in (('configurations', cases), ('comparisons', comparisons), ('sessions', sessions),
                       ('stages', stages), ('threads', threads)):
        table(target / (name + '.csv'), rows)
    lines = ['# YOLO：逐线程固定与核组内迁移', '',
        '固定切图：CPU0–2 → VTA3–6 → CPU7–8 → VTA9–12 → CPU13。CPU门控关闭，主机及控制线程共享四核，原等待机制。', '',
        '| CPU分配 | 绑定方式 | 第一轮FPS | 第二轮FPS | 汇总FPS | 前段隔离／流水 ms | 慢化倍数 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for row in cases:
        lines.append(f"| {LABELS[row['allocation']]} | {LABELS[row['affinity_mode']]} | {row['round1_FPS']:.4f} | {row['round2_FPS']:.4f} | {row['aggregate_FPS']:.4f} | {row['isolated_front_service_ms']:.3f}／{row['pipeline_front_service_ms']:.3f} | {row['front_slowdown_ratio']:.3f} |")
    lines += ['', '汇总FPS由两轮平均完成间隔的倒数计算，不直接平均FPS。每轮64个时间戳对应63个间隔。', '']
    for c in comparisons:
        direction = '两轮方向一致，迁移更快' if c['both_rounds_faster'] else '两轮方向一致，迁移更慢' if c['both_rounds_slower'] else '两轮方向不一致'
        lines += [f"- **{LABELS[c['allocation']]}**：迁移相对固定的汇总FPS变化 {c['group_FPS_change_percent']:+.2f}%；{direction}。两轮变化分别为 {c['round_gain_percent'][0]:+.2f}%、{c['round_gain_percent'][1]:+.2f}%。前段慢化倍数 {c['pinned_front_slowdown']:.3f} → {c['group_front_slowdown']:.3f}。"]
    lines += ['', '只验证放置许可的效果；没有实际迁移轨迹、精确调度等待或能耗测量。三线程分组中两个后段仍共享核心3，不代表全系统独占。',
        '', '20个原生会话，836次完整推理，44次隔离阶段调用。初始化空回调另计；本地QEMU四次CPU调用不混入板测数量。',
        '', '两轮来自同一次启动，不报告显著性或稳定收益。不将流水慢化回填profile，不自动改变执行默认、评分、DP或Top-20。',
        '', f'原始记录：`{run_dir}`', '',
        '逐阶段与窗口在途工作见 stages.csv；实际线程及允许集合见 threads.csv；所有原始凭据可由本入口重新验算。']
    (target / 'README.md').write_text('\n'.join(lines) + '\n')
    plot(target, cases)
    return dict(report=str(target / 'README.md'), comparisons=comparisons, costs=costs)


def plot(target, cases):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import font_manager
    import matplotlib.pyplot as plt
    font = Path('/usr/share/fonts/truetype/arphic-gbsn00lp/gbsn00lp.ttf')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus'] = False
    indexed = {c['case']: c for c in cases}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for j, mode in enumerate(('pinned', 'group')):
        selected = [indexed[a + '_' + mode] for a in ('overlap4', 'split3')]
        for ax, key in zip(axes, ('aggregate_FPS', 'front_slowdown_ratio')):
            bars = ax.bar([i + (j-.5)*.34 for i in range(2)], [c[key] for c in selected],
                          width=.32, label=LABELS[mode])
            ax.bar_label(bars, fmt='%.3f', padding=3)
            ax.set_xticks(range(2), [LABELS[a] for a in ('overlap4', 'split3')])
    axes[0].set_ylabel('FPS（两轮平均间隔的倒数）')
    axes[1].set_ylabel('CPU前段流水／隔离服务时间')
    axes[0].legend()
    fig.tight_layout()
    for ext in ('svg', 'png', 'pdf'):
        fig.savefig(target / ('affinity_comparison.' + ext), dpi=180)
    plt.close(fig)
