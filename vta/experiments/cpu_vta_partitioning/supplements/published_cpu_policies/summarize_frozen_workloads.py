"""Describe real isolated thread curves for the two frozen full-domain winners.

This report neither reads pipeline FPS nor prunes resource choices.  The four
curves for each subgraph remain its own measurements, including nonmonotonicity.
"""
from pathlib import Path
from collections import Counter
import json

import run_scope_rank
from cpu_vta_partitioning.common import read, write, sha256
from scheduler_gap_audit.curves import csv_write

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'artifacts/frozen_workloads_20261112_01'
REPORT = ROOT.parents[4] / 'paper/my_paper/analysis/20261112_workload_comparison'


def run():
    if OUT.exists():
        raise FileExistsError('preserve the workload report')
    source = ROOT / 'artifacts/scheduler_gap_audit_01/curves/THREAD_CURVES.json'
    frozen = ROOT / 'artifacts/run_scope_tail_comparison_20261112_01/LISTS.json'
    curves, lists = read(source), read(frozen)
    if not curves['no_pipeline_FPS_read'] or not curves['no_candidates_pruned']:
        raise ValueError('isolated full thread curves are required')
    for path, expected in curves['sources'].items():
        if sha256(path) != expected:
            raise ValueError('thread curve source changed')
    rows, groups = [], []
    fields = ('network', 'program', 'threads', 'service_ms', 'set_ms', 'run_ms', 'get_ms',
              'run_core_ms', 'boundary_core_ms', 'speedup_vs_one', 'run_core_growth',
              'nonmonotonic_vs_previous', 'input_tensors', 'output_tensors',
              'input_frontier_bytes', 'output_frontier_bytes', 'source_key', 'mask',
              'worker_core_map', 'core_cost_scope')
    for network, methods in lists.items():
        winner = methods['original_complete'][0]
        for stage in winner['stages']:
            if stage['device'] != 'cpu':
                continue
            matches = sorted([r for r in curves['rows'] if r['network'] == network and
                              r['program'] == stage['program']], key=lambda r: r['threads'])
            if [r['threads'] for r in matches] != [1, 2, 3, 4]:
                raise ValueError('each current subgraph needs four real isolated costs')
            selected = [dict({k: r[k] for k in fields},
                             complete_DP_selected_threads=stage['threads'],
                             complete_DP_selected_mask=stage['mask'],
                             placement_cost_is_homogeneous_alias=r['mask'] != stage['mask'])
                        for r in matches]
            rows.extend(selected)
            groups.append(dict(network=network, program=stage['program'], selected=stage['threads'],
                               configuration=winner['id'], rows=selected))
    if len(groups) != 8:
        raise ValueError('expected four CPU subgraphs in each frozen static winner')
    distributions = []
    for network in lists:
        summaries = [r for r in curves['summaries'] if r['network'] == network]
        distributions.append(dict(network=network, subgraphs=len(summaries),
            best_observed_threads=dict(Counter(r['best_observed_threads'] for r in summaries)),
            within_10_percent_threads=dict(Counter(r['first_threads_within_10pct_of_observed_best'] for r in summaries)),
            nonmonotonic_subgraphs=sum(r['nonmonotonic_pairs'] > 0 for r in summaries),
            thresholds_are_descriptive_not_candidate_pruning=True))
    result = dict(groups=groups, rows=rows, full_domain_thread_summary=distributions,
                  total_real_thread_curves=len(curves['rows']),
                  input0=True, no_pipeline_FPS_read=True, no_candidates_pruned=True,
                  network_name_is_not_a_policy_rule=True,
                  actual_participation_coverage_requires_unified_trace_inventory=True,
                  inventory=str(ROOT / 'artifacts/isolated_trace_inventory_20261112_01/INVENTORY.json'),
                  new_native_sessions=0, sources={str(source): sha256(source), str(frozen): sha256(frozen)})
    OUT.mkdir(parents=True); REPORT.mkdir(parents=True, exist_ok=True)
    write(OUT / 'RESULTS.json', result)
    csv_write(REPORT / 'current_CPU_thread_curves.csv', rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont('/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf')
    plt.rcParams.update({'font.family': ['DejaVu Sans', 'Droid Sans Fallback'], 'axes.unicode_minus': False})
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.1))
    for ax, group in zip(axes.flat, groups):
        values = group['rows']; x = [r['threads'] for r in values]
        ax.plot(x, [r['service_ms'] for r in values], 'o-', color='#2672a5', label='隔离服务时间')
        twin = ax.twinx()
        twin.plot(x, [r['run_core_ms'] for r in values], 's--', color='#cc8230', label='运行CPU核时')
        ax.axvline(group['selected'], color='#8b3c49', alpha=.65, linestyle=':', label='完整DP选中线程数')
        r = values[0]; name = 'ResNet18' if group['network'] == 'resnet18' else 'YOLOv3-tiny'
        program = group['program'].replace('cpu_', '').replace('_', '–')
        ax.set_title(f'{name} CPU{program}\n{r["input_tensors"]}输入／{r["output_tensors"]}输出', fontsize=11)
        ax.set_xticks(x); ax.set_xlabel('配置线程数'); ax.set_ylabel('服务时间（ms）', color='#2672a5')
        twin.set_ylabel('运行核时（核·ms）', color='#cc8230')
        ax.grid(alpha=.2); ax.set_ylim(bottom=0); twin.set_ylim(bottom=0)
    from matplotlib.lines import Line2D
    fig.legend([Line2D([], [], color='#2672a5', marker='o'),
                Line2D([], [], color='#8b3c49', linestyle=':'),
                Line2D([], [], color='#cc8230', marker='s', linestyle='--')],
               ['蓝：隔离服务时间', '点线：DP选中线程数', '橙虚线：运行CPU核时（右轴）'],
               loc='upper center', bbox_to_anchor=(.5, .955), ncol=3, frameon=False, fontsize=10)
    fig.suptitle('相同静态方法，按各子图真实曲线分配线程', y=.997, fontsize=15)
    fig.text(.5, .012, '所选布局均有4个CPU子图；其他同核数集合仍采用同构核近似。曲线不作平滑、不据此裁剪候选。',
             ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .04, 1, .91))
    for suffix in ('png', 'svg', 'pdf'):
        fig.savefig(REPORT / ('thread_curves.' + suffix), dpi=180)
    plt.close(fig)
    lines = ['两个网络的负载差异：冻结方法、分别使用真实profile', '',
             '这里整理的是已有隔离数据，不新增板测，也不按网络名称制定不同评分公式。',
             f'全表覆盖{len(curves["summaries"])}个CPU区间、{len(curves["rows"])}项1–4线程实测成本。下表取两个完整域静态第一名使用的8个CPU子图。',
             '格式：网络／CPU子图／1–4线程服务ms／1–4线程运行核时（核ms）／DP选择线程数']
    for group in groups:
        values = group['rows']
        lines.append('%s / %s / %s / %s / %s' % (group['network'], group['program'],
            ', '.join('%.3f' % r['service_ms'] for r in values),
            ', '.join('%.3f' % r['run_core_ms'] for r in values), group['selected']))
    lines.extend(['',
        'ResNet CPU20仅约2.8ms且多线程几乎不降服务；YOLO CPU12–13包含双头末端及两输入，服务从77.2ms降至约21.0ms。不能将两个网络的“尾段”当成相同工作。',
        'ResNet CPU0–2和CPU5–7均有明显并行收益，但静态第一名仍选择3线程与1线程：局部最快不是共享流水最优的充分条件。',
        'YOLO CPU2–4单线程约366.3ms，是此处最重的CPU子图；3线程约140.4ms，其计算量与ResNet轻尾段不是一个量级。',
        'YOLO最终8输出共862068字节，ResNet最终1输出共4000字节。输出工作量差异可以解释控制工作的差异，不能直接把输出CPU核时加到完成间隔上。',
        '这些是可测特征差异，并未单独证明哪种结构导致全部共享慢化。运行CPU核时包含运行时工作；线程负载、主机位置和依赖时序仍需分别看诊断证据。',
        '各阶段线程数与核心集合独立选择是原有能力。本次没有提出网络名称规则、删掉线程配置或缩放另一网络的成本。',
        '部分原始子图采用前缀核心测量，完整布局上的其他同核数集合仍属明确标注的同构核近似；实际参与线程只在有轨迹时判断。',
    ])
    (REPORT / 'FINDINGS_zh.txt').write_text('\n'.join(lines) + '\n')
    write(OUT / 'MANIFEST.json', dict(source_sha256=sha256(__file__),
        files={str(p): sha256(p) for p in [OUT / 'RESULTS.json', *REPORT.iterdir()] if p.is_file()}))
    print(json.dumps(dict(groups=len(groups), real_signatures=len(curves['rows']), distributions=distributions), ensure_ascii=False))


if __name__ == '__main__':
    run()
