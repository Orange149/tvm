"""Independent window audit and readable plots for the approved common cohort.

Consumes only the sealed, complete native analysis.  This code has no board
transport, ranking implementation, model fitting, or candidate selection path.
"""
from pathlib import Path
from collections import Counter
from statistics import mean, correlation
import json

import run_scope_tail_comparison as campaign
from run_scope_rank import assert_plan, inputs_for
from cpu_vta_partitioning.common import read, write, sha256, verify_inventory
from scheduler_gap_audit.curves import csv_write
from audit_latest_dp_window import window_inventory

ROOT = Path(__file__).resolve().parent
OUT = campaign.OUT / 'overview'
REPORT = ROOT.parents[4] / 'paper/my_paper/analysis/20261112_common_tail_comparison'
METHODS = ('tarnawski', 'original_complete', 'run_scope_complete')
LABELS = ('Tarnawski单核适配', '已有最强完整模型', '计算区间线程份额修正')
COLORS = ('#808b96', '#2579a7', '#d39831')


def average_ranks(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [None] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        for i in order[start:end]:
            result[i] = (start + 1 + end) / 2
        start = end
    return result


def spearman(left, right):
    a, b = average_ranks(left), average_ranks(right)
    return None if len(set(a)) < 2 or len(set(b)) < 2 else correlation(a, b)


def method_statistics(rows):
    if len(rows) != 20 or len({r['configuration'] for r in rows}) != 20:
        raise ValueError('exactly20 independently identified candidates are required')
    best = min(rows, key=lambda r: (r['actual_II_ms'], r['configuration']))
    return dict(mean_absolute_percent_II_error=mean(abs(r['signed_II_error_percent']) for r in rows),
                mean_signed_percent_II_error=mean(r['signed_II_error_percent'] for r in rows),
                spearman_predicted_actual_II=spearman([r['predicted_II_ms'] for r in rows],
                                                       [r['actual_II_ms'] for r in rows]),
                distinct_predicted_scores=len({r['predicted_II_ns'] for r in rows}),
                actual_best_static_rank=best['static_rank'], actual_best_configuration=best['configuration'],
                first_rank=rows[0]['configuration'], first_FPS=rows[0]['FPS'],
                best_FPS=best['FPS'], first_II_ms=rows[0]['actual_II_ms'], best_II_ms=best['actual_II_ms'])


def plot_all(budgets, candidates):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont('/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf')
    plt.rcParams.update({'font.family': ['DejaVu Sans', 'Droid Sans Fallback'],
                         'axes.unicode_minus': False, 'pdf.fonttype': 42})
    networks = ('resnet18', 'yolov3_tiny416')
    names = ('ResNet18', 'YOLOv3-tiny416')
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ni, (network, name) in enumerate(zip(networks, names)):
        ymax = max(r['FPS'] for r in budgets if r['network'] == network) * 1.16
        for ki, budget in enumerate((1, 5, 20)):
            ax = axes[ni, ki]
            values = [next(r['FPS'] for r in budgets if r['network'] == network and
                           r['method'] == method and r['budget'] == budget) for method in METHODS]
            bars = ax.bar(range(3), values, color=COLORS, width=.65)
            ax.bar_label(bars, fmt='%.3f', fontsize=11, padding=4)
            ax.set_xticks(range(3), ['Tarnawski\n单核适配', '已有最强\n完整模型', '计算区间\n线程份额修正'])
            ax.set_ylim(0, ymax); ax.set_ylabel('实测FPS'); ax.grid(axis='y', alpha=.2)
            title = '静态第一名' if budget == 1 else f'前{budget}项中实测最佳'
            ax.set_title(f'{name}：{title}')
    fig.suptitle('同批完整流程比较：首选质量与有限预算内最佳性能', fontsize=16)
    fig.text(.5, .013, '输入0，同一启动；每配置8帧预热＋16个完成时间戳＋24帧尾部。相同配置共用一次测量。无显著性结论。',
             ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .045, 1, .945))
    for suffix in ('png', 'svg', 'pdf'):
        fig.savefig(REPORT / ('common_budget_comparison.' + suffix), dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.7))
    for ni, (network, name) in enumerate(zip(networks, names)):
        group = [r for r in candidates if r['network'] == network]
        low = min(min(r['predicted_II_ms'], r['actual_II_ms']) for r in group) * .94
        high = max(max(r['predicted_II_ms'], r['actual_II_ms']) for r in group) * 1.05
        for mi, (method, label) in enumerate(zip(METHODS, LABELS)):
            ax = axes[ni, mi]; rows = [r for r in group if r['method'] == method]
            ax.scatter([r['predicted_II_ms'] for r in rows], [r['actual_II_ms'] for r in rows],
                       c=COLORS[mi], alpha=.8, edgecolors='white')
            ax.plot([low, high], [low, high], color='#777777', linestyle='--')
            ax.set_xlim(low, high); ax.set_ylim(low, high); ax.set_title(f'{name}：{label}', fontsize=11)
            ax.set_xlabel('静态预测完成间隔（ms）'); ax.set_ylabel('实测完成间隔（ms）'); ax.grid(alpha=.2)
    fig.suptitle('各方法严格Top-20：预测误差单独评价', fontsize=16)
    fig.text(.5, .012, '虚线为预测＝实测；点重合表示相同或接近的数据。本图不用于回填成本或修改排名。', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .045, 1, .94))
    for suffix in ('png', 'svg', 'pdf'):
        fig.savefig(REPORT / ('prediction_error.' + suffix), dpi=180)
    plt.close(fig)


def run():
    if OUT.exists():
        seal = read(OUT / 'MANIFEST.json')
        if seal['source_sha256'] != sha256(__file__):
            raise ValueError('sealed overview implementation changed')
        for path, wanted in seal['files'].items():
            if sha256(path) != wanted:
                raise ValueError('sealed overview artifact changed')
        cached = read(OUT / 'RESULTS.json')
        for path, wanted in cached['sources'].items():
            if sha256(path) != wanted:
                raise ValueError('sealed overview evidence changed')
        print(json.dumps(dict(status='VALIDATED_CACHED', configurations=cached['configurations'],
                              report=str(REPORT / 'FINDINGS_zh.txt')), ensure_ascii=False))
        return cached
    prepared = campaign.assert_sources()
    state = read(campaign.RUN / 'state.json')
    if state['status'] != 'COMPLETE' or state['completed_configurations'] != 88:
        raise ValueError('complete approved native cohort required')
    base = campaign.OUT / 'analysis'
    for path, wanted in read(base / 'MANIFEST.json')['files'].items():
        if sha256(path) != wanted:
            raise ValueError('sealed common analysis changed')
    results = read(base / 'RESULTS.json')
    lists, pool = read(campaign.OUT / 'LISTS.json'), read(campaign.OUT / 'POOL.json')
    sources = {str(base / 'RESULTS.json'): sha256(base / 'RESULTS.json')}
    measurements = {(r['network'], r['configuration']): r for r in results['rows']}
    if len(measurements) != 88 or len(results['sources']) != 176 or results['DNN_inferences'] != 4312:
        raise ValueError('native cohort cardinality changed')
    windows, stages, predictions, statistics = [], [], [], []
    cost_plan = assert_plan()
    isolated = {network: inputs_for(cost_plan, network, 'original')[1] for network in pool}
    for (network, cid), measured in measurements.items():
        path = measured['measurement_receipt']
        if sha256(path) != measured['measurement_receipt_sha256']:
            raise ValueError('accepted common performance receipt changed')
        receipt = read(path); verify_inventory(receipt['raw'], receipt['files'])
        doc = receipt['task']; cfg = pool[network][cid]
        if doc['configuration'] != cfg or receipt['boot_id'] != results['boot_id']:
            raise ValueError('window execution/boot differs')
        if [doc[k] for k in ('warmup', 'scored', 'tail')] != [8, 16, 24]:
            raise ValueError('unexpected finite-window protocol')
        rows = [json.loads(s) for s in (Path(receipt['raw']) / 'result.jsonl').read_text().splitlines()]
        if len(rows) != 48:
            raise ValueError('not48 complete frame outputs')
        inventory = window_inventory(rows, cfg['stages'], 8, 16)
        for i, row in enumerate(inventory['stages']):
            if measured['stage_service_ms'][i] <= 0:
                raise ValueError('invalid stage time')
            cost = isolated[network].get(cfg['stages'][i], cfg)
            selected = rows[8:24]
            stages.append(dict(network=network, configuration=cid, **row,
                threads=cfg['stages'][i]['threads'], mask=cfg['stages'][i]['mask'],
                isolated_service_ms=cost['service_ms'], isolated_run_ms=cost['run_ms'],
                isolated_run_CPU_ms=cost['run_core_ms'], isolated_source_key=cost['source_key'],
                cost_assumptions=cost['assumptions'],
                native_service_ms=measured['stage_service_ms'][i],
                native_run_ms=mean(r[f'stage{i}_run_ms'] for r in selected),
                native_set_ms=mean(r[f'stage{i}_set_ms'] for r in selected),
                native_get_ms=mean(r[f'stage{i}_get_ms'] for r in selected),
                native_over_isolated=measured['stage_service_ms'][i]/cost['service_ms'],
                stage_call_means_not_same_as_completion_window_resource_occupancy=True))
        windows.append(dict(network=network, configuration=cid,
                            FPS=measured['FPS'], II_ms=measured['II_ms'],
                            any_early_exhaustion=any(r['stage_exhausted_before_window_end'] for r in inventory['stages']),
                            maximum_inventory_change=max(abs(r['inventory_change']) for r in inventory['stages']),
                            inventory=inventory))
        sources[path] = sha256(path)
    for network, methods in lists.items():
        union_best = min(r['II_ms'] for r in results['rows'] if r['network'] == network)
        for method in METHODS:
            current = []
            for candidate in methods[method]:
                observed = measurements[(network, candidate['id'])]
                row = dict(network=network, method=method, configuration=candidate['id'],
                    static_rank=candidate['static_rank'], predicted_II_ns=candidate['score_ns'],
                    predicted_II_ms=candidate['score_ns']/1e6, actual_II_ms=observed['II_ms'],
                    signed_II_error_percent=100*(candidate['score_ns']/1e6/observed['II_ms']-1),
                    FPS=observed['FPS'], context=candidate['context_id'],
                    host_mask=candidate['host_mask'], control_mask=candidate['control_mask'],
                    CPU_gate=candidate['execution_mode'],
                    CPU_stages=[s for s in candidate['stages'] if s['device'] == 'cpu'])
                current.append(row); predictions.append(row)
            stats = method_statistics(current)
            statistics.append(dict(network=network, method=method, **stats,
                                   mean_error_is_not_selection_gain=True))
        for budget in (1, 5, 20):
            for method in METHODS:
                selected = next(r for r in results['budgets'] if r['network'] == network and
                                r['method'] == method and r['budget'] == budget)
                selected['II_loss_vs_observed_union_best_percent'] = 100*(selected['II_ms']/union_best-1)
                selected['reference_is_only_measured_union_not_global_optimum'] = True
    summary = {n: dict(configurations=sum(r['network'] == n for r in windows),
                       early_exhaustion_cases=sum(r['network'] == n and r['any_early_exhaustion'] for r in windows),
                       maximum_inventory_change=max(r['maximum_inventory_change'] for r in windows if r['network'] == n),
                       measured_context_counts=dict(Counter(c['context_id'] for c in pool[n].values())))
               for n in lists}
    OUT.mkdir(parents=True); REPORT.mkdir(parents=True, exist_ok=True)
    report = dict(statistics=statistics, budgets=results['budgets'], windows=windows,
                  window_summary=summary, sources=sources, boot=results['boot_id'],
                  configurations=88, native_sessions=176, DNN_inferences=4312,
                  no_new_native_execution=True, no_model_or_ranking_change=True,
                  selection_sha256=sha256(campaign.OUT / 'FREEZE.json'),
                  preparation=prepared['bundles'], raw_hashes_rechecked=True,
                  no_stability_or_significance_claim=True)
    write(OUT / 'RESULTS.json', report)
    csv_write(REPORT / 'budgets.csv', results['budgets'])
    csv_write(REPORT / 'all_Top20_predictions.csv', predictions)
    csv_write(REPORT / 'prediction_summary.csv', statistics)
    csv_write(REPORT / 'stage_inventory.csv', stages)
    plot_all(results['budgets'], predictions)
    text = ['两个网络、三种方法：同批完整Top-20实测', '',
            '本批88项不同执行配置、176会话、4312次完整推理。输入0，同一boot；每项单帧资格检查后8预热＋16完成时间戳＋24尾部。',
            '三份严格名单在实测前冻结。未加入历史优秀配置、未依据FPS改公式、未复用旧批FPS。相同配置共用本批一次测量。',
            '原最强完整模型＝非均匀份额＋主机CPU需求＋线程组服务约束＋主机集中约束。',
            '计算区间线程份额修正只改隔离数据中的份额来源；总核时、服务成本、完整公式、域及执行机制不变。这不是新的设备/调度时序预测器。',
            'Tarnawski为本平台单线程物理核映射适配；同核服务累计，单物理VTA累计，含共同DDR约束；不是原论文整套系统原样复现。', '',
            '网络／方法／选点预算／FPS／完成间隔ms／最佳在该静态名单中的名次']
    for r in results['budgets']:
        text.append('%s / %s / %s / %.4f / %.4f / %s' % (
            r['network'], LABELS[METHODS.index(r['method'])], r['budget'], r['FPS'], r['II_ms'], r['static_rank']))
    text += ['', '窗口逐项核查：', *[f'{n}：{v}' for n, v in summary.items()], '',
             '增加尾部避免了先前CPU阶段在窗口内耗尽本批任务的问题；仍需报告在途库存变化。这不是长时稳态、跨启动稳定性或显著性证明。',
             '静态第一名实测FPS反映首选质量，不单独证明数值预测准确；Top-20最佳反映该名单和预算内的选点，不能证明全域最优。',
             '所有约束最大值是规划评分，不是严格性能上限。预测误差见独立图表。',
             '本次只完成该冻结输入修正的完整选点检验；详细设备/线程时序回放仍有关键内核归因不准，未作为新DP模型推广。',
             '三个方法的组件profile与搜索成本另列；不能把这次同批共测解释为它们的总调优成本相同。',
             '电脑控制进程的分段恢复只回收/接续未验收任务，不改变板端boot或重跑已验收会话；恢复记录随原始证据交付。']
    (REPORT / 'FINDINGS_zh.txt').write_text('\n'.join(text) + '\n')
    write(OUT / 'MANIFEST.json', dict(source_sha256=sha256(__file__),
        files={str(p): sha256(p) for p in [OUT / 'RESULTS.json', *REPORT.iterdir()] if p.is_file()}))
    print(json.dumps(dict(window_summary=summary, statistics=statistics,
                          report=str(REPORT / 'FINDINGS_zh.txt')), ensure_ascii=False))
    return report


if __name__ == '__main__':
    run()
