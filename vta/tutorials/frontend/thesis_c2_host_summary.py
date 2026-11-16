#!/usr/bin/env python3
"""Read-only reconciliation and plots for independent C2 host prequalification."""
import argparse
import csv
import json
from pathlib import Path
import statistics

from thesis_supplement_run import sha256, write_new_json
from thesis_supplement_audit import reconcile


def load_run(directory):
    manifest = json.loads((directory / 'artifact_hashes.json').read_text())
    for relative, expected in manifest.items():
        path = directory / relative
        if path.stat().st_size != expected['bytes'] or sha256(path) != expected['sha256']:
            raise ValueError('Changed sealed artifact: ' + str(path))
    actual_files = {str(p.relative_to(directory)) for p in directory.rglob('*')
                    if p.is_file() and p.name != 'artifact_hashes.json'}
    if actual_files != set(manifest):
        raise ValueError('Sealed run file inventory drift')
    cost = reconcile(directory)
    saved = json.loads((directory / 'analysis/cost_audit.json').read_text())
    if cost != saved or cost['unfinished_action_ids']:
        raise ValueError('Cost or action completion drift')
    summary = json.loads((directory / 'summary.json').read_text())
    events = []
    for row in directory.glob('raw/run_host_lifecycle*.json'):
        action = json.loads(row.read_text())
        events = [json.loads(line) for line in (directory / action['stdout']).read_text().splitlines()]
    return {'directory': directory, 'summary': summary, 'cost': cost, 'events': events,
            'manifest_sha256': sha256(directory / 'artifact_hashes.json')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--runs', nargs=4, required=True,
                        help='baseline, LSan-failed amendment, accepted amendment, static validation')
    args = parser.parse_args()
    output = Path(args.output_dir)
    runs = [load_run(Path(path)) for path in args.runs]
    baseline, failed_amendment, accepted, static = runs
    assert [r['summary']['status'] for r in runs] == [
        'FAILED_NO_FORMAL_CERTIFICATE', 'FAILED_NO_FORMAL_CERTIFICATE',
        'HOST_PREQUALIFICATION_PASS', 'HOST_STATIC_PREQUALIFICATION_PASS']
    assert baseline['summary']['cases_failed'] == 10
    assert failed_amendment['summary']['cases_passed'] == 160
    assert accepted['summary']['cases_passed'] == 160
    assert all(r['summary']['actual_hardware_calls'] == 0 for r in runs)
    rows = []
    for role, run in zip(['baseline', 'amendment_LSan_failed', 'amendment_accepted', 'static'], runs):
        rows.append({'role': role, 'run_id': run['directory'].name,
                     'status': run['summary']['status'],
                     'passed': run['summary'].get('cases_passed'),
                     'failed': run['summary'].get('cases_failed'),
                     'measured_action_seconds': run['cost']['measured_action_total_seconds'],
                     'failed_actions': len(run['cost']['failed_actions_preserved']),
                     'artifact_manifest_sha256': run['manifest_sha256']})
    write_new_json(output / 'analysis/run_table.json', rows)
    with (output / 'analysis/run_table.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    groups = [('normal', 'normal_bundle'), ('backpressure', 'full_slots'),
              ('token', ('stale_generation', 'wrong_frame', 'wrong_edge', 'wrong_slot')),
              ('transition', ('duplicate_publish', 'early_release', 'wrong_release')),
              ('abort', 'abort_')]
    outcomes, durations = [], []
    for role, run in [('baseline', baseline), ('accepted', accepted)]:
        records = [e for e in run['events'] if e['event'] in ('case_pass', 'case_fail')]
        assert len(records) == 160 and len({e['case'] for e in records}) == 160
        starts = {e['case']: e['host_monotonic_ns'] for e in run['events'] if e['event'] == 'case_start'}
        for name, prefixes in groups:
            members = [e for e in records if e['case'].startswith(prefixes)]
            outcomes.append({'role': role, 'category': name, 'attempted': len(members),
                             'passed': sum(e['event'] == 'case_pass' for e in members)})
        for e in run['events']:
            if e['event'] == 'rejected' and e['case'].startswith(('stale_generation_', 'wrong_frame_')):
                durations.append({'role': role, 'case': e['case'],
                                  'test_start_to_rejection_ms':
                                  (e['host_monotonic_ns'] - starts[e['case']]) / 1e6})
    assert len(durations) == 40
    for name, records in [('case_outcomes', outcomes), ('token_rejection_durations', durations)]:
        write_new_json(output / 'analysis' / (name + '.json'), records)
        with (output / 'analysis' / (name + '.csv')).open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader(); writer.writerows(records)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 3.4), layout='constrained')
    names = [g[0] for g in groups]
    for offset, role, label in [(-0.18, 'baseline', 'Original'), (0.18, 'accepted', 'Amended')]:
        data = [r for r in outcomes if r['role'] == role]
        values = [r['passed'] for r in data]
        bars = ax.bar([i + offset for i in range(5)], values, width=.36, label=label)
        ax.bar_label(bars, labels=[f'{r["passed"]}/{r["attempted"]}' for r in data], fontsize=8)
    ax.set_xticks(range(5), names); ax.set_ylim(0, 57)
    ax.set_ylabel('Passed host cases'); ax.legend()
    ax.set_title('Production manager with CPU allocation adapters')
    for extension in ('pdf', 'svg'):
        fig.savefig(output / 'figures' / ('host_case_outcomes.' + extension))
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.6, 3.4), layout='constrained')
    for x, role in enumerate(('baseline', 'accepted')):
        data = [r['test_start_to_rejection_ms'] for r in durations if r['role'] == role]
        ax.scatter([x] * len(data), data, alpha=.6, s=18)
        ax.plot([x-.15, x+.15], [statistics.median(data)] * 2, color='black')
    ax.set_xticks([0, 1], ['Original: bounded timeout', 'Amended: identity check'])
    ax.set_yscale('log'); ax.set_ylabel('Test start to rejection (ms)')
    ax.set_title('20 token fault cases per condition; includes setup')
    for extension in ('pdf', 'svg'):
        fig.savefig(output / 'figures' / ('token_rejection_durations.' + extension))
    plt.close(fig)
    linked_reports = '\n'.join(f'- [{row["run_id"]}](../{row["run_id"]}/REPORT.md)' for row in rows)
    total = sum(row['measured_action_seconds'] for row in rows)
    medians = {role: statistics.median(r['test_start_to_rejection_ms'] for r in durations
                                     if r['role'] == role) for role in ('baseline', 'accepted')}
    static_logs = []
    for entry in static['summary']['results']:
        if entry['action'].startswith('test_'):
            static_logs.append((static['directory'] / entry['stdout']).read_text().strip())
    report = (
        '# C2 独立本机实验对照与核验\n\n'
        '用户要求先开展其他实验，因此将03中的本机生产核心预资格独立执行；'
        '01仍阻塞，02尚未完成，03正式板端矩阵尚未执行。\n\n'
        '修正前150/160 case通过，10次abort后Publish均意外成功；'
        '过期generation/错frame各10次按原实现等待1s后拒绝。'
        '修正加入abort发布门控和consumer入口身份检查，160/160 case通过；'
        '保留合法持有者abort后release，不强制busy→FREE。\n\n'
        '正常单/双tensor各10次×64帧；满两slot各10次通过predicate屏障证明第3个producer阻塞；'
        '另覆盖4类token、3类非法转换和5类abort情况。'
        '直接提取冻结生产manager，两处predicate仅加观察器且可逐字还原；'
        'NDArray/分配替换为CPU字节vector，不能证明VTA DMA或真实多tensor绑定。\n\n'
        '中间一次修正运行160断言全通过，但LeakSanitizer在ptrace环境致命退出，整体保留为失败。'
        '接受运行关闭leak扫描，继续开启地址及未定义行为检查，完整native runner语法检查通过。'
        '不主张已证明无内存泄漏。\n\n'
        '生产容量与shared-buffer plan header沿原相对目录逐字复制并运行既有C++测试：\n\n'
        + '\n'.join('- `' + line + '`' for text in static_logs for line in text.splitlines()) + '\n\n'
        '这些仅证明校验函数反例，依赖余额是必要条件，不是调度/完成证明；'
        '不认证真实命令峰值、Q1 program certificate、物理池或完整bundle。\n\n'
        f'20个token错误case的测试起点至拒绝中位数分别为{medians["baseline"]:.6f}ms和'
        f'{medians["accepted"]:.6f}ms，包含manager初始化和测试准备，仅描述本机故障响应，'
        '不属于推理性能label。10重复是机制检查，不视为独立板卡启动或统计泛化样本。\n\n'
        '图表来自原始host单调时钟事件：\n\n'
        '- [功能结果](figures/host_case_outcomes.svg)（[PDF](figures/host_case_outcomes.pdf)）\n'
        '- [token故障响应](figures/token_rejection_durations.svg)（[PDF](figures/token_rejection_durations.pdf)）\n\n'
        f'四个来源运行已付动作成本合计{total:.6f}s，含失败；核验/制图成本另见本run账本，'
        '禁止将来源成本重复归入核验动作。开发/人工等待为unknown/null，非在线T0墙钟。'
        '所有sealed文件清单/hash与账本独立复算一致，无未结束动作。\n\n'
        '完整原始运行（含两次失败）：\n\n' + linked_reports + '\n\n'
        '后续：恢复板卡完成01，完成02共同语义及真实准确率，重新构建包含本次改动的native runner；'
        '再接managed真实latch、认证完整bundle和当前FINISH后峰值/Q1，并完成03两boot矩阵。'
        '本轮可支持“生产核心在所测host故障下拒绝非法发布且保留安全release”的有限主张；'
        '不能宣称完整共享系统、内存节省或最终性能已成立。\n')
    (output / 'REPORT.md').write_text(report)
    write_new_json(output / 'summary.json', {
        'status': 'HOST_LOCAL_RESULTS_AUDITED', 'formal_phase03_completed': False,
        'actual_hardware_calls': 0, 'source_runs': rows,
        'source_paid_action_total_seconds': total,
        'source_costs_charged_again_to_this_run': False,
        'token_test_start_to_rejection_median_ms': medians,
        'all_sealed_source_hashes_and_costs_verified': True,
    })
    print(json.dumps({'source_runs_verified': 4, 'formal_phase03_completed': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
