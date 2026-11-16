"""Read-only reconciliation and export of a completed two-operator attempt."""
import argparse
from pathlib import Path

from c3_tps_sync_release_local import read, write, sha, verify, require
from c3_yolo416_comparison_protocol import analyse, canonical


def closeout(source, output):
    require(not output.exists(), 'RESULT_DIRECTORY_EXISTS')
    completion = read(source/'completion_manifest.json')
    require(completion['performance_conclusion_allowed'] is True, 'SOURCE_NOT_SUCCESSFUL')
    verify({str(source/name): digest for name, digest in completion['artifact_sha256'].items()})
    ledger = read(source/'call_manifest.json'); global_rows = read(source/'board/results.json')
    session_rows = []
    for ordinal, session in enumerate(ledger['sessions'], 1):
        folder = source/'board'/('session_%03d'%ordinal)
        summary = read(folder/'summary.json')
        require(summary['status'] == 'COMPLETE_FRESH_UNIT_SUCCESS' and
                summary['session'] == session and len(summary['results']) == 6,
                'FRESH_SESSION_CLOSURE_FAILED')
        for position, row in enumerate(summary['results'], 1):
            require(canonical(row) == canonical(read(folder/('call_%02d_end.json'%position))),
                    'INDIVIDUAL_CALL_AND_SESSION_DIFFER')
            execution = read(folder/('call_%02d_execution.json'%position))
            require(execution['helper_returned'] is True and
                    execution['metadata']['started_calls'] == execution['metadata']['completed_calls'] == position,
                    'HELPER_COMPLETION_EVIDENCE_MISSING')
            require(row['helper'] == execution, 'GLOBAL_EXECUTION_EVIDENCE_DIFFERS')
        session_rows.extend(summary['results'])
    require(canonical(session_rows) == canonical(global_rows), 'SESSION_AND_GLOBAL_ROWS_DIFFER')
    stats = analyse(ledger, session_rows)
    require(canonical(stats) == canonical(read(source/'statistics.json')), 'INDEPENDENT_STATISTICS_DIFFER')
    start, end = (read(source/'board'/name) for name in ('preflight.json', 'final_state.json'))
    require(start['boot_id'] == end['boot_id'] and not end['accelerator_occupancy_scan'].strip(),
            'FINAL_BOOT_OR_ACCELERATOR_OCCUPANCY_CHANGED')
    output.mkdir()
    write(output/'result.json', {'status': 'COMPLETE_INDEPENDENT_TWO_NODE_RECONCILIATION',
          'source': str(source), 'source_manifest_sha256': sha(source/'completion_manifest.json'),
          'source_artifacts_verified': len(completion['artifact_sha256']),
          'target_calls': len(global_rows), 'formal_samples': ledger['formal_samples'],
          'fresh_RPC_units': len(ledger['sessions']), 'new_target_calls': 0,
          'statistics': stats, 'three_way': read(source/'three_way_performance.json'),
          'boot_id': start['boot_id'], 'I_reuses_P_measurements': True})
    # A conventional exported chart; the hatched third series visibly shares P.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    items = stats['objects']; x = np.arange(len(items)); width = .24
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for offset, variant, label, color, hatch in (
            (-1, 'TH', 'TopHub', '#8c929a', None),
            (0, 'P', 'Local TPS', '#3977b8', None),
            (1, 'P', 'TPS + identity reuse*', '#81bca6', '//')):
        values = [item['T_ms'][variant] for item in items]
        bars = ax.bar(x+offset*width, values, width, label=label, color=color,
                      edgecolor='#333333', linewidth=.5, hatch=hatch)
        for bar, value in zip(bars, values):
            ax.annotate('%.3f'%value, (bar.get_x()+bar.get_width()/2, value),
                        xytext=(0, 4), textcoords='offset points', ha='center', fontsize=9)
    ax.set_xticks(x, ['Node '+item['object_id'][-3:].lstrip('0') for item in items])
    ax.set_ylabel('Complete fused-operator latency (ms)')
    ax.set_title('YOLOv3-tiny-416 on AXU5EVB')
    ax.set_ylim(0, max(item['T_ms'][v] for item in items for v in ('TH', 'P'))*1.23)
    ax.legend(frameon=False, fontsize=9); ax.spines[['top', 'right']].set_visible(False)
    fig.text(.5, .025, '* Identical executable to TPS; same measurements, no separate run.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .055, 1, 1))
    for extension in ('png', 'pdf', 'svg'):
        fig.savefig(output/('three_way_operator_latency.'+extension), dpi=180)
    plt.close(fig)
    lines = ['# YOLO416两个算子的三方案对照结果', '',
             '本次实测比较TopHub与本地适配TPS。身份复用在这两个TPS表示上均不改变程序，因此第三列共享TPS测量。', '',
             '|算子|TopHub ms|TPS ms|TPS加身份复用 ms|TPS相对TopHub配对改善|胜场|',
             '|---|---:|---:|---:|---:|---:|']
    for item in items:
        lines.append('|节点%s|%.6f|%.6f|%.6f（同TPS）|%+.4f%%|%d/6|' %
                     (item['object_id'][-3:].lstrip('0'), item['T_ms']['TH'], item['T_ms']['P'],
                      item['T_ms']['P'], item['Gain_P_over_TH_percent'], item['strict_wins']))
    lines += ['', '240次完整融合调用、40个fresh RPC单元及72个正式样本全部通过。'
              '独立核对逐调用、会话与全局记录一致，六轮统计一致，同一Linux启动，实验结束设备无外部占用。',
              'TPS加身份复用的额外收益在这两个固定表示上没有形成；第三列不是独立重复测量。'
              'TPS相对TopHub的差异不能计为身份复用贡献。当前结果是算子时延，不是整图FPS。',
              '[详细实验报告](%s)；[三方案数据CSV](%s)；[可导出图表](three_way_operator_latency.pdf)。' %
              (source/'REPORT.md', source/'three_way_performance.csv'), '', '本轮完成，暂停实验。']
    (output/'REPORT.md').open('x').write('\n'.join(lines)+'\n')
    write(output/'completion_manifest.json', {'status': 'SEALED_INDEPENDENT_TWO_NODE_RESULT',
          'source_script': str(Path(__file__).resolve()), 'source_script_sha256': sha(__file__),
          'artifact_sha256': {p.name: sha(p) for p in output.iterdir() if p.is_file()}})
    return {'status': 'COMPLETE', 'new_target_calls': 0, 'results': stats['objects']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path); args = parser.parse_args()
    print(closeout(args.source.resolve(), args.output.resolve()))
