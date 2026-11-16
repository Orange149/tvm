"""Remeasure only the frozen old/new full-DP lists, in a fresh native batch.

Reuse the certified preparation, native task builder and durable controller.
This smaller campaign excludes the forty additional Tarnawski-only layouts
from the deferred comparison. Neither the original lists nor sources change.
"""
from pathlib import Path
from types import FunctionType
from copy import deepcopy
import argparse
import json
import sys

import run_scope_board as parent
from cpu_vta_partitioning.common import read, write, sha256, digest, verify_inventory, now
from run_scope_rank import freeze, verify_one
from common_top20.runtime import task, summary
from analyze_run_scope_board import budget_choices, require_complete
from scheduler_gap_audit.curves import csv_write

ROOT = parent.ROOT
OUT = ROOT / 'artifacts/run_scope_retest_20261112_01'
RUN = ROOT / 'runs/run_scope_retest_20261112_01'
CONFIG = parent.CONFIG
METHODS = ('original_complete', 'run_scope_complete')
LABELS = {'original_complete': '已有最强完整模型', 'run_scope_complete': '计算区间线程份额修正'}


def sources():
    return {**parent.sources(), str(Path(__file__)): sha256(__file__),
            str(ROOT / 'tests/test_run_scope_retest.py'): sha256(ROOT / 'tests/test_run_scope_retest.py'),
            str(ROOT / 'analyze_run_scope_board.py'): sha256(ROOT / 'analyze_run_scope_board.py')}


def bind(name):
    original = getattr(parent, name)
    namespace = {**original.__globals__, 'OUT': OUT, 'RUN': RUN, 'sources': sources,
        'selection': selection, 'assert_sources': assert_sources, 'validate': validate,
        'bind_run': bind_run}
    return FunctionType(original.__code__, namespace, original.__name__, original.__defaults__)


def subset(lists, pool, order):
    selected_lists = {n: {m: deepcopy(rows[m]) for m in METHODS} for n, rows in lists.items()}
    wanted = {n: {r['id'] for rows in versions.values() for r in rows}
              for n, versions in selected_lists.items()}
    for network, versions in selected_lists.items():
        for rows in versions.values():
            if len(rows) != 20 or len({r['id'] for r in rows}) != 20:
                raise ValueError('twenty distinct ordered configurations required')
        if not wanted[network].issubset(pool[network]):
            raise ValueError('a frozen configuration is missing from the capsule pool')
    selected_pool = {n: {key: deepcopy(pool[n][key]) for key in sorted(ids)} for n, ids in wanted.items()}
    selected_order = [deepcopy(row) for row in order if row['configuration'] in wanted[row['network']]]
    expected = {(n, key) for n, ids in wanted.items() for key in ids}
    if (len(selected_order) != len(expected)
            or {(r['network'], r['configuration']) for r in selected_order} != expected):
        raise ValueError('subset order is incomplete or repeated')
    return selected_lists, selected_pool, selected_order


def freeze_selection():
    parent.selection()
    old_lists = read(parent.OUT / 'LISTS.json')
    lists, pool, order = subset(old_lists, read(parent.OUT / 'POOL.json'), read(parent.OUT / 'ORDER.json'))
    if {n: len(v) for n, v in pool.items()} != {'resnet18': 20, 'yolov3_tiny416': 28}:
        raise ValueError('frozen old/new list union changed')
    proofs = {}
    for network in lists:
        for method, variant in zip(METHODS, ('original', 'run_scope')):
            rows, _ = verify_one(network, variant)
            if [(r['id'], r['score_ns']) for r in rows] != [(r['id'], r['score_ns']) for r in lists[network][method]]:
                raise ValueError('subset does not exactly match verified full-domain DP')
            path = ROOT / 'artifacts/run_scope_rank_01' / network / variant / 'PROOF.json'
            proofs[str(path)] = sha256(path)
    protocol = deepcopy(read(parent.OUT / 'PROTOCOL.json'))
    protocol.update(experiment='run_scope_retest_20261112_01',
        current_state='FROZEN_OLD_NEW_DP_RETEST',
        additional_native_campaign_requires_user_approval=False,
        user_authorized_scope='Remeasure the new DP results with their old-list comparison; no new profiling or formula changes.',
        maximum_distinct_configurations=48, native_sessions=96, DNN_inferences=1392)
    plan = dict(experiment=protocol['experiment'], configurations=48, native_sessions=96,
        DNN_inferences=1392, configurations_by_network={n: len(v) for n, v in pool.items()},
        methods=list(METHODS), excluded_Tarnawski_only_configurations=40,
        old_source_selection_sha256=sha256(parent.OUT / 'FREEZE.json'),
        full_DP_proofs=proofs, no_new_profiles=True, no_formula_changes=True,
        no_reboot=True, no_reprogram=True, no_system_parameter_changes=True,
        measured_FPS_not_used_to_select_lists=True,
        estimated_minutes_with_deployment_and_collection=[20, 30])
    for name, document in [('PLAN.json', plan), ('POOL.json', pool), ('ORDER.json', order),
                           ('LISTS.json', lists), ('PROTOCOL.json', protocol)]:
        freeze(OUT / name, document)
    freeze(OUT / 'FREEZE.json', dict(files={name: sha256(OUT / name) for name in
        ('PLAN.json', 'POOL.json', 'ORDER.json', 'LISTS.json', 'PROTOCOL.json')}))
    freeze(OUT / 'AUTHORIZATION.json', dict(user_request='我认为现在先重新上板子实测一下新的dp结果',
        scope=plan, no_permission_needed_for_individual_sessions=True,
        user_authorization_precedes_first_native_task=True))
    return plan


def selection():
    return bind('selection')()


def assert_sources():
    return bind('assert_sources')()


def validate(bundle):
    return bind('validate')(bundle)


def bind_run(run_dir, boot, bundles):
    return bind('bind_run')(run_dir, boot, bundles)


def prepare():
    freeze_selection()
    return bind('prepare')()


def qemu():
    return bind('qemu')()


def board(run_dir=RUN, config=CONFIG, *, execute_board=False):
    authorization = read(OUT / 'AUTHORIZATION.json')
    plan, _, _ = selection()
    if authorization['scope'] != plan:
        raise ValueError('native scope differs from the authorized frozen campaign')
    return bind('board')(run_dir, config, execute_board=execute_board)


def analyze(run_dir=RUN):
    run_dir = Path(run_dir)
    prepared = assert_sources()
    plan, pool, _ = selection()
    state = read(run_dir / 'state.json')
    require_complete(state, plan)
    identity = read(run_dir / 'run.json')
    boot = state['boot_id']
    if (identity['boot_id'] != boot or identity['bundles'] != prepared['bundles']
            or identity['selection_sha256'] != sha256(OUT / 'FREEZE.json')
            or identity['preparation_sha256'] != sha256(OUT / 'PREPARATION.json')):
        raise ValueError('completed batch or preparation identity differs')
    output = OUT / 'analysis'
    if (output / 'MANIFEST.json').exists():
        for path, expected in read(output / 'MANIFEST.json')['files'].items():
            if sha256(path) != expected:
                raise ValueError('sealed result changed')
        return read(output / 'RESULTS.json')
    lists = read(OUT / 'LISTS.json')
    measurements = {n: {} for n in pool}
    sources_used, rows, expected_paths = {}, [], set()
    session_seconds = 0
    for network, candidates in pool.items():
        bundle = OUT / network / 'capsule'
        validate(bundle)
        for key, cfg in candidates.items():
            qualified = None
            for phase in ('qualify', 'short'):
                document = task(bundle, cfg, phase)
                path = run_dir / 'accepted' / boot / (document['task_id'] + '.json')
                expected_paths.add(path.resolve())
                receipt = read(path)
                if receipt['task'] != document or receipt['boot_id'] != boot:
                    raise ValueError('task or boot differs')
                verify_inventory(receipt['raw'], receipt['files'])
                checked = summary(receipt['raw'], document, bundle)
                if checked != receipt['summary']:
                    raise ValueError('raw acceptance no longer reproduces')
                sources_used[str(path)] = sha256(path)
                session_seconds += receipt['measurement_and_collection_seconds']
                if phase == 'qualify':
                    qualified = checked['output_fingerprints']
                else:
                    if checked['output_fingerprints'] != qualified:
                        raise ValueError('performance output differs from qualification')
                    measurements[network][key] = checked
                    rows.append(dict(network=network, configuration=key, FPS=checked['FPS'], II_ms=checked['II_ms'],
                        stage_service_ms=checked['stage_service_ms'],
                        CPU_ms_per_completion=checked['process_CPU_ms_per_completion'],
                        stage_completion_counts_at_window=checked['stage_completion_counts_at_window'],
                        context=cfg['context_id'], CPU_gate=cfg['execution_mode'],
                        CPU_stages=[s for s in cfg['stages'] if s['device'] == 'cpu'],
                        host_mask=cfg['host_mask'], control_mask=cfg['control_mask'],
                        receipt=str(path), receipt_sha256=sha256(path)))
    if {p.resolve() for p in (run_dir / 'accepted' / boot).glob('*.json')} != expected_paths:
        raise ValueError('foreign, missing or repeated accepted session')
    budgets, predictions = [], []
    for network in pool:
        budgets += [dict(network=network, **row) for row in budget_choices(lists[network], measurements[network])]
        for method in METHODS:
            for row in lists[network][method]:
                value = measurements[network][row['id']]
                predicted = row['score_ns'] / 1_000_000
                predictions.append(dict(network=network, method=method, configuration=row['id'],
                    static_rank=row['static_rank'], predicted_II_ms=predicted, actual_II_ms=value['II_ms'],
                    actual_FPS=value['FPS'], signed_error_ms=predicted - value['II_ms']))
    result = dict(boot_id=boot, rows=rows, budgets=budgets, predictions=predictions, sources=sources_used,
        source_sha256=sha256(__file__), selection_sha256=sha256(OUT / 'FREEZE.json'),
        native_sessions=96, DNN_inferences=1392, old_FPS_not_reused=True,
        no_formula_changes=True, shared_stage_timing_gap_not_claimed_solved=True,
        accepted_measurement_and_collection_seconds=session_seconds,
        last_controller_invocation_seconds=state['seconds'],
        single_input_single_boot_short_window=True,
        historical_13FPS_is_not_a_measurement_from_this_batch=True)
    output.mkdir(parents=True, exist_ok=True)
    freeze(output / 'STATE_SNAPSHOT.json', state)
    freeze(output / 'RESULTS.json', result)
    csv_write(output / 'candidates.csv', rows)
    csv_write(output / 'budgets.csv', budgets)
    csv_write(output / 'predictions.csv', predictions)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, network in zip(axes, pool):
        for j, method in enumerate(METHODS):
            values = [next(r['FPS'] for r in budgets if r['network'] == network and r['method'] == method
                           and r['budget'] == k) for k in (1, 5, 20)]
            bars = ax.bar([i + (j - .5) * .36 for i in range(3)], values, width=.36,
                          label=['Existing complete model', 'Run-scoped thread shares'][j])
            ax.bar_label(bars, fmt='%.3f', padding=2, fontsize=8)
        ax.set_xticks(range(3), ['1', '5', '20'])
        ax.set_xlabel('Full-pipeline measurement budget')
        ax.set_ylabel('Observed FPS')
        ax.set_title(network)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    h, labels = axes[0].get_legend_handles_labels()
    fig.legend(h, labels, loc='upper center', ncol=2, frameon=False)
    fig.tight_layout(rect=[0, .03, 1, .89])
    fig.savefig(output / 'budget_FPS.png', dpi=180)
    fig.savefig(output / 'budget_FPS.svg')
    plt.close(fig)
    lines = ['新旧完整DP名单共同实测：2026年11月12日', '',
        '新方法只修正隔离计算区间的线程份额；不是新时序公式，也未宣称解决隔离与流水阶段误差。',
        '两网络各两份严格Top-20，共48项去重配置；全部性能来自本次同一启动，不复用历史FPS。',
        '每项先做1帧全输出资格检查，然后8预热+16完成时间戳+4尾部，一轮；16时间戳对应15间隔。',
        '不重启、不重编模块或运行库、不重载FPGA；未加载详细诊断探针。', '',
        '网络 / 方法 / 预算 / FPS / II毫秒 / 选中静态名次']
    for row in budgets:
        lines.append(f"{row['network']} / {LABELS[row['method']]} / {row['budget']} / {row['FPS']:.5f} / {row['II_ms']:.4f} / {row['static_rank']}")
    lines += ['', 'ResNet新旧有序Top-20完全相同，性能曲线共用相同测量，不能把本批与历史的环境差别归给输入修正。',
        'YOLO新旧第一名也相同，预算一次必然共用结果；预算五次和二十次差异才检验名单/顺序变化。',
        '单输入、单启动、短窗口结果不构成显著性、稳定收益或稳态上限证明。',
        '本批不含Tarnawski对照，不能与其他批次拼接成共同测量的论文基线比较。',
        f'完整DNN推理1392次，原生会话96个；已验收会话测量与回收累计{session_seconds:.2f}秒。',
        '此前的13FPS配置仍按原身份保留，本批同配置结果单独报告。']
    (output / 'FINDINGS_zh.txt').write_text('\n'.join(lines) + '\n')
    write(output / 'MANIFEST.json', dict(files={str(p): sha256(p) for p in output.iterdir()
        if p.is_file() and p.name != 'MANIFEST.json'}))
    print(json.dumps(dict(budgets=budgets, report=str(output / 'FINDINGS_zh.txt')), ensure_ascii=False))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'validate', 'qemu', 'board', 'resume', 'status', 'analyze'))
    parser.add_argument('--execute-board', action='store_true')
    parser.add_argument('--run-dir', type=Path, default=RUN)
    parser.add_argument('--config', type=Path, default=CONFIG)
    args = parser.parse_args()
    if args.command == 'prepare':
        value = prepare()
    elif args.command == 'validate':
        value = {n: validate(OUT / n / 'capsule')['bundle_id'] for n in assert_sources()['bundles']}
    elif args.command == 'qemu':
        value = qemu()
    elif args.command in ('board', 'resume'):
        value = board(args.run_dir, args.config, execute_board=args.execute_board)
    elif args.command == 'analyze':
        value = analyze(args.run_dir)
    else:
        value = read(args.run_dir / 'state.json') if (args.run_dir / 'state.json').exists() else dict(status='NOT_STARTED')
    if args.command != 'analyze':
        print(json.dumps(value, ensure_ascii=False, indent=2))
