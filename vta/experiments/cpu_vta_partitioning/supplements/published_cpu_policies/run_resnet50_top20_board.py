#!/usr/bin/env python3
"""Measure the already-frozen ResNet50 Top-20 union in a separate boot cohort.

No compilation, calibration, ranking, or native execution changes are made.
Original cost/boot sources remain read-only; both methods share the new boot.
"""
import argparse
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
sys.path.insert(0, str(PROJECT / 'src'))

from cpu_vta_partitioning.common import inventory, locked, now, read, sha256, write
from cpu_vta_partitioning.prepare.build import copy_checked
from resnet50_validation.common import assert_controllers, frozen, timed_deploy
from resnet50_validation import campaign, runtime
from resnet50_validation.validation import validate_bundle

SOURCE = ROOT / 'artifacts/resnet50_v2_validation_native_history_02'
BATCH = 'resnet50_v2_top20_board_20261117_01'


def prepare(output, source):
    output, source = Path(output).resolve(), Path(source).resolve()
    if output == source:
        raise ValueError('a new execution cohort must not overwrite its profile/ranking source')
    science = assert_controllers(source)
    bundle = source / 'pipeline_bundle'
    manifest = validate_bundle(bundle)
    ranking = source / 'ranking'
    seal = read(ranking / 'FREEZE.json')
    if seal['candidate_count'] != 40 or seal['pipeline_FPS_used'] is not False:
        raise ValueError('the existing frozen forty-candidate union is required')
    for name, expected in seal['files'].items():
        if sha256(ranking / name) != expected:
            raise ValueError('frozen ranking changed')
        target = output / 'ranking' / name
        if target.exists() and sha256(target) != expected:
            raise ValueError('new cohort selection changed')
        if not target.exists():
            copy_checked(ranking / name, target, expected)
    for name in ('FREEZE.json', 'SEARCH_STATUS.json'):
        target = output / 'ranking' / name
        if not target.exists():
            copy_checked(ranking / name, target)
        elif sha256(target) != sha256(ranking / name):
            raise ValueError('source search or ranking identity changed')
    records = read(bundle / 'provenance/static_inputs/isolated_costs.json')['costs']
    provenance = dict(source_output=str(source), pipeline_bundle=str(bundle),
        pipeline_bundle_id=manifest['bundle_id'], pipeline_manifest_sha256=sha256(bundle / 'manifest.json'),
        ranking_freeze_sha256=sha256(ranking / 'FREEZE.json'), candidate_count=40,
        profile_source_boot_ids=sorted({r['source']['boot_id'] for r in records.values()}),
        profile_signatures=len(records), DMA_signatures=299,
        new_calibration_sessions=0, compilation=False, ranking_repeated=False,
        native_tasks_and_windows_unchanged=True,
        cross_boot_cost_reference=True, same_boot_performance_for_both_methods=True)
    frozen(output / 'SOURCE.json', provenance)
    frozen(output / 'CONTROLLER.json', dict(script=str(Path(__file__).resolve()),
        script_sha256=sha256(Path(__file__)), scientific_sources=science['sources'],
        source_controller_sha256=sha256(source / 'CONTROLLERS.json')))
    return provenance


def ensure_storage(ssh, boot):
    """Restore the existing SD bind mount after an external board restart."""
    source = '/media/sd-mmcblk1p3'
    target = ssh.config['storage_mount']
    q = shlex.quote
    script = '\n'.join(['set -eu',
        'test "$(cat /proc/sys/kernel/random/boot_id)" = ' + q(boot),
        "awk -v p=" + q(source) + " '$2 == p && $4 ~ /(^|,)rw(,|$)/ {ok=1} END {exit !ok}' /proc/mounts",
        'if ! awk -v p=' + q(target) + " '$2 == p {ok=1} END {exit !ok}' /proc/mounts; then",
        '  test ! -L ' + q(target),
        '  mkdir -p ' + q(target),
        '  mount --bind ' + q(source) + ' ' + q(target), 'fi',
        'test "$(stat -c %d ' + q(source) + ')" = "$(stat -c %d ' + q(target) + ')"',
        'df -Pk ' + q(target)])
    return ssh.command(script, timeout=30)


def analyze(output, run_dir):
    from resnet50_validation import analysis as original
    output, run_dir = Path(output), Path(run_dir)
    provenance = read(output / 'SOURCE.json')
    bundle = Path(provenance['pipeline_bundle'])
    validate_bundle(bundle)
    state = read(run_dir / 'pipeline/state.json')
    if state['status'] != 'COMPLETE':
        raise ValueError('all forty candidates must finish before final comparison')
    boot = state['boot_id']
    if read(run_dir / 'run.json')['boot_id'] != boot:
        raise ValueError('performance boot differs from the new cohort')
    lists = read(bundle / 'plans/rankings.json')['lists']
    pool = read(bundle / 'plans/pool.json')
    results, receipts, evidence = {}, [], {}
    for name, cfg in pool.items():
        qualification = None
        for phase in ('qualify', 'short'):
            doc = runtime.task(bundle, cfg, phase)
            path = run_dir / 'pipeline/accepted' / boot / (doc['task_id'] + '.json')
            item = original.receipt(path, doc, boot, bundle=bundle)
            receipts.append(item)
            evidence[str(path)] = sha256(path)
            if phase == 'qualify':
                qualification = item['summary']['output_fingerprints']
            elif item['summary']['output_fingerprints'] != qualification:
                raise ValueError('qualified and timed network outputs differ')
            else:
                results[name] = item['summary']
    projection = bundle / 'provenance/static_inputs'
    records = read(projection / 'isolated_costs.json')['costs']
    if sorted({r['source']['boot_id'] for r in records.values()}) != provenance['profile_source_boot_ids']:
        raise ValueError('original profile boot provenance changed')
    programs, ddr = read(projection / 'programs.json'), read(projection / 'ddr.json')
    rows = original.budgets(lists, results)
    target = output / 'analysis'
    target.mkdir(parents=True, exist_ok=True)
    original._csv(target / 'budgets.csv', rows)
    scores, constraints = original.score_rows(lists, pool, results, records, programs, ddr,
                                               read(bundle / 'plans/scores.json'))
    original._csv(target / 'configuration_scores.csv', scores)
    original._csv(target / 'constraint_scores.csv', constraints)
    original._csv(target / 'stage_isolated_pipeline.csv', original.stage_rows(pool, results, records, programs, ddr))
    curves = original.thread_curve_rows(records)
    original._csv(target / 'CPU_thread_curves.csv', curves)
    gains = {str(b): 100 * (next(r['FPS'] for r in rows if r['method'] == 'complete' and r['budget'] == b)
        / next(r['FPS'] for r in rows if r['method'] == 'tarnawski' and r['budget'] == b) - 1) for b in (1, 5, 20)}
    report = dict(status='COMPLETE', execution_cohort=BATCH, performance_boot_id=boot,
        profile_source_boot_ids=provenance['profile_source_boot_ids'],
        cross_boot_profile_reference=True, same_boot_performance_for_both_methods=True,
        budgets=rows, FPS_gain_percent=gains, unique_configurations=len(pool),
        native_pipeline_sessions=len(receipts), complete_board_DNN_inferences=49 * len(pool),
        new_calibration_sessions=0, source=provenance, receipt_sha256=evidence,
        measurement_cost=original.measurement_cost(receipts),
        one_input_one_boot_short_window=True, significance_claimed=False,
        profiles_and_performance_are_not_the_same_boot=True,
        phase_errors_may_include_cross_boot_environment_difference=True,
        global_measured_optimum_claimed=False)
    write(target / 'RESULTS.json', report, replace=True)
    text = ['ResNet50-v2：冻结Top-20的新启动共同实测', '',
            '两种方法在同一新启动中测量；隔离成本引用旧启动，来源身份保留。',
            '输入0；资格1帧；正式8预热＋16时间戳＋24尾帧；15个完成间隔。', '']
    for row in rows:
        text.append(f"{row['method']}／预算{row['budget']}：{row['FPS']:.6f} FPS，"
                    f"{row['II_ms']:.6f} ms，静态名次{row['static_rank']}。")
    text += ['', '完整模型相对Tarnawski适配的FPS变化：' + str(gains),
             '本次是单输入、单启动观察；隔离阶段与流水耗时比较同时受到跨启动条件影响。',
             '性能差距包含切图、线程／核心配置和评分共同作用，不单独归因某项约束。',
             '不宣称统计显著性、整个ResNet家族有效或全域实测最优。']
    (target / 'FINDINGS_zh.txt').write_text('\n'.join(text) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    for index, method in enumerate(('tarnawski', 'complete')):
        ax.bar([j + (index - .5) * .36 for j in range(3)],
               [r['FPS'] for r in rows if r['method'] == method], width=.36,
               label='Tarnawski adaptation' if method == 'tarnawski' else 'Frozen complete model')
    ax.set_xticks(range(3), ['Top-1', 'Best of top-5', 'Best of top-20'])
    ax.set_ylabel('Throughput (FPS)')
    ax.legend()
    fig.tight_layout()
    for extension in ('pdf', 'svg', 'png'):
        fig.savefig(target / ('budget_comparison.' + extension), dpi=250)
    plt.close(fig)
    return dict(status='COMPLETE', FPS_gain_percent=gains, results=str(target / 'RESULTS.json'))


def run(output, run_dir, source, local_config, *, execute_board=False):
    if not execute_board:
        raise ValueError('--execute-board is required')
    output, run_dir, source = map(Path, (output, run_dir, source))
    provenance = prepare(output, source)
    bundle = source / 'pipeline_bundle'
    ssh = runtime.transport(read(local_config)['board'], execute_board=True)
    with locked(run_dir):
        boot = ssh.boot_id()
        frozen(run_dir / 'run.json', dict(boot_id=boot, execution_cohort=BATCH,
            source_sha256=sha256(output / 'SOURCE.json'), controller_sha256=sha256(output / 'CONTROLLER.json'),
            profile_source_boot_ids=provenance['profile_source_boot_ids']))
        def state(stage, **extra):
            write(run_dir / 'state.json', dict(status=stage, boot_id=boot, updated_at=now(), **extra), replace=True)
        try:
            state('STORAGE')
            write(run_dir / 'storage.json', dict(boot_id=boot, raw=ensure_storage(ssh, boot)), replace=True)
            state('DEPLOYING')
            remote = timed_deploy(ssh, bundle, run_dir / 'pipeline')
            state('ENVIRONMENT')
            ssh.qualify_environment(remote, boot, run_dir / 'pipeline', bundle)
            state('PIPELINE')
            measured = campaign.pipeline(source, run_dir / 'pipeline', local_config, boot)
            state('ANALYZING', pipeline=measured)
            report = analyze(output, run_dir)
            state('COMPLETE', pipeline=measured, report=report)
            return report
        except Exception as error:
            state('STOPPED_ON_ERROR', error_type=type(error).__name__, error=str(error))
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run', 'resume', 'status', 'analyze'))
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts' / BATCH)
    parser.add_argument('--run-dir', type=Path, default=ROOT / 'runs' / BATCH)
    parser.add_argument('--source-output', type=Path, default=SOURCE)
    parser.add_argument('--local-config', type=Path, default=PROJECT / 'configs/local.json')
    parser.add_argument('--execute-board', action='store_true')
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        value = prepare(args.output, args.source_output)
    elif args.command == 'status':
        value = read(args.run_dir / 'state.json') if (args.run_dir / 'state.json').exists() else dict(status='NOT_STARTED')
    elif args.command == 'analyze':
        value = analyze(args.output, args.run_dir)
    else:
        value = run(args.output, args.run_dir, args.source_output, args.local_config,
                    execute_board=args.execute_board)
    print(__import__('json').dumps(value, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
