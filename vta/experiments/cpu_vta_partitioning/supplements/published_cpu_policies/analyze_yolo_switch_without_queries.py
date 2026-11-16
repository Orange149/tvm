"""Analyze the bounded counter-free observation without fitting a predictor."""
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
import json

import run_yolo_switch_without_queries as campaign
from cpu_vta_partitioning.common import read, write, sha256, verify_inventory
from phase_model.extract import project_call, average_calls
from analyze_caller_wait_budget import projected_metrics
from yolo_scheduler_trace.events import clip, Attribution
from yolo_scheduler_trace_repair.events import validate_profile
from scheduler_gap_audit.curves import csv_write

ROOT = campaign.ROOT
OUT = campaign.OUT / 'analysis'
REPORT = ROOT.parents[4] / 'paper/my_paper/analysis/20261111_yolo_switch_without_queries'


def model_budget(result, frames):
    totals = defaultdict(Fraction)
    walls = []
    for frame in frames:
        rows = [t for t in result['tasks'] if t['stage'] == 0 and t['frame'] == frame]
        a = min(t['ready_ns'] for t in rows if t['operation'] in ('serial', 'launch_prefix'))
        b = next(t['ready_ns'] for t in rows if t['operation'] == 'get')
        walls.append(b - a)
        for t in rows:
            if t['slot'] == 0 and t['operation'] not in ('set', 'get'):
                v = t['CPU']
                totals[t['operation']] += Fraction(v['numerator'], v['denominator'])
    scale = len(frames) * 1_000_000
    compute = float(totals['worker_compute'] / scale)
    other = float((totals['serial'] + totals['launch_prefix']) / scale)
    wait = float(totals['completion_wait'] / scale)
    wall = sum(walls) / scale
    return dict(caller_compute_CPU_ms=compute, caller_other_CPU_ms=other,
        completion_and_return_CPU_ms=wait, caller_run_CPU_ms=compute + other + wait,
        caller_run_wall_ms=wall, caller_off_CPU_ms=wall - compute - other - wait,
        model_join_continuation_and_native_return_markers_differ=True)


def run():
    if OUT.exists():
        raise FileExistsError('preserve the accepted counter-free analysis')
    campaign.validate()
    state = read(campaign.RUN / 'state.json')
    if state['status'] != 'COMPLETE' or state['accepted_sessions'] != 3 or state['DNN_inferences'] != 57:
        raise ValueError('bounded counter-free collection incomplete')
    identity = read(campaign.RUN / 'run.json')
    if identity['boot_id'] != state['boot_id'] or identity['plan_sha256'] != sha256(campaign.OUT / 'PLAN.json'):
        raise ValueError('campaign boot or frozen identity differs')
    receipts, sources = {}, {}
    for spec in campaign.specs():
        task = campaign.make_task(campaign.BUNDLE, spec)
        path = campaign.RUN / 'accepted' / state['boot_id'] / (task['task_id'] + '.json')
        receipt = read(path)
        if receipt['boot_id'] != state['boot_id'] or receipt['task'] != task:
            raise ValueError('accepted task or boot differs')
        verify_inventory(receipt['raw'], receipt['files'])
        if campaign.summary(receipt['raw'], task, campaign.BUNDLE) != receipt['summary']:
            raise ValueError('counter-free evidence no longer replays')
        receipts[spec['phase']] = receipt
        sources[str(path)] = sha256(path)
    if any(r['summary']['output_fingerprints'] != receipts['qualify']['summary']['output_fingerprints']
           for r in receipts.values()):
        raise ValueError('pipeline outputs differ')
    record = receipts['observed']
    raw = Path(record['raw'])
    affinity = read(raw / 'affinity.json')
    streams = validate_profile(read(raw / 'profile.json'), record['task'], affinity)['streams']
    attribution = Attribution(streams)
    lines = lambda p: [json.loads(x) for x in p.read_text().splitlines()]
    phases, trace = lines(raw / 'phase_trace.jsonl'), lines(raw / 'wait_trace.jsonl')
    launches = [r for r in trace if r['kind'] == 0]
    workers, barriers = defaultdict(list), defaultdict(list)
    for row in trace:
        if row['kind'] in (1, 2):
            (workers if row['kind'] == 1 else barriers)[row['launch']].append(row)
    selected = [p for p in phases if p['stage'] == 'cpu_00_02' and p['operation'] == 'run'
                and record['task']['warmup'] <= p['call'] < record['task']['warmup'] + record['task']['scored']]
    complete = [p for p in selected if all(clip(s, p['wall_start_ns'], p['wall_end_ns'])['fully_covered']
                                         for s in streams.values())]
    complete.sort(key=lambda p: p['call'])
    if not complete:
        raise ValueError('no fully covered front-stage frame; do not extrapolate or automatically remeasure')
    frames = [p['call'] for p in complete]
    projection = average_calls([project_call(p, launches, workers, barriers) for p in complete])
    observed_budget = projected_metrics(projection)
    parent = ROOT / 'artifacts/current_model_gap_01'
    plan = read(parent / 'PLAN.json')
    inputs = read(parent / 'INPUTS.json')
    if sha256(parent / 'INPUTS.json') != plan['input_sha256']:
        raise ValueError('sealed isolated inputs changed')
    path = parent / 'independent.json'
    if sha256(path) != read(parent / 'REPLAY_FREEZE.json')['files']['independent.json']:
        raise ValueError('sealed prediction changed')
    model = read(path)['result']
    sources[str(path)] = sha256(path)
    sources[str(parent / 'INPUTS.json')] = sha256(parent / 'INPUTS.json')
    isolated = inputs['case']['stages'][0]['profile']
    if [(k['function_offset'], k['tasks']) for k in isolated['kernels']] != [
            (k['function_offset'], k['tasks']) for k in projection['kernels']]:
        raise ValueError('kernel sequence differs from frozen isolated work')
    model_metrics = model_budget(model, frames)
    budgets = [dict(mode='isolated_profile', **projected_metrics(isolated)),
               dict(mode='sealed_model_same_frames', **model_metrics),
               dict(mode='pipeline_counter_free_observer', **observed_budget)]
    predicted = {(t['frame'], t['kernel'], t['slot']): t for t in model['tasks']
                 if t['stage'] == 0 and t['operation'] == 'worker_compute'}
    kernel_rows = []
    for phase in complete:
        group = sorted([l for l in launches if l['caller'] == phase['tid']
            and phase['wall_start_ns'] <= l['wall_start_ns'] <= l['wall_end_ns'] <= phase['wall_end_ns']],
            key=lambda l: l['wall_start_ns'])
        if len(group) != 12:
            raise ValueError('parallel sequence is incomplete')
        for kernel in (2, 6, 10):
            last = max(workers[group[kernel]['launch']], key=lambda w: (w['wall_end_ns'], -w['task']))
            span = clip(streams[str(last['tid'])], last['wall_start_ns'], last['wall_end_ns'])
            core = record['task']['stages'][0]['worker_core_map'][last['task']]
            role_cpu = defaultdict(int)
            for off in span['off']:
                for role, amount in attribution.other_work(off, core)['by_role_ns'].items():
                    role_cpu[role] += amount
            t = predicted[(phase['call'], kernel, last['task'])]
            kernel_rows.append(dict(frame=phase['call'], kernel=kernel, slot=last['task'],
                callback_CPU_ns=last['cpu_end_ns'] - last['cpu_start_ns'],
                callback_span_ns=last['wall_end_ns'] - last['wall_start_ns'],
                on_CPU_ns=span['on_CPU_ns'], off_CPU_ns=span['off_CPU_ns'],
                preempt_off_ns=span['preempt_off_ns'], nonpreempt_off_ns=span['nonpreempt_off_ns'],
                yield_calls=last['yield_calls'], barrier_calls=last['barrier_calls'],
                model_same_worker_span_ns=t['end_ns'] - t['first_service_ns'],
                coincident_roles_ns=dict(role_cpu), CPU_queue_counter_unavailable_by_design=True))
    summaries = []
    for kernel in (2, 6, 10):
        rows = [r for r in kernel_rows if r['kernel'] == kernel]
        summaries.append(dict(kernel=kernel, frames=len(rows),
            last_slots=dict(Counter(r['slot'] for r in rows)),
            mean_ms={key.replace('_ns', '_ms'): sum(r[key] for r in rows) / len(rows) / 1e6 for key in (
                'callback_CPU_ns', 'callback_span_ns', 'on_CPU_ns', 'off_CPU_ns',
                'preempt_off_ns', 'nonpreempt_off_ns', 'model_same_worker_span_ns')},
            yield_calls=sum(r['yield_calls'] for r in rows), barrier_calls=sum(r['barrier_calls'] for r in rows),
            coincident_roles_ms={role: sum(r['coincident_roles_ns'].get(role, 0) for r in rows) / len(rows) / 1e6
                                for role in sorted({role for r in rows for role in r['coincident_roles_ns']})}))
    clean = receipts['clean']['summary']
    measured = record['summary']
    differences = {k: observed_budget[k] - model_metrics[k] for k in (
        'caller_compute_CPU_ms', 'caller_other_CPU_ms', 'completion_and_return_CPU_ms',
        'caller_off_CPU_ms', 'caller_run_wall_ms')}
    if abs(sum(v for k, v in differences.items() if k != 'caller_run_wall_ms')
           - differences['caller_run_wall_ms']) > 1e-9:
        raise ValueError('same-caller difference partition does not conserve time')
    result = dict(boot_id=state['boot_id'], sources=sources, source_sha256=sha256(__file__),
        clean=dict(FPS=clean['fps'], II_ms=clean['ii_ms'], stage_service_ms=clean['stage_service_ms']),
        observer=dict(FPS_not_formal=measured['fps'], II_ms=measured['ii_ms'], stage_service_ms=measured['stage_service_ms']),
        front_service_probe_difference_ms=measured['stage_service_ms'][0] - clean['stage_service_ms'][0],
        queue_status=measured['queue_status'], complete_front_frames=frames,
        caller_budgets=budgets, same_caller_difference_ms=differences, critical_kernels=summaries,
        earlier_query_on_trace_is_different_boot_not_causal_matched_comparison=True,
        observer_CPU_counts_do_not_prove_pure_arithmetic_work=True,
        native_sessions=3, DNN_inferences=57, new_component_calls=0,
        no_formula_DP_or_Top20_change=True, native_probe_not_formal_FPS=True,
        additional_completion_CPU_is_endogenous_not_an_intrinsic_cost=True,
        previous_blocking_wait_test_did_not_improve_throughput=True)
    OUT.mkdir(parents=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    write(OUT / 'RESULTS.json', result)
    csv_write(OUT / 'caller_budgets.csv', budgets)
    csv_write(OUT / 'critical_callbacks.csv', kernel_rows)
    text = ['关闭逐次计数查询后，共享执行慢化仍存在', '',
        '本批已验收3个会话、57次推理。没有重启、重载FPGA、重建运行库/阶段模块，或改变线程布局、等待策略和DP。',
        '探针与无探针使用同一冻结执行器。排队计数接口enabled=false、reads=0、errors=0；字段零值不是没有排队的测量证据。',
        f'无探针正式结果：{clean["fps"]:.4f}FPS，前段阶段服务{clean["stage_service_ms"][0]:.3f}ms。',
        f'观测会话前段服务{measured["stage_service_ms"][0]:.3f}ms，差{result["front_service_probe_difference_ms"]:.3f}ms；关闭查询没有保证整个观测程序低扰动。',
        f'全部目标TID完整覆盖的原计入帧：{frames}。以下只使用这些帧，不挑最大差距帧。', '',
        '并行调用 / 最后线程回调跨度ms / 实际运行ms / 未运行ms / 模型同线程跨度ms']
    for row in summaries:
        m = row['mean_ms']
        text.append(f'{row["kernel"] + 1} / {m["callback_span_ms"]:.3f} / {m["on_CPU_ms"]:.3f} / '
                    f'{m["off_CPU_ms"]:.3f} / {m["model_same_worker_span_ms"]:.3f}')
    text += ['',
        '以上关键回调未观测到Yield或屏障调用，离核区间均带抢占标记；关闭TASKSTATS查询后竞争仍存在。',
        '没有TASKSTATS读数时不报告精确CPU排队计数。调度事件的运行/离核时长与线程CPU时钟仍有不同包络。', '',
        '同一个调用线程run墙钟的互不重复分解：计算回调CPU＋串行/启动CPU＋完成等待/返回CPU＋离核时间。',
        '口径 / 计算CPUms / 其他CPUms / 完成等待与返回CPUms / 离核ms / run墙钟ms']
    for row in budgets:
        text.append(row['mode'] + ' / ' + ' / '.join(f'{row[k]:.3f}' for k in (
            'caller_compute_CPU_ms', 'caller_other_CPU_ms', 'completion_and_return_CPU_ms',
            'caller_off_CPU_ms', 'caller_run_wall_ms')))
    text += ['',
        f'本批同帧调用线程的模型差{differences["caller_run_wall_ms"]:.3f}ms，其中完成等待/返回CPU包络差{differences["completion_and_return_CPU_ms"]:.3f}ms。',
        '模型已经生成约41ms这类等待CPU工作；不是完全漏计，不能把实测约74ms再整体加上去。',
        '关键worker更晚结束会延长调用线程的等待CPU占用；这种需求依赖其他线程的执行次序，不能视为每个子图不变的局部成本。',
        '较早隔离profile与本批跨boot、观测条件不同，原有带查询轨迹也非本批匹配对照；不将其差异全部归因于查询开关。',
        '本批只确认：逐次查询不是共享慢化现象成立的必要条件。仍未证明能准确预测CFS下每个关键线程的完成时刻。',
        '已有阻塞等待实验虽然减少CPU占用，却未改善FPS；不能由上述包络差直接推算可恢复的吞吐。',
        '后续只处理关键线程服务/完成次序的可预测缺口，不重复份额审计、换核、阻塞或全域板测。']
    report = REPORT / 'FINDINGS_zh.txt'
    report.write_text('\n'.join(text) + '\n')
    paths = [OUT / 'RESULTS.json', OUT / 'caller_budgets.csv', OUT / 'critical_callbacks.csv', report]
    write(OUT / 'MANIFEST.json', dict(files={str(p): sha256(p) for p in paths}))
    print(json.dumps(dict(clean_FPS=clean['fps'], complete_frames=frames,
                         same_caller_difference_ms=differences, report=str(report)), ensure_ascii=False))
    return result


if __name__ == '__main__':
    run()
