"""Locate a stage timing error using already accepted, counter-free events.

The sealed model's callback duration is applied to the *start* of the same
native callback. We ask how much CPU that native thread had actually received
by that relative deadline. This is an observational audit, not a predictor:
native service intervals must never become inputs to ranking or local costs.
"""
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
import csv
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from cpu_vta_partitioning.common import read, write, sha256, verify_inventory
from phase_model.simulator import rational
from scheduler_gap_audit.curves import csv_write
from yolo_scheduler_trace.events import clip, Attribution, intersection
from yolo_scheduler_trace_repair.events import validate_profile

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'artifacts/critical_worker_supply_01'
REPORT = ROOT.parents[4] / 'paper/my_paper/analysis/20261111_critical_worker_supply'
KERNELS = (2, 6, 10)


def cpu_crossing(spans, start, end, demand):
    """First observed timestamp delivering demand, without extending a trace.

    Fractional demand is retained. Missing supply returns None; it does not
    imply that a thread can continue at its final measured rate.
    """
    demand = Fraction(demand)
    if demand < 0 or end < start:
        raise ValueError('negative work or callback extent')
    selected = sorted((max(start, r['start_ns']), min(end, r['end_ns']))
                      for r in spans if intersection(start, end, r['start_ns'], r['end_ns']))
    if any(a[1] > b[0] for a, b in zip(selected, selected[1:])):
        raise ValueError('overlapping thread CPU service')
    if demand == 0:
        return Fraction(start)
    used = Fraction(0)
    for begin, finish in selected:
        if used + finish - begin >= demand:
            return Fraction(begin) + demand - used
        used += finish - begin
    return None


def deadline_supply(stream, start, end, model_duration, demand):
    if model_duration <= 0 or end <= start:
        raise ValueError('empty callback or model duration')
    stop = min(end, start + model_duration)
    at_deadline = clip(stream, start, stop)
    whole = clip(stream, start, end)
    if not whole['fully_covered'] or not at_deadline['fully_covered']:
        raise ValueError('callback scheduler coverage incomplete')
    crossing = cpu_crossing(whole['on'], start, end, demand)
    return dict(relative_model_deadline_ns=model_duration,
        deadline_clipped_to_callback_end=stop == end and start + model_duration > end,
        native_observed_prefix_wall_ns=stop - start,
        native_CPU_by_relative_deadline_ns=at_deadline['on_CPU_ns'],
        native_off_CPU_by_relative_deadline_ns=at_deadline['off_CPU_ns'],
        isolated_work_deficit_at_relative_deadline_ns=max(Fraction(0), demand - at_deadline['on_CPU_ns']),
        isolated_work_crossing_from_native_callback_start_ns=None if crossing is None else crossing - start,
        remaining_native_wall_after_work_crossing_ns=None if crossing is None else end - crossing,
        measured_supply_below_isolated_demand_at_callback_end=crossing is None,
        native_callback_on_CPU_ns=whole['on_CPU_ns'],
        native_callback_off_CPU_ns=whole['off_CPU_ns'])


def observed_competitors(stream, attribution, core, start, end):
    span = clip(stream, start, end)
    if not span['fully_covered']:
        raise ValueError('competitor observation is incomplete')
    roles = defaultdict(int)
    unknown = overlap = 0
    for interval in span['off']:
        row = attribution.other_work(interval, core)
        unknown += row['unobserved_ns']
        overlap += row['overlap_uncertainty_ns']
        for role, amount in row['by_role_ns'].items():
            roles[role] += amount
    if sum(roles.values()) + unknown - overlap != span['off_CPU_ns']:
        raise ValueError('observed competitor accounting does not conserve time')
    return dict(roles_ns=dict(roles), untracked_or_boundary_ns=unknown, overlap_uncertainty_ns=overlap)


def serializable(value):
    if isinstance(value, Fraction):
        return dict(numerator=value.numerator, denominator=value.denominator)
    if isinstance(value, dict):
        return {k: serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [serializable(v) for v in value]
    return value


def model_services(path):
    by_core = defaultdict(list)
    with path.open(encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            by_core[int(row['core'])].append(dict(start_ns=int(row['start_ns']),
                end_ns=int(row['end_ns']), identity=json.loads(row['identity'])))
    return by_core


def model_competitors(task, services, cfg):
    roles = defaultdict(int)
    own = 0
    core = task['allowed_mask'][0]
    if task['allowed_mask'] != [core]:
        raise ValueError('audit requires fixed worker placement')
    for row in services[core]:
        amount = intersection(task['first_service_ns'], task['end_ns'], row['start_ns'], row['end_ns'])
        if not amount:
            continue
        role, stage, frame, slot, op, kernel = row['identity']
        if role == 'stage' and (stage, slot) == (task['stage'], task['slot']):
            own += amount
        else:
            name = role if role != 'stage' else (cfg['stages'][stage]['program']
                + ('/host' if cfg['stages'][stage]['device'] == 'vta'
                   else '/caller' if slot == 0 else '/worker' + str(slot)))
            roles[name] += amount
    span = task['end_ns'] - task['first_service_ns']
    if own + sum(roles.values()) > span:
        raise ValueError('sealed model service ledger exceeds one core')
    return dict(on_CPU_ns=own, other_roles_ns=dict(roles), idle_or_unrecorded_ns=span - own - sum(roles.values()))


def run():
    if OUT.exists():
        raise FileExistsError('preserve the worker-supply evidence')
    accepted_path = ROOT / 'artifacts/yolo_switch_without_queries_01/analysis/RESULTS.json'
    parent_manifest = read(accepted_path.parent / 'MANIFEST.json')
    for path, expected in parent_manifest['files'].items():
        if sha256(path) != expected:
            raise ValueError('accepted counter-free analysis changed')
    accepted = read(accepted_path)
    frames = accepted['complete_front_frames']
    if frames != list(range(12, 24)):
        raise ValueError('do not select frames from the size of the error')
    sources = {str(accepted_path): sha256(accepted_path)}
    receipts = []
    for path, expected in accepted['sources'].items():
        if sha256(path) != expected:
            raise ValueError('sealed source changed')
        sources[path] = expected
        if '/accepted/' in path:
            item = read(path)
            verify_inventory(item['raw'], item['files'])
            if item['task']['spec']['phase'] == 'observed':
                receipts.append(item)
    if len(receipts) != 1:
        raise ValueError('counter-free observed task is ambiguous')
    receipt = receipts[0]
    raw = Path(receipt['raw'])
    if read(raw / 'queue_status.json') != accepted['queue_status'] or accepted['queue_status']['reads']:
        raise ValueError('unexpected CPU delay counter queries')
    inputs_path = ROOT / 'artifacts/current_model_gap_01/INPUTS.json'
    model_path = ROOT / 'artifacts/current_model_gap_01/independent.json'
    sealed = read(inputs_path)
    if sealed['native_trace_or_performance_included']:
        raise ValueError('sealed predictor input contains pipeline observations')
    model = read(model_path)['result']
    by_task = {(t['frame'], t['kernel'], t['slot']): t for t in model['tasks']
               if t['stage'] == 0 and t['operation'] == 'worker_compute'}
    ledger_path = ROOT / 'artifacts/critical_core_schedule_01/model_services.csv'
    expected = read(ledger_path.parent / 'MANIFEST.json')['files'][str(ledger_path)]
    if sha256(ledger_path) != expected:
        raise ValueError('sealed model service ledger changed')
    replay_check = read(ledger_path.parent / 'REPLAY_CHECK.json')
    if (not replay_check['exact_sealed_result_reproduced']
            or replay_check['original_sha256'] != sha256(model_path)):
        raise ValueError('service ledger refers to a different frozen replay')
    sources[str(ledger_path)] = expected
    services = model_services(ledger_path)
    affinity = read(raw / 'affinity.json')
    streams = validate_profile(read(raw / 'profile.json'), receipt['task'], affinity)['streams']
    attribution = Attribution(streams)
    caller = affinity['stages'][0]['caller_tid']
    tids = {w['slot']: w['tid'] for w in affinity['stages'][0]['worker_slots']}
    if sorted(tids) != list(range(4)) or tids[0] != caller:
        raise ValueError('caller and worker identities changed')
    lines = lambda p: [json.loads(s) for s in p.read_text().splitlines()]
    phases = lines(raw / 'phase_trace.jsonl')
    trace = lines(raw / 'wait_trace.jsonl')
    callbacks = defaultdict(list)
    launches = []
    for row in trace:
        if row['kind'] == 0 and row['caller'] == caller:
            launches.append(row)
        elif row['kind'] == 1:
            callbacks[row['launch']].append(row)
    rows = []
    for frame in frames:
        phase = [p for p in phases if p['stage'] == 'cpu_00_02' and p['operation'] == 'run' and p['call'] == frame]
        if len(phase) != 1:
            raise ValueError('front run interval missing or duplicated')
        phase = phase[0]
        group = sorted((r for r in launches if phase['wall_start_ns'] <= r['wall_start_ns']
            <= r['wall_end_ns'] <= phase['wall_end_ns']), key=lambda r: r['wall_start_ns'])
        if len(group) != 12:
            raise ValueError('parallel sequence is incomplete')
        for kernel in KERNELS:
            launch = group[kernel]
            profile = sealed['case']['stages'][0]['profile']['kernels'][kernel]
            if launch['function_offset'] != profile['function_offset']:
                raise ValueError('parallel callback module offset differs')
            workers = callbacks[launch['launch']]
            if sorted(w['task'] for w in workers) != list(range(4)):
                raise ValueError('duplicate or missing callback')
            last = max(workers, key=lambda w: (w['wall_end_ns'], -w['task']))['task']
            for worker in workers:
                slot = worker['task']
                if worker['tid'] != tids[slot]:
                    raise ValueError('logical worker TID differs')
                task = by_task[(frame, kernel, slot)]
                if profile['workers'][slot]['logical_worker'] != slot:
                    raise ValueError('isolated logical slot order differs')
                demand = sum((rational(n) for n in profile['workers'][slot]['compute_segments']), Fraction(0))
                if demand != rational(task['CPU']):
                    raise ValueError('model uses a different isolated compute demand')
                start, end = worker['wall_start_ns'], worker['wall_end_ns']
                duration = task['end_ns'] - task['first_service_ns']
                stream = streams[str(worker['tid'])]
                value = deadline_supply(stream, start, end, duration, demand)
                whole = clip(stream, start, end)
                cpu = worker['cpu_end_ns'] - worker['cpu_start_ns']
                value.update(frame=frame, kernel=kernel, slot=slot, actual_last=slot == last,
                    core=task['allowed_mask'][0], isolated_compute_CPU_ns=demand,
                    native_callback_CPU_clock_ns=cpu, callback_CPU_clock_minus_isolated_ns=cpu - demand,
                    native_clock_minus_scheduler_on_ns=cpu - whole['on_CPU_ns'],
                    native_callback_span_ns=end - start,
                    native_barrier_calls=worker['barrier_calls'], native_yield_calls=worker['yield_calls'],
                    native_preempt_off_ns=whole['preempt_off_ns'], native_nonpreempt_off_ns=whole['nonpreempt_off_ns'],
                    native_competitors=observed_competitors(stream, attribution, task['allowed_mask'][0], start, end),
                    model_competitors=model_competitors(task, services, sealed['case']['configuration']))
                rows.append(value)
    summaries = []
    fields = ('isolated_compute_CPU_ns', 'native_callback_CPU_clock_ns', 'native_callback_on_CPU_ns',
              'native_callback_off_CPU_ns', 'native_callback_span_ns', 'relative_model_deadline_ns',
              'isolated_work_deficit_at_relative_deadline_ns', 'native_clock_minus_scheduler_on_ns',
              'callback_CPU_clock_minus_isolated_ns')
    for kernel in KERNELS:
        selected = [r for r in rows if r['kernel'] == kernel and r['actual_last']]
        all_workers = [r for r in rows if r['kernel'] == kernel]
        mean = lambda collection, key: float(sum((Fraction(r[key]) for r in collection), Fraction(0)) / len(collection) / 1_000_000)
        available = [r for r in selected if r['isolated_work_crossing_from_native_callback_start_ns'] is not None]
        roles = sorted({role for r in selected for role in r['native_competitors']['roles_ns']})
        model_roles = sorted({role for r in selected for role in r['model_competitors']['other_roles_ns']})
        summaries.append(dict(kernel=kernel, frames=len(selected), last_slots=dict(Counter(r['slot'] for r in selected)),
            actual_last_worker_mean_ms={key.replace('_ns', '_ms'): mean(selected, key) for key in fields},
            all_four_workers_mean_CPU_ms={key.replace('_ns', '_ms'): mean(all_workers, key)
                for key in ('isolated_compute_CPU_ns', 'native_callback_CPU_clock_ns', 'native_callback_on_CPU_ns')},
            native_deficit_at_relative_deadline_count=sum(r['isolated_work_deficit_at_relative_deadline_ns'] > 0 for r in selected),
            isolated_work_crossing_found_count=len(available),
            crossing_sensitivity_mean_ms=None if not available else mean(available, 'isolated_work_crossing_from_native_callback_start_ns'),
            leftover_after_crossing_mean_ms=None if not available else mean(available, 'remaining_native_wall_after_work_crossing_ns'),
            native_competing_roles_mean_ms={role: sum(r['native_competitors']['roles_ns'].get(role, 0) for r in selected) / len(selected) / 1e6 for role in roles},
            model_competing_roles_mean_ms={role: sum(r['model_competitors']['other_roles_ns'].get(role, 0) for r in selected) / len(selected) / 1e6 for role in model_roles},
            native_untracked_or_boundary_ms=sum(r['native_competitors']['untracked_or_boundary_ns'] for r in selected) / len(selected) / 1e6,
            native_overlap_uncertainty_ms=sum(r['native_competitors']['overlap_uncertainty_ns'] for r in selected) / len(selected) / 1e6,
            yield_calls=sum(r['native_yield_calls'] for r in selected),
            barrier_calls=sum(r['native_barrier_calls'] for r in selected)))
    result = dict(sources=sources, source_sha256=sha256(__file__), boot_id=receipt['boot_id'],
        complete_frames=frames, audited_callbacks=len(rows), critical_kernels=list(KERNELS), summaries=summaries,
        unchanged_clean=accepted['clean'], observer=accepted['observer'],
        probe_effect_front_service_ms=accepted['front_service_probe_difference_ms'],
        no_new_native_sessions=True, new_DNN=0, no_model_DP_or_Top20_changed=True,
        new_predictor=False, observed_service_is_diagnostic_only=True,
        relative_callback_anchors_not_cross_session_absolute_clock_alignment=True,
        thread_CPU_clock_and_scheduler_service_have_different_envelopes=True,
        unknown_clock_error_not_assigned_to_memory_or_instructions=True,
        no_extrapolation_beyond_native_callback=True,
        concurrent_CPU_occupancy_not_causal_recoverable_delay=True,
        all_workers_reported_to_expose_last_worker_selection=True,
        old_isolated_profile_cross_boot_limit=True)
    OUT.mkdir(parents=True)
    write(OUT / 'RESULTS.json', result)
    write(OUT / 'callbacks.json', serializable(rows))
    csv_write(OUT / 'summary.csv', summaries)
    REPORT.mkdir(parents=True, exist_ok=True)
    text = ['隔离profile与流水差距：关键线程在预测完成时刻是否拿够CPU', '',
        '只读已验收的无TASKSTATS逐次查询轨迹及冻结隔离输入；不连接开发板、不重测、不修改模型或名单。',
        '对象：YOLO固定五阶段切图、CPU前段四线程重叠布局；不是原ResNet全域13FPS配置，也不是新全域评测。',
        '每次选真实最后完成的worker，在模型中查同一逻辑worker；并另列全部四个worker，避免把选出的拖尾者当平均线程。',
        '模型回调跨度移到真实回调起点作为相对截止时刻；两次运行的绝对钟不拼接。',
        '到该时刻获得的CPU服务由真实switch区间积分；与隔离计算核时比较，不用流水FPS拟合。', '',
        '并行调用序号 / 隔离所需CPUms / 实测回调CPU时钟ms / 实测运行ms / 实测未运行ms / 模型回调跨度ms / 实测回调跨度ms / 截止时平均CPU缺口ms']
    for r in summaries:
        m = r['actual_last_worker_mean_ms']
        text.append(str(r['kernel'] + 1) + ' / ' + ' / '.join(f'{m[k]:.3f}' for k in (
            'isolated_compute_CPU_ms', 'native_callback_CPU_clock_ms', 'native_callback_on_CPU_ms',
            'native_callback_off_CPU_ms', 'relative_model_deadline_ms', 'native_callback_span_ms',
            'isolated_work_deficit_at_relative_deadline_ms')))
        text.append('12帧中到模型相对截止时刻仍未拿够隔离核时：' + str(r['native_deficit_at_relative_deadline_count']) + '帧。')
        text.append('真实同核其他工作：' + str(r['native_competing_roles_mean_ms']) + '核ms/次；模型：' + str(r['model_competing_roles_mean_ms']))
        text.append('全部四个worker平均CPU（隔离/实测时钟/调度on）：' + str(r['all_four_workers_mean_CPU_ms']))
    text += ['',
        '解释边界：',
        '1. 未运行不等于全部精确排队；选出的最后回调的Yield/屏障及preempt标记单列，没有TASKSTATS增量。',
        '2. 线程CPU时钟、调度on区间与隔离探针的边界不同；CPU增长不能直接全归给缓存/访存，CPU基本不变也不能证明指令数不变。',
        '3. 在真实服务曲线上达到隔离核时的时刻只是诊断性敏感性值；不能把实测服务曲线拿去预测未测候选。缺少足够供给时返回未知，不外推。',
        '4. 模型与真实竞争工作的重合差异暴露了工作时序/执行位置近似；它们不是可以再次相加的惩罚或可直接消除的延迟。',
        f"5. 观测使前段服务比本批无探针多{result['probe_effect_front_service_ms']:.3f}ms；定量数据保留此限制。", '',
        '下一步针对关键worker之前的竞争任务到达、可运行及实际服务次序，而不是继续审计版本差异、刷新总核时或重复添加等待。',
        '没有得到新的通用修正式，也没有证明吞吐改善。此前换主机位置、阻塞等待和绑定对照不自动重复。']
    report = REPORT / 'FINDINGS_zh.txt'
    report.write_text('\n'.join(text) + '\n')
    files = [OUT / 'RESULTS.json', OUT / 'callbacks.json', OUT / 'summary.csv', report]
    write(OUT / 'MANIFEST.json', dict(files={str(p): sha256(p) for p in files}))
    print(json.dumps(dict(summaries=summaries, report=str(report)), ensure_ascii=False))
    return result


if __name__ == '__main__':
    run()
