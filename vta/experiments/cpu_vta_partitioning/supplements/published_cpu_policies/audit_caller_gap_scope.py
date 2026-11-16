"""Audit the remaining caller intervals using sealed, already accepted data.

This is a diagnostic partition, not a new predictor. No execution, ranking,
cost input, or scheduling decision is changed.
"""
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from cpu_vta_partitioning.common import read, sha256, verify_inventory, write
from scheduler_gap_audit.curves import csv_write
from yolo_scheduler_trace.events import Attribution, clip
from yolo_scheduler_trace_repair.events import validate_profile

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'artifacts/caller_gap_scope_01'
REPORT = ROOT.parents[4] / 'paper/my_paper/analysis/20261111_caller_gap_scope'


def caller_intervals(start, end, launches):
    """Partition time outside launch-to-last-callback envelopes exactly once."""
    cursor = start
    rows = []
    for index, launch in enumerate(launches):
        a, b, c = (launch[k] for k in ('start', 'last_callback_end', 'end'))
        if not cursor <= a <= b <= c <= end:
            raise ValueError('caller intervals overlap or reverse')
        rows.append(dict(kind='before_launch', kernel=index, start=cursor, end=a))
        rows.append(dict(kind='callback_end_to_return', kernel=index, start=b, end=c))
        cursor = c
    rows.append(dict(kind='after_last_launch', kernel=len(launches), start=cursor, end=end))
    outside = sum(r['end'] - r['start'] for r in rows)
    envelopes = sum(r['last_callback_end'] - r['start'] for r in launches)
    if outside + envelopes != end - start:
        raise ValueError('caller partition does not conserve wall time')
    return rows


def cpu_fraction(task):
    value = task['CPU']
    return Fraction(value['numerator'], value['denominator'])


def model_interval_budget(start, end, caller_tasks):
    selected = [t for t in caller_tasks if max(start, t['ready_ns']) < min(end, t['end_ns'])]
    selected.sort(key=lambda t: (t['ready_ns'], t['end_ns']))
    cursor = start
    cpu = Fraction(0)
    for task in selected:
        if task['ready_ns'] != cursor or task['end_ns'] > end:
            raise ValueError('model caller work crosses or misses an interval')
        cpu += cpu_fraction(task)
        cursor = task['end_ns']
    if cursor != end:
        raise ValueError('model caller interval is not covered')
    difference = Fraction(end - start) - cpu
    # Model task timestamps are rounded to integer ns; costs retain fractions.
    if difference < -max(1, len(selected)):
        raise ValueError('model CPU exceeds its wall envelope')
    return dict(wall_ns=end - start, CPU_ns=cpu,
                wall_minus_CPU_including_rounding_ns=difference,
                operation_CPU_ns={op: sum((cpu_fraction(t) for t in selected
                    if t['operation'] == op), Fraction(0))
                    for op in sorted({t['operation'] for t in selected})})


def means(rows, fields, frames):
    return {field.replace('_ns', '_ms'): float(sum(r[field] for r in rows) / frames / 1_000_000)
            for field in fields}


def fractions_as_records(value):
    if isinstance(value, Fraction):
        return dict(numerator=value.numerator, denominator=value.denominator)
    if isinstance(value, dict):
        return {key: fractions_as_records(item) for key, item in value.items()}
    if isinstance(value, list):
        return [fractions_as_records(item) for item in value]
    return value


def run():
    if OUT.exists():
        raise FileExistsError('preserve the accepted caller audit')
    geometry_path = ROOT / 'artifacts/current_gap_geometry_01/RESULTS.json'
    geometry_manifest = read(geometry_path.parent / 'MANIFEST.json')
    if sha256(geometry_path) != geometry_manifest['files'][str(geometry_path)]:
        raise ValueError('sealed gap geometry changed')
    geometry = read(geometry_path)
    sources = {str(geometry_path): sha256(geometry_path)}
    model_path = ROOT / 'artifacts/current_model_gap_01/independent.json'
    if sha256(model_path) != geometry['sources'][str(model_path)]:
        raise ValueError('sealed isolated-input replay changed')
    model_document = read(model_path)
    if model_document['no_pipeline_performance_read'] is not True:
        raise ValueError('prediction input isolation not established')
    model = model_document['result']
    sources[str(model_path)] = sha256(model_path)
    receipts = []
    for path, expected in geometry['sources'].items():
        if '/accepted/' not in path:
            continue
        if sha256(path) != expected:
            raise ValueError('diagnostic receipt changed')
        receipt = read(path)
        if receipt['task']['spec']['phase'] == 'observed':
            receipts.append(receipt)
            sources[path] = expected
    if len(receipts) != 1:
        raise ValueError('diagnostic source is missing or ambiguous')
    receipt = receipts[0]
    verify_inventory(receipt['raw'], receipt['files'])
    raw = Path(receipt['raw'])
    affinity = read(raw / 'affinity.json')
    profile = validate_profile(read(raw / 'profile.json'), receipt['task'], affinity)
    streams = profile['streams']
    caller = affinity['stages'][0]['caller_tid']
    mapping = {w['slot']: w['tid'] for w in affinity['stages'][0]['worker_slots']}
    if mapping.get(0) != caller or sorted(mapping) != list(range(4)):
        raise ValueError('logical caller/worker mapping changed')
    core = receipt['task']['stages'][0]['worker_core_map'][0]
    stream = streams[str(caller)]
    attribution = Attribution(streams)
    lines = lambda path: [json.loads(s) for s in path.read_text().splitlines()]
    phases = lines(raw / 'phase_trace.jsonl')
    trace = lines(raw / 'wait_trace.jsonl')
    launches = [t for t in trace if t['kind'] == 0]
    callbacks = defaultdict(list)
    for record in trace:
        if record['kind'] == 1:
            callbacks[record['launch']].append(record)
    frames = geometry['matched_frames']
    native_rows, model_rows = [], []
    for frame in frames:
        runs = [p for p in phases if p['stage'] == 'cpu_00_02'
                and p['operation'] == 'run' and p['call'] == frame]
        if len(runs) != 1:
            raise ValueError('native frame boundary is ambiguous')
        phase = runs[0]
        group = sorted([t for t in launches if t['caller'] == caller
            and phase['wall_start_ns'] <= t['wall_start_ns'] <= t['wall_end_ns'] <= phase['wall_end_ns']],
            key=lambda t: t['wall_start_ns'])
        if len(group) != 12:
            raise ValueError('native parallel launch sequence incomplete')
        native_launches = []
        for launch in group:
            workers = callbacks[launch['launch']]
            if (sorted(w['task'] for w in workers) != list(range(4))
                    or any(mapping[w['task']] != w['tid'] for w in workers)):
                raise ValueError('native worker set or identity changed')
            native_launches.append(dict(start=launch['wall_start_ns'],
                last_callback_end=max(w['wall_end_ns'] for w in workers), end=launch['wall_end_ns']))
        for interval in caller_intervals(phase['wall_start_ns'], phase['wall_end_ns'], native_launches):
            span = clip(stream, interval['start'], interval['end'])
            if not span['fully_covered']:
                raise ValueError('caller scheduler coverage incomplete')
            actors = defaultdict(int)
            unobserved = uncertainty = 0
            for off in span['off']:
                overlap = attribution.other_work(off, core)
                for role, amount in overlap['by_role_ns'].items():
                    actors[role] += amount
                unobserved += overlap['unobserved_ns']
                uncertainty += overlap['overlap_uncertainty_ns']
            native_rows.append(dict(frame=frame, **interval,
                wall_ns=interval['end'] - interval['start'],
                on_CPU_ns=span['on_CPU_ns'], off_CPU_ns=span['off_CPU_ns'],
                preempt_off_ns=span['preempt_off_ns'], nonpreempt_off_ns=span['nonpreempt_off_ns'],
                coincident_roles_ns=dict(actors), unobserved_other_ns=unobserved,
                overlap_uncertainty_ns=uncertainty))
        tasks = [t for t in model['tasks'] if t['stage'] == 0 and t['frame'] == frame]
        start = min(t['ready_ns'] for t in tasks if t['operation'] in ('serial', 'launch_prefix'))
        end = next(t['ready_ns'] for t in tasks if t['operation'] == 'get')
        model_launches = []
        for kernel in range(12):
            prefix = [t for t in tasks if t['operation'] == 'launch_prefix' and t['kernel'] == kernel]
            workers = [t for t in tasks if t['operation'] == 'worker_compute' and t['kernel'] == kernel]
            if len(prefix) != 1 or sorted(t['slot'] for t in workers) != list(range(4)):
                raise ValueError('sealed model launch/worker sequence incomplete')
            join = max(t['end_ns'] for t in workers)
            # The model resumes a continuation at the join; return CPU work
            # and queueing may already reside in its next serial interval.
            model_launches.append(dict(start=prefix[0]['ready_ns'], last_callback_end=join, end=join))
        caller_tasks = [t for t in tasks if t['slot'] == 0 and t['operation'] not in ('set', 'get')]
        for interval in caller_intervals(start, end, model_launches):
            model_rows.append(dict(frame=frame, **interval,
                                   **model_interval_budget(interval['start'], interval['end'], caller_tasks)))
    native = means(native_rows, ('wall_ns', 'on_CPU_ns', 'off_CPU_ns'), len(frames))
    predicted = means(model_rows, ('wall_ns', 'CPU_ns', 'wall_minus_CPU_including_rounding_ns'), len(frames))
    for name, value in [('native_observed', native), ('independent', predicted)]:
        expected = geometry['means'][name]['caller_and_other_outside_kernel_envelope_ms']
        if abs(value['wall_ms'] - expected) > 1e-9:
            raise ValueError('new partition does not reproduce sealed caller remainder')
    differences = dict(outside_wall_ms=native['wall_ms'] - predicted['wall_ms'],
        CPU_envelope_difference_ms=native['on_CPU_ms'] - predicted['CPU_ms'],
        off_CPU_and_rounding_difference_ms=native['off_CPU_ms'] - predicted['wall_minus_CPU_including_rounding_ms'])
    if abs(differences['outside_wall_ms'] - differences['CPU_envelope_difference_ms']
           - differences['off_CPU_and_rounding_difference_ms']) > 1e-9:
        raise ValueError('caller difference is not conserved')
    by_kind = {kind: means([r for r in native_rows if r['kind'] == kind],
        ('wall_ns', 'on_CPU_ns', 'off_CPU_ns', 'preempt_off_ns', 'nonpreempt_off_ns'), len(frames))
        for kind in ('before_launch', 'callback_end_to_return', 'after_last_launch')}
    positions = []
    for kernel in range(13):
        actual = [r for r in native_rows if r['kernel'] == kernel and r['kind'] != 'callback_end_to_return']
        forecast = [r for r in model_rows if r['kernel'] == kernel and r['kind'] != 'callback_end_to_return']
        positions.append(dict(kernel=kernel,
            native=means(actual, ('wall_ns', 'on_CPU_ns', 'off_CPU_ns'), len(frames)),
            model=means(forecast, ('wall_ns', 'CPU_ns'), len(frames))))
    result = dict(native=native, model=predicted, difference=differences, native_by_kind=by_kind,
        serial_positions=positions, matched_frames=frames, sources=sources,
        source_sha256=sha256(__file__), diagnostic_only=True, new_native_sessions=0, new_DNN=0,
        prediction_or_cost_or_DP_or_Top20_changed=False,
        native_CPU_event_and_model_CPU_clock_scopes_differ=True,
        last_callback_marker_is_not_actual_runtime_completion_signal=True,
        no_sum_of_all_worker_wall_wait=True, coincident_work_not_removable_latency=True,
        model_return_delay_may_be_in_next_serial_interval=True)
    OUT.mkdir(parents=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    write(OUT / 'RESULTS.json', result)
    csv_write(OUT / 'native_intervals.csv', native_rows)
    csv_write(OUT / 'model_intervals.csv', fractions_as_records(model_rows))
    text = [
        '剩余阶段误差：调用线程区间的离线核查', '',
        '本轮回到隔离profile与流水慢化，不执行已准备的88配置性能批次。',
        '只读既有隔离输入、封存回放及已验收轨迹；0次板端连接、0次推理。', '',
        '对象是YOLO CPU0–2四线程重叠配置；使用同一诊断帧12–23。',
        '探针窗口run实测225.355ms，冻结回放192.527ms，差32.828ms。探针数据不替代正式FPS。',
        '其中关键回调体跨度差20.882ms，剩余调用线程区间差12.091ms，进入区间抵消0.145ms。', '',
        '以下继续拆分12.091ms，不重放模型、不添加修正系数：',
        f"实际调用线程区间：墙钟{native['wall_ms']:.3f}ms，运行{native['on_CPU_ms']:.3f}ms，未运行{native['off_CPU_ms']:.3f}ms。",
        f"同口径模型区间：墙钟{predicted['wall_ms']:.3f}ms，CPU服务{predicted['CPU_ms']:.3f}ms。",
        f"差额为CPU包络差{differences['CPU_envelope_difference_ms']:.3f}ms，以及未运行/取整差{differences['off_CPU_and_rounding_difference_ms']:.3f}ms。", '',
        '实际区间 / 墙钟ms / 运行ms / 未运行ms']
    for kind, values in by_kind.items():
        text.append(kind + ' / ' + ' / '.join(f'{values[k]:.3f}' for k in ('wall_ms', 'on_CPU_ms', 'off_CPU_ms')))
    text += ['',
        '结论：不能把12.091ms全当作遗漏计算量。调用线程未运行是主要部分；单独改线程份额不足以解决。',
        '串行工作主要集中在第5、9次并行启动之前；但这些区间CPU事件与模型有效CPU工作边界不同，差额还含观测和运行时包络。',
        '全部未运行区间的switch-out带抢占标记；没有据此把整个区间都改名为精确排队时间。',
        '“最后回调结束”不是运行库真实完成通知；仍不能把该点到返回之间的6.760ms全部判为可删除的返回延迟。',
        '模型在完成汇合后立即进入下一串行段，等待可能移到下一段；因此按合并区间比较，不能把模型单独返回尾部为0当成全漏计。',
        '同核重合工作只用于定位竞争对象，不是可恢复时延，也不回填profile。', '',
        '下一步只核查这两处重串行区间和关键并行调用的边界/CPU服务：先分清实际工作增长、观测包络和调度次序误差。',
        '已有主机换核、阻塞等待、迁移和总核时份额实验不重复；无新的有效证据前不启动全域排名或整批Top20实测。',
        '尚未解决完整阶段误差，也没有声称提升吞吐或完成整个goal。']
    report = REPORT / 'FINDINGS_zh.txt'
    report.write_text('\n'.join(text) + '\n')
    paths = [OUT / name for name in ('RESULTS.json', 'native_intervals.csv', 'model_intervals.csv')] + [report]
    write(OUT / 'MANIFEST.json', dict(files={str(p): sha256(p) for p in paths}))
    print(json.dumps(dict(native=native, model=predicted, difference=differences, report=str(report)), ensure_ascii=False))
    return result


if __name__ == '__main__':
    run()
