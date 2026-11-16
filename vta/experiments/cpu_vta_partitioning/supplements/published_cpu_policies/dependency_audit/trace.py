"""Decode completion-order trace records; do not invent cross-clock anchors."""
from collections import Counter, defaultdict
from decimal import Decimal
import json
from pathlib import Path


def ns_from_ms(value):
    converted = Decimal(str(value)) * 1000000
    if not converted.is_finite() or converted < 0 or converted != converted.to_integral_value():
        raise ValueError('native millisecond value is not an exact nonnegative nanosecond')
    return int(converted)


def parse(events, affinity, native_rows, module_sha256, *, expected_kernels=12):
    """Map repeated same-caller sequences to ordinal frames, then check clocks."""
    if (not isinstance(module_sha256,str) or len(module_sha256)!=64
            or any(c not in '0123456789abcdef' for c in module_sha256)):
        raise ValueError('verified module SHA256 required')
    caller = affinity['caller_tid']
    mapping = {entry['cpus'][0]: entry['tid'] for entry in affinity['effective_affinities']
               if len(entry['cpus']) == 1}
    cores = affinity['requested_cpus']
    if (set(mapping) != set(cores) or len(mapping) != affinity['active_runtime_threads']
            or len(set(mapping.values())) != len(mapping) or mapping[cores[0]] != caller):
        raise ValueError('fixed logical thread ownership is not bijective')
    logical = {i: mapping[c] for i, c in enumerate(cores)}
    launches, callbacks, barriers = {}, defaultdict(list), defaultdict(list)
    non_target = Counter()
    for event in events:
        kind = event.get('kind')
        if type(kind) is not int or kind not in (0, 1, 2):
            raise ValueError('unknown event kind')
        module=event.get('module','')
        if event.get('caller')==caller and '/programs/' in module and not module.endswith('/programs/cpu_00_02/module.so'):
            raise ValueError('unexpected module executed on the target stage caller')
        if event.get('caller') != caller or not module.endswith('/programs/cpu_00_02/module.so'):
            if kind == 0:
                non_target[event.get('module', '')] += 1
            continue
        for key in ('launch', 'tid', 'caller', 'task', 'tasks', 'wall_start_ns', 'wall_end_ns',
                    'cpu_start_ns', 'cpu_end_ns', 'function_offset', 'result'):
            if type(event.get(key)) is not int:
                raise ValueError('trace identity and clocks must be integers')
        if (event['result'] != 0 or min(event[k] for k in ('wall_start_ns','wall_end_ns','cpu_start_ns','cpu_end_ns')) < 0
                or event['wall_start_ns'] > event['wall_end_ns']
                or event['cpu_start_ns'] > event['cpu_end_ns']
                or event['cpu_end_ns']-event['cpu_start_ns'] > event['wall_end_ns']-event['wall_start_ns']+10000):
            raise ValueError('trace clocks or result invalid')
        if event['tasks'] != len(cores):
            raise ValueError('active task count differs from the fixed group')
        key = event['launch']
        if kind == 0:
            if key in launches:
                raise ValueError('duplicate launch')
            launches[key] = event
        elif kind == 1:
            callbacks[key].append(event)
        else:
            barriers[key].append(event)
    if not launches or set(callbacks) != set(launches) or barriers:
        raise ValueError('missing launch/callback or unsupported internal barrier')
    ordered = sorted(launches.values(), key=lambda e: e['wall_start_ns'])
    previous_cpu, details = {}, []
    for index, launch in enumerate(ordered):
        group = callbacks[launch['launch']]
        if (len(group) != len(cores) or {v['task'] for v in group} != set(logical)
                or len({v['tid'] for v in group}) != len(group)):
            raise ValueError('missing or duplicate callback')
        if index and (ordered[index-1]['wall_end_ns'] > launch['wall_start_ns']
                      or ordered[index-1]['cpu_end_ns'] > launch['cpu_start_ns']):
            raise ValueError('next launch started before the previous return')
        group.sort(key=lambda e: e['task'])
        for task in group:
            if task['tid'] != logical[task['task']] or task['function_offset'] != launch['function_offset']:
                raise ValueError('logical thread or kernel identity differs')
            if (task['wall_start_ns'] < launch['wall_start_ns'] or task['wall_end_ns'] > launch['wall_end_ns']
                    or task['cpu_start_ns'] < previous_cpu.get(task['tid'], 0)):
                raise ValueError('callback clock/order escaped its launch')
            previous_cpu[task['tid']] = task['cpu_end_ns']
        task0 = group[0]
        if (task0['cpu_start_ns'] < launch['cpu_start_ns'] or task0['cpu_end_ns'] > launch['cpu_end_ns']
                or task0['tid'] != caller):
            raise ValueError('caller callback CPU clock not enclosed by launch')
        if launch.get('last_task_end_ns') != max(t['wall_end_ns'] for t in group):
            raise ValueError('last callback end does not match the recorded join')
        details.append(dict(launch_id=launch['launch'], function_offset=launch['function_offset'],
            module_sha256=module_sha256, caller_tid=caller,
            wall_start_ns=launch['wall_start_ns'], wall_end_ns=launch['wall_end_ns'],
            caller_cpu_start_ns=launch['cpu_start_ns'], caller_cpu_end_ns=launch['cpu_end_ns'],
            worker_cpu_ns=[t['cpu_end_ns']-t['cpu_start_ns'] for t in group],
            worker_wall_ns=[t['wall_end_ns']-t['wall_start_ns'] for t in group],
            worker_wall_start_ns=[t['wall_start_ns'] for t in group],
            worker_wall_end_ns=[t['wall_end_ns'] for t in group],
            launch_wall_ns=launch['wall_end_ns']-launch['wall_start_ns'],
            start_skew_ns=max(t['wall_start_ns'] for t in group)-min(t['wall_start_ns'] for t in group),
            body_span_ns=max(t['wall_end_ns'] for t in group)-min(t['wall_start_ns'] for t in group),
            largest_task_cpu_ns=max(t['cpu_end_ns']-t['cpu_start_ns'] for t in group),
            after_last_callback_ns=launch['wall_end_ns']-max(t['wall_end_ns'] for t in group),
            caller_tail_cpu_ns=launch['cpu_end_ns']-task0['cpu_end_ns'],
            caller_tail_wall_ns=launch['wall_end_ns']-task0['wall_end_ns'],
            last_worker=max(group, key=lambda t: (t['wall_end_ns'], -t['task']))['task'],
            accounted_launch_cpu_ns=launch['cpu_end_ns']-launch['cpu_start_ns']+
                sum(t['cpu_end_ns']-t['cpu_start_ns'] for t in group[1:])))
    count = len(native_rows)
    pattern = [d['function_offset'] for d in details[:expected_kernels]]
    frame_mapping = (len(details) == count*expected_kernels and len(pattern) == expected_kernels
        and all([d['function_offset'] for d in details[i*expected_kernels:(i+1)*expected_kernels]] == pattern
                for i in range(count)))
    result = dict(module_sha256=module_sha256, caller_tid=caller, threads=len(cores), core_map=cores,
        sequence=pattern, frame_mapping=frame_mapping, non_target_launches_by_module=dict(non_target), launches=details,
        frame_mapping_basis='verified repeated kernel sequence and serial caller; no invented absolute clock origin',
        exact_cross_clock_anchor=False, frames=[])
    if not frame_mapping:
        result['mapping_failure'] = 'kernel sequence/count does not establish ordinal frame mapping'
        return result
    offset_lower, offset_upper = [], []
    for index, row in enumerate(native_rows):
        group = details[index*expected_kernels:(index+1)*expected_kernels]
        if row['frame_id'] != index or row['completion_index'] != index:
            raise ValueError('native frame order differs from the serial stage')
        gaps = [b['caller_cpu_start_ns']-a['caller_cpu_end_ns'] for a, b in zip(group, group[1:])]
        if min(gaps, default=0) < 0:
            raise ValueError('caller CPU gap is negative')
        offset_lower.append(group[-1]['wall_end_ns']-ns_from_ms(row['stage0_run_end_ms']))
        offset_upper.append(group[0]['wall_start_ns']-ns_from_ms(row['stage0_run_start_ms']))
        work = sum(d['accounted_launch_cpu_ns'] for d in group)+sum(gaps)
        isolated = row.get('stage_cpu_time_scope') == 'exclusive_process_cpu_time'
        run_cpu = ns_from_ms(row['stage0_run_process_cpu_ms']) if isolated else None
        # This residual includes unobserved runtime/probe work; it is not a
        # prefix/suffix duration and cannot be assigned a position in the DAG.
        residual = run_cpu-work if isolated else None
        if residual is not None and residual < -10000:
            raise ValueError('accounted launch work exceeds isolated process CPU time')
        result['frames'].append(dict(frame_id=index, launches=group, interlaunch_caller_cpu_ns=gaps,
            accounted_run_cpu_ns=work, native_run_cpu_ns=run_cpu, unlocated_CPU_residual_ns=residual,
            native_run_wall_ns=ns_from_ms(row['stage0_run_ms']),
            native_service_ns=sum(ns_from_ms(row['stage0_'+op+'_ms']) for op in ('set','run','get'))))
    interval = dict(lower_ns=max(offset_lower), upper_ns=min(offset_upper))
    result['clock_offset_consistency_interval'] = interval
    if interval['lower_ns'] > interval['upper_ns']:
        result['frame_mapping'] = False
        result['mapping_failure'] = 'ordinal alignment contradicts the enclosing native run windows'
        result['frames'] = []
    return result


def load_trace(raw, module_sha256):
    raw = Path(raw)
    native = [json.loads(s) for s in (raw/'result.jsonl').read_text().splitlines() if s.strip()]
    native.sort(key=lambda row: row['completion_index'])
    report = json.loads((raw/'affinity.json').read_text())
    records = [json.loads(s) for s in (raw/'wait_trace.jsonl').read_text().splitlines() if s.strip()]
    return parse(records, report['stages'][0], native, module_sha256)
