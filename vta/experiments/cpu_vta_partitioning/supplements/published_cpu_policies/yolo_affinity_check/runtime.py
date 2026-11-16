"""One executor, two affinity flags; validate logical slots in each process."""
from copy import deepcopy
from pathlib import Path
from types import FunctionType
import math
import statistics

from cpu_vta_partitioning.common import digest, read
from cpu_vta_partitioning.measurement import (calibration, correctness, frames,
    input_sequence, window)
from cpu_vta_partitioning.transport import SSH
from yolo_transfer.pipeline import check_outputs
from yolo_validation.pipeline import work_boundaries
from .common import protocol, validate_config, ENV


def make_task(bundle, spec):
    bundle = Path(bundle)
    p, manifest = read(bundle / 'plans/protocol.json'), read(bundle / 'manifest.json')
    if spec not in read(bundle / 'plans/task_order.json'):
        raise ValueError('task is not in the frozen experiment order')
    component = spec['scope'] == 'component'
    qualify = spec['phase'] == 'qualify'
    cfg = read(bundle / 'plans/configurations.json')['components' if component else 'pipelines'][spec['case']]
    validate_config(cfg, read(bundle / 'plans/programs.json'), component=component)
    timing = p['qualification' if qualify else 'component' if component else 'performance']
    task = dict(kind=('qualify_component' if qualify else 'calibration') if component else
                     ('qualify_pipeline' if qualify else 'performance'),
        configuration=deepcopy(cfg), stages=deepcopy(cfg['stages']), spec=deepcopy(spec),
        qualification=qualify, input_indices=[0], **timing, environment=deepcopy(ENV),
        experiment=p['experiment'], bundle_id=manifest['bundle_id'],
        executor_sha256=manifest['runner_sha256'], runtime_sha256=manifest['runtime_sha256'],
        initialization_affinity_probe=True, wait_policy='original_without_adapter',
        output_mode='all_stage_dumps' if qualify else 'all_final_hashes')
    task['task_id'] = digest(task)[:24]
    return task


def native_command(doc, bundle, remote_bundle, remote_job):
    from cpu_vta_partitioning.measurement.tasks import native_command as base
    if doc != make_task(bundle, doc['spec']):
        raise ValueError('task identity differs')
    argv, inputs = base(doc, bundle, remote_bundle, remote_job)
    at = argv.index(remote_bundle + '/runtime/ld-linux-aarch64.so.1')
    argv[at:at] = ['TVM_BIND_THREADS=1', 'TVM_EXCLUDE_WORKER0=1']
    argv += ['--allow-cpu-affinity-superset', '--control-cpu-affinity', '0,1,2,3',
             '--resource-affinity-probe']
    if doc['configuration']['affinity_mode'] == 'pinned':
        argv.append('--cpu-worker-pinning')
    if doc['spec']['scope'] == 'pipeline' and not doc['qualification']:
        argv[argv.index('--output-mode') + 1] = 'raw'
    if any(x in argv for x in ('--queue-timing', '--process-cpu-affinity', '--session-profile-json')):
        raise ValueError('unrequested timing probe or process affinity')
    if next(x for x in argv if x.startswith('LD_PRELOAD=')) != 'LD_PRELOAD=' + remote_bundle + '/runtime/libvta.so':
        raise ValueError('waiting adapter must not be loaded')
    return argv, inputs


def matched_ssh(config, *, execute_board=False):
    measure = FunctionType(SSH.measure.__code__, {**SSH.measure.__globals__, 'native_command': native_command},
                           SSH.measure.__name__, SSH.measure.__defaults__, SSH.measure.__closure__)
    return type('YOLOAffinitySSH', (SSH,), {'measure': measure})(config, execute_board=execute_board)


def finite(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError('invalid nonnegative clock/activity')
    return value


def affinity(report, cfg, *, qemu=False):
    """Logical identity comes from a real launch, never from a core number."""
    pinned = cfg['affinity_mode'] == 'pinned'
    if report.get('cpu_worker_pinning') is not pinned or len(report.get('stages', [])) != len(cfg['stages']):
        raise ValueError('affinity mode/stage coverage differs')
    owned, ledger, idle = set(), [], []
    for index, (stage, row) in enumerate(zip(cfg['stages'], report['stages'])):
        tids, entries, caller = row.get('thread_ids', []), row.get('effective_affinities', []), row.get('caller_tid')
        if (row.get('requested_cpus') != stage['mask'] or row.get('active_runtime_threads') != stage['threads']
                or any(type(t) is not int or t <= 0 for t in tids) or type(caller) is not int
                or caller not in tids or len(tids) != len(set(tids)) or owned.intersection(tids)
                or len(entries) != len(tids) or {e['tid'] for e in entries} != set(tids)):
            raise ValueError('stage ownership/thread request differs')
        owned.update(tids)
        by_tid = {e['tid']: e for e in entries}
        slots = row.get('worker_slots', []) if stage['device'] == 'cpu' else []
        if stage['device'] == 'cpu':
            if (len(slots) != stage['threads'] or any(type(s.get('slot')) is not int for s in slots)
                    or sorted(s['slot'] for s in slots) != list(range(stage['threads']))
                    or len({s['tid'] for s in slots}) != len(slots)):
                raise ValueError('active logical worker slots missing/duplicated')
            for slot in slots:
                expected = [stage['worker_core_map'][slot['slot']]] if pinned else stage['mask']
                if (slot['tid'] not in by_tid or slot['cpus'] != expected
                        or by_tid[slot['tid']]['cpus'] != expected
                        or (slot['slot'] == 0) != (slot['tid'] == caller)):
                    raise ValueError('caller/worker actual allowed mask differs')
        active = {s['tid']: s['slot'] for s in slots}
        for entry in entries:
            cpus, delta = entry.get('cpus'), finite(entry.get('cpu_time_delta_ms'))
            if not cpus or cpus != sorted(set(cpus)) or any(type(c) is not int for c in cpus):
                raise ValueError('invalid actual CPU mask')
            is_idle = stage['device'] == 'cpu' and entry['tid'] not in active
            # QEMU's emulator helper is not a guest compute slot.  This special
            # local-only allowance is never enabled by board acceptance.
            if not (qemu and is_idle) and not set(cpus) <= set(stage['mask']):
                raise ValueError('thread escaped the requested stage group')
            if is_idle and not qemu and delta != 0:
                raise ValueError('unconfigured CPU helper accumulated work')
            if stage['device'] == 'vta' and cpus != stage['mask']:
                raise ValueError('VTA host/helper allowed set differs')
            item = dict(stage=index, program=stage['program'], tid=entry['tid'],
                role='caller' if entry['tid'] == caller else 'worker', logical_worker=active.get(entry['tid']),
                requested_mask=stage['mask'], actual_mask=cpus, cpu_time_delta_ms=delta,
                scope='whole_session_including_startup_warmup_tail',
                participation='idle_auxiliary' if is_idle else 'configured_CPU_slot' if active else 'VTA_host_or_helper')
            ledger.append(item)
            if is_idle or (stage['device'] == 'vta' and entry['tid'] != caller and delta == 0):
                idle.append(item)
    if len(cfg['stages']) > 1:
        controls = report.get('control_threads', [])
        if (len(controls) != 2 or {c.get('role') for c in controls} != {'producer', 'collector'}
                or len({c.get('tid') for c in controls}) != 2
                or any(type(c.get('tid')) is not int or c['tid'] <= 0 or c['tid'] in owned
                       or c.get('cpus') != cfg['control_mask'] for c in controls)):
            raise ValueError('control thread placement differs')
        ledger.extend(dict(role=c['role'], tid=c['tid'], actual_mask=c['cpus'],
                           cpu_time_delta_ms=None, scope='not_collected') for c in controls)
    return ledger, idle


def summary(raw, doc, bundle, *, qemu=False):
    raw, bundle = Path(raw), Path(bundle)
    if doc != make_task(bundle, doc['spec']):
        raise ValueError('summary task changed')
    rows = frames(raw / 'result.jsonl', sum(doc[k] for k in ('warmup', 'scored', 'tail')))
    if any(type(r.get(k)) is not int for r in rows for k in ('frame_id', 'completion_index', 'input_index')):
        raise ValueError('invalid frame identities')
    input_sequence(rows, doc)
    cfg = doc['configuration']
    ledger, idle = affinity(read(raw / 'affinity.json'), cfg, qemu=qemu)
    programs, expected = read(bundle / 'plans/programs.json'), read(bundle / 'plans/expected_outputs.json')
    component = doc['spec']['scope'] == 'component'
    for row in rows:
        if row.get('cpu_stage_gate_mode', 'off') != 'off':
            raise ValueError('unexpected gate mode')
        if row.get('queue_timing_enabled') is not False or row.get('defer_result_write_enabled') is not True:
            raise ValueError('timing/logging settings differ')
        if row.get('stage_cpu_time_scope') != ('exclusive_process_cpu_time' if component else 'disabled_concurrent_pipeline'):
            raise ValueError('stage CPU clock scope differs')
        check_outputs(row.get('raw_outputs', []), programs[cfg['stages'][-1]['program']], expected)
    if any((raw / p).exists() for p in ('wait_status.json', 'wait_trace.jsonl')):
        raise ValueError('unexpected waiting adapter records')
    if doc['qualification']:
        correctness(rows, doc, bundle, raw)
        for row in rows:
            if len(row['stage_raw_outputs']) != len(cfg['stages']):
                raise ValueError('stage frontier incomplete')
            for stage, output in zip(cfg['stages'], row['stage_raw_outputs']):
                check_outputs(output['outputs'], programs[stage['program']], expected)
        result = dict(correct=True)
    elif component:
        cost = calibration(rows, doc['warmup'], doc['scored'])
        for op in ('set', 'run', 'get'):
            for suffix, field in (('_ms', '_ms'), ('_core_ms', '_process_cpu_ms')):
                cost[op + suffix] = statistics.mean(r['stage0_' + op + field] for r in rows[doc['warmup']:])
        result = dict(correct=True, cost=cost, stage_service_ms=[cost['service_ms']])
    else:
        result = window(rows, doc['warmup'], doc['scored'], doc['tail'])
        clocks = [finite(r.get('completion_process_cpu_ms')) for r in rows]
        if any(b < a for a, b in zip(clocks, clocks[1:])):
            raise ValueError('process CPU clock decreased')
        first, last = doc['warmup'], doc['warmup'] + doc['scored'] - 1
        work = clocks[last] - clocks[first]
        wall = rows[last]['completion_ms'] - rows[first]['completion_ms']
        if work <= 0 or work / wall > 4.05:
            raise ValueError('invalid whole-process CPU capacity')
        services = [statistics.mean(sum(finite(r[f'stage{i}_{op}_ms']) for op in ('set', 'run', 'get'))
                    for r in rows[first:last + 1]) for i in range(len(cfg['stages']))]
        result.update(correct=True, fps=1000 / result['ii_ms'], stage_service_ms=services,
            process_core_ms_per_completion=work / (doc['scored'] - 1), average_cpu_cores=work / wall,
            cpu_time_scope='whole_process_between_scored_completions', window_work=work_boundaries(rows, doc))
    return dict(result, thread_ledger=ledger, idle_auxiliary_threads=idle)
