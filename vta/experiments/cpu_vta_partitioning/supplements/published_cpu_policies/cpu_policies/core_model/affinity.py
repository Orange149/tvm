"""Validate sorted allowed masks and independently ordered logical workers."""
import math


def validate_affinity(report, config, qualification):
    if report.get('cpu_worker_pinning') is not (config['affinity_mode'] == 'pinned'):
        raise ValueError('native pinning mode differs')
    if len(report.get('stages', [])) != len(config['stages']):
        raise ValueError('affinity stage coverage differs')
    ledger, idle, owned = [], [], set()
    for index, (stage, row) in enumerate(zip(config['stages'], report['stages'])):
        if row.get('requested_cpus') != stage['mask'] or row.get('active_runtime_threads') != stage['threads']:
            raise ValueError('stage mask/thread request differs')
        tids, entries = row.get('thread_ids', []), row.get('effective_affinities', [])
        caller = row.get('caller_tid')
        if (not tids or any(type(t) is not int or t <= 0 for t in tids)
                or type(caller) is not int or caller not in tids
                or len(tids) != len(set(tids)) or len(entries) != len(tids)
                or {e['tid'] for e in entries} != set(tids) or owned.intersection(tids)):
            raise ValueError('invalid or overlapping stage TID ownership')
        owned.update(tids)
        for entry in entries:
            cpus, value = entry.get('cpus'), entry.get('cpu_time_delta_ms')
            if (not cpus or any(type(c) is not int for c in cpus) or cpus != sorted(set(cpus))
                    or not set(cpus) <= set(stage['mask'])):
                raise ValueError('actual affinity escaped stage mask')
            if (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
                raise ValueError('invalid thread activity increment')
            item = {'stage': stage['program'], 'stage_index': index, 'tid': entry['tid'],
                    'role': 'caller' if entry['tid'] == caller else 'worker',
                    'requested_mask': stage['mask'], 'actual_mask': cpus,
                    'cpu_time_delta_ms': value, 'scope': 'whole_session_including_warmup_and_tail'}
            if stage['device'] == 'cpu':
                if len(cpus) != 1:
                    raise ValueError('CPU logical worker is not pinned to one core')
                item['logical_worker'] = stage['worker_core_map'].index(cpus[0])
                item['participation'] = 'configured_compute_slot'
                if (entry['tid'] == caller) != (item['logical_worker'] == 0):
                    raise ValueError('caller differs from logical worker zero')
            elif value == 0 and entry['tid'] != caller:
                item['participation'] = 'VTA_idle_auxiliary_without_observed_increment'
                idle.append(item)
            ledger.append(item)
        if stage['device'] == 'cpu':
            mapping = stage['worker_core_map']
            if len(tids) != stage['threads'] or sorted(e['cpus'][0] for e in entries) != sorted(mapping):
                raise ValueError('CPU worker/core mapping is not bijective')
            if qualification:
                slots = row.get('worker_slots', [])
                if (len(slots) != stage['threads'] or any(type(v.get('slot')) is not int for v in slots)
                        or sorted(v['slot'] for v in slots) != list(range(stage['threads']))
                        or len({v['tid'] for v in slots}) != len(slots)):
                    raise ValueError('missing or duplicate logical worker probe')
                for slot in slots:
                    if (slot['tid'] not in tids or slot['cpus'] != [mapping[slot['slot']]]
                            or ((slot['slot'] == 0) != (slot['tid'] == caller))):
                        raise ValueError('actual logical slot mapping differs')
    if len(config['stages']) > 1:
        controls = report.get('control_threads', [])
        if (len(controls) != 2 or {c.get('role') for c in controls} != {'collector', 'producer'}
                or len({c.get('tid') for c in controls}) != 2
                or any(type(c.get('tid')) is not int or c['tid'] <= 0 or c['tid'] in owned
                       or c.get('cpus') != config['control_mask'] for c in controls)):
            raise ValueError('control role/affinity differs')
        ledger.extend({'role': c['role'], 'tid': c['tid'], 'actual_mask': c['cpus'],
                       'cpu_time_delta_ms': None, 'scope': 'not_collected_by_frozen_executor'} for c in controls)
    return ledger, idle
