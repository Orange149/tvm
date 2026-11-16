"""Low-overhead driver and single host-thread counters, never process CPU."""
import math
import statistics as st
from analyze_cpu_vta_all_stage_gate_validation import analyze as gate_analyze


def analyze(rows, enabled, scope, discard=7):
    if not rows or [r['frame_id'] for r in rows]!=list(range(len(rows))):
        raise ValueError('Missing/reordered frame')
    gate=gate_analyze(rows,'exclusive',scope,discard)
    if not enabled:
        if any('benchmark_monotonic_origin_ms' in r or any(f'stage{i}_driver_snapshot' in r for i in range(3)) for r in rows):
            raise ValueError('Default/off schema drift')
        return {'gate_metrics':gate,'enabled':False}
    origins={r.get('benchmark_monotonic_origin_ms') for r in rows}
    if len(origins)!=1 or None in origins:raise ValueError('Clock origin drift')
    origin=next(iter(origins));tids=set();frames=[]
    for r in rows:
        for i in range(3):
            components=sum(r[f'stage{i}_{k}_ms'] for k in ('set','run','get'))
            if abs(components-r[f'stage{i}_ms'])>.01:
                raise ValueError('Probe cost lost from inclusive stage service')
        if r.get('stage0_driver_snapshot') is not None or r.get('stage2_driver_snapshot') is not None:
            raise ValueError('CPU stage incorrectly attributed')
        if any(r.get(f'stage{i}_process_cpu_ms') is not None for i in range(3)):
            raise ValueError('Pipeline process CPU attribution')
        x=r.get('stage1_driver_snapshot')
        if not isinstance(x,dict):raise ValueError('Missing VTA counters')
        for k,v in x.items():
            if not isinstance(v,(int,float)) or not math.isfinite(v) or v<0:
                raise ValueError('Invalid snapshot '+k)
        if x['driver_run_calls']!=15 or x['driver_run_insns']!=3397 or x['driver_timeout_calls']:
            raise ValueError('Frozen 03:17 command/completion identity drift')
        if not isinstance(x['tid'],int) or x['tid']<=0:raise ValueError('Invalid host TID')
        tids.add(x['tid'])
        start=origin+r['stage1_run_start_ms'];end=origin+r['stage1_run_end_ms']
        if not x['bracket_start_monotonic_ms']<=start<=end<=x['bracket_end_monotonic_ms']:
            raise ValueError('Clock bracket/order failure')
        bracket=x['bracket_end_monotonic_ms']-x['bracket_start_monotonic_ms']
        if abs(end-start-r['stage1_run_ms'])>1e-4:raise ValueError('Run clock mismatch')
        if x['driver_total_ms']>r['stage1_run_ms']+.02:raise ValueError('Driver outside run')
        children=sum(x[k] for k in ('driver_mmio_ms','driver_post_start_sleep_ms','driver_poll_wait_ms'))
        if children>x['driver_total_ms']+.02:raise ValueError('Nested driver timers outside parent')
        if x['thread_cpu_ms']+x['sched_runqueue_ms']>bracket+.1:
            raise ValueError('Thread CPU/runqueue counters outside bracket')
        if x['probe_before_ms']+x['probe_after_ms']>bracket-r['stage1_run_ms']+.03:
            raise ValueError('Probe budget outside bracket')
        frames.append({'frame_id':r['frame_id'],'run_ms':r['stage1_run_ms'],
                       **{k:x[k] for k in ('driver_total_ms','driver_mmio_ms','driver_post_start_sleep_ms',
                                         'driver_poll_wait_ms','thread_cpu_ms','sched_runqueue_ms',
                                         'sched_dispatches','probe_before_ms','probe_after_ms')},
                       'driver_other_ms':x['driver_total_ms']-children,
                       'host_outside_driver_ms':r['stage1_run_ms']-x['driver_total_ms'],
                       'snapshot_bracket_ms':bracket,
                       'thread_non_cpu_ms':bracket-x['thread_cpu_ms'],
                       'thread_non_cpu_not_runqueue_ms':bracket-x['thread_cpu_ms']-x['sched_runqueue_ms']})
    if len(tids)!=1:raise ValueError('VTA worker identity drift')
    window=frames[discard:]
    if not window:raise ValueError('Empty stable window')
    return {'gate_metrics':gate,'enabled':True,'tid':next(iter(tids)),
            'means':{k:st.mean(r[k] for r in window) for k in frames[0] if k!='frame_id'},
            'per_frame':frames,'stable_frame_ids':[r['frame_id'] for r in window],
            'sched_counter_observable':any(r['sched_runqueue_ms']>0 for r in window) and all(r['sched_dispatches']>0 for r in window),
            'clock':'steady-clock absolute origin and brackets; thread CPU clock separate',
            'scope':'Driver children partition driver_total. CPU and schedstat describe overlapping host-thread bracket, never add to driver time.',
            'unknown':['fine-grained scheduler timeline (kernel FTRACE disabled)',
                       'hardware compute-only','physical DDR/DMA','timer wakeup versus polling sleep versus hardware wait in remaining non-CPU wall']}
