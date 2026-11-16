"""Validate original MMIO observations and conservatively assign frames.

The frozen runner's std::chrono::steady_clock and the observer both use Linux
CLOCK_MONOTONIC (checked in the packaged libstdc++ disassembly). The runner
exports times relative to an undisclosed session origin. Retain *all* origins
consistent with every event and phase; never choose one near measured FPS.
Only frame identities invariant across all admissible origins are projected.
"""
from collections import defaultdict
from decimal import Decimal,ROUND_HALF_UP
from completion_profile.workflow import completion_bounds


def ns(value):
    return int((Decimal(str(value))*1_000_000).to_integral_value(rounding=ROUND_HALF_UP))


def intersect_intervals(left,right):
    # Closed integer intervals. Do not merge adjacent intervals: an event's
    # frame assignment could change across two adjacent admissible origins.
    return sorted(set((max(a,c),min(b,d)) for a,b in left for c,d in right if max(a,c)<=min(b,d)))


def frame_alignment(events_by_stage,frames):
    possible=[(0,2**63-1)];runs={}
    for stage,events in events_by_stage.items():
        if not events:raise ValueError('a VTA stage has no device events')
        phases=[(ns(r[f'stage{stage}_run_start_ms']),ns(r[f'stage{stage}_run_end_ms'])) for r in frames]
        if (any(b<a for a,b in phases) or any(a[1]>b[0] for a,b in zip(phases,phases[1:]))):
            raise ValueError('native stage run order invalid')
        if any(a['wall_end_ns']>b['wall_start_ns'] for a,b in zip(events,events[1:])):
            raise ValueError('same-stage device events overlap')
        runs[stage]=phases
        # All actual graph calls must have hardware work. Thus the first and
        # last observed events must lie in the first and last graph calls.
        for e,phase in [(events[0],phases[0]),(events[-1],phases[-1])]:
            possible=intersect_intervals(possible,[(e['wall_end_ns']-phase[1],e['wall_start_ns']-phase[0])])
        for e in events:
            allowed=[(e['wall_end_ns']-end,e['wall_start_ns']-begin) for begin,end in phases
                     if e['wall_end_ns']-end<=e['wall_start_ns']-begin]
            possible=intersect_intervals(possible,allowed)
            if not possible:raise ValueError('device events cannot be mapped into native graph runs')
    mappings=[];valid=[]
    for lower,upper in possible:
        projected=[];complete=True
        for stage,events in sorted(events_by_stage.items()):
            coverage=set()
            for ordinal,e in enumerate(events):
                candidates=[frame for frame,(begin,end) in enumerate(runs[stage])
                    if lower+begin<=e['wall_start_ns']<=e['wall_end_ns']<=lower+end]
                if len(candidates)!=1:raise ValueError('admissible origin has ambiguous event containment')
                frame=candidates[0];coverage.add(frame);projected.append((stage,ordinal,frame))
            if coverage!=set(range(len(frames))):complete=False
        if complete:valid.append((lower,upper));mappings.append(projected)
    if not valid:raise ValueError('a graph invocation lacks device-event coverage')
    reliable=all(m==mappings[0] for m in mappings)
    return dict(relative_origin_intervals_ns=valid,per_frame_mapping_reliable=reliable,
        mapping=mappings[0] if reliable else None,no_origin_selected_to_fit_performance=True,
        absolute_clock_origin_not_assumed=True)


def validate_events(lines,document,affinity,frames):
    if not lines or lines[0].get('kind')!='status':raise ValueError('device status missing')
    status=lines[0]
    if (status.get('overflow') or status.get('events')!=len(lines)-1
            or status.get('events',0)>4096 or status.get('stats_bytes')!=64
            or status.get('extra_MMIO_reads')!=0 or not status.get('per_poll_probe')
            or not status.get('completion_brackets') or not status.get('provider','').endswith('/runtime/libvta.so')):
        raise ValueError('device observer capacity/provider/ABI invalid')
    targets={row['caller_tid']:i for i,(s,row) in enumerate(zip(document['stages'],affinity['stages'])) if s['device']=='vta'}
    if len(targets)!=sum(s['device']=='vta' for s in document['stages']):
        raise ValueError('duplicate VTA host TID')
    events=defaultdict(list)
    for row in lines[1:]:
        if (row.get('kind')!='device' or row.get('result')!=0 or row.get('tid') not in targets
                or row.get('instructions',0)<1 or row.get('driver_run_calls')!=1
                or not row.get('counters_monotonic') or row.get('driver_timeout_calls')!=0):
            raise ValueError('device event identity or execution invalid')
        wall=row['wall_end_ns']-row['wall_start_ns'];cpu=row['cpu_end_ns']-row['cpu_start_ns']
        if not 0<=cpu<=wall+10_000 or wall<=0:raise ValueError('device clock invalid')
        bounds=completion_bounds(row)
        events[targets[row['tid']]].append({**row,'completion_bounds':bounds})
    if set(events)!=set(targets.values()):raise ValueError('VTA device-event stage coverage missing')
    for group in events.values():group.sort(key=lambda e:e['wall_start_ns'])
    all_events=sorted([e for group in events.values() for e in group],key=lambda e:e['wall_start_ns'])
    if any(a['wall_end_ns']>b['wall_start_ns'] for a,b in zip(all_events,all_events[1:])):
        raise ValueError('multiple physical VTA calls overlap')
    alignment=frame_alignment(events,frames)
    mapped=[]
    if alignment['per_frame_mapping_reliable']:
        for stage,ordinal,frame in alignment['mapping']:
            mapped.append(dict(stage=stage,frame=frame,**events[stage][ordinal]))
    return dict(status=status,alignment=alignment,mapped=mapped,
                whole_session_events=[dict(stage=i,**e) for i,group in events.items() for e in group])


def device_projection(validated,frames,document):
    if not validated['alignment']['per_frame_mapping_reliable']:
        return dict(per_frame_projection_supported=False,eligible_for_static_prediction=False,
                    reason='ambiguous frame association',whole_session_only=True)
    a,b=document['warmup'],document['warmup']+document['scored']
    result={}
    for i,s in enumerate(document['stages']):
        if s['device']!='vta':continue
        per_frame=[]
        for frame in range(a,b):
            events=sorted([e for e in validated['mapped'] if e['stage']==i and e['frame']==frame],key=lambda e:e['wall_start_ns'])
            driver_wall=sum(e['wall_end_ns']-e['wall_start_ns'] for e in events)
            run_wall=ns(frames[frame][f'stage{i}_run_end_ms'])-ns(frames[frame][f'stage{i}_run_start_ms'])
            if driver_wall>run_wall:raise ValueError('driver spans exceed graph run')
            per_frame.append(dict(frame=frame,device_calls=len(events),
                instructions=[e['instructions'] for e in events],run_wall_ns=run_wall,
                driver_wall_ns=driver_wall,driver_CPU_ns=sum(e['cpu_end_ns']-e['cpu_start_ns'] for e in events),
                outside_driver_run_wall_ns=run_wall-driver_wall,
                completion_lower_sum_ns=sum(e['completion_bounds']['lower_ns'] for e in events),
                completion_upper_sum_ns=sum(e['completion_bounds']['upper_ns'] for e in events),
                read_clock_envelope_ns=sum(e['poll_read_envelope_ns'] for e in events)))
        count=len(per_frame)
        fields=('run_wall_ns','driver_wall_ns','driver_CPU_ns','outside_driver_run_wall_ns',
                'completion_lower_sum_ns','completion_upper_sum_ns','read_clock_envelope_ns')
        means={key:dict(numerator=sum(r[key] for r in per_frame),denominator=count) for key in fields}
        result[s['program']]=dict(frames=per_frame,means=means,
            device_counts=[r['device_calls'] for r in per_frame],
            pure_device_CPU_wait_split_not_exact=True,
            outside_driver_wall_is_not_all_CPU_or_scheduler_wait=True)
    return dict(per_frame_projection_supported=True,eligible_for_static_prediction=False,stages=result,
                cost_source_kind=document.get('kind','unknown'),
                independent_probe_quality_required_before_any_prediction=True,
                pipeline_observations_forbidden_as_general_isolated_cost=True,
                stage_frame_scope_not_completion_window_scope=True)
