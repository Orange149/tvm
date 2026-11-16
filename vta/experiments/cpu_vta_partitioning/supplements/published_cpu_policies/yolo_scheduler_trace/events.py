"""Decode four-core task events, leaving unobserved boundary intervals unknown."""
from bisect import bisect_right,bisect_left
from collections import defaultdict
import struct

OUT=1<<13
PREEMPT=1<<14


def intersection(a,b,c,d):return max(0,min(b,d)-max(a,c))


def decode_stream(stream,pid):
    if stream['overflow']:raise ValueError('scheduler ring overflow')
    bounds=[stream[k] for k in ('enable_start_ns','enable_end_ns','disable_start_ns','disable_end_ns')]
    if any(type(x) is not int or x<=0 for x in bounds) or bounds!=sorted(bounds):raise ValueError('scheduler stream clock invalid')
    head,tail=stream['head'],stream['tail']
    if not 0<=tail<=head or head-tail>stream['data_size']:raise ValueError('scheduler ring head/tail invalid')
    try:raw=bytes.fromhex(stream['records_hex'])
    except ValueError as e:raise ValueError('scheduler ring bytes invalid') from e
    if len(raw)!=head-tail:raise ValueError('scheduler ring byte count mismatch')
    rows=[];offset=0;previous=bounds[0]
    while offset<len(raw):
        if len(raw)-offset<8:raise ValueError('partial scheduler record header')
        kind,misc,size=struct.unpack_from('<IHH',raw,offset)
        if size<8 or size%8 or offset+size>len(raw):raise ValueError('partial scheduler record')
        if kind in (2,13):raise ValueError('lost scheduler records')
        if kind!=14 or size!=32:raise ValueError('unexpected scheduler ABI record')
        record_pid,tid,when,cpu,reserved=struct.unpack_from('<IIQII',raw,offset+8)
        if record_pid!=pid or tid!=stream['tid'] or stream['pid']!=pid:raise ValueError('scheduler TID/PID differs')
        if cpu not in stream['allowed'] or reserved or not previous<=when<=bounds[-1]:raise ValueError('scheduler CPU/time differs')
        if misc&PREEMPT and not misc&OUT:raise ValueError('preempt bit on switch-in')
        direction='out' if misc&OUT else 'in'
        if rows and rows[-1]['direction']==direction:raise ValueError('missing or duplicate scheduler direction')
        rows.append(dict(tid=tid,time_ns=when,cpu=cpu,direction=direction,preempt=bool(misc&PREEMPT)))
        previous=when;offset+=size
    on=[];off=[]
    for a,b in zip(rows,rows[1:]):
        if a['direction']=='in':
            if a['cpu']!=b['cpu']:raise ValueError('CPU changed without switch-out/in')
            on.append(dict(start_ns=a['time_ns'],end_ns=b['time_ns'],cpu=a['cpu'],tid=stream['tid']))
        else:off.append(dict(start_ns=a['time_ns'],end_ns=b['time_ns'],from_cpu=a['cpu'],to_cpu=b['cpu'],preempt=a['preempt'],tid=stream['tid']))
    # The observed task was not the thread enabling the descriptor. Never
    # infer its initial/final running state from the first/last direction.
    coverage=[rows[0]['time_ns'],rows[-1]['time_ns']] if len(rows)>=2 else None
    counted=stream['counter_after']['value']-stream['counter_before']['value']
    if counted<0:raise ValueError('task CPU counter moved backwards')
    known_CPU=sum(t['end_ns']-t['start_ns'] for t in on)
    unknown=(coverage[0]-bounds[0]+bounds[-1]-coverage[1]) if coverage else bounds[-1]-bounds[0]
    tolerance=max(2000000,counted//50)
    if known_CPU>counted+tolerance or counted>known_CPU+unknown+tolerance:raise ValueError('event timeline violates task-clock envelope')
    return dict(tid=stream['tid'],events=rows,on=on,off=off,coverage_ns=coverage,
        task_CPU_ns=counted,known_CPU_ns=known_CPU,boundary_unknown_ns=unknown,allowed=stream['allowed'],
        no_initial_or_final_state_invented=True)


def role_map(doc,affinity):
    roles={}
    def add(tid,**data):
        if tid in roles:raise ValueError('logical thread role duplicated')
        roles[tid]=data
    for i,(stage,record) in enumerate(zip(doc['stages'],affinity['stages'])):
        if stage['device']=='cpu':
            for slot in record['worker_slots']:
                add(slot['tid'],stage=i,slot=slot['slot'],role=stage['program']+('/caller' if slot['slot']==0 else '/worker'+str(slot['slot'])),
                    kind='CPU_stage',allowed=[stage['worker_core_map'][slot['slot']]])
        else:add(record['caller_tid'],stage=i,slot=0,role=stage['program']+'/host',kind='VTA_host',allowed=stage['mask'])
    for record in affinity['control_threads']:
        add(record['tid'],stage=-1,slot=0,role=record['role'],kind='control',allowed=doc['configuration']['control_mask'])
    return roles


def validate_profile(profile,doc,affinity):
    if profile.get('status')!='MEASURED' or profile.get('clock')!='CLOCK_MONOTONIC' or profile.get('byte_order')!='little' or profile.get('sample_id_layout')!='TID,TIME,CPU' or profile.get('periodic_sampling') is not False:
        raise ValueError('scheduler observer unavailable or unexpected')
    roles=role_map(doc,affinity);rows=profile['streams'];ids=[r['tid'] for r in rows]
    if len(roles)!=12 or len(ids)!=len(set(ids)) or set(ids)!=set(roles):raise ValueError('scheduler role coverage incomplete')
    setup=[profile[k] for k in ('completion_start_ns','setup_start_ns','setup_end_ns','completion_end_ns','stop_start_ns','stop_end_ns')]
    if any(type(v) is not int or v<=0 for v in setup) or setup!=sorted(setup):raise ValueError('scheduler completion anchors/clock invalid')
    streams={}
    for r in rows:
        if r['allowed']!=roles[r['tid']]['allowed'] or not r['alive_at_stop']:raise ValueError('scheduler target affinity/lifetime differs')
        if not setup[1]<=r['enable_start_ns']<=r['enable_end_ns']<=setup[2] or not setup[4]<=r['disable_start_ns']<=r['disable_end_ns']<=setup[5]:raise ValueError('scheduler stream outside setup/stop envelope')
        for key in ('value','enabled','running'):
            if not 0<=r['counter_before'][key]<=r['counter_after'][key]:raise ValueError('scheduler task counter invalid')
        decoded=decode_stream(r,profile['pid']);decoded['role']=roles[r['tid']];streams[str(r['tid'])]=decoded
    by_core=defaultdict(list)
    for r in streams.values():
        for span in r['on']:by_core[span['cpu']].append(span)
    max_overlap=0
    for spans in by_core.values():
        ordered=sorted(spans,key=lambda s:(s['start_ns'],s['end_ns']));previous=None
        for span in ordered:
            if previous is not None and previous['tid']!=span['tid']:max_overlap=max(max_overlap,previous['end_ns']-span['start_ns'])
            if previous is None or span['end_ns']>previous['end_ns']:previous=span
    if max_overlap>100000:raise ValueError('two recorded TIDs execute on same core simultaneously')
    total=sum(len(r['events']) for r in streams.values())
    if not total:raise ValueError('scheduler event interception produced no records')
    return dict(streams=streams,total_events=total,
        max_same_core_overlap_ns=max_overlap,completion_start_ns=setup[0],completion_end_ns=setup[3],
        setup_after_completion_ns=[setup[1]-setup[0],setup[2]-setup[0]],stop_after_completion_ns=[setup[4]-setup[3],setup[5]-setup[3]],
        exact_wakeup_times=False,off_CPU_is_not_all_runqueue=True,uncovered_boundaries_remain_unknown=True)


def clip(stream,start,end):
    if end<start:raise ValueError('negative diagnostic window')
    on=[{**r,'start_ns':max(start,r['start_ns']),'end_ns':min(end,r['end_ns'])} for r in stream['on'] if intersection(start,end,r['start_ns'],r['end_ns'])]
    off=[{**r,'start_ns':max(start,r['start_ns']),'end_ns':min(end,r['end_ns'])} for r in stream['off'] if intersection(start,end,r['start_ns'],r['end_ns'])]
    run=sum(r['end_ns']-r['start_ns'] for r in on);away=sum(r['end_ns']-r['start_ns'] for r in off)
    unknown=end-start-run-away
    if unknown<0:raise ValueError('scheduler intervals double counted')
    return dict(on=on,off=off,on_CPU_ns=run,off_CPU_ns=away,unknown_ns=unknown,fully_covered=unknown==0,
        preempt_off_ns=sum(r['end_ns']-r['start_ns'] for r in off if r['preempt']),
        nonpreempt_off_ns=sum(r['end_ns']-r['start_ns'] for r in off if not r['preempt']))


class Attribution:
    def __init__(self,streams):
        self.streams=streams;self.cores={}
        for core in range(4):
            rows=sorted([r for stream in streams.values() for r in stream['on'] if r['cpu']==core],key=lambda r:r['start_ns'])
            maximum=[];end=0
            for row in rows:end=max(end,row['end_ns']);maximum.append(end)
            self.cores[core]=(rows,maximum)

    def other_work(self,off,pinned_core):
        if off['from_cpu']!=pinned_core or off['to_cpu']!=pinned_core:raise ValueError('fixed worker migrated')
        rows,ends=self.cores[pinned_core];a,b=off['start_ns'],off['end_ns'];totals=defaultdict(int)
        i=bisect_right(ends,a)
        while i<len(rows) and rows[i]['start_ns']<b:
            row=rows[i]
            if row['tid']!=off['tid']:
                name=self.streams[str(row['tid'])]['role']['role'];totals[name]+=intersection(a,b,row['start_ns'],row['end_ns'])
            i+=1
        used=sum(totals.values());excess=max(0,used-(b-a))
        if excess>100000:raise ValueError('competing actor intervals overlap excessively')
        return dict(by_role_ns=dict(totals),unobserved_ns=max(0,b-a-used),overlap_uncertainty_ns=excess,
            coincident_CPU_work_not_proof_of_all_wait_cause=True)
