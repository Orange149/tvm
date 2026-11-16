"""Strict task-scoped SWITCH records; off-CPU is not synonymous with runqueue."""
from collections import Counter
import struct

OUT=1<<13
PREEMPT=1<<14


def decode(stream):
    if stream.get('error'):raise ValueError('native stream error: '+stream['error'])
    if stream['overflow']:raise ValueError('perf ring overflow')
    if not 0<=stream['tail']<=stream['head'] or stream['head']-stream['tail']>stream['data_size']:raise ValueError('invalid ring head/tail')
    try:raw=bytes.fromhex(stream['records_hex'])
    except ValueError as e:raise ValueError('invalid ring bytes') from e
    if len(raw)!=stream['head']-stream['tail']:raise ValueError('ring byte count mismatch')
    rows=[];pos=0;previous=stream['enable_start_ns']
    while pos<len(raw):
        if len(raw)-pos<8:raise ValueError('truncated perf header')
        kind,misc,size=struct.unpack_from('<IHH',raw,pos)
        if size<8 or size%8 or pos+size>len(raw):raise ValueError('truncated or invalid perf record')
        if kind in (2,13):raise ValueError('lost perf records')
        if kind!=14 or size!=32:raise ValueError('unexpected record type/layout; no silent skipping')
        pid,tid,when,cpu,reserved=struct.unpack_from('<IIQII',raw,pos+8)
        if (pid,tid)!=(stream['pid'],stream['tid']):raise ValueError('wrong task identity in switch record')
        if cpu not in (0,1) or reserved or not previous<=when<=stream['disable_end_ns']:raise ValueError('CPU or timestamp outside native envelope')
        if misc&PREEMPT and not misc&OUT:raise ValueError('preemption flag on switch-in')
        rows.append(dict(tid=tid,time_ns=when,cpu=cpu,direction='out' if misc&OUT else 'in',preempt=bool(misc&PREEMPT),misc=misc))
        previous=when;pos+=size
    return rows


def intervals(stream,events):
    start,end=stream['enable_start_ns'],stream['disable_end_ns']
    first_core=stream['segments'][0]['actual_start_core']
    running=not events or events[0]['direction']=='out'
    core=events[0]['cpu'] if running and events else first_core
    cursor=start;on=[];off=[];left=None
    for e in events:
        if e['direction']=='out':
            if not running or core!=e['cpu']:raise ValueError('duplicate switch-out or inconsistent running CPU')
            on.append(dict(start_ns=cursor,end_ns=e['time_ns'],cpu=core));running=False;left=e
        else:
            if running:raise ValueError('duplicate switch-in')
            if left is not None:off.append(dict(start_ns=left['time_ns'],end_ns=e['time_ns'],preempt=left['preempt'],from_cpu=left['cpu'],to_cpu=e['cpu']))
            running=True;core=e['cpu']
        cursor=e['time_ns']
    if running:on.append(dict(start_ns=cursor,end_ns=end,cpu=core))
    # Initial/final incomplete off-CPU spans are intentionally not categorized.
    return on,off


def intersection(a,b,c,d):return max(0,min(b,d)-max(a,c))


def validate_stream(stream):
    bounds=[stream[k] for k in ('enable_start_ns','enable_end_ns','disable_start_ns','disable_end_ns')]
    if any(type(v) is not int or v<=0 for v in bounds) or bounds!=sorted(bounds):raise ValueError('invalid perf enable/disable clock')
    if stream['policy']!=0:raise ValueError('controlled test is not SCHED_OTHER')
    events=decode(stream);on,off=intervals(stream,events);segments=[]
    for s in stream['segments']:
        a,b=s['wall_start_ns'],s['wall_end_ns'];cpu=s['cpu_end_ns']-s['cpu_start_ns'];core=s['requested_core']
        if not bounds[1]<=a<b<=bounds[2] or cpu<0 or cpu>b-a+10000:raise ValueError('invalid workload clocks')
        if s['actual_start_core']!=core or s['actual_end_core']!=core:raise ValueError('controlled affinity differs')
        if any(intersection(a,b,r['start_ns'],r['end_ns']) and r['cpu']!=core for r in on):raise ValueError('switch CPU differs from fixed segment')
        run=sum(intersection(a,b,r['start_ns'],r['end_ns']) for r in on)
        if abs(run-cpu)>max(2000000,int(.02*cpu)):raise ValueError('switch timeline CPU does not match thread CPU clock')
        segments.append(dict(**s,reconstructed_CPU_ns=run,CPU_clock_ns=cpu,difference_ns=run-cpu,
            off_CPU_wall_ns=b-a-run,off_CPU_is_not_all_runqueue_delay=True))
    return dict(case=stream['case'],index=stream['index'],tid=stream['tid'],events=events,on_CPU=on,off_CPU=off,
        switches=len(events),preempt_out=sum(e['direction']=='out' and e['preempt'] for e in events),
        nonpreempt_out=sum(e['direction']=='out' and not e['preempt'] for e in events),segments=segments)


def check(document):
    if document.get('status')!='MEASURED':return dict(passed=False,reason='native event interface unavailable, not zero waiting')
    if document.get('DNN_inferences')!=0 or document.get('periodic_sampling') is not False or document.get('byte_order')!='little' or document.get('sample_id_layout')!='TID,TIME,CPU':
        raise ValueError('unexpected recording protocol')
    streams=document['streams'];expected=[('alone',0),('contended',0),('contended',1),('sleep',0),('migrate',0)]
    if [(s['case'],s['index']) for s in streams]!=expected:raise ValueError('controlled event cases incomplete')
    if len({s['tid'] for s in streams})!=len(streams):raise ValueError('TID reused across finite cases; preserve distinct stream identity')
    rows=[validate_stream(s) for s in streams]
    for row in rows:
        kinds=[s['kind'] for s in row['segments']]
        want=['busy','busy'] if row['case']=='migrate' else ['sleep'] if row['case']=='sleep' else ['busy']
        if kinds!=want:raise ValueError('wrong workload segment coverage')
        if row['case'] in ('alone','contended') and row['segments'][0]['CPU_clock_ns']<100000000:raise ValueError('busy positive control insufficient')
    busy=[r for r in rows if r['case']=='contended'];preemption=all(r['preempt_out']>0 for r in busy)
    same_core_overlap=0
    sa=busy[0]['segments'][0];sb=busy[1]['segments'][0]
    for a in busy[0]['on_CPU']:
        for b in busy[1]['on_CPU']:
            if a['cpu']==b['cpu']:
                same_core_overlap=max(same_core_overlap,intersection(max(a['start_ns'],sa['wall_start_ns']),min(a['end_ns'],sa['wall_end_ns']),
                    max(b['start_ns'],sb['wall_start_ns']),min(b['end_ns'],sb['wall_end_ns'])))
    sleep=next(r for r in rows if r['case']=='sleep');s=sleep['segments'][0]
    sleeping_out=any(not e['preempt'] and intersection(s['wall_start_ns'],s['wall_end_ns'],e['start_ns'],e['end_ns'])>=100000000 for e in sleep['off_CPU'])
    sleep_not_CPU=s['CPU_clock_ns']<10000000 and s['reconstructed_CPU_ns']<10000000
    migration=next(r for r in rows if r['case']=='migrate')
    migrated=([s['requested_core'] for s in migration['segments']]==[0,1] and
        all(s['CPU_clock_ns']>=50000000 for s in migration['segments']) and
        any(e['from_cpu']==0 and e['to_cpu']==1 for e in migration['off_CPU']))
    return dict(passed=preemption and same_core_overlap<=100000 and sleeping_out and sleep_not_CPU and migrated,
        preemption_positive_control=preemption,max_same_core_TID_overlap_ns=same_core_overlap,
        known_sleep_nonpreempt_event=sleeping_out,sleep_not_CPU=sleep_not_CPU,migration_cpu_records=migrated,
        streams=rows,total_switch_records=sum(r['switches'] for r in rows),
        exact_wakeup_instants_available=False,nonpreempt_out_does_not_always_mean_sleep=True,
        no_lost_records=True,no_periodic_sampling=True)
