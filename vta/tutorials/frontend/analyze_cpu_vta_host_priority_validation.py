"""Priority causal contrast with boot-paired means and explicit worker identities."""
import statistics as st
from analyze_cpu_vta_driver_snapshot_validation import analyze


def verify_priority(rows, nice, initialization, affinity):
    if nice not in (0,-10):raise ValueError('Unregistered priority')
    if [r['frame_id'] for r in rows]!=list(range(len(rows))):raise ValueError('Missing/reordered frames')
    if initialization['requested_nice']!=nice:raise ValueError('Initialization request drift')
    callers=[x['caller_tid'] for x in affinity['stages']]
    if len(callers)!=3 or len(set(callers))!=3 or initialization['vta_tid']!=callers[1]:
        raise ValueError('Distinct stage worker identity failure')
    tasks=initialization['threads'];tids=[x['tid'] for x in tasks]
    if len(set(tids))!=len(tids) or not tasks:raise ValueError('Duplicate or missing task audit')
    if not set(t for s in affinity['stages'] for t in s['thread_ids']).issubset(tids):
        raise ValueError('CPU pool task absent from initialization audit')
    for t in tasks:
        if type(t['tid'])!=int or t['tid']<=0 or t['nice']!=(nice if t['tid']==callers[1] else 0):
            raise ValueError('Initialization actual nice drift')
    for r in rows:
        for i in range(3):
            x=r.get(f'stage{i}_host_priority')
            if x!={'tid':callers[i],'nice':nice if i==1 else 0}:raise ValueError('Per-frame actual nice or TID drift')
        snap=r.get('stage1_driver_snapshot')
        if snap is not None and snap['tid']!=callers[1]:raise ValueError('Snapshot/priority worker identity mismatch')
    return {'passed':True,'frame_checks':len(rows)*3,'initialization_tasks':len(tasks),
            'callers':callers,'requested_vta_nice':nice,'CPU_pool_workers_nice':0}


def summarize(sessions):
    perf=[r for r in sessions if r['mode']=='pipeline']
    bootids=sorted({r['boot_index'] for r in perf});cal=[];pairs=[];blocks=[]
    for boot in bootids:
        for block in range(2):
            arms={(r['condition'],r['driver_snapshots_enabled']):r for r in perf if r['boot_index']==boot and r['block'].startswith(f'pair{block}_')}
            for name in ('N0','N1'):
                if (name,True) in arms and (name,False) in arms:
                    on,off=arms[name,True],arms[name,False]
                    cal.append({'boot':boot,'block':block,'condition':name,
                        'ii_change_fraction':on['metrics']['ii_ms']/off['metrics']['ii_ms']-1,
                        'vta_run_change_fraction':on['gate_metrics']['stages'][1]['run_mean_ms']/off['gate_metrics']['stages'][1]['run_mean_ms']-1})
            for enabled in (False,True):
                if ('N0',enabled) not in arms or ('N1',enabled) not in arms:continue
                a,b=arms['N0',enabled],arms['N1',enabled]
                ar=a['gate_metrics']['stages'][1]['run_mean_ms'];br=b['gate_metrics']['stages'][1]['run_mean_ms']
                x={'boot':boot,'block':block,'snapshots':enabled,
                   'vta_run_N0_ms':ar,'vta_run_N1_ms':br,'vta_run_reduction_ms':ar-br,
                   'vta_run_reduction_fraction':1-br/ar,'ii_N0_ms':a['metrics']['ii_ms'],'ii_N1_ms':b['metrics']['ii_ms'],
                   'ii_reduction_ms':a['metrics']['ii_ms']-b['metrics']['ii_ms'],
                   'front_run_reduction_ms':a['gate_metrics']['stages'][0]['run_mean_ms']-b['gate_metrics']['stages'][0]['run_mean_ms'],
                   'tail_run_reduction_ms':a['gate_metrics']['stages'][2]['run_mean_ms']-b['gate_metrics']['stages'][2]['run_mean_ms']}
                if enabled:
                    ma,mb=a['snapshot_metrics']['means'],b['snapshot_metrics']['means']
                    x.update(runqueue_N0_ms=ma['sched_runqueue_ms'],runqueue_N1_ms=mb['sched_runqueue_ms'],
                        runqueue_reduction_ms=ma['sched_runqueue_ms']-mb['sched_runqueue_ms'],
                        driver_total_reduction_ms=ma['driver_total_ms']-mb['driver_total_ms'],
                        host_cpu_reduction_ms=ma['thread_cpu_ms']-mb['thread_cpu_ms'])
                blocks.append(x)
        for enabled in (False,True):
            xs=[x for x in blocks if x['boot']==boot and x['snapshots']==enabled]
            if len(xs)==2:pairs.append({'boot':boot,'snapshots':enabled,**{k:st.mean(x[k] for x in xs) for k in xs[0] if k not in ('boot','block','snapshots')}})
    calibration=len(cal)==len(bootids)*4 and bool(cal) and max(max(abs(x['ii_change_fraction']),abs(x['vta_run_change_fraction'])) for x in cal)<=.02
    signal=calibration and bool(pairs) and all(x['vta_run_reduction_fraction']>=.05 for x in pairs) and all(x['runqueue_reduction_ms']>0 for x in pairs if x['snapshots'])
    ci={}
    if len(bootids)==3:
        for enabled in (False,True):
            xs=[x for x in pairs if x['snapshots']==enabled]
            if len(xs)==3:
                values=[x['vta_run_reduction_ms'] for x in xs];mean=st.mean(values);margin=4.302652729911275*st.stdev(values)/(3**.5)
                ci['on' if enabled else 'off']={'mean_reduction_ms':mean,'95_percent_CI_ms':[mean-margin,mean+margin],'df':2,'sampling_unit':'kernel boot paired block means'}
    return {'boots':len(bootids),'sessions':len(sessions),'calibration_pairs':cal,'calibration_passed':calibration,
            'block_pairs':blocks,'boot_pairs':pairs,'qualified_pilot_signal':signal,'boot_level_CI':ci,
            'performance_source':'snapshot off','complete_boots':len(sessions)//13,
            'mechanism_counter_scope':'rq and driver overlap; do not sum. Priority changes CPU scheduling and may change memory traffic.'}
