"""Fixed mask contrast; matched service and bootstrap units remain separate."""
import statistics as st
from analyze_cpu_vta_all_stage_gate_validation import IDENTITY,effect


def runs(r):
    if r['mode']=='serial':return [s['run_ms']['mean'] for s in r['metrics']['stages']]
    return [s['run_mean_ms'] for s in r['gate_metrics']['stages']]


def matching(a,b,allow_affinity=False):
    for k in IDENTITY:
        if k=='affinity' and allow_affinity:continue
        if a.get(k) is None or a[k]!=b.get(k):raise ValueError('Identity drift '+k)


def serial_drift(before,after):
    fractions={}
    for n in ('S','D'):
        matching(before[n],after[n]);fractions[n]=[y/x-1 for x,y in zip(runs(before[n]),runs(after[n]))]
    return {'fractions':fractions,'passed':max(abs(v) for vs in fractions.values() for v in vs)<=.05}


def dedicated_inflation(before,after,performance):
    matching(before,after)
    if before['condition']!='D' or before['mode']!='serial' or after['mode']!='serial' or len(performance)!=2:
        raise ValueError('Two dedicated off blocks and matched serial required')
    ref=[st.mean(v) for v in zip(runs(before),runs(after))]
    for r in performance:
        matching(before,r)
        if r['condition']!='D' or r['driver_snapshots_enabled'] or r['gate_scope']!='cpu':raise ValueError('Diagnostic identity')
    observed=[st.mean(v) for v in zip(*(runs(r) for r in performance))]
    fractions=[y/x-1 for x,y in zip(ref,observed)]
    return {'serial_run_reference_ms':ref,'pipeline_run_mean_ms':observed,'inflation_fractions':fractions,
            'trigger':max(fractions[0],fractions[2])>.05,'threshold':.05,'scope':'CPU run, dedicated mask, snapshots off'}


def summarize(sessions):
    if len({r['session_id'] for r in sessions})!=len(sessions):raise ValueError('Duplicate session')
    perf=[r for r in sessions if r['mode']=='pipeline'];boots=sorted({r['boot_index'] for r in perf})
    blocks=[];cal=[];coactivity=[];references=[]
    for boot in boots:
        group=[r for r in sessions if r['boot_index']==boot]
        if len({r['boot_id'] for r in group})!=1:raise ValueError('Boot identity drift')
        for i in range(2):
            arms={(r['condition'],r['driver_snapshots_enabled']):r for r in group if r['mode']=='pipeline' and r['block'].startswith(f'pair{i}_')}
            if len(arms)!=4 or set(arms)!={(n,e) for n in ('S','D') for e in (False,True)}:raise ValueError('Incomplete balanced primary block')
            for n in ('S','D'):
                on,off=arms[n,True],arms[n,False];matching(on,off)
                cal.append({'boot':boot,'block':i,'condition':n,'ii_change_fraction':on['metrics']['ii_ms']/off['metrics']['ii_ms']-1,
                            'run_change_fractions':[y/x-1 for x,y in zip(runs(off),runs(on))]})
            for enabled in (False,True):
                a,b=arms['S',enabled],arms['D',enabled];matching(a,b,True)
                if a['affinity']!=[[0,1,2]]*3 or b['affinity']!=[[0,1,2],[3],[0,1,2]]:raise ValueError('Unregistered masks')
                ar,br=runs(a),runs(b)
                row={'boot':boot,'block':i,'snapshots':enabled,'run_S_ms':ar,'run_D_ms':br,
                     'run_reduction_ms':[x-y for x,y in zip(ar,br)],'run_reduction_fractions':[1-y/x for x,y in zip(ar,br)],
                     'ii_S_ms':a['metrics']['ii_ms'],'ii_D_ms':b['metrics']['ii_ms'],
                     'ii_reduction_ms':a['metrics']['ii_ms']-b['metrics']['ii_ms'],
                     'CPU_overlap_ms':[a['gate_metrics']['cpu_stage_overlap_ms'],b['gate_metrics']['cpu_stage_overlap_ms']],
                     'CPU_VTA_overlap_ms':[sum(r['gate_metrics']['pairwise_stage_overlap_ms'][k] for k in ('0:1','1:2')) for r in (a,b)]}
                if enabled:row['counter_means']={n:r['snapshot_metrics']['means'] for n,r in [('S',a),('D',b)]}
                blocks.append(row)
            arms2={r['condition']:r for r in group if r['mode']=='pipeline' and r['block']==f'coactivity{i}_off'}
            if arms2:
                if set(arms2)!={'D','D_all'}:raise ValueError('Incomplete coactivity pair')
                coactivity.append({'boot':boot,'block':i,**effect(arms2['D'],arms2['D_all'])})
        for n in ('S','D'):
            baseline=[r for r in group if r['mode']=='serial' and r['condition']==n]
            by={r['block']:r for r in baseline}
            if not {'matched_before','matched_checkpoint'}.issubset(by):raise ValueError('Missing matched serial')
            a,b=by['matched_before'],by.get('matched_after',by['matched_checkpoint']);matching(a,b)
            service=[st.mean([a['metrics']['stages'][i]['busy_ms']['mean'],b['metrics']['stages'][i]['busy_ms']['mean']]) for i in range(3)]
            references.append({'boot':boot,'condition':n,'matched_service_ms':service,
                               'matched_resource_bound_ms':max(service[0]+service[2],service[1]),
                               'assumption':'matched serial service transfers; CPU gate serializes CPU stages',
                               'serial_outer_ms':st.mean(r['metrics']['outer_residual_ms']['mean'] for r in (a,b))})
    if len({r['boot_id'] for r in sessions})!=len(boots):raise ValueError('Kernel boots are not independent')
    bootpairs=[]
    for boot in boots:
        xs=[r for r in blocks if r['boot']==boot and not r['snapshots']]
        bootpairs.append({'boot':boot,'run_reduction_ms':[st.mean(r['run_reduction_ms'][i] for r in xs) for i in range(3)],
                          'run_reduction_fractions':[st.mean(r['run_reduction_fractions'][i] for r in xs) for i in range(3)],
                          'ii_reduction_ms':st.mean(r['ii_reduction_ms'] for r in xs),
                          'trace_passed':all(max(r['CPU_overlap_ms'])==0 and min(r['CPU_VTA_overlap_ms'])>0 for r in xs)})
    ci={}
    if len(boots)==3:
        for i in range(3):
            values=[r['run_reduction_ms'][i] for r in bootpairs];mean=st.mean(values);half=4.302652729911275*st.stdev(values)/3**.5
            ci[str(i)]={'mean_ms':mean,'CI95_ms':[mean-half,mean+half],'df':2}
    return {'boots':len(boots),'sessions':len(sessions),'calibration_pairs':cal,
            'calibration_passed':bool(cal) and max(max(abs(x['ii_change_fraction']),*(abs(v) for v in x['run_change_fractions'])) for x in cal)<=.02,
            'block_pairs':blocks,'boot_pairs':bootpairs,'matched_serial_references':references,'coactivity_pairs':coactivity,
            'qualified_pilot_signal':bool(bootpairs) and any(abs(v)>=.05 for v in bootpairs[0]['run_reduction_fractions']) and bootpairs[0]['trace_passed'],
            'boot_level_CI':ci,'performance_source':'snapshots off','inference':'exploratory' if len(boots)<3 else 'boot paired df2 intervals',
            'unknown':['physical DDR/internal DMA','hardware compute-only','unique runqueue mediation'],
            'counter_scope':'driver and runqueue overlap; never add or subtract their totals from II'}
