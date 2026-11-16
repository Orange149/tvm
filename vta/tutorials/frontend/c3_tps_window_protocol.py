"""Pure fixed V4 B/N/G call ledger and paired analysis; no tool/board access."""
from collections import Counter
from statistics import median, pstdev


SEEDS=(0,20250901,20260910)
BASIS=('TPS_B','TPS_N','TPS_G','Q1','Q8','official_B','official_N','official_G')


def manifest(programs,attempt):
    if not programs or len(programs)>8 or len(set(programs))!=len(programs) or not set(programs)<=set(BASIS):
        raise ValueError('FIXED_PROGRAM_POOL_INVALID')
    programs=[p for p in BASIS if p in programs]
    calls=[];sessions=[]
    def session(phase,program,rotation,round_id,specs,mode):
        sid=f'{phase}:{len(sessions):03d}'
        sessions.append({'session':sid,'phase':phase,'program':program,'mode':mode,
                         'rotation':rotation,'round':round_id,'calls':len(specs)})
        for position,(seed,role,timed,weight_variant) in enumerate(specs,1):
            calls.append({'attempt':attempt,'session':sid,'round':round_id,
                'operator':'official' if program.startswith('official') else 'node24_224',
                'program':program,'variant':program.split('_')[-1],
                'seed':seed,'phase':role,'call_position':position,'call_index':len(calls),
                'timed':timed,'formal_sample':role=='formal',
                'helper_kind':'timer' if timed else 'diagnostic',
                'collection_mode':mode,'weight_variant':weight_variant})
    for program in programs:
        specs=[(s,'qualification',True,'W0') for s in SEEDS]
        specs += [(SEEDS[0],'weight_W0_W1_W0',True,w) for w in ('W0','W1','W0')]
        session('qualification',program,None,None,specs,'detailed')
    for rotation in range(3):
        seeds=SEEDS[rotation:]+SEEDS[:rotation]
        for program in programs:
            specs=[(s,'diagnostic_prefix',False,'W0') for s in seeds]
            specs += [(s,'warm_diagnostic',False,'W0') for s in seeds]
            session('diagnostic',program,rotation,None,specs,'detailed')
    orders=[]
    for r in range(6):
        order=programs[r%len(programs):]+programs[:r%len(programs)] if r<5 else list(reversed(programs))
        orders.append(order)
        rotation=r%3;seeds=SEEDS[rotation:]+SEEDS[:rotation]
        for program in order:
            specs=[(s,'explicit_warmup',True,'W0') for s in seeds]
            specs += [(s,'formal',True,'W0') for s in seeds]
            session('formal',program,rotation,r+1,specs,'minimal')
    if len(calls)!=60*len(programs) or sum(c['formal_sample'] for c in calls)!=18*len(programs):
        raise ValueError('CALL_BUDGET')
    return {'schema':'c3_tps_window_fixed_call_manifest_v1','programs':programs,
            'attempt':attempt,'calls':calls,'sessions':sessions,'orders':orders,
            'phase_counts':dict(Counter(c['phase'] for c in calls)),
            'total_target_calls':len(calls),'implicit_calls_allowed':0}


def analyse(ledger,results):
    if len(results)!=len(ledger['calls']):raise ValueError('INCOMPLETE_ATTEMPT_NO_PERFORMANCE_RESULT')
    data={}
    for expected,actual in zip(ledger['calls'],results):
        for key in ('attempt','session','round','program','seed','phase','call_position','call_index','timed','formal_sample','helper_kind','collection_mode','weight_variant'):
            if actual.get(key)!=expected[key]:raise ValueError('CALL_MANIFEST_MISMATCH:'+key)
        if actual.get('status')!='PASS' or actual.get('target_calls')!=1:
            raise ValueError('FAILED_OR_UNACCOUNTED_TARGET_CALL')
        if expected['formal_sample']:
            value=actual.get('latency_ms')
            if not isinstance(value,(int,float)) or not value>0:raise ValueError('INVALID_FORMAL_TIME')
            data[(expected['program'],expected['round'],expected['seed'])]=value
    medians={p:[median(data[(p,r,s)] for s in SEEDS) for r in range(1,7)] for p in ledger['programs']}
    stats={p:{'round_medians_ms':v,'T_ms':median(v),'range_ms':[min(v),max(v)],
              'population_stdev_ms':pstdev(v),'MAD_ms':median(abs(x-median(v)) for x in v)} for p,v in medians.items()}
    comparisons={}
    pairs=[('Q8','Q1'),('Q8','TPS_B'),('Q8','TPS_G')]
    for obj in ('TPS','official'):
        pairs += [(obj+'_N',obj+'_B'),(obj+'_G',obj+'_N'),(obj+'_G',obj+'_B')]
    for candidate,baseline in pairs:
        if candidate not in medians or baseline not in medians:continue
        cv,bv=medians[candidate],medians[baseline]
        gains=[100*(1-a/b) for a,b in zip(cv,bv)]
        wins=sum(a<b for a,b in zip(cv,bv))
        comparisons[f'{candidate};{baseline}']={'round_gains_percent':gains,
            'Gain_percent':median(gains),'wins':wins,'paired_absolute_saving_ms':median(b-a for a,b in zip(cv,bv)),
            'engineering_goal_met':median(gains)>=2 and wins>=5,
            'statistical_significance_claimed':False,'independent_boot_confirmation':False}
    return {'status':'COMPLETE_SINGLE_BOOT_PAIRED_RESULTS','programs':stats,'comparisons':comparisons}
