"""Pure TH/P operator and graph comparison ledger; no target execution.

One unique program consumes ten fresh sessions of six calls, sixty total.
Structural aliases are declared with an exact, already certified execution
identity; aliases never produce fictitious zero-percent timing samples.
"""
from collections import Counter
import json
import math
from statistics import median, pstdev

SEEDS = (0, 20250901, 20260910)


def require(ok, message):
    if not ok: raise ValueError(message)


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def manifest(attempt, pool):
    require(isinstance(attempt,str) and bool(attempt.strip()),'ATTEMPT_REQUIRED')
    require(isinstance(pool,list) and bool(pool),'POOL_REQUIRED')
    pool=json.loads(canonical(pool)); seen=set(); representatives={}
    for group in pool:
        oid=group['object_id'];require(oid not in seen,'DUPLICATE_OBJECT');seen.add(oid)
        require(group['kind'] in ('operator','graph'),'INVALID_TARGET_KIND')
        require(set(group['programs'])=={'TH','P'},'EXACT_TH_P_PAIR_REQUIRED')
        aliases=group.get('representatives',{'TH':'TH','P':'P'})
        require(aliases in ({'TH':'TH','P':'P'},{'TH':'TH','P':'TH'}),'INVALID_ALIAS')
        if aliases['P']=='TH':
            a,b=(group['programs'][v].get('execution_identity') for v in ('TH','P'))
            require(isinstance(a,dict) and bool(a) and canonical(a)==canonical(b),
                    'UNCERTIFIED_STRUCTURAL_ALIAS')
        representatives[oid]=[v for v in ('TH','P') if aliases[v]==v]
    calls=[];sessions=[]

    def add(group,variant,phase,rotation,round_id,specs,mode):
        oid=group['object_id'];sid=attempt+':s%04d'%(len(sessions)+1)
        first=len(calls)+1
        for position,(seed,role,weight) in enumerate(specs,1):
            calls.append({'call_index':len(calls)+1,'session_id':sid,'object_id':oid,
                'kind':group['kind'],'variant':variant,'program_key':oid+'/'+variant,
                'program_identity':group['programs'][variant],'phase':role,'round':round_id,
                'rotation':rotation,'seed':seed,'position':position,'collection':mode,
                'weight':weight,'timed':role=='formal'})
        sessions.append({'session_id':sid,'object_id':oid,'kind':group['kind'],'variant':variant,
            'phase':phase,'round':round_id,'rotation':rotation,'collection':mode,
            'first_call_index':first,'last_call_index':len(calls),'target_calls':len(specs)})

    for group in pool:
        for variant in representatives[group['object_id']]:
            specs=[(s,'qualification','W0') for s in SEEDS]
            specs += [(0,'weight_check',w) for w in ('W0','W1','W0')]
            add(group,variant,'qualification',None,None,specs,'detailed')
    for rotation in range(3):
        seeds=SEEDS[rotation:]+SEEDS[:rotation]
        for group in pool:
            for variant in representatives[group['object_id']]:
                specs=[(s,'diagnostic_prefix','W0') for s in seeds]
                specs += [(s,'diagnostic_warm','W0') for s in seeds]
                add(group,variant,'diagnostic',rotation,None,specs,'detailed')
    # Operators and the graph are separate comparison strata. A larger pool
    # must never move the graph into a selectively timed subset of layers.
    strata=[[g for g in pool if g['kind']==kind] for kind in ('operator','graph')]
    for round_id in range(1,7):
        rotation=(round_id-1)%3;seeds=SEEDS[rotation:]+SEEDS[:rotation]
        for stratum in strata:
            if not stratum:continue
            offset=(round_id-1)%len(stratum)
            order=(stratum[offset:]+stratum[:offset]) if round_id<6 else list(reversed(stratum))
            for group in order:
                variants=representatives[group['object_id']]
                if round_id%2==0:variants=list(reversed(variants))
                for variant in variants:
                    specs=[(s,'warmup','W0') for s in seeds]+[(s,'formal','W0') for s in seeds]
                    add(group,variant,'formal',rotation,round_id,specs,'minimal')
    unique=sum(map(len,representatives.values()))
    return {'schema':'c3_yolo416_TH_P_comparison_v1','attempt':attempt,'pool':pool,
        'representatives':representatives,'calls':calls,'sessions':sessions,
        'total_target_calls':60*unique,'formal_samples':18*unique,
        'unique_programs':unique,'phase_counts':dict(Counter(r['phase'] for r in calls)),
        'implicit_calls_allowed':0,'seeds':list(SEEDS)}


def validate(ledger, results):
    require(canonical(ledger)==canonical(manifest(ledger['attempt'],ledger['pool'])),
            'FROZEN_LEDGER_CHANGED')
    require(len(results)==len(ledger['calls']),'INCOMPLETE_ATTEMPT')
    for expected,actual in zip(ledger['calls'],results):
        require(all(k in actual and canonical(actual[k])==canonical(v)
                    for k,v in expected.items()),'CALL_IDENTITY_OR_ORDER_CHANGED')
        require(actual.get('status')=='PASS' and type(actual.get('target_calls')) is int
                and actual['target_calls']==1,'TARGET_NOT_QUALIFIED_ONCE')
        elapsed=actual.get('elapsed_ms')
        if elapsed is None and expected['phase'].startswith('diagnostic_'):continue
        require(type(elapsed) in (int,float) and math.isfinite(elapsed) and elapsed>0,
                'INVALID_ELAPSED_TIME')
    return True


def analyse(ledger,results):
    validate(ledger,results);report=[]
    samples={(r['object_id'],r['variant'],r['round'],r['seed']):r['elapsed_ms']
             for r in results if r['timed']}
    for group in ledger['pool']:
        oid=group['object_id'];variants=ledger['representatives'][oid]
        if variants==['TH']:
            report.append({'object_id':oid,'kind':group['kind'],
                'status':'STRUCTURALLY_IDENTICAL_NO_DIFFERENTIAL_MEASUREMENT'});continue
        m={v:[median(samples[oid,v,r,s] for s in SEEDS) for r in range(1,7)] for v in variants}
        gains=[100*(1-p/th) for p,th in zip(m['P'],m['TH'])]
        absolute=[th-p for p,th in zip(m['P'],m['TH'])]
        center=median(gains)
        report.append({'object_id':oid,'kind':group['kind'],'status':'MEASURED_FULL_CALL',
            'round_medians_ms':m,'T_ms':{v:median(x) for v,x in m.items()},
            'round_gains_percent':gains,'Gain_P_over_TH_percent':center,
            'strict_wins':sum(p<th for p,th in zip(m['P'],m['TH'])),
            'round_absolute_ms':absolute,'median_absolute_ms':median(absolute),
            'gain_range_percent':[min(gains),max(gains)],
            'gain_MAD_percent':median(abs(x-center) for x in gains),
            'round_dispersion_ms':{v:{'range':[min(x),max(x)],'population_stdev':pstdev(x),
                 'MAD':median(abs(y-median(x)) for y in x)} for v,x in m.items()}})
    return {'status':'COMPLETE_SINGLE_BOOT_COMPARISON','objects':report,
            'target_calls':len(results),'formal_samples':ledger['formal_samples'],
            'statistical_significance_claimed':False,'graph_time_is_sum_of_operators':False}
