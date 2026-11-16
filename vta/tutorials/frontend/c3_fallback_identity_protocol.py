"""Pure F/T/U comparison of two YOLO224 complete operators; no target execution.

One unique program consumes ten fresh sessions of six calls, sixty total.
Structural aliases are declared with an exact, already certified execution
identity; aliases never produce fictitious zero-percent timing samples.
"""
from collections import Counter
import json
import math
from statistics import median, pstdev

SEEDS = (0, 20250901, 20260910)
ROUND_ORDERS = (('F','T','U'), ('T','U','F'), ('U','F','T'),
                ('U','T','F'), ('T','F','U'), ('F','U','T'))


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
        require(set(group['programs'])=={'F','T','U'},'EXACT_F_T_U_REQUIRED')
        require(group['kind']=='operator','ONLY_TWO_OPERATORS')
        representatives[oid]=['F','T','U']
    require([g['object_id'] for g in pool] == ['yolo224_node34','yolo224_node46'], 'EXACT_TWO_NODE_POOL')
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
                variants=ROUND_ORDERS[round_id-1]
                for variant in variants:
                    specs=[(s,'warmup','W0') for s in seeds]+[(s,'formal','W0') for s in seeds]
                    add(group,variant,'formal',rotation,round_id,specs,'minimal')
    unique=sum(map(len,representatives.values()))
    return {'schema':'c3_yolo224_fallback_TPS_identity_v1','attempt':attempt,'pool':pool,
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


def analyse(ledger, results):
    validate(ledger,results)
    samples={(r['object_id'],r['variant'],r['round'],r['seed']):r['elapsed_ms']
             for r in results if r['timed']}
    reports=[]
    for group in ledger['pool']:
        oid=group['object_id']
        m={v:[median(samples[oid,v,r,s] for s in SEEDS) for r in range(1,7)] for v in ('F','T','U')}
        pairs={}
        for candidate, baseline in (('T','F'), ('U','T'), ('U','F')):
            gains=[100*(1-c/b) for c,b in zip(m[candidate],m[baseline])]
            absolute=[b-c for c,b in zip(m[candidate],m[baseline])]
            center=median(gains)
            pairs[candidate+'/'+baseline]={'Gain_percent':center,'round_gains_percent':gains,
                'strict_wins':sum(c<b for c,b in zip(m[candidate],m[baseline])),
                'median_absolute_ms':median(absolute),'round_absolute_ms':absolute,
                'range_percent':[min(gains),max(gains)],
                'MAD_percent':median(abs(x-center) for x in gains)}
        reports.append({'object_id':oid,'T_ms':{v:median(x) for v,x in m.items()},
            'round_medians_ms':m,'pairwise':pairs,
            'dispersion_ms':{v:{'range':[min(x),max(x)],'stdev':pstdev(x),
                              'MAD':median(abs(y-median(x)) for y in x)} for v,x in m.items()}})
    return {'status':'COMPLETE_F_T_U_SINGLE_BOOT_COMPARISON','objects':reports,
            'target_calls':len(results),'formal_samples':ledger['formal_samples'],
            'operator_inverse_is_not_network_FPS':True,'statistical_significance_claimed':False}


def transport_contract(ledger, base_port=19000):
    require(canonical(ledger)==canonical(manifest(ledger['attempt'],ledger['pool'])), 'INVALID_COMPARISON_LEDGER')
    require(type(base_port) is int and 1024 <= base_port <= 32000, 'INVALID_FROZEN_PORT_BASE')
    sessions=ledger['sessions']
    require(len(sessions)==60 and base_port+len(sessions)<65536, 'FIXED_SESSION_COUNT')
    return {'schema':'c3_fallback_TPS_identity_transport_v1','base_port':base_port,
        'sessions':[{'ordinal':i,'session_id':s['session_id'],'port':base_port+i-1}
                    for i,s in enumerate(sessions,1)],
        'rpc_connect_attempts_per_session':1,'target_retries':0,'port_reuse_within_attempt':False,
        'shutdown':'release every RPC-backed reference, close connection explicitly, stop owned server only'}
