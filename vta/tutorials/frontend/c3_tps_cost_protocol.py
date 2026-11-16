"""Frozen 72-call R-only instrumentation comparison; no target invocation."""
import math
from statistics import median, pstdev
from c3_tps_cost_probe import split_boundary, require

SEEDS = (0, 20250901, 20260910)
ORDERS = (('off','on'), ('on','off'), ('on','off'), ('off','on'))


def manifest(attempt, identities):
    sessions, calls = [], []
    def add(mode, phase, rnd=0):
        sid=attempt+':s%03d'%(len(sessions)+1)
        collection='detailed' if phase=='diagnostic' else 'minimal'
        sessions.append(dict(session_id=sid, program=mode, boundary_mode=mode,
                             phase=phase, round=rnd, collection=collection))
        for position in range(1,7):
            weight='W1' if phase=='qualification' and position==5 else 'W0'
            seed=SEEDS[(position-1)%3] if phase!='qualification' or position<=3 else 0
            kind=('qualification' if position<=3 else 'weight_check') if phase=='qualification' else (
                  ('diagnostic_prefix' if position<=3 else 'diagnostic_warm') if phase=='diagnostic' else
                  ('warmup' if position<=3 else 'formal'))
            calls.append(dict(attempt=attempt,session_id=sid,call_index=len(calls)+1,position=position,
                              program=mode,boundary_mode=mode,operator='R',collection=collection,
                              phase=kind,round=rnd,rotation=0,seed=seed,weight=weight,timed=kind=='formal',
                              program_identity=identities[mode]))
    for phase in ('qualification','diagnostic'):
        for mode in ('off','on'):add(mode,phase)
    for rnd,order in enumerate(ORDERS,1):
        for mode in order:add(mode,'measurement',rnd)
    return dict(schema='c3_tps_cost_board_72_v1',attempt=attempt,sessions=sessions,calls=calls,
                round_orders=[list(o) for o in ORDERS],target_calls=72,formal_samples=24,implicit_calls_allowed=0)


def check_manifest(ledger):
    require(len(ledger['calls'])==72,'CALL_BUDGET')
    identities={p:next(r['program_identity'] for r in ledger['calls'] if r['program']==p) for p in ('off','on')}
    # JSON normalizes tuples to lists.
    import json
    require(ledger==json.loads(json.dumps(manifest(ledger['attempt'],identities))),'FROZEN_LEDGER_CHANGED')


def transport(ledger):
    check_manifest(ledger)
    return {'schema':'c3_tps_cost_transport_v1','rpc_connect_attempts_per_session':1,
            'sessions':[{'ordinal':i,'session_id':s['session_id'],'boundary_mode':s['boundary_mode'],
                         'port':9600+i-1} for i,s in enumerate(ledger['sessions'],1)]}


def validate_results(ledger,rows):
    check_manifest(ledger)
    require(len(rows)==72,'MISSING_OR_EXTRA_CALL')
    for expected,row in zip(ledger['calls'],rows):
        require(all(row.get(k)==v for k,v in expected.items()),'ROW_ORDER_OR_IDENTITY_CHANGED')
        require(row.get('status')=='PASS' and row.get('target_calls')==1,'TARGET_CALL_NOT_PASSED')
        elapsed=row.get('elapsed_ms')
        if row['phase'].startswith('diagnostic_'):
            require(elapsed is None,'DIAGNOSTIC_NOT_A_TIMING_SAMPLE')
        else:
            require(isinstance(elapsed,(int,float)) and math.isfinite(elapsed) and elapsed>0,'INVALID_TIME')
        split=split_boundary(row['helper_boundary'],row['driver_boundary'],row['boundary_mode'],
                             row['position'],elapsed)
        require(row['boundary_split']==split,'SPLIT_RECORD_CHANGED')
    return True


def analyse(ledger,rows):
    validate_results(ledger,rows)
    formal=[r for r in rows if r['timed']]
    medians={p:[median(r['elapsed_ms'] for r in formal if r['program']==p and r['round']==i)
                for i in range(1,5)] for p in ('off','on')}
    delta=[100*(b/a-1) for a,b in zip(medians['off'],medians['on'])]
    absolute=[b-a for a,b in zip(medians['off'],medians['on'])]
    sensitive=abs(median(delta))>.5 or any(abs(x)>1 for x in delta)
    parts={}
    for field in ('before_driver_ms','driver_run_and_wait_ms','after_driver_ms'):
        samples=[r for r in formal if r['program']=='on']
        values=[r['boundary_split'][field] for r in samples]
        shares=[100*r['boundary_split'][field]/r['elapsed_ms'] for r in samples]
        parts[field]={'sample_ms':values,'sample_percent':shares,'median_ms':median(values),
                      'range_ms':[min(values),max(values)],'median_percent':median(shares)}
    return {'round_medians_ms':medians,'T_ms':{p:median(v) for p,v in medians.items()},
            'round_on_minus_off_percent':delta,'median_on_minus_off_percent':median(delta),
            'round_on_minus_off_ms':absolute,'median_on_minus_off_ms':median(absolute),
            'instrument_sensitivity_gate_passed':not sensitive,
            'not_statistical_equivalence':True,'pure_hardware_compute_measured':False,
            'on_formal_parts':parts,'target_calls':72,'formal_samples':24,
            'dispersion':{p:{'range_ms':[min(v),max(v)],'MAD_ms':median(abs(x-median(v)) for x in v),
                              'stddev_ms':pstdev(v)} for p,v in medians.items()}}
