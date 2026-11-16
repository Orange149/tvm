"""Unmeasured resource-layout validation of a frozen completion predictor."""
from copy import deepcopy
from pathlib import Path
import argparse,json,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from cpu_vta_partitioning.common import read,write,sha256,digest
from finite_service.common import freeze
from prepare_finite_service_holdout import collect_known,fingerprint
from policy_branch_eval.common import isolated_inputs_only
from cpu_policies.open_resources.domain import canonical_tuple
from device_completion_replay import CompletionFinite
import run_device_completion_replay as development

ROOT=development.ROOT
OUTPUT=ROOT/'artifacts/device_completion_holdout_01'
MODES=('aggregate','lower','midpoint','upper')


def sources():
    result=development.sources()
    for leaf in ('prepare_completion_holdout.py','prepare_finite_service_holdout.py',
                 'tests/test_completion_holdout.py'):
        result[str(ROOT/leaf)]=sha256(ROOT/leaf)
    return result


def choose(scores,configs):
    rankings={k:sorted(scores,key=lambda cid:(scores[cid][k],canonical_tuple(configs[cid]['configuration'])))
        for k in ('static','aggregate','midpoint','lower','upper')}
    selected=list(dict.fromkeys(rankings[k][0] for k in ('static','aggregate','midpoint')))
    if len(selected)<2:
        candidates=sorted(scores,key=lambda cid:(-max(abs(rankings[a].index(cid)-rankings[b].index(cid))
            for a in rankings for b in rankings),canonical_tuple(configs[cid]['configuration'])))
        for cid in candidates:
            if cid not in selected:selected.append(cid)
            if len(selected)>=min(2,len(configs)):break
    if len(selected)>3:raise ValueError('fresh completion validation exceeds three configurations per network')
    return rankings,selected


def prepare(output=OUTPUT):
    output=Path(output)
    if (output/'PLAN.json').exists():
        plan=read(output/'PLAN.json')
        if plan['sources']!=sources():raise ValueError('fresh completion predictor changed')
        return plan['counts']
    if not read(development.OUTPUT/'DELIVERY.json')['development_gate_passed']:
        raise ValueError('known-case development gate failed; no automatic native expansion')
    known=collect_known(output)
    pool=read(development.POOL/'ISOLATED_POOL.json')
    projected=read(development.OUTPUT/'INPUTS.json');templates={}
    for item in projected['cases'].values():
        for cfg,s in zip(item['configuration']['stages'],item['stages']):
            if cfg['device']=='vta':templates[item['network'],cfg['program']]=s
    cases={};counts={}
    for network,rows in pool['networks'].items():
        cases[network]={};excluded=0
        for cid,item in rows.items():
            if fingerprint(item['configuration']) in known:excluded+=1;continue
            new=deepcopy(item)
            for index,(cfg,s) in enumerate(zip(new['configuration']['stages'],new['stages'])):
                if cfg['device']=='vta':
                    t=templates[network,cfg['program']]
                    if s['mask']!=t['mask']:raise ValueError('new host mask lacks matching completion profile')
                    new['stages'][index]=deepcopy(t)
            cases[network][cid]=new
        if len(cases[network])<2:raise ValueError('insufficient genuinely unmeasured canonical layouts')
        counts[network]=dict(available=len(rows),known_excluded=excluded,unmeasured=len(cases[network]))
    freeze(output/'INPUTS.json',dict(networks=cases,window=pool['window'],quanta_ns=pool['quanta_ns'],
        contains_pipeline_performance=False,original_full_domain_unchanged=True,
        known_development_projection_sha256=sha256(development.OUTPUT/'INPUTS.json')))
    freeze(output/'PLAN.json',dict(sources=sources(),counts=counts,inputs_sha256=sha256(output/'INPUTS.json'),
        known_metadata_sha256=sha256(output/'KNOWN_EXECUTIONS.json'),
        predictor_plan_sha256=sha256(development.OUTPUT/'PLAN.json'),
        expected_boot_id=read(development.OUTPUT/'PLAN.json')['profile_boot'],
        primary='midpoint',baseline='static',matched_chronology_baseline='aggregate',modes=MODES,
        selection='union of static, aggregate and midpoint first choices; if fewer than two choices, add largest rank disagreement; at most three per network',
        max_configurations=6,max_native_sessions=12,max_DNN_inferences=174,
        qualification=dict(warmup=0,scored=1,tail=0),performance=dict(warmup=8,scored=16,tail=4),
        success='report same-session top1 and regret; both-network top1 not worse than original static and at least one improved is a positive observation, not statistical proof',
        scope='unmeasured resource layouts under two known fixed topologies, not full-domain Top20 or unseen DNNs',
        no_pipeline_performance_read=True,no_extra_profile=True,no_runtime_executor_change=True,
        no_reboot=True,no_anchors=True,no_automatic_repeats=True,execution_seed=20261111))
    return counts


def assert_plan(output):
    p=read(output/'PLAN.json')
    if p['sources']!=sources() or p['inputs_sha256']!=sha256(output/'INPUTS.json'):
        raise ValueError('fresh completion plan/input/source differs')
    if p['predictor_plan_sha256']!=sha256(development.OUTPUT/'PLAN.json'):
        raise ValueError('completion predictor was revised after development')
    return p


def rank(output=OUTPUT):
    output=Path(output);plan=assert_plan(output);inputs=read(output/'INPUTS.json')
    results={};selected={};files={};start=time.monotonic()
    for network,items in inputs['networks'].items():
        scores={}
        for number,(cid,item) in enumerate(items.items(),1):
            record=dict(static=item['static_score_ns'])
            for mode in MODES:
                path=output/'predictions'/network/(cid+'_'+mode+'.json')
                if not path.exists():
                    with isolated_inputs_only():
                        sim=CompletionFinite(item['stages'],quanta=inputs['quanta_ns'],placement='wake_boundary',
                            completion_model='aggregate' if mode=='aggregate' else 'independent',
                            completion_point='midpoint' if mode=='aggregate' else mode,
                            runnable_completion_wait=True,controls=None,gate=item['configuration']['execution_mode'],
                            host_order='before',**inputs['window']).run()
                    freeze(path,dict(input_sha256=digest(item),result=sim,
                        score_ns=max(item['static_score_ns'],sim['replay']['interval_ns']),
                        plan_sha256=sha256(output/'PLAN.json'),pipeline_performance_read=False))
                d=read(path)
                if d['plan_sha256']!=sha256(output/'PLAN.json') or d['input_sha256']!=digest(item):
                    raise ValueError('fresh completion prediction identity differs')
                record[mode]=d['score_ns'];files[str(path.relative_to(output))]=sha256(path)
            scores[cid]=record
            print('未测资源配置排名',network,number,'/',len(items),flush=True)
        rankings,ids=choose(scores,items);selected[network]=[items[cid]['configuration'] for cid in ids]
        results[network]=dict(rankings=rankings,scores=scores,selected_ids=ids)
    freeze(output/'SELECTION_FREEZE.json',dict(networks=results,selected_configurations=selected,
        plan_sha256=sha256(output/'PLAN.json'),prediction_files=files,
        no_pipeline_performance_read=True,selection_before_native=True,
        DNN_budget=sum(len(v) for v in selected.values())*29))
    write(output/'RANK_COST.json',dict(seconds=time.monotonic()-start),replace=True)
    return dict(counts={n:len(v) for n,v in selected.items()},DNN_budget=sum(len(v) for v in selected.values())*29)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','rank'))
    p.add_argument('--output',type=Path,default=OUTPUT);a=p.parse_args()
    print(json.dumps(globals()[a.command](a.output),ensure_ascii=False,indent=2))
