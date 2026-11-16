"""Known-layout hotspot intervention, preceded by two isolated CPU references.

Reuses the unchanged native executor and validation/transport components.
No global search, model update, automatic repeats or reboot. CPU refresh and
all static predictions are frozen before the first full-network call.
"""
from copy import deepcopy
from pathlib import Path
from types import FunctionType
import argparse
import json
import random
import shlex
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from cpu_vta_partitioning.common import read,write,sha256,digest,locked,now,sync_tree
from finite_service.common import freeze
from policy_branch_eval.ranking import inputs
from policy_branch_eval.common import isolated_inputs_only
from cpu_policies.open_resources.costs import Costs,FIELDS
from staged_ablation.model import score
from phase_profile import runtime as phase
from yolo_validation.components import hardware_check
import finite_service_holdout_board as parent
from audit_three_thread_host_gap import OUT as POOL

ROOT=parent.ROOT
RUN=ROOT/'runs/three_thread_host_gap_01'
CONFIG=parent.CONFIG
OUTPUT=parent.OUTPUT


def protocol():
    p=parent.protocol();p.update(experiment='three_thread_host_gap_01',
        component=dict(warmup=2,scored=8,tail=0),CPU_stage_calls=22,
        only_local_pool_validation=False,known_layout_host_hotspot_diagnostic=True,
        new_components=2,new_DNN=87,max_native_sessions=10,
        predictions_before_first_pipeline_qualification=True,
        no_score_or_search_change=True,no_detailed_observer=True)
    return p


def sources():
    return {**parent.sources(),**{str(p):sha256(p) for p in (
        Path(__file__),ROOT/'tests/test_three_thread_host_gap.py',ROOT/'audit_three_thread_host_gap.py')}}


def selection():
    x=read(POOL/'ISOLATED_SELECTION.json');history=read(POOL/'HISTORY_AUDIT.json')
    if x['source_sha256']!=sha256(ROOT/'audit_three_thread_host_gap.py') or history['selection_sha256']!=sha256(POOL/'ISOLATED_SELECTION.json'):
        raise ValueError('isolated selection/history identity changed')
    if history['pipeline_host_pairs']!=['[[0, 1, 2, 3], [0, 1, 2, 3]]']:
        raise ValueError('matched fixed-host records already exist; review before measuring')
    if len(x['configurations'])!=3 or set(x['components'])!={'3','4'}:raise ValueError('diagnostic scope changed')
    return dict(selected_configurations={'yolov3_tiny416':x['configurations']},components=x['components'],
        roles=x['roles'],DNN_budget=87)


def specs():
    pairs=[c['id'] for c in selection()['selected_configurations']['yolov3_tiny416']]
    random.Random(protocol()['seed']).shuffle(pairs)
    rows=[dict(network='yolov3_tiny416',case=n,scope='component',phase=p,runner='phase',probe=False)
          for n in ('3','4') for p in ('qualify','clean')]
    rows += [dict(network='yolov3_tiny416',case=cid,scope='pipeline',phase=p,runner='phase',probe=False)
             for cid in pairs for p in ('qualify','clean')]
    return rows


def adapt(name):
    f=getattr(parent,name)
    g=FunctionType(f.__code__,{**f.__globals__,**globals()},f.__name__,f.__defaults__)
    g.__kwdefaults__=f.__kwdefaults__;return g


def prepare():
    # The reused packager stores a selected_configurations seal by this name.
    freeze(POOL/'SELECTION_FREEZE.json',dict(**selection(),sources=sources(),
        isolated_selection_sha256=sha256(POOL/'ISOLATED_SELECTION.json'),
        historical_audit_sha256=sha256(POOL/'HISTORY_AUDIT.json'),
        no_pipeline_FPS_read=True,CPU_stage_call_budget=22,max_native_sessions=10))
    return adapt('prepare')()


def validate(bundle):return adapt('validate')(bundle)
def qemu():return adapt('qemu')()


def make_task(bundle,spec):
    bundle=Path(bundle);m=read(bundle/'manifest.json')
    if spec['network']!=m['network'] or spec['runner']!='phase' or spec['probe']:
        raise ValueError('executor or observer mode differs')
    if spec['scope']=='component':cfg=selection()['components'][spec['case']]
    elif spec['scope']=='pipeline':cfg=read(bundle/'plans/configurations.json')['pipelines'][spec['case']]
    else:raise ValueError('unknown scope')
    if spec['phase'] not in ('qualify','clean'):raise ValueError('unknown phase')
    q=spec['phase']=='qualify';serial=spec['scope']=='component'
    kind=('qualify_component' if q else 'calibration') if serial else ('qualify_pipeline' if q else 'performance')
    doc=dict(kind=kind,configuration=deepcopy(cfg),stages=deepcopy(cfg['stages']),spec=deepcopy(spec),
        environment=protocol()['environment'],**protocol()['qualification' if q else 'component' if serial else 'pipeline'],
        input_indices=[0],executor_sha256=m['phase_runner_sha256'],adapter_sha256=None,
        runtime_sha256=m['runtime_sha256'],bundle_id=m['bundle_id'],experiment=protocol()['experiment'],
        startup_affinity_probe=True)
    doc['task_id']=digest(doc)[:24];return doc


def native_command(doc,bundle,remote_bundle,remote_job):return adapt('native_command')(doc,bundle,remote_bundle,remote_job)
def matched_ssh(config,*,execute_board=False):return adapt('matched_ssh')(config,execute_board=execute_board)
summary=parent.summary
accept=parent.accept


class RefreshedCosts(Costs):
    def __init__(self,*args,overrides=None,uniform=False,**kwargs):
        super().__init__(*args,**kwargs);self.overrides=overrides or {};self.uniform=uniform

    def get(self,row,context_value='shared4'):
        v=super().get(row,context_value)
        if row['device']=='cpu' and row['program']=='cpu_00_02' and str(row['threads']) in self.overrides:
            v.update({k:value for k,value in self.overrides[str(row['threads'])].items() if k in FIELDS})
            v['assumptions'].append('current_isolated_time_overlay_original_share_proxy_preserved')
        if self.uniform and row['device']=='cpu':
            v['cpu_core_weights']=[1./row['threads']]*row['threads']
            v['assumptions'].append('uniform_share_counterfactual_not_measured_thread_data')
        return v


def freeze_predictions(bundle,boot):
    projections={};proofs={}
    for n in ('3','4'):
        spec=next(s for s in specs() if s['scope']=='component' and s['case']==n and s['phase']=='clean')
        doc=make_task(bundle,spec);p=RUN/'accepted'/boot/(doc['task_id']+'.json');r=read(p)
        if r['task']!=doc or r['boot_id']!=boot or 'fps' in r['summary']:raise ValueError('component projection includes wrong scope')
        projections[n]=r['summary']['cost'];proofs[n]=dict(receipt=str(p),receipt_sha256=sha256(p),task_hash=digest(doc))
    freeze(POOL/'CURRENT_ISOLATED_CPU_ONLY.json',dict(costs=projections,proofs=proofs,boot_id=boot,
        only_isolated_component_values=True,no_pipeline_performance=True))
    results={}
    with isolated_inputs_only():
        programs,records,ddr,hashes,*_=inputs('yolov3_tiny416')
        if hashes!=read(POOL/'ISOLATED_SELECTION.json')['component_hashes']:raise ValueError('original costs changed')
        variants=dict(original=RefreshedCosts(records,programs=programs,ddr=ddr,distribution='session_weights'),
            refreshed=RefreshedCosts(records,programs=programs,ddr=ddr,distribution='session_weights',overrides=projections),
            refreshed_uniform=RefreshedCosts(records,programs=programs,ddr=ddr,distribution='session_weights',overrides=projections,uniform=True))
        for cfg in selection()['selected_configurations']['yolov3_tiny416']:
            results[cfg['id']]={label:score(cfg,cost,'concentrated') for label,cost in variants.items()}
            if results[cfg['id']]['original']!=read(POOL/'ISOLATED_SELECTION.json')['prior_predictions'][cfg['id']]:
                raise ValueError('original scalar score did not replay')
    freeze(POOL/'PIPELINE_PREDICTIONS_FREEZE.json',dict(results=results,
        CPU_projection_sha256=sha256(POOL/'CURRENT_ISOLATED_CPU_ONLY.json'),
        input_selection_sha256=sha256(POOL/'SELECTION_FREEZE.json'),
        original_scoring_source_sha256=sha256(ROOT/'staged_ablation/model.py'),
        no_pipeline_FPS_read=True,same_formula_all_variants=True,
        uniform_is_input_counterfactual_only=True,original_cost_records_unchanged=True))
    return results


def board(*,execute_board=False):
    if not execute_board:raise ValueError('--execute-board is required')
    p=read(POOL/'BOARD_PLAN.json')
    if p['sources']!=sources() or p['selection_sha256']!=sha256(POOL/'SELECTION_FREEZE.json'):
        raise ValueError('frozen controller differs')
    if not all(v['passed'] for v in read(POOL/'QEMU.json').values()):raise ValueError('QEMU check required')
    ssh=matched_ssh(read(CONFIG)['board'],execute_board=True);boot=ssh.boot_id()
    if boot!=p['expected_boot_id']:raise ValueError('boot changed; stop without mixing')
    bundle=POOL/'yolov3_tiny416/capsule';validate(bundle);started=time.monotonic()
    with locked(RUN):
        freeze(RUN/'run.json',dict(boot_id=boot,board_plan_sha256=sha256(POOL/'BOARD_PLAN.json'),protocol=protocol()))
        def state(status,**kw):write(RUN/'state.json',dict(status=status,boot_id=boot,updated_at=now(),**kw),replace=True)
        try:
            (RUN/'yolov3_tiny416').mkdir(parents=True,exist_ok=True)
            remote=ssh.deploy(bundle,RUN/'yolov3_tiny416')
            ssh.qualify_environment(remote,boot,RUN/'yolov3_tiny416',bundle)
            write(RUN/'hardware.json',hardware_check(ssh,bundle,boot),replace=True)
            expected={};count=CPU=0;todo=specs()
            for index,spec in enumerate(todo,1):
                if spec['scope']=='pipeline':freeze_predictions(bundle,boot)
                doc=make_task(bundle,spec);key=spec['scope'],spec['case']
                state('RUNNING',accepted_sessions=index-1,total_sessions=len(todo),current_spec=spec)
                if spec['phase']=='clean' and key not in expected:raise ValueError('measurement precedes qualification')
                item=accept(ssh,doc,bundle,remote,RUN,boot,expected.get(key))
                if spec['phase']=='qualify':expected[key]=item['summary']['output_fingerprints']
                calls=sum(doc[k] for k in ('warmup','scored','tail'))
                if spec['scope']=='pipeline':count+=calls
                else:CPU+=calls
                print('三线程主机竞争匹配',index,'/',len(todo),spec['scope'],spec['phase'],flush=True)
            if count!=87 or CPU!=22:raise ValueError('native call budget differs')
            ssh.idle_command('cd '+shlex.quote(remote)+' && sha256sum -c board.sha256 > three_thread_post_verify.log && sync',boot=boot,timeout=300)
            if ssh.boot_id()!=boot:raise ValueError('boot changed at end')
            state('COMPLETE',accepted_sessions=len(todo),DNN_inferences=count,CPU_stage_calls=CPU,
                seconds=time.monotonic()-started,no_reboot=True,no_automatic_extra_measurement=True)
            sync_tree(RUN)
        except Exception as e:
            state('STOPPED_ON_ERROR',error_type=type(e).__name__,error=str(e));raise
    return read(RUN/'state.json')


def analyze():
    from cpu_vta_partitioning.common import verify_inventory
    s=read(RUN/'state.json')
    if s['status']!='COMPLETE' or s['DNN_inferences']!=87 or s['CPU_stage_calls']!=22:raise ValueError('native batch incomplete')
    b=POOL/'yolov3_tiny416/capsule';rows=[]
    for spec in specs():
        if spec['phase']!='clean':continue
        doc=make_task(b,spec);p=RUN/'accepted'/s['boot_id']/(doc['task_id']+'.json');r=read(p)
        if r['task']!=doc or r['boot_id']!=s['boot_id']:raise ValueError('receipt changed')
        verify_inventory(r['raw'],r['files'])
        if summary(r['raw'],doc,b)!=r['summary']:raise ValueError('accepted summary no longer replays')
        rows.append(dict(scope=spec['scope'],case=spec['case'],role=selection()['roles'].get(spec['case']),
            measurement=r['summary'],receipt=str(p),receipt_sha256=sha256(p)))
    out=dict(rows=rows,predictions=read(POOL/'PIPELINE_PREDICTIONS_FREEZE.json'),
        boot_id=s['boot_id'],new_DNN=87,new_CPU_stage_calls=22,new_native_sessions=10,
        no_new_formula=True,not_complete_search_or_Top20=True,single_boot_short_window=True,
        source_sha256=sha256(__file__))
    write(POOL/'analysis/RESULTS.json',out,replace=True);return out


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','qemu','board','analyze'))
    p.add_argument('--execute-board',action='store_true');a=p.parse_args()
    if a.command=='board' and not a.execute_board:p.error('--execute-board is required')
    print(json.dumps(globals()[a.command](execute_board=True) if a.command=='board' else globals()[a.command](),ensure_ascii=False,indent=2))
