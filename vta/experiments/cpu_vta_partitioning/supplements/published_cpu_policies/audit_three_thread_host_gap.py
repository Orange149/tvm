"""Choose a host-overlap intervention from isolated work, then audit history."""
from collections import Counter
from pathlib import Path
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from cpu_vta_partitioning.common import read,sha256
from finite_service.common import freeze
from model_semantics_audit.common import metadata_projection
from policy_branch_eval.ranking import inputs
from cpu_policies.open_resources.costs import Costs,quantize_ns,allocate_core_ns
from cpu_policies.open_resources.domain import stage,candidate,context,component
from staged_ablation.model import score
from phase_model.simulator import rational

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/three_thread_host_gap_01'


def target_pipeline(c):
    if not isinstance(c,dict) or not isinstance(c.get('stages'),list):return False
    if any(not isinstance(s,dict) for s in c['stages']):return False
    if [(s.get('device'),s.get('begin'),s.get('end')) for s in c['stages']]!=[
        ('cpu',0,3),('vta',3,7),('cpu',7,9),('vta',9,13),('cpu',13,14)]:return False
    if [(s.get('threads'),s.get('mask'),s.get('worker_core_map')) for s in c['stages'] if s['device']=='cpu']!=[
        (3,[0,1,2],[0,1,2]),(1,[3],[3]),(1,[3],[3])]:return False
    return c.get('execution_mode')=='off' and c.get('affinity_mode')=='pinned' and c.get('control_mask')==[0,1,2,3]


def target_component(task):
    c=task.get('configuration');s=task.get('stages')
    if not isinstance(c,dict) or not isinstance(s,list) or len(s)!=1:return False
    r=s[0]
    if not isinstance(r,dict) or (r.get('device'),r.get('begin'),r.get('end'))!=('cpu',0,3):return False
    n=r.get('threads')
    return n in (3,4) and r.get('mask')==list(range(n)) and r.get('worker_core_map')==list(range(n)) and c.get('affinity_mode')=='pinned'


def run():
    programs,records,ddr,hashes,*_=inputs('yolov3_tiny416')
    costs=Costs(records,programs=programs,ddr=ddr,distribution='session_weights')
    front=stage('cpu',0,3,3,[0,1,2]);v=costs.get(front)
    demand=allocate_core_ns(quantize_ns(v['run_core_ms']),v['cpu_core_weights'],3)
    hot=max((1,2),key=lambda k:(demand[k],-k));light=min((1,2),key=lambda k:(demand[k],k));outside=3
    if hot==light:raise ValueError('distinct worker-load intervention not supported')
    intrinsic=ROOT/'artifacts/model_semantics_audit_01/ISOLATED_INPUTS.json'
    kernels=read(intrinsic)['cases']['yolov3_tiny416/split']['stages'][0]['profile']['kernels']
    actual=[sum((rational(seg) for k in kernels if i<k['tasks'] for seg in k['workers'][i]['compute_segments']),0) for i in range(3)]
    if actual[hot]<=actual[light]:raise ValueError('isolated per-kernel work disagrees with coarse worker ranking')
    configs=[];roles={};prior_predictions={}
    for role,core in [('host_on_heavier_worker',hot),('host_on_lighter_worker',light),('host_outside_front_group',outside)]:
        ctx='host_shared_'+str(core);m=context(ctx)['host_mask']
        c=candidate([front,stage('vta',3,7,1,m),stage('cpu',7,9,1,[3]),
            stage('vta',9,13,1,m),stage('cpu',13,14,1,[3])],ctx,'off')
        configs.append(c);roles[c['id']]=role
        prior_predictions[c['id']]=score(c,costs,'concentrated')
    components={str(n):component(stage('cpu',0,3,n,list(range(n)))) for n in (3,4)}
    freeze(OUT/'ISOLATED_SELECTION.json',dict(configurations=configs,roles=roles,components=components,
        prior_predictions=prior_predictions,component_hashes=hashes,
        front_cost_source_key=v['source_key'],front_service_ms=v['service_ms'],front_run_core_ms=v['run_core_ms'],
        front_CPU_share_proxy=v['cpu_core_weights'],front_proxy_run_per_slot_ns=list(demand),
        isolated_kernel_compute_per_slot_ms=[float(a/1_000_000) for a in actual],
        intrinsic_source=str(intrinsic),intrinsic_source_sha256=sha256(intrinsic),
        chosen_heavier_noncaller_core=hot,chosen_lighter_noncaller_core=light,chosen_outside_front_core=outside,
        control_mask=[0,1,2,3],gate='off',no_pipeline_FPS_used=True,
        no_new_score_or_domain=True,source_sha256=sha256(__file__),
        source_basis='Pipe-it fixed computation groups; GCAPS host CPU/device split, adapted not fully reproduced',
        outside_front_not_full_system_exclusivity=True))
    inventory=ROOT/'artifacts/device_completion_holdout_01/KNOWN_EXECUTIONS.json';known=read(inventory)
    paths={Path(p) for p in known['files'] if '/accepted/' in p}
    paths.update((ROOT/'runs').glob('**/accepted/**/*.json'))
    found=[];components_found=[];parsed=0;unparsed=0
    for p in sorted(paths):
        h=sha256(p)
        if str(p) in known['files'] and h!=known['files'][str(p)]:raise ValueError('indexed receipt changed')
        try:m=metadata_projection(p,{'task','boot_id'})
        except ValueError:unparsed+=1;continue
        parsed+=1;t=m['task'];c=t.get('configuration')
        entry=dict(receipt=str(p),receipt_sha256=h,boot_id=m.get('boot_id'),kind=t.get('kind'),
            phase=t.get('spec',{}).get('phase'),scope=t.get('spec',{}).get('scope'),
            input_indices=t.get('input_indices'),executor_sha256=t.get('executor_sha256'),
            runtime_sha256=t.get('runtime_sha256'),window={k:t.get(k) for k in ('warmup','scored','tail')},
            no_FPS_decoded=True)
        if target_pipeline(c) and target_pipeline({**c,'stages':t.get('stages')}):
            found.append(dict(**entry,host_masks=[s['mask'] for s in c['stages'] if s['device']=='vta'],
                configuration_id=c.get('id')))
        if target_component(t):components_found.append(dict(**entry,threads=t['stages'][0]['threads']))
    result=dict(pipeline_matches=found,component_matches=components_found,parsed_receipts=parsed,
        unparsed_receipts=unparsed,selection_sha256=sha256(OUT/'ISOLATED_SELECTION.json'),
        inventory_sha256=sha256(inventory),source_sha256=sha256(__file__),no_pipeline_FPS_used=True,
        pipeline_host_pairs=sorted({json.dumps(r['host_masks']) for r in found}))
    freeze(OUT/'HISTORY_AUDIT.json',result)
    current_boot=read(ROOT/'artifacts/clock_scope_audit_01/RECEIPT.json')['boot_id']
    exact=[r for r in components_found if r['boot_id']==current_boot and r['executor_sha256']=='20ca357bbbce14cf41b2ebc7f13bcbd8e082e878af82f043762ce9cf5350d187'
           and r['input_indices']==[0] and r['window']==dict(warmup=2,scored=8,tail=0)]
    return dict(pipeline_matches=len(found),pipeline_host_pairs=result['pipeline_host_pairs'],
        current_boot_component_matches=exact,roles=roles,
        selected_cores=dict(heavier_worker=hot,lighter_worker=light,outside_front=outside),
        old_static_scores_ms={cid:r['score_ns']/1e6 for cid,r in prior_predictions.items()})


if __name__=='__main__':print(json.dumps(run(),ensure_ascii=False,indent=2))
