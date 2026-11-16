"""Cross-network regression on all four previously accepted resource layouts.

No native work, re-selection, new topology or full-domain performance claim.
"""
from pathlib import Path
import json
import shutil
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from cpu_vta_partitioning.common import read,write,sha256,verify_inventory,digest
from finite_service.common import freeze
from policy_branch_eval.common import isolated_inputs_only
from compiled_sleep_model import CompiledFinite
from scheduler_gap_audit.curves import csv_write
import completion_holdout_board as old_board

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/compiled_kernel_crosscheck_01'
PARENT=ROOT/'artifacts/device_completion_holdout_01'


def run():
    started=time.monotonic();source=PARENT/'INPUTS.json';parameter=ROOT/'artifacts/compiled_sleeper_01/COMPILED_SLEEPER.json'
    with isolated_inputs_only():
        p=read(parameter);x=read(source);selected=read(PARENT/'SELECTION_FREEZE.json');original_plan=read(PARENT/'PLAN.json')
        if original_plan['inputs_sha256']!=sha256(source):raise ValueError('original isolated inputs changed')
        cases={network+'/'+c['id']:x['networks'][network][c['id']]
               for network,rows in selected['selected_configurations'].items() for c in rows}
        if len(cases)!=4 or x['window']!=dict(frames=28,queue_depth=2,scored=16,warmup=8):raise ValueError('regression scope/window changed')
        plan=dict(source_input_sha256=sha256(source),parameter_sha256=sha256(parameter),
            selection_sha256=sha256(PARENT/'SELECTION_FREEZE.json'),cases=list(cases),
            source_hashes={str(f):sha256(f) for f in (Path(__file__),ROOT/'compiled_sleep_model.py')},
            constants=dict(wakeup_granularity_ns=p['wakeup_granularity_ns'],sleeper_credit_ns=p['sleeper_credit_ns']),
            resource_layouts_chosen_before_current_rule_repair=True,
            known_measurements_development_regression_not_unseen_validation=True,
            local_fixed_topology_result_not_13_FPS_global_baseline=True,
            no_new_CPU_profiles_or_native_sessions=True,parameter_values_not_searched=True)
        freeze(OUT/'PLAN.json',plan);paths={};old_paths={}
        for key,c in cases.items():
            network,cid=key.split('/');old=PARENT/'predictions'/network/(cid+'_midpoint.json')
            rel=str(old.relative_to(PARENT));h=selected['prediction_files'][rel]
            if sha256(old)!=h:raise ValueError('old prediction changed')
            old_paths[key]=dict(path=str(old),sha256=h)
            path=OUT/'replays'/(network+'_'+cid+'.json')
            if not path.exists():
                result=CompiledFinite(c['stages'],quanta=x['quanta_ns'],wake=True,sleeper=True,
                    wakeup_granularity_ns=p['wakeup_granularity_ns'],sleeper_credit_ns=p['sleeper_credit_ns'],
                    placement='wake_boundary',completion_model='independent',completion_point='midpoint',
                    runnable_completion_wait=True,controls=None,gate='off',host_order='before',**x['window']).run()
                freeze(path,dict(result=result,plan_sha256=sha256(OUT/'PLAN.json'),no_pipeline_performance_read=True))
            if read(path)['plan_sha256']!=sha256(OUT/'PLAN.json'):raise ValueError('cached correction input differs')
            paths[key]=dict(path=str(path),sha256=sha256(path))
        freeze(OUT/'REPLAY_FREEZE.json',dict(paths=paths,old_paths=old_paths,plan_sha256=sha256(OUT/'PLAN.json'),
            no_pipeline_performance_read=True))
    # Accepted native data are read only after the new predictions are sealed.
    measurements={};receipts={};qual={}
    state=read(old_board.RUN/'state.json')
    if state['status']!='COMPLETE':raise ValueError('original native cohort incomplete')
    for spec in old_board.specs():
        bundle=old_board.POOL/spec['network']/'capsule';task=old_board.make_task(bundle,spec)
        path=old_board.RUN/'accepted'/state['boot_id']/(task['task_id']+'.json');r=read(path)
        if r['task']!=task or r['boot_id']!=state['boot_id']:raise ValueError('native identity differs')
        verify_inventory(r['raw'],r['files'])
        if old_board.summary(r['raw'],task,bundle)!=r['summary']:raise ValueError('native evidence no longer validates')
        if (Path(r['raw'])/'task_hash').read_text().strip()!=digest(task):raise ValueError('native task hash differs')
        key=spec['network']+'/'+spec['case'];receipts[str(path)]=sha256(path)
        if spec['phase']=='qualify':qual[key]=r['summary']['output_fingerprints']
        else:
            if r['summary']['output_fingerprints']!=qual[key]:raise ValueError('output differs from qualification')
            measurements[key]=r['summary']
    rows=[];stages=[]
    for key,c in cases.items():
        network,cid=key.split('/');n=measurements[key]
        old=read(old_paths[key]['path'])['result']['replay'];new=read(paths[key]['path'])['result']['replay']
        rows.append(dict(network=network,configuration_id=cid,
            CPU_threads=[s['threads'] for s in c['configuration']['stages'] if s['device']=='cpu'],
            CPU_masks=[s['mask'] for s in c['configuration']['stages'] if s['device']=='cpu'],
            host_mask=c['configuration']['host_mask'],
            old_front_ms=old['stage_service_ns'][0]/1e6,new_front_ms=new['stage_service_ns'][0]/1e6,
            native_front_ms=n['stage_service_ms'][0],
            old_front_error_ms=old['stage_service_ns'][0]/1e6-n['stage_service_ms'][0],
            new_front_error_ms=new['stage_service_ns'][0]/1e6-n['stage_service_ms'][0],
            old_II_ms=old['interval_ns']/1e6,new_II_ms=new['interval_ns']/1e6,
            native_II_ms=n['ii_ms'],native_FPS=n['fps'],
            scope='already measured local resource layouts, not global Top1/Top20'))
        for i,s in enumerate(c['configuration']['stages']):
            stages.append(dict(network=network,configuration_id=cid,stage=s['program'],
                old_ms=old['stage_service_ns'][i]/1e6,new_ms=new['stage_service_ns'][i]/1e6,
                native_ms=n['stage_service_ms'][i]))
    out=dict(rows=rows,stages=stages,source_receipts=receipts,new_native_sessions=0,new_DNN=0,
        source_sha256=sha256(__file__),plan_sha256=sha256(OUT/'PLAN.json'),
        previous_13_FPS_configuration_not_remeasured=True,
        both_networks_same_compiled_constants=True,no_parameter_tuning=True,
        no_complete_domain_or_Top20_result=True,compute_seconds=time.monotonic()-started)
    write(OUT/'RESULTS.json',out,replace=True);csv_write(OUT/'phase_comparison.csv',rows);csv_write(OUT/'all_stages.csv',stages)
    report=ROOT.parents[4]/'paper/my_paper/analysis/20261111_compiled_kernel_crosscheck';report.mkdir(parents=True,exist_ok=True)
    for name in ('RESULTS.json','phase_comparison.csv','all_stages.csv','PLAN.json','REPLAY_FREEZE.json'):
        shutil.copy2(OUT/name,report/name)
    return out


if __name__=='__main__':
    r=run();print(json.dumps({k:v for k,v in r.items() if k not in ('stages','source_receipts')},ensure_ascii=False,indent=2))
