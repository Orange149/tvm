"""Matched-component capacity and temporal predictions; no pipeline FPS reads."""
from copy import deepcopy
import json
from pathlib import Path

from cpu_vta_partitioning.common import read,sha256
from cpu_policies.open_resources.costs import Costs,quantize_ns
from cpu_policies.open_resources.domain import component,component_key,stage,configuration
from device_phase.predict_development import DeviceReplay
from phase_model.predict import build_stages
from phase_model.simulator import Replay,rational
from policy_branch_eval.common import isolated_inputs_only
from policy_branch_eval.ranking import inputs
from staged_ablation.model import score
from workload_model.common import OUTPUT,freeze


def fresh_records(records,network,CPU,device):
    result=deepcopy(records.get('costs',records));replaced={}
    for profile in CPU['profiles'].values():
        identity=profile['identity']
        if identity['network']!=network:continue
        s=profile['configuration']['stages'][0];key=component_key(component(s))
        if key not in result:raise ValueError('new isolated CPU has no original comparison')
        result[key]={**result[key],'cost':profile['clean_cost'],
            'session_thread_weights':profile['fresh_session_thread_weights'],
            'kind':'matched_new_isolated_session','local_profile_receipt':profile['provenance']['clean_receipt']}
        replaced[key]=dict(program=s['program'],threads=s['threads'],type='CPU')
    for profile in device['profiles'].values():
        if profile['network']!=network:continue
        begin,end=map(int,profile['program'].split('_')[1:])
        s=stage('vta',begin,end+1,1,[0,1,2,3]);key=component_key(component(s))
        if key not in result:raise ValueError('new isolated VTA has no original comparison')
        result[key]={**result[key],'cost':profile['clean_cost'],'kind':'matched_new_isolated_session',
                     'local_profile_receipt':profile['clean_receipt']}
        replaced[key]=dict(program=s['program'],threads=1,type='VTA')
    return result,replaced


def order_key(value):
    return json.dumps(configuration(value),sort_keys=True,separators=(',',':'))


def predict(output):
    output=Path(output).resolve()
    with isolated_inputs_only():
        plan=read(output/'PLAN.json');CPU=read(output/'ISOLATED_CPU_INPUTS.json')
        device_path=OUTPUT/'device_phase_02/ISOLATED_DEVICE_PHASES.json';device=read(device_path)
        if not CPU['no_pipeline_FPS'] or not device['no_pipeline_FPS']:raise ValueError('non-isolated prediction input')
        for path,expected in plan['model_sources_frozen'].items():
            if sha256(path)!=expected:raise ValueError('predeclared mechanism changed')
        networks={};rankings={}
        for network,cases in plan['networks'].items():
            programs,records,ddr,hashes,_,_=inputs(network)
            for profile in CPU['profiles'].values():
                identity=profile['identity']
                if identity['network']==network and identity['module_sha256']!=programs[identity['program']]['files']['module.so']['sha256']:
                    raise ValueError('CPU phase work belongs to a different module')
            for profile in device['profiles'].values():
                if profile['network']==network and profile['module_sha256']!=programs[profile['program']]['files']['module.so']['sha256']:
                    raise ValueError('VTA phase work belongs to a different module')
            replaced,provenance=fresh_records(records,network,CPU,device)
            old=Costs(records,programs=programs,ddr=ddr,distribution='session_weights')
            fresh=Costs(replaced,programs=programs,ddr=ddr,distribution='session_weights')
            networks[network]={}
            for case,cfg in cases.items():
                base_old=score(cfg,old,'fine_host');base=score(cfg,fresh,'fine_host')
                stages,assumptions=build_stages(network,cfg,CPU['profiles'],fresh)
                aggregate=deepcopy(stages)
                for i,(s,row) in enumerate(zip(stages,cfg['stages'])):
                    if row['device']!='vta':continue
                    profile=device['profiles'][network+'/'+row['program']]
                    s['device_profile']=profile['stages'];parts=profile['stages']
                    total_host=sum(rational(t) for t in parts['host_serial_CPU'])
                    total_driver_CPU=sum(rational(t['CPU']) for t in parts['devices'])
                    total_driver_wall=sum(rational(t['wall']) for t in parts['devices'])
                    aggregate[i]['run_cpu_ns']=total_host+total_driver_CPU
                    aggregate[i]['run_nonrunning_ns']=total_driver_wall-total_driver_CPU
                simulations={}
                for order in ('before','after'):
                    run=DeviceReplay(stages,host_order=order,runnable_completion_wait=True).run()
                    simulations['interleaved_'+order]={**run,'score_ns':max(base['score_ns'],run['interval_ns'])}
                    run=Replay(aggregate,host_order=order,runnable_completion_wait=True).run()
                    simulations['aggregate_'+order]={**run,'score_ns':max(base['score_ns'],run['interval_ns'])}
                row=dict(configuration=cfg,old_capacity=base_old,fresh_capacity=base,simulations=simulations,
                         assumptions=assumptions,components_replaced=provenance,old_component_hashes=hashes)
                networks[network][case]=row
            versions={'old_capacity':lambda r:r['old_capacity']['score_ns'],
                'fresh_capacity':lambda r:r['fresh_capacity']['score_ns'],
                **{key:(lambda r,k=key:r['simulations'][k]['score_ns']) for key in simulations}}
            rankings[network]={name:sorted(cases,key=lambda c:(value(networks[network][c]),order_key(cases[c])))
                               for name,value in versions.items()}
        document=dict(networks=networks,rankings=rankings,pipeline_FPS_read=False,
            same_new_component_costs_for_fresh_capacity_and_temporal_models=True,
            primary='interleaved_before',ablation='aggregate_before',host_sensitivity='interleaved_after',
            component_inputs_sha256=sha256(output/'ISOLATED_CPU_INPUTS.json'),device_inputs_sha256=sha256(device_path))
        freeze(output/'PREDICTIONS.json',document)
        model_paths=[Path(__file__),Path(__file__).parents[1]/'phase_model/simulator.py',
                     Path(__file__).parents[1]/'device_phase/predict_development.py',
                     Path(__file__).parents[1]/'phase_model/predict.py']
        freeze(output/'PREDICTION_FREEZE.json',dict(files={p.name:sha256(p) for p in
            (output/'PLAN.json',output/'PREDICTIONS.json',output/'ISOLATED_CPU_INPUTS.json',output/'PIPELINE_TASKS.json')},
            model_sources={str(p):sha256(p) for p in model_paths},
            pipeline_FPS_read=False,full_domain=False))
    return {n:{method:dict(first=ranking[0],order=ranking) for method,ranking in rs.items()} for n,rs in rankings.items()}
