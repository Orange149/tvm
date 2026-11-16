"""Post-hoc diagnosis with the existing frozen completion/CPU replay model.

Only project known isolated profiles onto the three measured configurations.
No model changes, parameter fitting, input rescaling, native runs or searching.
"""
from copy import deepcopy
from pathlib import Path
import json
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from cpu_vta_partitioning.common import read,write,sha256
from finite_service.common import freeze
from device_completion_replay import CompletionFinite
from match_current_model_gap import project
from policy_branch_eval.common import isolated_inputs_only

ROOT=Path(__file__).resolve().parent
PARENT=ROOT/'artifacts/three_thread_host_gap_01'
OUT=PARENT/'existing_model_diagnostic'


def run():
    dev=ROOT/'artifacts/finite_service_01/development'
    current=ROOT/'artifacts/device_completion_replay_01'
    start=time.monotonic()
    with isolated_inputs_only():
        old=read(dev/'INPUTS.json');latest=read(current/'INPUTS.json')
        plan=read(current/'PLAN.json')
        if sha256(dev/'INPUTS.json')!=read(dev/'PLAN.json')['inputs_sha256']:
            raise ValueError('isolated CPU inputs changed')
        if sha256(current/'INPUTS.json')!=plan['inputs_sha256']:
            raise ValueError('isolated device inputs changed')
        for p,h in plan['sources'].items():
            if sha256(p)!=h:raise ValueError('existing replay implementation changed')
        if latest['contains_pipeline_performance'] or old['quanta_ns']!=latest['quanta_ns']:
            raise ValueError('input domain or CPU dispatch calibration differs')
        selection=read(PARENT/'ISOLATED_SELECTION.json')
        base=old['isolated']['cases']['yolov3_tiny416/split']
        donor=next(c for k,c in latest['cases'].items() if k.startswith('yolov3_tiny416/'))
        projected=project(base,donor);cases={}
        for cfg in selection['configurations']:
            stages=deepcopy(projected['stages'])
            for original,requested,stage in zip(base['configuration']['stages'],cfg['stages'],stages):
                if original['device']=='cpu':
                    if original!=requested:raise ValueError('CPU mapping differs from original isolated profile')
                else:stage['mask']=requested['mask']
            cases[cfg['id']]=stages
        inputs=dict(cases=cases,window=dict(frames=28,warmup=8,scored=16,queue_depth=2),
            quanta_ns=old['quanta_ns'],sources={str(dev/'INPUTS.json'):sha256(dev/'INPUTS.json'),
                str(current/'INPUTS.json'):sha256(current/'INPUTS.json'),
                str(PARENT/'ISOLATED_SELECTION.json'):sha256(PARENT/'ISOLATED_SELECTION.json')},
            replay_sources=plan['sources'],source_sha256=sha256(__file__),
            no_native_wall_or_FPS_input=True,post_hoc_diagnostic_only=True,
            CPU_work_and_dispatch_parameters_unchanged=True,
            shared_VTA_profile_transferred_to_pinned_hosts=True,
            old_isolated_CPU_profile_not_refitted_to_new_component_cost=True,
            controls_unmodeled=True)
        freeze(OUT/'INPUTS.json',inputs)
        outputs={}
        for cid,stages in cases.items():
            path=OUT/(cid+'.json')
            if not path.exists():
                result=CompletionFinite(stages,quanta=inputs['quanta_ns'],placement='wake_boundary',
                    completion_model='independent',completion_point='midpoint',runnable_completion_wait=True,
                    controls=None,gate='off',host_order='before',**inputs['window']).run()
                freeze(path,dict(result=result,input_sha256=sha256(OUT/'INPUTS.json'),
                    no_pipeline_FPS_read=True))
            if read(path)['input_sha256']!=sha256(OUT/'INPUTS.json'):raise ValueError('cached replay input differs')
            outputs[cid]=dict(path=str(path),sha256=sha256(path))
        freeze(OUT/'REPLAY_FREEZE.json',dict(outputs=outputs,input_sha256=sha256(OUT/'INPUTS.json'),
            no_pipeline_wall_or_FPS_read=True,post_hoc_diagnostic=True))
    # Native data become available only after all three replays are sealed.
    native=read(PARENT/'analysis/GAP_COMPARISON.json')
    rows=[]
    for n in native['rows']:
        pred=read(outputs[n['configuration_id']]['path'])['result']['replay']
        rows.append(dict(role=n['role'],configuration_id=n['configuration_id'],
            predicted_stage_ms=[v/1e6 for v in pred['stage_service_ns']],
            predicted_II_ms=pred['interval_ns']/1e6,
            native_stage_ms=n['pipeline_all_stage_ms'],native_II_ms=n['pipeline_II_ms'],
            front_error_ms=pred['stage_service_ns'][0]/1e6-n['pipeline_front_ms']))
    result=dict(rows=rows,input_sha256=sha256(OUT/'INPUTS.json'),new_native_sessions=0,
        new_model=False,post_hoc_diagnostic_not_independent_validation=True,
        control_omission_and_profile_transfer_remain=True,
        native_source_sha256=sha256(PARENT/'analysis/GAP_COMPARISON.json'),
        compute_seconds=time.monotonic()-start)
    write(OUT/'RESULTS.json',result,replace=True)
    return result


if __name__=='__main__':print(json.dumps(run(),ensure_ascii=False,indent=2))
