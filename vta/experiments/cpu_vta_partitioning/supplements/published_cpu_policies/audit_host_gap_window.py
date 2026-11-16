"""Match all three known-layout replays to the actual 28-frame intervention.

Post-hoc diagnostic, not model selection or prospective accuracy validation.
No new parameters, measurements, window extensions, or runtime changes.
"""
from copy import deepcopy
from pathlib import Path
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from cpu_vta_partitioning.common import read,sha256,write
from finite_service.common import freeze
from device_completion_replay import CompletionFinite
from policy_branch_eval.common import isolated_inputs_only

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/host_gap_intervention_01/matching_native_window'


def run():
    source=ROOT/'artifacts/current_model_gap_01/INPUTS.json';inputs=read(source)
    parent=OUT.parent;s=read(parent/'SELECTION_FREEZE.json');p=read(parent/'BOARD_PLAN.json')['protocol']
    w=p['pipeline'];window=dict(frames=sum(w.values()),warmup=w['warmup'],scored=w['scored'],queue_depth=2)
    if window!=dict(frames=28,warmup=8,scored=16,queue_depth=2):raise ValueError('native window changed')
    cases={}
    for cfg in s['selected_configurations']['yolov3_tiny416']:
        stages=deepcopy(inputs['case']['stages'])
        for c,t in zip(cfg['stages'],stages):
            if c['device']=='vta':t['mask']=c['mask']
        cases[s['roles'][cfg['id']]]=stages
    freeze(OUT/'INPUTS.json',dict(cases=cases,window=window,quanta_ns=inputs['quanta_ns'],
        source_sha256=sha256(__file__),isolated_source_sha256=sha256(source),
        premeasurement_configuration_selection_sha256=sha256(parent/'SELECTION_FREEZE.json'),
        no_FPS_or_native_wall_time_input=True,post_hoc_diagnostic=True,
        controls_unmodeled_as_in_original=True,shared_profile_transferred_to_pinned_hosts_assumption=True))
    paths={}
    for label,stages in cases.items():
        path=OUT/(label+'.json')
        if not path.exists():
            with isolated_inputs_only():
                value=CompletionFinite(stages,quanta=inputs['quanta_ns'],placement='wake_boundary',
                    completion_model='independent',completion_point='midpoint',runnable_completion_wait=True,
                    controls=None,gate='off',host_order='before',**window).run()
            freeze(path,dict(result=value,input_sha256=sha256(OUT/'INPUTS.json'),no_native_FPS_read=True))
        if read(path)['input_sha256']!=sha256(OUT/'INPUTS.json'):raise ValueError('cached replay differs')
        paths[label]=dict(path=str(path),sha256=sha256(path))
    freeze(OUT/'REPLAY_FREEZE.json',dict(paths=paths,input_sha256=sha256(OUT/'INPUTS.json'),
        same_window_as_native=True,no_native_wall_or_FPS_read=True))
    native=read(parent/'analysis/RESULTS.json');rows=[]
    for n in native['rows']:
        r=read(paths[n['role']]['path'])['result']['replay']
        rows.append(dict(role=n['role'],predicted_stage_ms=[v/1e6 for v in r['stage_service_ns']],
            native_stage_ms=n['stage_services_ms'],predicted_II_ms=r['interval_ns']/1e6,
            native_II_ms=n['ii_ms'],native_FPS=n['fps'],
            front_error_ms=r['stage_service_ns'][0]/1e6-n['stage_services_ms'][0]))
    result=dict(rows=rows,same_native_window=True,post_hoc_diagnostic=True,new_model=False,new_native_sessions=0,
        profile_transfer_and_control_omission_remain=True,source_sha256=sha256(__file__),
        source_native_results_sha256=sha256(parent/'analysis/RESULTS.json'))
    write(OUT/'RESULTS.json',result,replace=True)
    return result


if __name__=='__main__':print(json.dumps(run(),ensure_ascii=False,indent=2))
