"""Post-intervention diagnostic: exactly match both fixed host configurations.

No new predictor or native session. Uses the already-frozen selection metadata,
never FPS, to construct inputs. Shared-host isolated shape is explicitly a
transfer assumption, not a newly measured fixed-host device profile.
"""
from copy import deepcopy
from pathlib import Path
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from cpu_vta_partitioning.common import read,sha256
from finite_service.common import freeze
from device_completion_replay import CompletionFinite
from policy_branch_eval.common import isolated_inputs_only

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/host_gap_intervention_01/matched_model_postcheck'


def run():
    source=ROOT/'artifacts/current_model_gap_01/INPUTS.json';inputs=read(source)
    selection=ROOT/'artifacts/host_gap_intervention_01/SELECTION_FREEZE.json';s=read(selection)
    cases={}
    for cfg in s['selected_configurations']['yolov3_tiny416']:
        if s['roles'][cfg['id']]=='shared':continue
        stages=deepcopy(inputs['case']['stages'])
        for a,b in zip(cfg['stages'],stages):
            if a['device']=='vta':b['mask']=a['mask']
            elif a!=inputs['case']['configuration']['stages'][cfg['stages'].index(a)]:raise ValueError('CPU mapping changed')
        cases[s['roles'][cfg['id']]]=stages
    freeze(OUT/'INPUTS.json',dict(cases=cases,window=inputs['window'],quanta_ns=inputs['quanta_ns'],
        source_sha256=sha256(__file__),isolated_source_sha256=sha256(source),
        premeasurement_configuration_selection_sha256=sha256(selection),
        no_pipeline_wall_or_FPS_input=True,post_intervention_diagnosis_not_prospective_prediction=True,
        shared_host_isolated_shape_transferred_to_pinned_hosts_assumption=True))
    result={}
    for label,stages in cases.items():
        path=OUT/(label+'.json')
        if not path.exists():
            with isolated_inputs_only():
                value=CompletionFinite(stages,quanta=inputs['quanta_ns'],placement='wake_boundary',
                    completion_model='independent',completion_point='midpoint',runnable_completion_wait=True,
                    controls=None,gate='off',host_order='before',**inputs['window']).run()
            freeze(path,dict(result=value,input_sha256=sha256(OUT/'INPUTS.json'),no_pipeline_FPS_read=True))
        d=read(path)
        if d['input_sha256']!=sha256(OUT/'INPUTS.json'):raise ValueError('cached postcheck differs')
        r=d['result']['replay'];result[label]=dict(predicted_II_ms=r['interval_ns']/1e6,
            predicted_stage_ms=[x/1e6 for x in r['stage_service_ns']],replay_sha256=sha256(path))
    original=read(ROOT/'artifacts/current_model_gap_01/observed_placement_diagnostic/REPLAY.json')['result']['replay']
    fixed=read(OUT/'observed_modal_hosts_3_2.json')['result']['replay']
    # Only shape metadata differs; both existing ways of fixing placement
    # must produce the same event timing and CPU accounting.
    if fixed!=original:raise ValueError('fixed masks and retained observed cores do not give identical timing')
    freeze(OUT/'REPLAY_FREEZE.json',dict(results=result,input_sha256=sha256(OUT/'INPUTS.json'),
        no_native_performance_read=True,observed_fixed_core_reconstruction_exact=True))
    return result


if __name__=='__main__':print(json.dumps(run(),ensure_ascii=False,indent=2))
