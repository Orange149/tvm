"""One observed-placement oracle diagnostic, never a deployable predictor.

Uses the existing retain-placement mode with initial host cores chosen only
from actual CPU placement observations. No FPS selects cores or coefficients.
CPU input, device model, execution masks and scheduling parameters are fixed.
"""
from pathlib import Path
import json
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from cpu_vta_partitioning.common import read,sha256,write
from finite_service.common import freeze
from device_completion_replay import CompletionFinite

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/current_model_gap_01/observed_placement_diagnostic'


def run():
    parent=OUT.parent;inputs=read(parent/'INPUTS.json');plan=read(parent/'PLAN.json')
    if sha256(parent/'INPUTS.json')!=plan['input_sha256']:raise ValueError('frozen inputs changed')
    for path,h in plan['original_model_sources'].items():
        if sha256(path)!=h:raise ValueError('frozen existing model changed')
    audit=ROOT/'artifacts/host_burst_gap_01/RESULTS.json';data=read(audit)
    selected={}
    for index,cfg in enumerate(inputs['case']['configuration']['stages']):
        if cfg['device']!='vta':continue
        rows=[r for r in data['host_phase_costs'] if r['host']==cfg['program']+'/host']
        if len(rows)!=3:raise ValueError('complete native host phases required')
        work=[sum(r['native_core_distribution_ms_per_call'][c] for r in rows) for c in range(4)]
        core=max(range(4),key=lambda c:(work[c],-c))
        if core not in cfg['mask']:raise ValueError('observed host escaped allowed mask')
        selected[str(index)]=dict(core=core,core_work_ms=work,selection_uses_FPS=False)
    if set(selected)!={'1','3'}:raise ValueError('diagnostic stage layout differs')
    freeze(OUT/'INPUT_FREEZE.json',dict(parent_inputs_sha256=sha256(parent/'INPUTS.json'),
        source_placement_audit=str(audit),source_placement_sha256=sha256(audit),selected=selected,
        contains_pipeline_placement_and_CPU_distribution=True,contains_stage_wall_time_or_FPS=False,
        diagnostic_only=True,new_static_method=False,
        observed_migrations_not_reconstructed=True,existing_retain_placement_hypothesis=True,
        source_sha256=sha256(__file__)))
    started=time.monotonic();path=OUT/'REPLAY.json'
    if not path.exists():
        case=inputs['case']
        model=CompletionFinite(case['stages'],quanta=inputs['quanta_ns'],placement='retain',
            completion_model='independent',completion_point='midpoint',runnable_completion_wait=True,
            controls=None,gate=case['configuration']['execution_mode'],host_order='before',**inputs['window'])
        if model.actor_core or model.active:raise ValueError('initial replay already has active actors')
        model.actor_core.update({('stage',int(i),0):s['core'] for i,s in selected.items()})
        value=model.run()
        for row in value['actor_totals']:
            if str(row['stage']) in selected:
                if set(row['core_CPU'])!={str(selected[str(row['stage'])]['core'])}:
                    raise ValueError('host placement hypothesis not retained')
        freeze(path,dict(result=value,input_sha256=sha256(OUT/'INPUT_FREEZE.json'),
            diagnostic_only=True,no_DNN=True,not_static_prediction=True))
    else:
        if read(path)['input_sha256']!=sha256(OUT/'INPUT_FREEZE.json'):raise ValueError('cached diagnostic differs')
    freeze(OUT/'REPLAY_FREEZE.json',dict(path_sha256=sha256(path),input_sha256=sha256(OUT/'INPUT_FREEZE.json'),
        complete_before_reading_native_stage_times=True))
    # Read the reference only after the diagnostic replay itself is frozen.
    base=read(parent/'analysis/RESULTS.json')
    p=read(path)['result']['replay'];normal=next(r for r in base['rows'] if r['mode']=='independent')
    result=dict(normal_prediction_ms=normal['stage_services_ms'],diagnostic_prediction_ms=[v/1e6 for v in p['stage_service_ns']],
        native_clean_stage_ms=normal['native_clean_stage_services_ms'],
        normal_interval_ms=normal['predicted_II_ms'],diagnostic_interval_ms=p['interval_ns']/1e6,
        native_clean_interval_ms=normal['native_clean_II_ms'],
        diagnostic_only=True,observed_modal_cores=selected,
        exact_host_migration_not_reconstructed=True,new_model=False,new_DNN=0,
        source_sha256=sha256(__file__),seconds=time.monotonic()-started,
        stages_other_than_hosts_use_same_pinned_placement=True,
        no_empirical_duration_multiplier=True)
    write(OUT/'RESULTS.json',result,replace=True)
    return result


if __name__=='__main__':print(json.dumps(run(),ensure_ascii=False,indent=2))
