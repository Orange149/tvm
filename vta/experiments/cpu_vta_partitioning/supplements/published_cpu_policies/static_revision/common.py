from pathlib import Path
from cpu_vta_partitioning.common import read,write,sha256
ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'artifacts/static_revision_01'
REPORT=ROOT.parents[4]/'paper/my_paper/analysis/20261111_static_revision'
INPUT=ROOT/'artifacts/model_semantics_audit_01'
TRACE=ROOT/'artifacts/yolo_scheduler_trace_repair_01'


def freeze(path,value):
    path=Path(path)
    if path.exists():
        if read(path)!=value:raise ValueError('revision checkpoint differs: '+str(path))
    else:write(path,value)
    return value


def protocol():
    return dict(experiment='static_revision_01',first_stage='formula and matched-window trajectory audit',
        window=dict(frames=48,warmup=8,scored=16,queue_depth=2),
        replay_modes=['instant','retain','wake_boundary'],controls='unmodeled, no pipeline input',
        cases=['resnet18/static_first','yolov3_tiny416/overlap','yolov3_tiny416/split'],
        replay_count=9,new_native_sessions=0,new_DNN_inferences=0,
        development_replay_not_independent_validation=True,
        no_score_change_in_audit=True,no_search_domain_change=True,
        no_pipeline_work_or_wait_fitted=True,no_reboot=True,
        future_changes_require_frozen_hypothesis_and_validation=True)


def sources():
    files=[ROOT/'run_static_revision.py',*sorted(Path(__file__).parent.glob('*.py')),
           ROOT/'tests/test_static_revision.py',ROOT/'staged_ablation/model.py',ROOT/'staged_ablation/common.py',
           ROOT/'cpu_policies/open_resources/costs.py',ROOT/'resource_envelope/model.py',
           ROOT/'thread_history_audit/simulator.py',ROOT/'model_semantics_audit/simulator.py',
           ROOT/'pool_temporal/simulator.py',ROOT/'phase_model/simulator.py']
    return {str(p):sha256(p) for p in files}
