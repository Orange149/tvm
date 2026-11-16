from pathlib import Path
from cpu_vta_partitioning.common import read,write,sha256
from workload_model.common import ROOT

PARENT=ROOT/'artifacts/scheduler_gap_audit_01'
OUTPUT=ROOT/'artifacts/yolo_scheduler_trace_01'
RUN=ROOT/'runs/yolo_scheduler_trace_01'
REPORT=ROOT.parents[4]/'paper/my_paper/analysis/20261111_yolo_scheduler_trace'
CAPABILITY=ROOT/'artifacts/scheduler_event_audit_01'
CONFIG=ROOT/'artifacts/cross_network_model_01/board.local.json'


def protocol():
    return dict(experiment='yolo_scheduler_trace_01',revision=1,input_index=0,
        qualification=dict(warmup=0,scored=1,tail=0),pipeline=dict(warmup=8,scored=16,tail=24),
        max_native_sessions=5,max_DNN_inferences=146,ring_data_pages=32,expected_roles=12,
        byte_order='little',clock='CLOCK_MONOTONIC',sampling_timer=False,background_sampling_threads=0,
        original_wait=True,no_reboot=True,score_changed=False,DP_changed=False,Top20_changed=False,
        no_automatic_remeasurement=True,no_anchor=True,no_drift_rejection=True,
        probe_quality_fraction=0.2,
        boundary_policy='only IN-to-OUT and OUT-to-IN interior intervals are reconstructed; boundary spans stay unknown',
        next_step_not_automatic=True)


def specs():
    return [dict(network='yolov3_tiny416',scope='pipeline',case='overlap',phase=p,runner=r,probe=p=='observed')
        for p,r in [('qualify','phase'),('qualify','old'),('clean','phase'),('observed','phase'),('clean','old')]]


def freeze(path,value):
    path=Path(path)
    if path.exists():
        if read(path)!=value:raise ValueError('scheduler trace identity changed: '+str(path))
    else:write(path,value)
    return value


def sources():
    files=[ROOT/'run_yolo_scheduler_trace.py',*sorted(Path(__file__).parent.glob('*.py')),
        *sorted(Path(__file__).parent.glob('*.h')),ROOT/'tests/test_yolo_scheduler_trace.py',
        ROOT/'phase_profile/runtime.py',ROOT/'phase_profile/worker.py',ROOT/'phase_profile/campaign.py',
        ROOT/'scheduler_gap_audit/runtime.py',ROOT/'host_placement_check/prepare.py']
    return {str(p):sha256(p) for p in files}
