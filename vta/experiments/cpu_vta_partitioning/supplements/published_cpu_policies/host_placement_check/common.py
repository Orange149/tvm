from pathlib import Path
from cpu_vta_partitioning.common import read,sha256,write
from workload_model.common import ROOT

PARENT=ROOT/'artifacts/scheduler_gap_audit_01'
CAPABILITY=ROOT/'artifacts/placement_alignment_01/counter'
OUTPUT=ROOT/'artifacts/yolo_host_placement_01'
RUN=ROOT/'runs/yolo_host_placement_01'
REPORT=ROOT.parents[4]/'paper/my_paper/analysis/20261110_yolo_host_placement'
CONFIG=ROOT/'artifacts/cross_network_model_01/board.local.json'


def protocol():
    return dict(experiment='yolo_host_placement_01',qualification=dict(warmup=0,scored=1,tail=0),
        pipeline=dict(warmup=8,scored=64,tail=32),max_native_sessions=5,max_DNN_inferences=314,
        no_reboot=True,score_changed=False,DP_changed=False,Top20_changed=False,
        interface='PERF_COUNT_SW_TASK_CLOCK, per TID/all CPUs and per TID/core 0..3',
        sampling=False,background_sampling_threads=0,automatic_remeasurement=False,
        window='collector dequeues completion 8 to completion 71; per-TID read envelopes retained',
        probe_quality_reference='20% of clean isolation-to-pipeline front service gap, diagnostic only')


def freeze(path,value):
    path=Path(path)
    if path.exists():
        if read(path)!=value:raise ValueError('frozen host-placement identity changed: '+str(path))
    else:write(path,value)
    return value


def specs():
    return [dict(phase='qualify',probe=False,runner='phase'),dict(phase='qualify',probe=False,runner='old'),
        dict(phase='clean',probe=False,runner='phase'),dict(phase='observed',probe=False,runner='phase'),
        dict(phase='clean',probe=False,runner='old')]


def sources():
    files=[ROOT/'run_host_placement_check.py',*sorted(Path(__file__).parent.glob('*.py')),
           *sorted(Path(__file__).parent.glob('*.h')),ROOT/'tests/test_host_placement_check.py',
           ROOT/'phase_profile/runtime.py',ROOT/'phase_profile/campaign.py']
    return {str(p):sha256(p) for p in files}
