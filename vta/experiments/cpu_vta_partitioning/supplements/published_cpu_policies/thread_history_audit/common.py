from pathlib import Path
from cpu_vta_partitioning.common import read,write,sha256
from workload_model.common import ROOT

PARENT=ROOT/'artifacts/model_semantics_audit_01'
OUTPUT=ROOT/'artifacts/thread_history_audit_01'
REPORT=ROOT.parents[4]/'paper/my_paper/analysis/20261110_thread_history_audit'


def protocol():
    return dict(experiment='thread_history_audit_01',revision=1,new_native_sessions=0,new_DNN_inferences=0,
        window=dict(frames=104,warmup=8,scored=64,queue_depth=2),
        placements=['instant','retain','wake_boundary'],runnable_wait=[True,False],
        control_modes=['zero','diagnostic_observed'],scenario_count=36,
        initial_placement='minimum currently assigned runnable count, tie lowest allowed core',
        retain='logical TID retains initial core throughout session',
        wake_boundary='retain while runnable; reconsider after positive nonrunning interval or queue/resource blocking; retain previous core on equal load',
        CPU_share='continuous equal share within a core, unchanged from frozen model',
        nonrunning_intervals_are_only_potential_wakeup_boundaries=True,
        no_Linux_CFS_reproduction_claim=True,no_timeslice_parameter=True,no_fitted_CPU_ratio=True,
        score_changed=False,DP_changed=False,Top20_changed=False,
        primary='all three cases; show stage error, caller CPU, host placement and completion order, not closest-fit picking')


def freeze(path,value):
    path=Path(path)
    if path.exists():
        if read(path)!=value:raise ValueError('offline history identity changed: '+str(path))
    else:write(path,value)
    return value


def sources():
    files=[ROOT/'run_thread_history_audit.py',*sorted(Path(__file__).parent.glob('*.py')),
           ROOT/'tests/test_thread_history_audit.py',ROOT/'model_semantics_audit/simulator.py',
           ROOT/'model_semantics_audit/replay.py',ROOT/'model_semantics_audit/common.py',
           ROOT/'pool_temporal/simulator.py',ROOT/'phase_model/simulator.py']
    return {str(p):sha256(p) for p in files}
