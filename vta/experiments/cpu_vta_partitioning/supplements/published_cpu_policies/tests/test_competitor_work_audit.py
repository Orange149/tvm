from copy import deepcopy
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_competitor_work_audit as a


def fixture():
    worker=lambda n:dict(logical_worker=n,compute_segments=[n+1])
    kernel=dict(function_offset=10,tasks=3,barrier_generations=0,
        workers=[worker(i) for i in range(3)],launch_prefix_cpu=8,isolated_completion_wait_and_return_cpu=20)
    stages=[dict(program='cpu_a',profile=dict(kernels=[kernel],serial_cpu_segments=[5,7]),set_cpu_ns=11)]
    p=dict(schema='diagnostic-observed-callback-work-v1',diagnostic_only=True,contains_pipeline_measurements=True,
        not_a_static_cost_source=True,CPU_callbacks={'0':dict(program='cpu_a',kernels=[dict(function_offset=10,tasks=3,
        barrier_generations=0,workers=[dict(logical_worker=i,compute_segments=[10+i]) for i in range(3)])])})
    return stages,p


def test_substitution_changes_only_selected_callback_cpu_not_waits_or_inputs():
    stages,p=fixture();before=deepcopy(stages)
    changed=a.substitute(stages,p,{0:[1]})
    expected=deepcopy(stages);expected[0]['profile']['kernels'][0]['workers'][1]['compute_segments']=[11]
    assert changed==expected and stages==before
    assert a.substitute(stages,p,{})==stages


@pytest.mark.parametrize('fault',['extra_wall','wrong_slot','kernel','negative','not_diagnostic'])
def test_no_implicit_wall_penalty_or_identity_change(fault):
    stages,p=fixture()
    if fault=='extra_wall':p['pipeline_wall_ns']=100
    elif fault=='wrong_slot':p['CPU_callbacks']['0']['kernels'][0]['workers'][1]['logical_worker']=0
    elif fault=='kernel':p['CPU_callbacks']['0']['kernels'][0]['function_offset']=99
    elif fault=='negative':p['CPU_callbacks']['0']['kernels'][0]['workers'][1]['compute_segments']=[-1]
    else:p['diagnostic_only']=False
    with pytest.raises(ValueError):a.substitute(stages,p,{0:[1]})
