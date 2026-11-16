from copy import deepcopy
from pathlib import Path
from fractions import Fraction
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_isolated_sequence_audit as a


def profile(work):
    return dict(serial_cpu_segments=[3,5],kernels=[dict(module='/programs/cpu_a/module.so',function_offset=1,
        tasks=1,barrier_generations=0,launch_prefix_cpu=2,workers=[dict(logical_worker=0,compute_segments=[work])])])


def inputs():
    stages=[dict(device='cpu',profile=profile(Fraction(9,2)))]
    samples=dict(schema='isolated-sequence-work-v1',contains_pipeline_measurements=False,
        CPU_samples={'0':[profile(n) for n in range(1,9)]})
    return stages,samples


def test_every_origin_has_identical_exact_work_without_smoothing_samples():
    stages,samples=inputs();a.validate_samples(stages,samples)
    for origin in range(8):
        v=[a.cpu_vector(samples['CPU_samples']['0'][(frame+origin)%8]) for frame in range(8,24)]
        assert [sum(row[j] for row in v)/16 for j in range(len(v[0]))]==a.cpu_vector(stages[0]['profile'])
    assert [p['kernels'][0]['workers'][0]['compute_segments'][0] for p in samples['CPU_samples']['0']]==list(range(1,9))


@pytest.mark.parametrize('fault',['pipeline','wall','mean','missing','worker'])
def test_invalid_source_and_implicit_wall_cost_are_rejected(fault):
    stages,samples=inputs()
    if fault=='pipeline':samples['contains_pipeline_measurements']=True
    elif fault=='wall':samples['CPU_samples']['0'][0]['run_wall']=999
    elif fault=='mean':samples['CPU_samples']['0'][0]['kernels'][0]['workers'][0]['compute_segments']=[9]
    elif fault=='missing':samples['CPU_samples']['0'].pop()
    else:samples['CPU_samples']['0'][0]['kernels'][0]['workers'][0]['logical_worker']=1
    with pytest.raises(ValueError):a.validate_samples(stages,samples)
