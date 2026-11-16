"""Supplemental end-to-end checks without changing the frozen input-order audit."""
from copy import deepcopy
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_isolated_sequence_audit as a
from test_isolated_sequence_audit import profile


def setup(stages):
    return dict(quanta={str(i):10 for i in range(2,7)},wake=True,sleeper=True,
        wakeup_granularity_ns=3,sleeper_credit_ns=9,placement='wake_boundary',
        completion_model='independent',completion_point='midpoint',runnable_completion_wait=True,
        controls=None,gate='off',host_order='before',frames=28,warmup=8,scored=16,queue_depth=2,
        collector_mask=[0,1,2,3],isolated_output_cost=dict(schema='isolated-output-control-v1',
            origin='isolated_output_serialization',network='resnet18',ring_count=1,CPU_ns=0,
            warmup=2,scored=8,output_bytes=4,output_count=1,receipt_sha256='0'*64,
            plan_sha256='1'*64,runtime_sha256='2'*64,serializer_source_sha256='3'*64,
            contains_pipeline_data=False,measured_scope='ResultJSON_and_deferred_string_insertion'))


def test_constant_samples_exactly_recover_original_engine_and_keep_inputs_immutable():
    stages=[dict(device='cpu',mask=[0],worker_core_map=[0],set_cpu_ns=1,get_cpu_ns=2,profile=profile(4))]
    source=deepcopy(stages);samples=dict(schema='isolated-sequence-work-v1',contains_pipeline_measurements=False,
        CPU_samples={'0':[profile(4) for _ in range(8)]})
    expected=a.IsolatedOutputFinite(deepcopy(stages),**setup(stages)).run()
    for origin in (0,7):
        got=a.SequencedInputs(stages,isolated_samples=samples,origin=origin,**setup(stages)).run()
        assert got==expected
        assert stages==source
    assert expected['replay']['interval_ns']==17


def test_variable_samples_are_executed_once_per_frame_with_equal_window_cpu_budget():
    from fractions import Fraction
    stages=[dict(device='cpu',mask=[0],worker_core_map=[0],set_cpu_ns=1,get_cpu_ns=2,profile=profile(Fraction(9,2)))]
    samples=dict(schema='isolated-sequence-work-v1',contains_pipeline_measurements=False,
        CPU_samples={'0':[profile(n) for n in range(1,9)]})
    for origin in (0,3,7):
        result=a.SequencedInputs(deepcopy(stages),isolated_samples=samples,origin=origin,**setup(stages)).run()
        tasks=[t for t in result['tasks'] if t['operation']=='worker_compute']
        assert [t['frame'] for t in tasks]==list(range(8,24))
        assert [a.rational(t['CPU']) for t in tasks]==[(f+origin)%8+1 for f in range(8,24)]
        assert sum(a.rational(t['CPU']) for t in tasks)==72
