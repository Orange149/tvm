from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit_current_profile_coverage import cpu_support


def test_rejects_other_modules_and_keeps_mapping_approximation_explicit():
    stage=dict(program='cpu_x',threads=2,worker_core_map=[1,3])
    profile=dict(identity=dict(network='yolo',program='cpu_x',threads=2,module_sha256='actual'),
        configuration={'stages':[dict(worker_core_map=[0,1])]})
    profiles={'one':profile}
    assert cpu_support(stage,'other','yolo',profiles)['kind']=='missing_multithread_fork_join_structure'
    assert cpu_support(stage,'actual','resnet',profiles)['kind']=='missing_multithread_fork_join_structure'
    assert 'permutation' in cpu_support(stage,'actual','yolo',profiles)['kind']
    stage['worker_core_map']=[0,1]
    assert cpu_support(stage,'actual','yolo',profiles)['kind']=='matching_CPU_phase_structure'


def test_single_thread_aggregate_is_not_invented_as_a_fine_trace():
    stage=dict(program='cpu_x',threads=1,worker_core_map=[0])
    value=cpu_support(stage,'actual','yolo',{})
    assert value['kind']=='single_thread_no_fork_join_but_only_aggregate_cost'
    assert value['aggregate_not_a_measured_continuous_busy_interval']
