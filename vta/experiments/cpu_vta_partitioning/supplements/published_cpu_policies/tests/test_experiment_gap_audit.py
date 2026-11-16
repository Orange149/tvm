from copy import deepcopy
from audit_experiment_gaps import execution_key,host_guard_and_capacity


def test_matched_gate_pair_cannot_change_threads_or_host_placement():
    a=dict(stages=[dict(device='cpu',begin=0,end=3,threads=2,mask=[0,1],worker_core_map=[0,1])],
           affinity_mode='pinned',execution_mode='off',host_mask=[2],control_mask=[0,1,2,3],
           compute_pool=[0,1,2,3],reserved_cores=[])
    b=dict(a,execution_mode='resource_group')
    assert execution_key(a,('gate',))==execution_key(b,('gate',))
    assert execution_key(a,('gate',))!=execution_key(dict(b,host_mask=[3]),('gate',))
    c=deepcopy(b);c['stages'][0]['threads']=1
    assert execution_key(a,('gate',))!=execution_key(c,('gate',))


def test_fixed_host_concentration_is_already_in_singleton_capacity():
    for demand in ([10,2,3,8],[1,1,1,1],[0,0,0,0]):
        for host in (0,5,31):
            for core in range(4):
                a,b=host_guard_and_capacity(demand,host,[core])
                assert a==b
    guard,capacity=host_guard_and_capacity([10,2,3,8],15,[0,1,2,3])
    assert guard>capacity
