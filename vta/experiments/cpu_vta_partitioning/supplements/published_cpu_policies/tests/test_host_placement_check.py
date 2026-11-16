from pathlib import Path
from copy import deepcopy
import sys,pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT.parents[1]/'src')]
from cpu_vta_partitioning.common import read,write
from host_placement_check.common import specs,protocol,PARENT
from host_placement_check.campaign import full_specs,run
from host_placement_check.runtime import make_task,native_command,validate_profile
from host_placement_check.prepare import validate_input_paths


def test_bounded_task_identity_and_qualification():
    assert len(full_specs())==5
    seen=set();total=0
    for s in full_specs():
        if s['phase']=='qualify':seen.add(s['runner'])
        else:assert s['runner'] in seen
        total+=sum(protocol()['qualification' if s['phase']=='qualify' else 'pipeline'].values())
    assert total==314 and all(not s['probe'] for s in specs())


def bundle(tmp_path):
    parent=PARENT/'yolov3_tiny416/capsule'
    for name in ('programs','inputs','configurations'):write(tmp_path/'plans'/(name+'.json'),read(parent/'plans'/(name+'.json')))
    m=read(parent/'manifest.json');m['phase_runner_sha256']='new';m['old_runner_sha256']='old';m['adapter_sha256']=None
    write(tmp_path/'manifest.json',m);return tmp_path


def test_probe_is_a_separate_execution_identity_and_uses_same_runner(tmp_path):
    b=bundle(tmp_path);clean,observed=map(lambda s:make_task(b,s),full_specs()[2:4])
    assert clean['configuration']==observed['configuration'] and clean['executor_sha256']==observed['executor_sha256']
    assert clean['task_id']!=observed['task_id'] and not clean['counter_probe'] and observed['counter_probe']
    a,_=native_command(observed,b,'/bundle','/job');off,_=native_command(clean,b,'/bundle','/job')
    assert '--session-profile-json' in a and '--session-profile-json' not in off
    assert '--session-profile-window' in a and a[a.index('--session-profile-window')+1]=='8,64'
    assert '--phase-event-json' not in a and not any('libwait_adapter' in x for x in a)
    assert '--process-cpu-affinity' not in a and '--cpu-worker-pinning' in a


def test_both_executors_qualify_without_old_waiting_adapter(tmp_path):
    b=bundle(tmp_path)
    for spec in full_specs()[:2]:
        doc=make_task(b,spec);a,_=native_command(doc,b,'/bundle','/job')
        assert doc['kind']=='qualify_pipeline' and (doc['warmup'],doc['scored'],doc['tail'])==(0,1,0)
        assert a[a.index('--output-mode')+1]=='raw_all_stages' and '--output-dump-dir' in a
        assert '--phase-event-json' not in a and doc['adapter_sha256'] is None
        assert doc['executor_sha256']==('new' if spec['runner']=='phase' else 'old')


def fixture():
    stages=[dict(program='cpu_00_02',device='cpu',mask=[0,1],worker_core_map=[0,1]),dict(program='vta_03_06',device='vta',mask=[0,1,2,3])]
    affinity=dict(stages=[dict(caller_tid=1,worker_slots=[dict(slot=0,tid=1,cpus=[0]),dict(slot=1,tid=2,cpus=[1])]),
                         dict(caller_tid=3,worker_slots=[])],control_threads=[dict(role='producer',tid=4,cpus=[0,1,2,3]),dict(role='collector',tid=5,cpus=[0,1,2,3])])
    rows=[]
    for tid,allowed,core in [(1,[0],0),(2,[1],1),(3,[0,1,2,3],2),(4,[0,1,2,3],3),(5,[0,1,2,3],3)]:
        zero=[dict(value=0,enabled=10**8,running=0) for _ in range(5)]
        after=deepcopy(zero)
        for x in after:x['enabled']=2*10**8
        for i in (0,core+1):after[i].update(value=50000000,running=50000000)
        rows.append(dict(tid=tid,allowed=allowed,before=dict(read_start_ns=100000000+tid*10000,read_end_ns=100001000+tid*10000,counts=zero),
            after=dict(read_start_ns=201000000+tid*10000,read_end_ns=201001000+tid*10000,counts=after)))
    p=dict(status='MEASURED',sampling=False,scaled=False,cpus=[0,1,2,3],setup_start_ns=100000000,start_reads_end_ns=101000000,
        stop_reads_start_ns=201000000,stop_reads_end_ns=202000000,completion_start_ns=99000000,completion_end_ns=200000000,tasks=rows)
    return p,affinity,dict(stages=stages)


def test_distinct_roles_and_pinned_callers_use_their_actual_single_core():
    p,a,d=fixture();r=validate_profile(p,a,d)
    assert len(r['roles'])==5 and r['no_rescaling']
    assert r['roles'][0]['requested']==[0] and r['roles'][2]['by_core_ns']==[0,0,50000000,0]
    assert r['start_after_completion_ns']==[1000000,2000000]


def test_missing_or_duplicate_tid_and_affinity_escape_rejected():
    p,a,d=fixture();p['tasks'].pop()
    with pytest.raises(ValueError):validate_profile(p,a,d)
    p,a,d=fixture();p['tasks'].append(deepcopy(p['tasks'][0]))
    with pytest.raises(ValueError):validate_profile(p,a,d)
    p,a,d=fixture();p['tasks'][0]['allowed']=[0,1]
    with pytest.raises(ValueError):validate_profile(p,a,d)


def test_bad_conservation_cpu_filter_and_backwards_counters_rejected():
    p,a,d=fixture();p['tasks'][0]['after']['counts'][2]['value']=50000000
    with pytest.raises(ValueError):validate_profile(p,a,d)
    p,a,d=fixture();p['tasks'][0]['before']['counts'][0]['value']=60000000
    with pytest.raises(ValueError):validate_profile(p,a,d)
    p,a,d=fixture();p['tasks'][0]['after']['counts'][1]['value']=0;p['tasks'][0]['after']['counts'][2]['value']=50000000
    with pytest.raises(ValueError):validate_profile(p,a,d)


def test_unsupported_scaled_or_wrong_window_not_zero_work():
    p,a,d=fixture();p['status']='UNAVAILABLE'
    with pytest.raises(ValueError):validate_profile(p,a,d)
    p,a,d=fixture();p['scaled']=True
    with pytest.raises(ValueError):validate_profile(p,a,d)
    p,a,d=fixture();p['completion_start_ns']=300000000
    with pytest.raises(ValueError):validate_profile(p,a,d)


def test_no_native_connection_without_explicit_gate(tmp_path):
    with pytest.raises(ValueError,match='execute-board'):run(tmp_path,tmp_path,tmp_path)


def test_frontier_references_must_be_present_and_listed_in_payload(tmp_path):
    write(tmp_path/'plans/inputs.json',dict(records=[dict(input='inputs/a',frontiers={'70':dict(raw='references/a.bin',npy='references/a.npy')})]))
    m=dict(files={n:{} for n in ('inputs/a','references/a.bin')})
    for n in m['files']:
        p=tmp_path/n;p.parent.mkdir(exist_ok=True);p.write_bytes(b'input')
    with pytest.raises(ValueError,match='reference not packaged'):validate_input_paths(tmp_path,m)
    (tmp_path/'references/a.npy').write_bytes(b'reference')
    with pytest.raises(ValueError):validate_input_paths(tmp_path,m)
    m['files']['references/a.npy']={};validate_input_paths(tmp_path,m)
