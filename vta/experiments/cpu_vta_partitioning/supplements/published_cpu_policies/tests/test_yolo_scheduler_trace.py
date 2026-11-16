from pathlib import Path
from copy import deepcopy
import struct,sys,pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT.parents[1]/'src')]
from cpu_vta_partitioning.common import read,write
from yolo_scheduler_trace.common import PARENT,specs,protocol
from yolo_scheduler_trace.events import OUT,PREEMPT,decode_stream,clip,Attribution
from yolo_scheduler_trace.runtime import make_task,native_command
from yolo_scheduler_trace.campaign import run


def event(tid,t,core=3,out=False,preempt=False,pid=90):
    return struct.pack('<IHHIIQII',14,(OUT if out else 0)|(PREEMPT if preempt else 0),32,pid,tid,t,core,0)


def stream(raw,tid=91,core=3,CPU=100):
    return dict(tid=tid,pid=90,allowed=[core],overflow=False,enable_start_ns=1,enable_end_ns=10,
        disable_start_ns=900,disable_end_ns=1000,head=len(raw),tail=0,data_size=131072,records_hex=raw.hex(),
        counter_before=dict(value=0,enabled=0,running=0),counter_after=dict(value=CPU,enabled=900,running=CPU))


def test_interior_intervals_leave_both_boundaries_unknown():
    raw=event(91,100,out=True,preempt=True)+event(91,200)+event(91,300,out=True)
    d=decode_stream(stream(raw),90);assert d['coverage_ns']==[100,300]
    full=clip(d,50,350)
    assert (full['on_CPU_ns'],full['off_CPU_ns'],full['unknown_ns'])==(100,100,100)
    assert full['preempt_off_ns']==100 and not full['fully_covered']
    interior=clip(d,110,290)
    assert interior['fully_covered'] and interior['unknown_ns']==0


def test_empty_stream_is_unknown_not_fake_running_or_zero_wait():
    d=decode_stream(stream(b''),90);v=clip(d,100,500)
    assert v['unknown_ns']==400 and v['on_CPU_ns']==v['off_CPU_ns']==0
    assert d['coverage_ns'] is None


def test_four_core_and_nonpreempt_semantics():
    d=decode_stream(stream(event(91,100,out=True)+event(91,400)),90)
    x=clip(d,100,400)
    assert x['nonpreempt_off_ns']==300 and x['preempt_off_ns']==0
    assert not x['on'] and all(r['from_cpu']==3 for r in x['off'])


def test_lost_partial_duplicate_and_wrong_tid_are_rejected():
    for raw in [struct.pack('<IHH',2,0,8),event(91,100)[:-1],event(999,100),event(91,200)+event(91,100,out=True),
                event(91,100,out=True)+event(91,200,out=True)]:
        with pytest.raises(ValueError):decode_stream(stream(raw),90)
    s=stream(b'');s['overflow']=True
    with pytest.raises(ValueError,match='overflow'):decode_stream(s,90)


def test_core_migration_requires_out_then_in():
    s=stream(event(91,100,core=2)+event(91,200,core=3,out=True));s['allowed']=[2,3]
    with pytest.raises(ValueError,match='without switch'):decode_stream(s,90)
    s=stream(event(91,100,core=2,out=True)+event(91,200,core=3));s['allowed']=[2,3]
    assert decode_stream(s,90)['off'][0]['to_cpu']==3


def test_attribution_separates_known_work_from_unobserved_time():
    target=decode_stream(stream(event(91,200,out=True,preempt=True)+event(91,400)),90)
    host=decode_stream(stream(event(92,210)+event(92,300,out=True),tid=92,CPU=90),90)
    other=decode_stream(stream(event(93,300)+event(93,360,out=True),tid=93,CPU=60),90)
    streams={'91':{**target,'role':dict(role='front/worker3')},'92':{**host,'role':dict(role='VTA/host')},'93':{**other,'role':dict(role='other_CPU')}}
    a=Attribution(streams).other_work(target['off'][0],3)
    assert a['by_role_ns']=={'VTA/host':90,'other_CPU':60} and a['unobserved_ns']==50
    assert a['coincident_CPU_work_not_proof_of_all_wait_cause']


def test_bounded_sessions_and_thread_lifetime_tail():
    seen=set();count=0
    for s in specs():
        if s['phase']=='qualify':seen.add(s['runner'])
        else:assert s['runner'] in seen
        count+=sum(protocol()['qualification' if s['phase']=='qualify' else 'pipeline'].values())
    assert count==146 and len(specs())==5
    assert protocol()['pipeline']['tail']>6*2+5+1


def bundle(tmp):
    parent=PARENT/'yolov3_tiny416/capsule'
    for name in ('programs','inputs','configurations'):write(tmp/'plans'/(name+'.json'),read(parent/'plans'/(name+'.json')))
    m=read(parent/'manifest.json');m.update(phase_runner_sha256='new',old_runner_sha256='old',adapter_sha256='adapter')
    write(tmp/'manifest.json',m);return tmp


def test_clean_qualifications_and_observed_identity_are_independent(tmp_path):
    b=bundle(tmp_path);documents=[make_task(b,s) for s in specs()]
    assert len({d['task_id'] for d in documents})==5
    assert documents[2]['executor_sha256']==documents[3]['executor_sha256']=='new'
    for doc in documents[:2]:
        a,_=native_command(doc,b,'/bundle','/job')
        assert '--output-dump-dir' in a and '--phase-event-json' not in a and '--session-profile-json' not in a
        assert '--cpu-worker-pinning' in a and '--resource-affinity-probe' in a
    a,_=native_command(documents[3],b,'/bundle','/job')
    assert a[a.index('--session-profile-window')+1]=='8,16' and '--phase-event-json' in a
    assert 'VTA_QUEUE_COUNTERS=1' in a and 'VTA_WAIT_POLICY=original' in a
    assert '--process-cpu-affinity' not in a and '--queue-timing' not in a


def test_no_board_without_explicit_gate(tmp_path):
    with pytest.raises(ValueError,match='execute-board'):run(tmp_path,tmp_path,tmp_path)
