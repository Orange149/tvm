from pathlib import Path
from copy import deepcopy
import struct,sys,pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT.parents[1]/'src')]
from scheduler_event_audit.decode import OUT,PREEMPT,decode,intervals,validate_stream,check
from scheduler_event_audit.workflow import run,qualify

MS=1000000


def record(tid,when,cpu=0,out=False,preempt=False,pid=99):
    return struct.pack('<IHHIIQII',14,(OUT if out else 0)|(PREEMPT if preempt else 0),32,pid,tid,when,cpu,0)


def segment(a,b,cpu,CPU,kind='busy'):
    return dict(kind=kind,requested_core=cpu,actual_start_core=cpu,actual_end_core=cpu,
        wall_start_ns=a,wall_end_ns=b,cpu_start_ns=0,cpu_end_ns=CPU)


def stream(case,index,tid,segments,events):
    start=segments[0]['wall_start_ns'];end=segments[-1]['wall_end_ns'];raw=b''.join(events)
    return dict(case=case,index=index,pid=99,tid=tid,policy=0,nice=0,error='',enable_start_ns=start-MS,
        enable_end_ns=start-500000,disable_start_ns=end+500000,disable_end_ns=end+MS,
        head=len(raw),tail=0,data_size=65536,overflow=False,segments=segments,records_hex=raw.hex())


def document():
    a=1000*MS;b=2000*MS;c=3000*MS;d=4000*MS
    rows=[stream('alone',0,10,[segment(a,a+150*MS,0,150*MS)],[]),
        stream('contended',0,11,[segment(b,b+300*MS,0,150*MS)],
            [record(11,b+75*MS,out=True,preempt=True),record(11,b+150*MS),record(11,b+225*MS,out=True,preempt=True),record(11,b+300*MS)]),
        stream('contended',1,12,[segment(b+75*MS,b+300*MS,0,150*MS)],
            [record(12,b+150*MS,out=True,preempt=True),record(12,b+225*MS)]),
        stream('sleep',0,13,[segment(c,c+150*MS,0,50000,'sleep')],
            [record(13,c+20000,out=True),record(13,c+150*MS-30000)]),
        stream('migrate',0,14,[segment(d,d+100*MS,0,100*MS),segment(d+105*MS,d+205*MS,1,100*MS)],
            [record(14,d+100*MS+1000,out=True),record(14,d+105*MS-1000,cpu=1)])]
    return dict(status='MEASURED',DNN_inferences=0,periodic_sampling=False,byte_order='little',sample_id_layout='TID,TIME,CPU',streams=rows)


def replace_raw(s,raw):s.update(records_hex=raw.hex(),head=s['tail']+len(raw))


def test_complete_positive_controls_and_timeline_CPU_conservation():
    v=check(document());assert v['passed'] and v['total_switch_records']==10
    assert v['preemption_positive_control'] and v['known_sleep_nonpreempt_event'] and v['migration_cpu_records']
    assert v['max_same_core_TID_overlap_ns']==0 and not v['exact_wakeup_instants_available']
    assert all(s['difference_ns']==0 for r in v['streams'] for s in r['segments'])


def test_no_switches_in_isolated_busy_run_is_valid_not_failed_observation():
    s=document()['streams'][0];v=validate_stream(s)
    assert v['switches']==0 and v['segments'][0]['reconstructed_CPU_ns']==150*MS


def test_no_events_in_sleep_does_not_mean_no_wait():
    s=document()['streams'][3];replace_raw(s,b'')
    with pytest.raises(ValueError,match='CPU clock'):validate_stream(s)


@pytest.mark.parametrize('kind',[2,13])
def test_lost_record_types_and_ring_overflow_are_rejected(kind):
    s=document()['streams'][0];replace_raw(s,struct.pack('<IHH',kind,0,8))
    with pytest.raises(ValueError,match='lost'):decode(s)
    s=document()['streams'][0];s['overflow']=True
    with pytest.raises(ValueError,match='overflow'):decode(s)


def test_truncated_record_unknown_type_and_wrong_task_rejected():
    s=document()['streams'][1];raw=bytes.fromhex(s['records_hex']);replace_raw(s,raw[:-1])
    with pytest.raises(ValueError):decode(s)
    s=document()['streams'][0];replace_raw(s,struct.pack('<IHH',9,0,8))
    with pytest.raises(ValueError):decode(s)
    s=document()['streams'][1];replace_raw(s,record(999,s['segments'][0]['wall_start_ns'],out=True))
    with pytest.raises(ValueError,match='task identity'):decode(s)


def test_event_order_direction_and_preemption_flags_checked():
    s=document()['streams'][1];t=s['segments'][0]['wall_start_ns']
    replace_raw(s,record(11,t+2*MS,out=True)+record(11,t+MS))
    with pytest.raises(ValueError,match='timestamp'):decode(s)
    replace_raw(s,record(11,t+MS,out=True)+record(11,t+2*MS,out=True))
    with pytest.raises(ValueError,match='duplicate switch-out'):validate_stream(s)
    replace_raw(s,record(11,t+MS,preempt=True))
    with pytest.raises(ValueError,match='switch-in'):decode(s)


def test_linearized_wrapped_ring_preserves_all_record_bytes():
    s=document()['streams'][1];raw=bytes.fromhex(s['records_hex']);s['tail']=65520;s['head']=65520+len(raw)
    assert len(decode(s))==4


def test_wrong_clock_affinity_or_missing_migration_not_accepted():
    s=document()['streams'][1];s['enable_start_ns']=s['disable_end_ns']+1
    with pytest.raises(ValueError):validate_stream(s)
    s=document()['streams'][1];s['segments'][0]['actual_end_core']=1
    with pytest.raises(ValueError,match='affinity'):validate_stream(s)
    s=document()['streams'][4];replace_raw(s,b'')
    with pytest.raises(ValueError,match='CPU differs'):validate_stream(s)


def test_nonpreempt_out_not_automatically_sleep_and_lack_of_preempt_is_failure():
    d=document()
    for s in d['streams'][1:3]:
        raw=bytearray.fromhex(s['records_hex'])
        for i in range(0,len(raw),32):
            misc=struct.unpack_from('<H',raw,i+4)[0];struct.pack_into('<H',raw,i+4,misc&~PREEMPT)
        replace_raw(s,raw)
    assert not check(d)['passed']
    assert check(document())['nonpreempt_out_does_not_always_mean_sleep']


def test_missing_streams_and_unsupported_interfaces_are_not_zero_wait():
    d=document();d['streams'].pop()
    with pytest.raises(ValueError,match='incomplete'):check(d)
    assert not check(dict(status='UNAVAILABLE'))['passed']
    d=document();d['streams'][0]['overflow']=True
    v=qualify(d);assert not v['passed'] and v['invalid_records_not_zero_wait']


def test_board_access_requires_explicit_gate(tmp_path):
    with pytest.raises(ValueError,match='execute-board'):run(tmp_path)
