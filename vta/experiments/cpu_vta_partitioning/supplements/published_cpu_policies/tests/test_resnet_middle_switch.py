from copy import deepcopy
from pathlib import Path
import struct
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_resnet_middle_switch as w
from cpu_vta_partitioning.common import read


def profile_fixture():
    cfg=w.selection()['selected_configurations']['resnet18'][0]
    affinity=dict(stages=[],control_threads=[]);next_tid=100
    for s in cfg['stages']:
        if s['device']=='cpu':
            slots=[dict(slot=i,tid=next_tid+i) for i in range(s['threads'])]
            affinity['stages'].append(dict(caller_tid=next_tid,worker_slots=slots));next_tid+=s['threads']
        else:affinity['stages'].append(dict(caller_tid=next_tid));next_tid+=1
    affinity['control_threads']=[dict(role='producer',tid=next_tid),dict(role='collector',tid=next_tid+1)]
    doc=dict(stages=cfg['stages'],configuration=cfg)
    roles=w.old_events.role_map(doc,affinity);streams=[]
    for i,(tid,role) in enumerate(roles.items()):
        start=20000+i*10000;core=role['allowed'][0]
        raw=b''.join(struct.pack('<IHHIIQII',14,misc,32,90,tid,t,core,0)
            for misc,t in [(0,start),(8192,start+1000)])
        streams.append(dict(tid=tid,pid=90,allowed=role['allowed'],overflow=False,alive_at_stop=True,
            enable_start_ns=1000,enable_end_ns=1500,disable_start_ns=212000,disable_end_ns=213000,
            data_size=131072,head=64,tail=0,retained_bytes=64,records_hex=raw.hex(),
            counter_before=dict(value=0,enabled=0,running=0),counter_after=dict(value=1000,enabled=200000,running=200000),
            drain_chunks=[dict(tail=0,head=64,head_after_copy=64,head_after_publish=64,offset=0,begin_ns=218000,end_ns=219000)]))
    profile=dict(status='MEASURED',clock='CLOCK_MONOTONIC',byte_order='little',sample_id_layout='TID,TIME,CPU',
        periodic_sampling=False,pid=90,ring_trace_revision=2,drain_policy='collector_completion',retention_end_ns=230000,
        completion_start_ns=800,setup_start_ns=900,setup_end_ns=2000,completion_end_ns=200000,
        stop_start_ns=210000,stop_end_ns=220000,streams=streams)
    return profile,doc,affinity


def test_all_thirteen_roles_and_strict_physical_record_decoder():
    p,d,a=profile_fixture();r=w.validate_events(p,d,a)
    assert len(r['streams'])==13 and r['total_events']==26
    for change in ('missing','lost','near_full','wrong_core'):
        bad=deepcopy(p)
        if change=='missing':bad['streams'].pop()
        elif change=='lost':
            s=bad['streams'][0];raw=bytes.fromhex(s['records_hex']);s['records_hex']=(struct.pack('<IHH',2,0,32)+raw[8:]).hex()
        elif change=='near_full':bad['streams'][0]['drain_chunks'][0]['head_after_publish']=131040
        else:bad['streams'][0]['allowed']=[3]
        with pytest.raises(ValueError):w.validate_events(bad,d,a)


def test_disabled_queries_are_explicit_unknowns_not_zero_wait_evidence():
    fields=dict.fromkeys(('queue_before_ns','queue_after_ns','queue_read_start0_ns','queue_read_end0_ns','queue_read_start1_ns','queue_read_end1_ns'),0)
    assert w.disabled_queue_rows([fields],scope='worker')['zero_fields_are_not_queue_measurements']
    with pytest.raises(ValueError):w.disabled_queue_rows([{**fields,'queue_after_ns':1}],scope='worker')
    with pytest.raises(ValueError):w.disabled_queue_rows([],scope='worker')


def test_native_port_is_only_target_count_and_no_implicit_board():
    text=(w.FROZEN/'sources/vta_session_profile.h').read_text();new,proof=w.port_header(text)
    assert new.replace('targets_.size()!=13','targets_.size()!=12')==text
    assert proof['other_native_recorder_code_unchanged']
    with pytest.raises(ValueError,match='execute-board'):w.board()


def test_one_executor_and_no_taskstats_in_observed_command():
    b=w.target.POOL/'resnet18/capsule'
    docs=[w.make_task(b,s) for s in w.specs()]
    assert len({d['executor_sha256'] for d in docs})==1
    assert sum(sum(d[k] for k in ('warmup','scored','tail')) for d in docs)==57
    argv,_=w.native_command(docs[-1],b,'/bundle','/job')
    assert 'VTA_QUEUE_COUNTERS=0' in argv and 'VTA_WAIT_POLICY=original' in argv
    assert argv[argv.index('--session-profile-window')+1]=='8,16'
    clean,_=w.native_command(docs[1],b,'/bundle','/job')
    assert '--session-profile-json' not in clean and '--phase-event-json' not in clean
