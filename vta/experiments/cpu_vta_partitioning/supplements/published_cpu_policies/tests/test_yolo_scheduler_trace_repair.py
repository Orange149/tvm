from copy import deepcopy
from pathlib import Path
import shutil,struct,subprocess,sys
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT.parents[1]/'src')]
from yolo_scheduler_trace_repair.audit import capacity_row
from yolo_scheduler_trace_repair.header_patch import patched_header,once
from yolo_scheduler_trace_repair.events import decode_stream,validate_chunks
from yolo_scheduler_trace_repair.common import specs,protocol
from yolo_scheduler_trace_repair.campaign import run


def stream():
    raw=b''.join(struct.pack('<IHHIIQII',14,8192 if i%2 else 0,32,90,91,100+i*100,3,0) for i in range(8))
    return dict(tid=91,pid=90,allowed=[3],overflow=False,
        enable_start_ns=1,enable_end_ns=10,disable_start_ns=900,disable_end_ns=1000,
        head=256,tail=128,data_size=256,retained_bytes=256,records_hex=raw.hex(),
        counter_before=dict(value=0,enabled=0,running=0),counter_after=dict(value=400,enabled=900,running=400),
        drain_chunks=[dict(tail=0,head=128,head_after_copy=128,head_after_publish=128,offset=0,begin_ns=500,end_ns=510),
                      dict(tail=128,head=256,head_after_copy=256,head_after_publish=256,offset=128,begin_ns=1001,end_ns=1010)])


def test_continuous_chunks_can_exceed_physical_buffer_without_faking_capacity():
    s=stream();d=decode_stream(s,90)
    assert d['physical_capacity_bytes']==256 and d['retained_bytes']==256
    assert d['known_CPU_ns']==400 and d['coverage_ns']==[100,800]
    assert s['head']-s['tail']==128 # Physical last snapshot is distinct from retained bytes.


def test_original_full_buffer_without_lost_is_flagged():
    s=dict(tid=1,head=131040,tail=0,data_size=131072,disable_end_ns=100,
           records_hex=(b'\0'*131040).hex())
    d=dict(role=dict(role='host'),events=[0]*4095,coverage_ns=[1,20])
    assert capacity_row(s,d)['capacity_suspect']
    s['head']=64;s['records_hex']=(b'\0'*64).hex()
    assert not capacity_row(s,d)['capacity_suspect']


@pytest.mark.parametrize('change',('cursor','offset','near_full','copy_fill','publish_fill','time','bytes','capacity','final_tail'))
def test_invalid_drain_is_rejected(change):
    s=stream()
    if change=='cursor':s['drain_chunks'][1]['tail']=120
    elif change=='offset':s['drain_chunks'][1]['offset']=120
    elif change=='near_full':s['drain_chunks'][0]['head']=224
    elif change=='copy_fill':s['drain_chunks'][0]['head_after_copy']=224
    elif change=='publish_fill':s['drain_chunks'][0]['head_after_publish']=224
    elif change=='time':s['drain_chunks'][1]['begin_ns']=400
    elif change=='bytes':s['records_hex']=s['records_hex'][:-2]
    elif change=='capacity':s['data_size']=255
    elif change=='final_tail':s['tail']=0
    with pytest.raises(ValueError):validate_chunks(s)


def test_lost_marker_and_record_corruption_remain_rejected():
    s=stream();raw=bytes.fromhex(s['records_hex']);s['records_hex']=(struct.pack('<IHH',2,0,32)+raw[8:]).hex()
    with pytest.raises(ValueError,match='lost'):decode_stream(s,90)


def test_checked_patch_retains_original_wait_and_thread_creation():
    text=(ROOT/'yolo_scheduler_trace/session_profile.h').read_text();patched=patched_header(text)
    assert 'void Checkpoint()' in patched and 'DrainTask(t)' in patched
    assert 'ring_trace_revision' in patched and 'retention_end_ns' in patched
    assert 'std::thread' not in patched and 'sleep(' not in patched
    with pytest.raises(ValueError):once('x x','x','y')


def test_exact_three_sessions_qualification_clean_observed():
    s=specs();p=protocol()
    assert [x['phase'] for x in s]==['qualify','clean','observed']
    assert len(s)==3 and p['max_DNN_inferences']==1+2*(8+16+24)==97
    assert not p['new_sampling_thread'] and not p['new_periodic_timer']


def test_no_implicit_board_connection(tmp_path):
    with pytest.raises(ValueError,match='execute-board'):run(tmp_path,tmp_path,tmp_path)


def test_native_ring_reader_wrap_loss_guard_and_copy_race(tmp_path):
    cc=shutil.which('c++')
    if not cc:pytest.skip('host C++ compiler absent')
    src=tmp_path/'test.cc';exe=tmp_path/'test'
    src.write_text(r'''
#include "ring_drain.h"
#include <cassert>
using namespace scheduler_ring;
int main(){
 unsigned char data[256];for(int i=0;i<256;++i)data[i]=i;
 Retained s;U t=0,h=128,clock=1;
 auto head=[&](){return h;};auto now=[&](){return clock++;};auto pub=[&](U n){t=n;};
 Drain(s,data,256,t,head,now,pub);assert(t==128&&s.bytes.size()==128);
 h=256;Drain(s,data,256,t,head,now,pub);
 h=384;Drain(s,data,256,t,head,now,pub);
 assert(s.bytes.size()==384&&s.chunks.size()==3&&t==384);
 for(int i=0;i<384;++i)assert(s.bytes[i]==i%256);
 // A ring can be effectively full with no explicit LOST and head-tail<capacity.
 bool refused=false;h=t+224;
 try{Drain(s,data,256,t,head,now,pub);}catch(...){refused=true;}
 assert(refused&&t==384);
 // Also refuse filling between head snapshot and tail publication.
 Retained raced;U calls=0,pt=0;
 auto advancing=[&](){return ++calls==1?128ULL:224ULL;};
 refused=false;try{Drain(raced,data,256,0,advancing,now,[&](U v){pt=v;});}catch(...){refused=true;}
 assert(refused&&pt==0);
 // A preemption between final pre-publication check and publishing the tail.
 Retained late;calls=0;pt=0;
 auto late_head=[&](){return ++calls<=2?128ULL:224ULL;};
 refused=false;try{Drain(late,data,256,0,late_head,now,[&](U v){pt=v;});}catch(...){refused=true;}
 assert(refused); // Published cursor is irrelevant: this session is rejected.
 refused=false;try{Drain(s,data,256,t-1,head,now,pub);}catch(...){refused=true;}
 assert(refused);
}
''')
    p=subprocess.run([cc,'-std=c++14','-I',str(ROOT/'yolo_scheduler_trace_repair'),str(src),'-o',str(exe)],capture_output=True,text=True)
    assert p.returncode==0,p.stderr
    p=subprocess.run([str(exe)],capture_output=True,text=True,timeout=10)
    assert p.returncode==0,p.stderr
