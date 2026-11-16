from copy import deepcopy
import json
from pathlib import Path
import sys
import subprocess
import shutil
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from frame_stream.domain import cases,program_names
from frame_stream.common import DEFAULT_PARENT,api
from frame_stream.runtime import validate_rows
from frame_stream.model import rank

def fixture(name='validation/split2'):
    actors=cases()[name]['actors'];doc={'actors':actors}
    affinity={'cpu_worker_pinning':True,'stages':[],
        'control_threads':[{'role':'collector','tid':9000,'cpus':[0,1,2,3]},
                           {'role':'producer','tid':9001,'cpus':[0,1,2,3]}]}
    rows=[]
    for i,s in enumerate(actors):
        tids=[100+i*20+j for j in range(s['threads'])]
        affinity['stages'].append({'requested_cpus':s['mask'],'active_runtime_threads':s['threads'],
            'caller_tid':tids[0],'thread_ids':tids,
            'effective_affinities':[{'tid':tid,'cpus':[core],'cpu_time_delta_ms':10.} for tid,core in zip(tids,s['mask'])],
            'worker_slots':[{'slot':j,'tid':tid,'cpus':[core]} for j,(tid,core) in enumerate(zip(tids,s['mask']))]})
        for f in range(14):
            base=f*10_000_000;pub=base+i*10
            rows.append({'schema':'cpu_frame_stream_v1','actor':i,'frame_id':f,'input_index':0,
                'program':s['program'],'push_start_ns':pub,'push_return_ns':pub+5,
                'wait_start_ns':base+100,'dequeue_ns':base+200,'start_ns':base+300,
                'end_ns':base+1_001_000+i*2000,'set_ms':.1,'run_ms':.8,'get_ms':.1,'service_ms':1.,
                'qualification_passed':True,'final_output_passed':True,'calls':14,'extra_calls':0})
    return doc,rows,affinity

def test_eight_cases_include_fixed_tail():
    assert len(cases())==8 and len(program_names())==4
    assert sum(len(c['actors'])*15 for c in cases().values())==300
    for name,c in cases().items():
        assert all(s['device']=='cpu' for s in c['actors'])
        if name.startswith('explanation'):
            assert c['actors'][-1]['program']=='cpu_20_20' and c['actors'][-1]['mask']==[0]

@pytest.mark.parametrize('name',['validation/split2','explanation/split31'])
def test_group_clock_and_per_actor_frames(name):
    doc,rows,aff=fixture(name);s=validate_rows(rows,doc,aff)
    assert s['cpu_ii_ns']==10_000_000 and s['completion_gaps']==7 and s['scored_completion_count']==8
    assert s['stage_calls']==len(doc['actors'])*15
    assert len(s['group_completion_ns'])==14

def test_integer_ceiling_only_after_seven_gap_sum():
    doc,rows,aff=fixture();rows[14+9]['end_ns']+=3
    assert validate_rows(rows,doc,aff)['cpu_ii_ns']==10_000_001

@pytest.mark.parametrize('failure',['missing','duplicate','extra','wrong_role','wrong_core','wrong_worker',
    'wrong_input','qualification','final_output','nan_clock','backwards','source_order','bad_service','overlap'])
def test_invalid_evidence_rejected(failure):
    doc,rows,aff=fixture()
    if failure=='missing':rows.pop()
    if failure=='duplicate':rows[1]['frame_id']=0
    if failure=='extra':rows[0]['calls']=15
    if failure=='wrong_role':aff['control_threads'][0]['role']='producer'
    if failure=='wrong_core':aff['stages'][0]['effective_affinities'][0]['cpus']=[3]
    if failure=='wrong_worker':aff['stages'][0]['worker_slots'][0]['slot']=2
    if failure=='wrong_input':rows[0]['input_index']=1
    if failure=='qualification':rows[0]['qualification_passed']=False
    if failure=='final_output':rows[0]['final_output_passed']=False
    if failure=='nan_clock':rows[0]['end_ns']=float('nan')
    if failure=='backwards':rows[0]['dequeue_ns']=rows[0]['start_ns']+1
    if failure=='source_order':rows[14]['push_start_ns']=0
    if failure=='bad_service':rows[0]['run_ms']=-1.
    if failure=='overlap':rows[0]['end_ns']=rows[1]['wait_start_ns']+1
    with pytest.raises(ValueError):validate_rows(rows,doc,aff)

def model_fixture():
    values={'overlap4':10,'overlap2':16,'split2':13,'split31':12}
    models={'costs':{},'original_predictions':{},'reference_rankings':{}}
    summaries={}
    for name,c in cases().items():
        models['original_predictions'][name]={'original_score_ns':values[name.split('/')[1]],'saturated_score_ms':.001}
        summaries[name]={'cpu_ii_ns':11,'completion_gaps':7}
    for topology in ('explanation','validation'):
        names=[n for n in cases() if n.startswith(topology+'/')]
        models['reference_rankings'][topology]={m:sorted(names) for m in ('original','saturated')}
    return models,summaries

def test_scoring_has_no_io_or_fps_access(monkeypatch):
    model,data=model_fixture()
    import builtins
    def forbidden(*a,**kw):raise AssertionError('score must not open files')
    monkeypatch.setattr(builtins,'open',forbidden)
    result=rank(model,data)
    assert result['pipeline_feedback_used'] is False
    for name,row in result['predictions'].items():
        assert row['frame_stream_ns']==max(model['original_predictions'][name]['original_score_ns'],11)
    model['fps']=1
    with pytest.raises(ValueError):rank(model,data)

def test_actual_eight_task_argument_closure(tmp_path):
    from frame_stream.runtime import task,native_command,validate_document
    c=api(DEFAULT_PARENT);(tmp_path/'plans').mkdir()
    for directory in ('sources/base_controller/src','sources/cpu_strategy'):(tmp_path/directory).mkdir(parents=True)
    (tmp_path/'manifest.json').write_text(json.dumps({'bundle_id':'fixture'}))
    (tmp_path/'plans/executors.json').write_text(json.dumps({'frame_stream':{'path':'runner/new','sha256':'fixed'}}))
    for name in ('programs.json','inputs.json'):
        shutil.copy2(DEFAULT_PARENT/'plans'/name,tmp_path/'plans'/name)
    for name in cases():
        doc=task(tmp_path,name);argv,spec=native_command(doc,tmp_path,'/bundle','/job')
        assert '--cpu-frame-stream-spec' in argv and '--contention-spec' not in argv
        assert '--process-cpu-affinity' not in argv and '--cpu-worker-pinning' in argv
        spec=json.loads(spec);assert len(spec['actors'])==len(cases()[name]['actors'])
        for s,a in zip(cases()[name]['actors'],spec['actors']):
            if s['program']=='cpu_17_20':assert len(a['inputs'])==2
        bad=deepcopy(doc);bad['actors'].pop()
        with pytest.raises(ValueError):validate_document(bad,tmp_path)

def test_native_queue_no_frame_barrier_and_cancellation(tmp_path):
    # Exercise the exact existing BoundedQueue class, not a mirror implementation.
    source_path=DEFAULT_PARENT/'sources/contention_native/vta_stage_pipeline_runner_contention.cc'
    if not source_path.is_file():source_path=DEFAULT_PARENT/'sources/frame_stream_native/vta_stage_pipeline_runner_frame_stream.cc'
    source=source_path.read_text()
    begin=source.index('template <typename T>\nclass BoundedQueue')
    end=source.index('uint64_t EmptyStageTransform',begin)
    queue=source[begin:end]
    stub='''struct QueueTiming {void PushRequested(double){} void Enqueued(double){}
      void PushReturned(double){} void Dequeued(double){}}; double NowMillis(){return 0;}
'''
    test=r'''
#include "flow.h"
#include <cassert>
#include <chrono>
using cpu_frame_stream::Flow;
int main(){
  Flow<BoundedQueue> flow(2,2,14);std::mutex m;std::condition_variable cv;
  bool first=false,second=false;std::atomic<int> a{0},b{0};
  auto clock=[](){return 0LL;};
  std::thread slow([&]{flow.Consume(1,clock,[&](int f,auto,auto){
    if(f==0){std::unique_lock<std::mutex> l(m);cv.wait(l,[&]{return second;});}
    ++b;
  });});
  std::thread fast([&]{flow.Consume(0,clock,[&](int f,auto,auto){
    ++a;if(f==1){std::lock_guard<std::mutex> l(m);second=true;cv.notify_all();}
  });});
  std::thread producer([&]{flow.Produce(clock,[](auto,auto,auto,auto){});});
  producer.join();fast.join();slow.join();assert(a==14&&b==14&&second);
  // Empty consumers and a producer blocked on a full queue are cancellable.
  Flow<BoundedQueue> full(2,2,100);std::atomic<int> sent{0};
  std::thread blocked([&]{full.Produce(clock,[&](auto,auto,auto,auto){++sent;});});
  while(sent<4)std::this_thread::yield();full.Cancel();blocked.join();
  Flow<BoundedQueue> empty(2,2,14);
  std::thread waiting([&]{empty.Consume(0,clock,[](auto,auto,auto){assert(false);});});
  empty.Cancel();waiting.join();
  // An actor failure closes every queue and releases the producer and other actors.
  Flow<BoundedQueue> bad(2,2,14);
  std::thread broken([&]{try{bad.Consume(0,clock,[](auto,auto,auto){throw 1;});}catch(...){bad.Cancel();}});
  std::thread other([&]{bad.Consume(1,clock,[](auto,auto,auto){});});
  std::thread feeding([&]{bad.Produce(clock,[](auto,auto,auto,auto){});});
  feeding.join();broken.join();other.join();
}
'''
    includes='\n'.join('#include <'+h+'>' for h in ('atomic','memory','vector','mutex','condition_variable','deque','functional','stdexcept','thread','utility'))+'\n'
    path=tmp_path/'flow.cc';path.write_text(includes+stub+queue+test)
    binary=tmp_path/'flow';subprocess.run([shutil.which('c++'),'-std=c++17','-pthread','-I',str(ROOT/'frame_stream'),
        '-include',str(ROOT/'frame_stream/flow.h'),str(path),'-o',str(binary)],capture_output=True,check=True)
    subprocess.run([str(binary)],check=True,timeout=4)

def fake_bundle(tmp_path):
    c=api(DEFAULT_PARENT);bundle=tmp_path/'capsule';(bundle/'plans').mkdir(parents=True)
    for directory in ('sources/base_controller/src','sources/cpu_strategy'):(bundle/directory).mkdir(parents=True)
    (bundle/'manifest.json').write_text(json.dumps({'bundle_id':'fixture'}))
    (bundle/'plans/executors.json').write_text(json.dumps({'frame_stream':{'path':'runner/new','sha256':'fixed'}}))
    for name in ('programs.json','inputs.json'):shutil.copy2(DEFAULT_PARENT/'plans'/name,bundle/'plans'/name)
    return bundle

def test_receipt_replay_does_not_execute_twice_and_rejects_hash(tmp_path,monkeypatch):
    from frame_stream import campaign
    from frame_stream.runtime import task
    bundle=fake_bundle(tmp_path);root=tmp_path/'run';root.mkdir();doc=task(bundle,'validation/split2')
    state={'bundle':str(bundle),'boot_id':'boot-A','frozen_files':{}}
    summary={'correct':True,'stage_calls':30}
    monkeypatch.setattr(campaign,'validate_summary',lambda *a:deepcopy(summary))
    class Transport:
        calls=0
        def measure(self,d,b,r,local,boot):
            self.calls+=1;p=local/'raw'/boot/d['task_id']/'collected';p.mkdir(parents=True)
            (p/'result.jsonl').write_text('raw\n');return p
        def acknowledge(self,*a):pass
    t=Transport();record=campaign.Recorder(root,state,t,'/remote')
    assert record.saved(doc)==summary and record.saved(doc)==summary and t.calls==1
    raw=next((root/'raw').rglob('result.jsonl'));raw.write_text('changed\n')
    with pytest.raises(ValueError):record.saved(doc)

def test_wrong_boot_stops_before_deployment(tmp_path,monkeypatch):
    from frame_stream import campaign,prepare
    c=api(DEFAULT_PARENT);state={'bundle':str(DEFAULT_PARENT),'boot_id':'boot-A','frozen_files':{},'status':'INTERRUPTED'}
    monkeypatch.setattr(campaign,'validate_state',lambda *a:state)
    monkeypatch.setattr(prepare,'validate_bundle',lambda *a:None)
    class Transport:
        def boot_id(self):return 'boot-B'
        def wait_idle(self,*a):raise AssertionError('must stop before deployment')
    with pytest.raises(ValueError,match='boot'):campaign.run_current(tmp_path,Transport())
    assert c.read(tmp_path/'state.json')['status']=='STOPPED_INVALID_RESULT'

def test_disconnect_after_completion_recollects_without_launch(tmp_path):
    from frame_stream.runtime import task,native_command,_ssh_class
    from cpu_vta_partitioning.transport import TransportError
    bundle=fake_bundle(tmp_path);doc=task(bundle,'validation/split2');run=tmp_path/'run';run.mkdir()
    c=api(bundle);cls=_ssh_class(str(bundle));transport=object.__new__(cls)
    transport.config={'remote_base':'/board'};transport.boot_id=lambda:'boot-A'
    native_hash='';launched=0;collections=0
    def command(script):
        nonlocal native_hash,launched
        if 'nohup sh execute.sh' in script:native_hash=c.digest(doc);launched+=1;return ''
        if 'test -f ' in script and '/task_hash' in script:return native_hash
        if 'test -f ' in script and '/exit_code' in script:return '0'
        return ''
    transport.command=command;transport.upload=lambda *a:None;transport.check_space=lambda *a:None
    def collect(job,raw):
        nonlocal collections
        collections+=1
        if collections==1:raise TransportError('simulated disconnect during collection')
        out=raw/'collected';out.mkdir()
        for name,text in [('task_hash',c.digest(doc)),('boot_id','boot-A'),('inputs.txt',(raw/'inputs.txt').read_text())]:
            (out/name).write_text(text)
        return out
    transport.collect=collect
    with pytest.raises(TransportError):transport.measure(doc,bundle,'/bundle',run,'boot-A')
    transport.measure(doc,bundle,'/bundle',run,'boot-A')
    assert launched==1 and collections==2
