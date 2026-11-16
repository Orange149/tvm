from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from frame_stream_top20.common import FULL_PARENT, FRAME_PARENT, api
from frame_stream_top20.domain import pool_from_rankings, groups
from frame_stream_top20.runtime import validate_rows, cpu_task, pipeline_task, native_command, validate_document
from frame_stream_top20.model import rank

def static():
    c=api(FULL_PARENT)
    return c.read(ROOT/'runs/resnet18_open_cpu_resource_02/rankings.json')

def inputs():
    c=api(FULL_PARENT)
    return c.read(FULL_PARENT/'plans/programs.json'),c.read(FULL_PARENT/'plans/inputs.json')

def plan():
    pool=pool_from_rankings(static());p,i=inputs()
    return pool,groups(pool,p,i,{'path':'runner/new','sha256':'frozen'})

def fixture(gated=False,reserved=False,four=False):
    pool,gp=plan()
    cfg=next(c for c in pool.values() if (not gated or c['execution_mode']=='resource_group')
             and (not reserved or len(c['control_mask'])==1)
             and (not four or sum(s['device']=='cpu' for s in c['stages'])==4))
    doc={'actors':[s for s in cfg['stages'] if s['device']=='cpu'],
         'control_mask':cfg['control_mask'],'gate':cfg['execution_mode']}
    aff={'cpu_worker_pinning':True,'stages':[],
         'control_threads':[{'role':'collector','tid':9000,'cpus':cfg['control_mask']},
                            {'role':'producer','tid':9001,'cpus':cfg['control_mask']}]}
    rows=[]
    for i,s in enumerate(doc['actors']):
        tids=[100+i*20+j for j in range(s['threads'])]
        aff['stages'].append({'requested_cpus':s['mask'],'active_runtime_threads':s['threads'],
            'caller_tid':tids[0],'thread_ids':tids,
            'effective_affinities':[{'tid':tid,'cpus':[core],'cpu_time_delta_ms':10.} for tid,core in zip(tids,s['mask'])],
            'worker_slots':[{'slot':j,'tid':tid,'cpus':[core]} for j,(tid,core) in enumerate(zip(tids,s['mask']))]})
        for f in range(14):
            base=f*10_000_000;pub=base+i*10;start=base+10_000+i*100_000
            rows.append({'schema':'cpu_frame_stream_top20_v1','actor':i,'frame_id':f,'input_index':0,
                'program':s['program'],'push_start_ns':pub,'push_return_ns':pub+5,
                'wait_start_ns':base+100,'dequeue_ns':base+200,'gate_request_ns':base+300,
                'gate_acquire_ns':start,'start_ns':start,'end_ns':start+90_000,
                'gate_release_ns':start+90_001,'gate_mode':doc['gate'],
                'set_ms':.01,'run_ms':.07,'get_ms':.01,'service_ms':.09,
                'qualification_passed':True,'final_output_passed':True,'calls':14,'extra_calls':0})
    return doc,rows,aff

def test_complete_frozen_DP_domain_and_pool():
    s=static();pool,gp=plan()
    assert s['counts']=={'contexts':9,'enhanced':966240306,'prefix':6121596,'single_core':6121596,'topologies':4623}
    assert len(pool)==28 and len(gp['groups'])==28
    assert any(len(v['actors'])==4 for v in gp['groups'].values())
    assert any(v['gate']=='resource_group' for v in gp['groups'].values())
    assert any(len(v['control_mask'])==1 for v in gp['groups'].values())

@pytest.mark.parametrize('kw',[{'four':True},{'gated':True},{'reserved':True},{}])
def test_frame_conservation_group_clock_and_resource_context(kw):
    doc,rows,aff=fixture(**kw);r=validate_rows(rows,doc,aff)
    assert r['cpu_ii_ns']==10_000_000 and r['stage_calls']==15*len(doc['actors'])
    assert r['completion_gaps']==7 and len(r['group_completion_ns'])==14

@pytest.mark.parametrize('bad',['missing','duplicate','extra','wrong_output','wrong_core','wrong_control',
                               'wrong_worker','wrong_gate','wrong_input','clock','service'])
def test_invalid_CPU_evidence_rejected(bad):
    doc,rows,aff=fixture(four=True)
    if bad=='missing':rows.pop()
    if bad=='duplicate':rows[1]['frame_id']=0
    if bad=='extra':rows[0]['calls']=15
    if bad=='wrong_output':rows[0]['final_output_passed']=False
    if bad=='wrong_core':aff['stages'][0]['effective_affinities'][0]['cpus']=[9]
    if bad=='wrong_control':aff['control_threads'][0]['cpus']=[2]
    if bad=='wrong_worker':aff['stages'][0]['worker_slots'][0]['slot']=9
    if bad=='wrong_gate':rows[0]['gate_mode']='exclusive'
    if bad=='wrong_input':rows[0]['input_index']=1
    if bad=='clock':rows[0]['gate_acquire_ns']=rows[0]['end_ns']+1
    if bad=='service':rows[0]['get_ms']=float('nan')
    with pytest.raises(ValueError):validate_rows(rows,doc,aff)

def test_same_group_overlapping_work_rejected():
    doc,rows,aff=fixture(gated=True)
    # Force a pair of actual equal-mask actors to overlap, preserving local clocks.
    a,b=next((i,j) for i,s in enumerate(doc['actors']) for j,t in enumerate(doc['actors'][:i]) if s['mask']==t['mask'])
    for field in ('gate_acquire_ns','start_ns','end_ns','gate_release_ns'):
        rows[a*14][field]=rows[b*14][field]
    with pytest.raises(ValueError,match='same-group'):validate_rows(rows,doc,aff)

def test_ceiling_after_seven_intervals():
    doc,rows,aff=fixture(four=True);rows[-14+9]['end_ns']+=3;rows[-14+9]['gate_release_ns']+=3
    assert validate_rows(rows,doc,aff)['cpu_ii_ns']==10_000_001

def test_equivalent_CPU_work_can_share_host_only_change():
    from cpu_policies.open_resources.domain import candidate
    c=api(FULL_PARENT);p,i=inputs();old=next(iter(pool_from_rankings(static()).values()))
    first=candidate(old['stages'],'shared4',old['execution_mode'])
    stages=deepcopy(first['stages'])
    for s in stages:
        if s['device']=='vta':s['mask']=[0]
    second=candidate(stages,'host_shared_0',old['execution_mode'])
    gp=groups({first['id']:first,second['id']:second},p,i,{'sha256':'same'})
    assert len(gp['groups'])==1 and len(next(iter(gp['groups'].values()))['members'])==2
    second['control_mask']=[1]
    with pytest.raises(ValueError):groups({second['id']:second},p,i,{'sha256':'same'})

def fake_bundle(tmp_path):
    c=api(FULL_PARENT);b=tmp_path/'bundle';(b/'plans').mkdir(parents=True)
    for d in ('sources/base_controller/src','sources/cpu_strategy'):(b/d).mkdir(parents=True)
    pool,gp=plan()
    for name,value in [('static_pool.json',pool),('cpu_groups.json',gp),
                       ('executors.json',{'frame_stream':{'path':'runner/new','sha256':'frozen'},
                                          'policy':{'path':'runner/old','sha256':'old'}})]:
        (b/'plans'/name).write_text(json.dumps(value))
    (b/'manifest.json').write_text(json.dumps({'bundle_id':'fixture'}))
    for name in ('programs.json','inputs.json'):shutil.copyfile(FULL_PARENT/'plans'/name,b/'plans'/name)
    return b

def test_all_tasks_have_exact_frontiers_roles_and_closed_identity(tmp_path):
    b=fake_bundle(tmp_path);pool,gp=plan()
    for group,value in gp['groups'].items():
        doc=cpu_task(b,group);argv,serialized=native_command(doc,b,'/capsule','/job');spec=json.loads(serialized)
        assert '--process-cpu-affinity' not in argv
        assert argv[argv.index('--cpu-stage-gate')+1]==value['gate']
        assert argv[argv.index('--control-cpu-affinity')+1]==','.join(map(str,value['control_mask']))
        for row,actor in zip(value['actors'],spec['actors']):
            p=api(b).read(b/'plans/programs.json')[row['program']]
            assert len(actor['inputs'])==len(p['input_names']) and len(actor['references'])==len(p['output_node_ids'])
        bad=deepcopy(doc);bad['executor_sha256']='wrong'
        with pytest.raises(ValueError):validate_document(bad,b)
    for name in pool:
        for qualification in (True,False):
            doc=pipeline_task(b,name,qualification);argv,_=native_command(doc,b,'/capsule','/job')
            assert '--process-cpu-affinity' not in argv and doc['campaign']=='resnet18_dp_frame_stream_top20'
            assert doc['scored']==(1 if qualification else 16)

def test_model_no_IO_or_pipeline_FPS_access(monkeypatch):
    pool,gp=plan();measurements={g:{'cpu_ii_ns':100_000_000,'completion_gaps':7} for g in gp['groups']}
    def forbidden(*a,**kw):raise AssertionError('ranker must not open files')
    monkeypatch.setattr('builtins.open',forbidden)
    result=rank(pool,gp,measurements)
    assert len(result['top20'])==20 and not result['pipeline_feedback_used']
    for row in result['rankings']:assert row['score_ns']==max(row['original_score_ns'],100_000_000)
    measurements.pop(next(iter(measurements)))
    with pytest.raises(ValueError):rank(pool,gp,measurements)

def test_freezing_rankings_cannot_read_pipeline_observations(tmp_path,monkeypatch):
    from frame_stream_top20 import campaign
    b=fake_bundle(tmp_path);root=tmp_path/'run';root.mkdir();c=api(b)
    state={'bundle':str(b),'frozen_files':{}}
    records={g:{'cpu_ii_ns':100_000_000,'completion_gaps':7} for g in plan()[1]['groups']}
    monkeypatch.setattr(campaign,'cpu_records',lambda *args:records)
    old=c.read
    def read(path,*args,**kwargs):
        if Path(path).name not in ('static_pool.json','cpu_groups.json'):
            raise AssertionError('rank attempted to read observations: '+str(path))
        return old(path,*args,**kwargs)
    monkeypatch.setattr(c,'read',read)
    value=campaign.freeze_predictions(root,state)
    assert len(value['top20'])==20 and (root/'prediction_freeze.json').is_file()

def test_raw_receipt_hash_and_no_duplicate_measurement(tmp_path,monkeypatch):
    from frame_stream_top20 import campaign
    b=fake_bundle(tmp_path);r=tmp_path/'run';r.mkdir();doc=cpu_task(b,next(iter(plan()[1]['groups'])))
    state={'bundle':str(b),'boot_id':'boot-A','frozen_files':{}}
    monkeypatch.setattr(campaign,'validate_summary',lambda *args:{'correct':True})
    class Transport:
        calls=0
        def measure(self,d,bundle,remote,root,boot):
            self.calls+=1;p=root/'raw'/boot/d['task_id']/'collected';p.mkdir(parents=True)
            (p/'result.jsonl').write_text('raw\n');return p
        def acknowledge(self,*a):pass
    t=Transport();rec=campaign.Recorder(r,state,t,'/board')
    assert rec.saved(doc)==rec.saved(doc) and t.calls==1
    next((r/'raw').rglob('result.jsonl')).write_text('tampered')
    with pytest.raises(ValueError):rec.saved(doc)

def test_disconnect_recollects_finished_session_without_relaunch(tmp_path):
    from frame_stream_top20.runtime import _ssh_class
    b=fake_bundle(tmp_path);doc=cpu_task(b,next(iter(plan()[1]['groups'])));r=tmp_path/'run';r.mkdir()
    c=api(b);transport=object.__new__(_ssh_class(str(b)))
    from cpu_vta_partitioning.transport import TransportError
    transport.config={'remote_base':'/board'};transport.boot_id=lambda:'boot-A'
    remote_hash='';launched=0;collected=0
    def command(script):
        nonlocal remote_hash,launched
        if 'nohup sh execute.sh' in script:launched+=1;remote_hash=c.digest(doc)
        if 'test -f ' in script and '/task_hash' in script:return remote_hash
        if 'test -f ' in script and '/exit_code' in script:return '0'
        return ''
    transport.command=command;transport.check_space=lambda *a:None;transport.upload=lambda *a:None
    transport.failure_snapshot=lambda *a:'retained'
    def collect(job,raw):
        nonlocal collected
        collected+=1
        if collected==1:raise TransportError('disconnect')
        out=raw/'collected';out.mkdir()
        for name,text in [('task_hash',c.digest(doc)),('boot_id','boot-A'),('inputs.txt',(raw/'inputs.txt').read_text())]:
            (out/name).write_text(text)
        return out
    transport.collect=collect
    with pytest.raises(TransportError):transport.measure(doc,b,'/bundle',r,'boot-A')
    transport.measure(doc,b,'/bundle',r,'boot-A')
    assert launched==1 and collected==2

def test_wrong_boot_rejected_before_deployment(tmp_path,monkeypatch):
    from frame_stream_top20 import campaign,prepare
    state={'bundle':str(FULL_PARENT),'boot_id':'old','frozen_files':{},'status':'INTERRUPTED'}
    monkeypatch.setattr(campaign,'validate_state',lambda *a:state)
    monkeypatch.setattr(prepare,'validate_bundle',lambda *a:None)
    class Transport:
        def boot_id(self):return 'new'
        def wait_idle(self,*a):raise AssertionError('must reject before deployment')
    with pytest.raises(ValueError,match='boot'):campaign.run_current(tmp_path,Transport())
    assert api().read(tmp_path/'state.json')['status']=='STOPPED_INVALID_RESULT'

def test_four_actor_native_queue_group_serial_parallel_and_cancel(tmp_path):
    source=(FRAME_PARENT/'sources/frame_stream_native/vta_stage_pipeline_runner_frame_stream.cc').read_text()
    queue=source[source.index('template <typename T>\nclass BoundedQueue'):source.index('uint64_t EmptyStageTransform')]
    includes='\n'.join('#include <'+h+'>' for h in ('atomic','memory','vector','mutex','condition_variable','deque','functional','stdexcept','thread','utility','cassert'))+'\n'
    stub='struct QueueTiming {void PushRequested(double){} void Enqueued(double){} void PushReturned(double){} void Dequeued(double){}}; double NowMillis(){return 0;}\n'
    test=r'''
#include "frame_stream_flow.h"
#include "vta_cpu_resource_groups.h"
int main(){
 cpu_frame_stream::Flow<BoundedQueue> f(4,2,14);
 vta_deploy::CpuResourceGroups groups({{0},{0},{1},{2}});
 std::atomic<int> active{0},done{0};std::atomic<bool> other{false};
 std::vector<std::thread> workers;auto clock=[](){return 0LL;};
 for(int i=0;i<4;i++) workers.emplace_back([&,i]{f.Consume(i,clock,[&](int frame,auto,auto){
   auto lease=groups.Acquire(i);
   if(i<2){assert(++active==1);if(i==0&&frame==0)while(!other)std::this_thread::yield();--active;}
   if(i==2)other=true;
   ++done;
 });});
 std::thread producer([&]{f.Produce(clock,[](auto,auto,auto,auto){});});
 producer.join();for(auto& t:workers)t.join();assert(done==56&&other);
 vta_deploy::CpuResourceGroups cancelled({{0},{0}});auto held=cancelled.Acquire(0);
 std::atomic<bool> entered{false},caught{false};
 std::thread blocked([&]{entered=true;try{auto l=cancelled.Acquire(1);}catch(...){caught=true;}});
 while(!entered)std::this_thread::yield();cancelled.Cancel();blocked.join();assert(caught);
}
'''
    p=tmp_path/'groups.cc';p.write_text(includes+stub+queue+test);binary=tmp_path/'groups'
    subprocess.run([shutil.which('c++'),'-std=c++17','-pthread','-I',str(FRAME_PARENT/'sources/frame_stream_native'),
                    str(p),'-o',str(binary)],capture_output=True,check=True)
    subprocess.run([str(binary)],check=True,timeout=5)

def test_frozen_runner_patch_does_not_change_old_sources():
    from frame_stream_top20.native_patch import header
    p=ROOT/'frame_stream/native.h';original=p.read_text();updated=header(original)
    assert 'actors.size()<=4' in updated and 'groups.Acquire(index)' in updated and 'groups.Cancel()' in updated
    assert p.read_text()==original

def test_selection_quality_reports_both_baselines_and_extra_cost():
    from frame_stream_top20.analysis import evaluate
    ids=[str(i) for i in range(28)];measurements={n:{'ii_ms':100-int(n),'fps':1000/(100-int(n))} for n in ids}
    value=evaluate({'pure_static':ids[:20],'diverse':ids[8:28],
                    'frame_stream':list(reversed(ids[8:28]))},measurements)
    assert value['budget20_not_worse_than_both'] and value['positive_single_boot_observation']
    assert not value['independent_confirmation'] and value['extra_CPU_profile_cost_must_be_reported']
