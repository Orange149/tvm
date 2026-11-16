"""Frozen coverage, accounting, concurrent evidence rejection, and selection."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cpu_policies.contention import domain
from cpu_policies.contention.runtime import validate_micro
from cpu_policies.contention.analysis import evaluate
from cpu_policies.contention.patch_runner import patch
from cpu_policies.contention.common import DEFAULT_PARENT

def fixture():
    actors=domain.groups()['explanation/split2/cpu_pair']['actors']
    doc={'kind':'concurrent','actors':actors,'targets':[0,1]}
    affinity={'cpu_worker_pinning':True,'stages':[]}
    rows=[]
    for i,s in enumerate(actors):
        tids=[100+i*10+j for j in range(s['threads'])]
        entries=[{'tid':tid,'cpus':[core],'cpu_time_delta_ms':1.} for tid,core in zip(tids,s['mask'])]
        affinity['stages'].append({'requested_cpus':s['mask'],'active_runtime_threads':s['threads'],
            'caller_tid':tids[0],'thread_ids':tids,'effective_affinities':entries,
            'worker_slots':[{'slot':j,'tid':tid,'cpus':[core]} for j,(tid,core) in enumerate(zip(tids,s['mask']))]})
        for f in range(8):
            rows.append({'schema':'controlled_contention_v1','actor':i,'iteration':f,'program':s['program'],
                'qualification_passed':True,'final_output_passed':True,'start_ms':f*3.,'end_ms':f*3.+3.,
                'service_ms':3.,'set_ms':1.,'run_ms':1.,'get_ms':1.,'active_start_ms':0.,'active_end_ms':30.,
                'calls':9,'extra_calls':1,'process_cpu_valid':False})
    return doc,affinity,rows

def test_predeclared_counts_and_multifrontier():
    assert len(domain.components())==15
    assert len(domain.groups())==20
    assert len(domain.diagnostics())==4
    assert len(domain.configurations())==8
    assert len(domain.program_names())==7
    for cfg in domain.configurations().values():
        assert cfg['execution_mode']=='off'
        assert cfg['host_mask']==[0,1,2,3]
        assert all(s['worker_core_map']==s['mask'] for s in cfg['stages'] if s['device']=='cpu')

def test_concurrent_extra_calls_not_extra_samples():
    d,a,r=fixture(); result=validate_micro(r,d,a)
    assert result['actors']['0']['calls']==9
    assert result['actors']['0']['samples']==8
    assert 'run_core_ms' not in result['actors']['0']
    assert {entry['stage_index'] for entry in result['thread_ledger']}=={0,1}

@pytest.mark.parametrize('failure',['missing','duplicate','hash','affinity','caller','ownership','clock','time','background','extra','unexpected'])
def test_invalid_raw_rejected(failure):
    d,a,r=fixture()
    if failure=='missing': r.pop()
    if failure=='duplicate': r[1]['iteration']=0
    if failure=='hash': r[0]['qualification_passed']=False
    if failure=='affinity': a['stages'][0]['effective_affinities'][0]['cpus']=[3]
    if failure=='caller': a['stages'][0]['caller_tid']=a['stages'][0]['thread_ids'][1]
    if failure=='ownership': a['stages'][1]=deepcopy(a['stages'][0])
    if failure=='clock': r[0]['process_cpu_valid']=True
    if failure=='time': r[0]['run_ms']=-1
    if failure=='background':
        for row in r[8:]: row['active_end_ms']=2
    if failure=='extra': r[0]['extra_calls']=5
    if failure=='unexpected': r[0]['actor']=9
    with pytest.raises((ValueError,KeyError)): validate_micro(r,d,a)

def test_native_patch_opt_in_and_first_frame_output():
    src=(DEFAULT_PARENT/'sources/core_model_native/vta_stage_pipeline_runner_core_model.cc').read_text()
    out=patch(src)
    assert out.index('controlled_contention::Run')<out.index('std::vector<InputRecord> inputs = LoadInputs')
    assert 'if(frame!=0)' in out
    with pytest.raises(ValueError): patch(out)

def test_success_requires_both_topologies_and_direction():
    p={'predictions':{},'rankings':{}}; measured={}
    for topology in domain.TOPOLOGIES:
        names=[topology+'/'+n for n in domain.PLACEMENTS]
        actual=[12,15,10,11]
        p['rankings'][topology]={m:sorted(names,key=lambda n:actual[names.index(n)] if m!='original' else names.index(n)) for m in ('original','cpu_corun','cpu_vta_corun')}
        for n,x in zip(names,actual):
            p['predictions'][n]={'estimates_ms':{'original':8.,'cpu_corun':x,'cpu_vta_corun':x+1}}
            measured[n]={'ii_ms':x,'fps':1000/x}
    result=evaluate(p,measured)
    assert result['recommended_followup']=='cpu_corun'
    p['predictions']['validation/overlap2']['estimates_ms']['cpu_corun']=1
    p['predictions']['validation/overlap2']['estimates_ms']['cpu_vta_corun']=1
    assert evaluate(p,measured)['recommended_followup'] is None

def test_actual_task_plan_and_qemu_argument_translation(tmp_path):
    # An offline manifest with real frozen programs is sufficient to validate all argv.
    from cpu_policies.contention.common import api
    from cpu_policies.contention.campaign import profile_plan,final_plan
    from cpu_policies.contention.runtime import native_command,validate_document
    c=api(); parent=json.loads((DEFAULT_PARENT/'plans/programs.json').read_text())
    full=DEFAULT_PARENT.parent/'8a942b2047c031339168f9d2'
    programs=json.loads((full/'plans/programs.json').read_text())
    (tmp_path/'plans').mkdir()
    (tmp_path/'sources/base_controller/src').mkdir(parents=True)
    (tmp_path/'sources/cpu_strategy').mkdir(parents=True)
    for name,data in [('programs.json',{n:programs[n] for n in domain.program_names()}),
                      ('inputs.json',json.loads((DEFAULT_PARENT/'plans/inputs.json').read_text())),
                      ('executors.json',{'contention':{'sha256':'a','path':'runner/vta_stage_pipeline_runner_contention'},'policy':{'sha256':'b','path':'runner/vta_stage_pipeline_runner_pinned'}})]:
        (tmp_path/'plans'/name).write_text(json.dumps(data))
    (tmp_path/'manifest.json').write_text(json.dumps({'bundle_id':'fixture'}))
    plan=profile_plan(tmp_path)+final_plan(tmp_path)
    assert len(plan)==48 and len({d['task_id'] for d in plan})==48
    for doc in plan:
        argv,inputs=native_command(doc,tmp_path,'/bundle','/job')
        assert '--process-cpu-affinity' not in argv
        assert '--cpu-worker-pinning' in argv
        assert '--resource-affinity-probe' in argv
        validate_document(doc,tmp_path)
        if doc['kind'] not in ('pipeline','bridge'):
            spec=json.loads(inputs)
            assert len(spec['actors'])==len(doc['actors'])
            for s,a in zip(doc['actors'],spec['actors']):
                if s['program']=='cpu_17_20': assert len(a['inputs'])==2
    bad=deepcopy(plan[0]); bad['actors'][0]['threads']+=1
    with pytest.raises(ValueError): validate_document(bad,tmp_path)

def test_durable_receipt_resume_hash_and_boot(tmp_path,monkeypatch):
    from cpu_policies.contention import campaign, runtime
    from cpu_policies.contention.common import api
    from cpu_policies.contention.campaign import profile_plan
    # A real parent manifest supplies frozen input, program, and executor identities.
    common=api(DEFAULT_PARENT)
    bundle=tmp_path/'capsule'; (bundle/'plans').mkdir(parents=True)
    (bundle/'sources/base_controller/src').mkdir(parents=True)
    (bundle/'sources/cpu_strategy').mkdir(parents=True)
    (bundle/'manifest.json').write_text(json.dumps({'bundle_id':'fixture'}))
    (bundle/'plans/executors.json').write_text(json.dumps({'contention':{'path':'runner/new','sha256':'fixed'},'policy':{'path':'runner/old','sha256':'old'}}))
    doc=profile_plan(bundle)[0]
    state={'bundle':str(bundle),'boot_id':'test-boot','frozen_files':{}}
    summary={'actors':{'0':{'service_ms':1}},'correct':True}
    monkeypatch.setattr(campaign,'validate_summary',lambda *a:deepcopy(summary))
    class Transport:
        calls=0
        def measure(self,document,bundle,remote,root,boot):
            self.calls+=1
            raw=root/'raw'/boot/document['task_id']/'collected';raw.mkdir(parents=True)
            (raw/'result.jsonl').write_text('immutable raw evidence\n')
            return raw
        def acknowledge(self,*args): pass
    transport=Transport(); recorder=campaign.Recorder(tmp_path,state,transport,'/remote')
    assert recorder.saved(doc)==summary
    assert recorder.saved(doc)==summary and transport.calls==1
    receipt=tmp_path/'accepted/test-boot'/(doc['task_id']+'.json')
    item=common.read(receipt);item['boot_id']='wrong-boot';common.write(receipt,item,replace=True)
    with pytest.raises(ValueError,match='boot'): recorder.saved(doc)
    item['boot_id']='test-boot';common.write(receipt,item,replace=True)
    raw=tmp_path/item['raw']/'result.jsonl';raw.write_text('modified\n')
    with pytest.raises(ValueError): recorder.saved(doc)

def test_service_constraints_are_max_not_additive():
    from cpu_policies.contention.model import predictions
    records={}; concurrent={}
    for key,cfg in domain.components().items():
        n=cfg['stages'][0]['threads']
        records[key]={'cost':{'service_ms':10.,'run_core_ms':8.*n,'boundary_core_ms':2.,
                             'run_ms':8.,'set_ms':1.,'get_ms':1.,'set_core_ms':1.,'get_core_ms':1.}}
    for name in domain.configurations():
        for mode in ('cpu_pair','cpu_pair_vta'):
            concurrent[name+'/'+mode]={'actors':{'0':{'service_ms':40.},'1':{'service_ms':20.}}}
    result=predictions(records,concurrent,None,None)
    for value in result['predictions'].values():
        scores=value['estimates_ms']
        assert scores['cpu_corun']==max(scores['original'],40.)
        assert scores['cpu_corun']==scores['cpu_vta_corun']
        assert value['baseline_ledger']['accounting']['unit']=='integer_nanosecond'
    assert result['pipeline_feedback_used'] is False

def test_native_device_gate_independent_completion_and_cancellation(tmp_path):
    import subprocess,shutil
    header=Path(__file__).resolve().parents[1]/'cpu_policies/contention'
    source=tmp_path/'gate.cc'
    source.write_text(r'''
#include "turn_gate.h"
#include <thread>
#include <chrono>
#include <cassert>
#include <stdexcept>
using namespace controlled_contention;
int main() {
  // A continuously ready single VTA actor cannot block the CPU stop flag.
  std::atomic<bool> stop{false}; TurnGate one({1}); std::atomic<int> calls{0};
  std::thread device([&]{while(one.Execute(1,stop,[&]{++calls; std::this_thread::sleep_for(std::chrono::microseconds(100));})) {}});
  while(calls.load()<5) std::this_thread::yield();
  stop=true; one.NotifyStop(); device.join();
  assert(calls.load()>=5);
  // An actor waiting for its turn is released on cancellation.
  stop=false; TurnGate two({1,2}); std::atomic<bool> returned{false};
  std::thread waiting([&]{assert(!two.Execute(2,stop,[]{assert(false);}));returned=true;});
  std::this_thread::sleep_for(std::chrono::milliseconds(2));
  stop=true; two.NotifyStop(); waiting.join(); assert(returned);
  // A throwing device action releases its mutex; cancellation remains possible.
  stop=false; bool threw=false;
  try {two.Execute(1,stop,[]{throw std::runtime_error("failure");});} catch(...) {threw=true;}
  assert(threw); stop=true; two.NotifyStop();
  assert(!two.Execute(2,stop,[]{assert(false);}));
}
''')
    binary=tmp_path/'gate'
    subprocess.run([shutil.which('c++'),'-std=c++17','-pthread','-I',str(header),str(source),'-o',str(binary)],check=True,capture_output=True)
    subprocess.run([str(binary)],check=True,timeout=3)

def test_first_warmup_pipeline_correctness_envelope(monkeypatch,tmp_path):
    from types import SimpleNamespace
    from cpu_policies.contention import runtime
    from cpu_policies.contention.common import api
    api(DEFAULT_PARENT)
    from cpu_vta_partitioning import measurement
    cfg=domain.configurations()['validation/overlap2']
    doc={'kind':'pipeline','configuration':cfg}
    rows=[]
    for i in range(28):
        rows.append({'frame_id':i,'completion_index':i,'input_index':0,
            'completion_ms':i*10.,'completion_process_cpu_ms':i*10.,'total_latency_ms':30.,
            'queue_timing_enabled':False,**{f'stage{j}_ms':3. for j in range(3)}})
    (tmp_path/'result.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    monkeypatch.setattr(runtime,'validate_document',lambda *a:None)
    monkeypatch.setattr(runtime,'api',lambda *a:SimpleNamespace(read=lambda p:{}))
    monkeypatch.setattr(runtime,'validate_affinity',lambda *a:None)
    monkeypatch.setattr(runtime,'validate_final_hashes',lambda *a:None)
    checked=[]
    def check(first,task,*args):
        measurement.input_sequence(first,task)
        checked.append((task['warmup'],task['scored'],task['tail']))
    monkeypatch.setattr(measurement,'correctness',check)
    summary=runtime.validate_summary(tmp_path,doc,DEFAULT_PARENT)
    assert summary['correct'] and summary['fps']==100
    assert checked==[(0,1,0)]
