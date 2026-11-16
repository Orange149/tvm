"""Interpose the real runtime; keep all performance and diagnosis modes separate."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT.parents[1]/'src')]
from cpu_vta_partitioning.common import digest,read,write
from wait_diagnosis.common import protocol,configurations,PROFILE,PARENT
from wait_diagnosis.prepare import task_order
from wait_diagnosis.runtime import make_task,native_command,trace_summary
from wait_diagnosis.campaign import decide,accept


@pytest.fixture(scope='module')
def host_native(tmp_path_factory):
    work=tmp_path_factory.mktemp('wait-native');repo=ROOT.parents[4]
    # ROOT=.../tvm/vta/experiments/cpu_vta_partitioning/supplements/published_cpu_policies
    repo=ROOT.parents[4]
    runtime=repo/'build';source=ROOT/'wait_diagnosis'
    subprocess.run(['c++','-std=c++17','-O2',str(source/'harness.cc'),'-o',str(work/'harness'),
        '-L'+str(runtime),'-Wl,-rpath,'+str(runtime),'-ltvm_runtime','-pthread'],check=True,capture_output=True)
    for name,flags in [('normal',[]),('spurious',['-DWAIT_ADAPTER_SPURIOUS_ONCE']),('overflow',['-DWAIT_TRACE_CAPACITY=8'])]:
        subprocess.run(['c++','-std=c++17','-O2','-shared','-fPIC',*flags,str(source/'adapter.cc'),
            '-o',str(work/(name+'.so')),'-ldl','-pthread'],check=True,capture_output=True)
    return work


def invoke(work,policy='original',probe=True,error=False,library='normal'):
    tag=policy+str(probe)+str(error)+library
    env=dict(os.environ,LD_PRELOAD=str(work/(library+'.so')),TVM_NUM_THREADS='4',TVM_BIND_THREADS='0',
        TVM_EXCLUDE_WORKER0='1',TVM_THREAD_POOL_SPIN_COUNT='0',VTA_WAIT_POLICY=policy,
        VTA_WAIT_PROBE=str(int(probe)),VTA_WAIT_TRACE=str(work/(tag+'.jsonl')),VTA_WAIT_STATUS=str(work/(tag+'.json')))
    p=subprocess.run([str(work/'harness'),str(int(error))],env=env,text=True,capture_output=True,timeout=30)
    assert p.returncode==0,p.stderr
    return read(work/(tag+'.json')),work/(tag+'.jsonl')


@pytest.mark.parametrize('policy,probe,error',[('original',True,False),('blocking',True,False),
    ('blocking',False,False),('blocking',True,True)])
def test_real_runtime_concurrent_groups_repeated_barriers_and_errors(host_native,policy,probe,error):
    status,path=invoke(host_native,policy,probe,error)
    assert status['errors']==status['overflow']==0
    assert status['launches']==(4 if error else 15)
    if policy=='original':assert status['yield_calls']>0 and status['futex_waits']==0
    else:assert status['futex_waits']>0
    if probe:
        rows=[json.loads(s) for s in path.read_text().splitlines()]
        assert sum(r['kind']==1 for r in rows)==status['callbacks']
        assert sum(r['kind']==2 for r in rows)==status['barriers']
        launches={r['launch']:r for r in rows if r['kind']==0}
        for k,r in launches.items():
            tasks=[x for x in rows if x['kind']==1 and x['launch']==k]
            assert len(tasks)==r['tasks'] and {x['task'] for x in tasks}==set(range(r['tasks']))
            assert max(x['wall_end_ns'] for x in tasks)<=r['wall_end_ns']
        if not error:assert {r['tasks'] for r in launches.values()}=={1,2,3,4}
    else:assert status['records']==0 and not path.exists()


def test_spurious_wake_predicate_recheck(host_native):
    status,_=invoke(host_native,'blocking',True,False,'spurious')
    assert status['errors']==0 and status['futex_waits']>0


def test_buffer_overflow_is_explicit_not_out_of_bounds(host_native):
    status,path=invoke(host_native,'original',True,False,'overflow')
    assert status['overflow']>0 and len(path.read_text().splitlines())==8


def test_adapter_creates_no_sampling_threads(host_native):
    symbols=subprocess.run(['nm','-D','-u',str(host_native/'normal.so')],capture_output=True,text=True,check=True).stdout
    assert 'pthread_create' not in symbols


@pytest.fixture
def capsule(tmp_path):
    cfg=configurations();write(tmp_path/'plans/configurations.json',cfg);write(tmp_path/'plans/protocol.json',protocol())
    for leaf in ('inputs.json','programs.json'):
        write(tmp_path/'plans'/leaf,read(PROFILE/'plans'/leaf))
    write(tmp_path/'manifest.json',dict(bundle_id='fixture',runner_sha256='runner',runtime_sha256='runtime',adapter_sha256='adapter'))
    return tmp_path


@pytest.mark.parametrize('scope,policy,probe,qual,total',[('component','original',False,False,10),
    ('component','original',True,True,1),('pipeline','original',False,False,104),
    ('pipeline','original',True,False,104),('pipeline','blocking',False,False,104),
    ('pipeline','blocking',True,True,1)])
def test_exact_modes_windows_and_no_process_affinity(capsule,scope,policy,probe,qual,total):
    spec=dict(scope=scope,policy=policy,probe=probe,case='overlap4',stage=1 if policy=='original' else 2)
    doc=make_task(capsule,spec,qual);argv,inputs=native_command(doc,capsule,'/bundle','/job')
    assert argv[argv.index('--runs')+1]==str(total)
    assert '--process-cpu-affinity' not in argv and '--queue-timing' not in argv
    assert ('--serial' in argv)==(scope=='component')
    preload=next(x for x in argv if x.startswith('LD_PRELOAD='))
    assert ('libwait_adapter.so' in preload)==(probe or policy=='blocking')
    assert ('--resource-affinity-probe' in argv)==qual
    assert inputs=='/bundle/inputs/000.bin\n'
    bad=deepcopy(doc);bad['spec']['probe']=not probe
    with pytest.raises(ValueError):native_command(bad,capsule,'/bundle','/job')


def test_task_identity_separates_all_modes_but_config_does_not(capsule):
    docs=[make_task(capsule,dict(scope='pipeline',policy=p,probe=q,case='overlap4',stage=1 if p=='original' else 2))
          for p in ('original','blocking') for q in (False,True)]
    assert len({d['task_id'] for d in docs})==4 and all(d['configuration']==docs[0]['configuration'] for d in docs)


def test_frozen_order_and_budget_counts():
    order=task_order();assert order==task_order() and len(order)==12
    assert all(s['policy']=='original' for s in order[:8])
    full=sum(105 for s in order if s['scope']=='pipeline');stage=sum(11 for s in order if s['scope']=='component')
    assert 2*len(order)==24 and full==840 and stage==44


def stage_one(gap=80,perturbation=10,yields=100):
    return {('component','overlap4','original',False):dict(cost={'service_ms':140}),
        ('pipeline','overlap4','original',False):dict(stage_service_ms=[140+gap]),
        ('pipeline','overlap4','original',True):dict(stage_service_ms=[140+gap+perturbation],
            wait={'stages':{'0':dict(barrier_yield_calls=yields,completion_yield_calls=0)}})}


@pytest.mark.parametrize('gap,delta,count,allowed',[(80,10,100,True),(80,16,100,True),
    (80,16.1,100,False),(0,0,100,False),(-1,0,100,False),(80,1,0,False)])
def test_frozen_trigger_no_remeasurement_or_FPS_selection(gap,delta,count,allowed):
    r=decide(stage_one(gap,delta,count));assert r['execute_blocking'] is allowed
    assert r['decision_is_not_causal_proof'] and r['no_extra_measurements_on_failure']


def test_resume_receipt_no_repeat_and_wrong_hash_or_boot_rejected(tmp_path,monkeypatch):
    monkeypatch.setitem(accept.__globals__,'summary',lambda *a:{'correct':True})
    boot='A';doc=dict(task_id='task')
    class Transport:
        calls=0
        def measure(self,*a):
            self.calls+=1;p=tmp_path/'raw';p.mkdir();(p/'boot_id').write_text(boot)
            (p/'task_hash').write_text(digest(doc));(p/'result.jsonl').write_text('valid');return p
        def acknowledge(self,*a):pass
    t=Transport();receipt=accept(t,doc,'bundle','remote',tmp_path,boot)
    assert receipt==accept(t,doc,'bundle','remote',tmp_path,boot) and t.calls==1
    (tmp_path/'raw/result.jsonl').write_text('wrong')
    with pytest.raises(ValueError):accept(t,doc,'bundle','remote',tmp_path,boot)


def test_overflow_or_missing_interposition_cannot_mean_no_wait(tmp_path):
    status=dict(policy='original',probe=True,errors=0,overflow=1,launches=1,callbacks=1)
    write(tmp_path/'wait_status.json',status)
    with pytest.raises(ValueError):trace_summary(tmp_path,dict(spec={'policy':'original','probe':True}),'bundle',{})
    status['overflow']=0;status['launches']=0;write(tmp_path/'wait_status.json',status,replace=True)
    with pytest.raises(ValueError):trace_summary(tmp_path,dict(spec={'policy':'original','probe':True}),'bundle',{})


def test_board_cli_requires_explicit_execute():
    from run_yolo_wait_diagnosis import main
    with pytest.raises(SystemExit):main(['board','run'])


def test_trace_requires_real_module_and_exact_logical_worker_TID(tmp_path):
    bundle=tmp_path/'bundle';raw=tmp_path/'raw';raw.mkdir()
    module=bundle/'programs/cpu_00_02/module.so';module.parent.mkdir(parents=True);module.write_bytes(b'module')
    runner=bundle/'runner/vta_stage_pipeline_runner';runner.parent.mkdir();runner.write_bytes(b'runner')
    base=dict(launch=1,caller=100,tid=100,tasks=1,module='/bundle/programs/cpu_00_02/module.so',
        function_offset=32,wall_start_ns=1,wall_end_ns=100,cpu_start_ns=1,cpu_end_ns=60,
        barrier_wall_ns=0,barrier_cpu_ns=0,barrier_calls=0,yield_calls=0,completion_yield_calls=1,
        caller_tail_wall_start_ns=70,caller_tail_cpu_start_ns=50,last_task_end_ns=70,result=0)
    rows=[dict(base,kind=0,task=-1),dict(base,kind=1,task=0,wall_start_ns=20,wall_end_ns=70,cpu_start_ns=10,cpu_end_ns=50)]
    status=dict(policy='original',probe=True,errors=0,overflow=0,launches=1,callbacks=1,barriers=0,records=2)
    write(raw/'wait_status.json',status)
    def save(data):(raw/'wait_trace.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in data))
    save(rows);doc=dict(spec={'policy':'original','probe':True},stages=[dict(device='cpu',program='cpu_00_02',threads=4)])
    affinity=dict(stages=[dict(caller_tid=100,worker_slots=[dict(slot=i,tid=100+i) for i in range(4)])])
    result=trace_summary(raw,doc,bundle,affinity)
    assert result['stages']['0']['kernel_launches']==1 and result['offCPU_not_separated']
    bad=deepcopy(affinity);bad['stages'][0]['worker_slots'][0]['tid']=999
    with pytest.raises(ValueError,match='TID'):trace_summary(raw,doc,bundle,bad)
    badrows=[dict(r,module='/bundle/runner/vta_stage_pipeline_runner') for r in rows];save(badrows)
    with pytest.raises(ValueError,match='actual target'):trace_summary(raw,doc,bundle,affinity)
    save(rows[:1])
    with pytest.raises(ValueError,match='length'):trace_summary(raw,doc,bundle,affinity)
