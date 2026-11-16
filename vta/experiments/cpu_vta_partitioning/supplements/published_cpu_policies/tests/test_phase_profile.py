"""Optional phase clock scope, disabled behavior, and authenticated patching."""
from pathlib import Path
import json
import subprocess
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT.parents[1]/'src')]
from phase_profile.patch import patch


def test_patch_preserves_compute_calls_and_disabled_binding():
    source=(ROOT/'artifacts/bundles/abbcccc62af230ee844745ac/sources/vta/apps/native_deploy/vta_stage_pipeline_runner.cc').read_text()
    result=patch(source)
    assert result.count('stage->run();')==source.count('stage->run();')
    assert '--cpu-worker-pinning' in result and '--phase-event-json' in result
    assert result.count('phase_run.End();')==1
    assert 'ConfigureThreadPool' in result
    with pytest.raises(ValueError):patch(source+'\n')


@pytest.mark.parametrize('enabled',[True,False])
def test_same_clock_recorder_multithreaded(tmp_path,enabled):
    target=tmp_path/'phase.jsonl';code=tmp_path/'harness.cc'
    code.write_text('#include "phase_probe.h"\n#include <thread>\n#include <vector>\n'
        'int main(int argc,char** argv){ vta_workload_phase::Configure(argv[1]); '
        'std::vector<std::thread> ts;for(int i=0;i<2;++i)ts.emplace_back([i]{'
        'std::string name="cpu_"+std::to_string(i);for(int j=0;j<3;++j){'
        'auto c=vta_workload_phase::Next(name);vta_workload_phase::Span s(name,c,"run");'
        'for(volatile int k=0;k<5000;++k){}s.End();}});for(auto& t:ts)t.join();'
        'vta_workload_phase::State().Flush();}')
    executable=tmp_path/'harness'
    subprocess.run(['c++','-std=c++17','-O2','-pthread','-I',str(ROOT/'phase_profile'),str(code),'-o',str(executable)],check=True,capture_output=True)
    subprocess.run([str(executable),str(target) if enabled else ''],check=True,capture_output=True)
    if not enabled:assert not target.exists();return
    rows=[json.loads(s) for s in target.read_text().splitlines()]
    assert len(rows)==6
    for name in ('cpu_0','cpu_1'):
        a=[r for r in rows if r['stage']==name]
        assert [r['call'] for r in a]==[0,1,2] and len({r['tid'] for r in a})==1
    assert len({r['tid'] for r in rows})==2
    for r in rows:
        assert r['wall_start_ns']<=r['wall_end_ns']
        assert 0<=r['cpu_end_ns']-r['cpu_start_ns']<=r['wall_end_ns']-r['wall_start_ns']


def test_protocol_has_complete_multi_input_component_and_bounded_tasks():
    from phase_profile.prepare import configurations,task_specs
    configs=configurations('yolov3_tiny416')
    assert configs['components']['cpu_07_08_t3']['stages'][0]['threads']==3
    assert configs['components']['cpu_13_13_t1']['stages'][0]['threads']==1
    tasks=task_specs()
    assert len(tasks)==30
    assert len({(t['network'],t['case'],t['phase']) for t in tasks})==30
    for network in ('resnet18','yolov3_tiny416'):
        assert sum(t['network']==network for t in tasks)==15


def test_phase_trace_rejects_duplicate_missing_and_wrong_caller(tmp_path):
    from phase_profile.runtime import phase_summary
    doc=dict(stages=[dict(program='cpu_00_02')],warmup=0,scored=1,tail=0,spec=dict(probe=True))
    affinity=dict(stages=[dict(caller_tid=5)])
    rows=[dict(stage='cpu_00_02',call=0,operation=op,tid=5,
               wall_start_ns=i*100,wall_end_ns=i*100+80,cpu_start_ns=i*20,cpu_end_ns=i*20+10)
          for i,op in enumerate(('set','run','get'))]
    path=tmp_path/'phase_trace.jsonl'
    def save(values):path.write_text(''.join(json.dumps(r)+'\n' for r in values))
    save(rows);assert phase_summary(tmp_path,doc,affinity)['events']==3
    for bad in [rows[:-1],rows+[rows[0]],[{**r,'tid':8} for r in rows]]:
        save(bad)
        with pytest.raises(ValueError):phase_summary(tmp_path,doc,affinity)


def test_generic_worker_parser_keeps_kernel_module_requirement():
    import inspect
    from phase_profile.worker import parser
    from wait_diagnosis.runtime import trace_summary
    function,proof=parser()
    assert function is not trace_summary
    assert not proof['clock_callback_module_TID_and_affinity_checks_changed']
    assert 'cpu_00_02' in inspect.getsource(trace_summary)


def test_accept_is_idempotent_and_rejects_changed_raw(tmp_path,monkeypatch):
    from phase_profile import campaign
    from cpu_vta_partitioning.common import digest
    task=dict(task_id='test_task',kind='calibration');boot='test_boot';raw=tmp_path/'rawdata'
    result=dict(correct=True,output_fingerprints=[[0,[1],'int8','00']])
    monkeypatch.setattr(campaign,'summary',lambda *args:result)
    class FakeSSH:
        calls=0
        def measure(self,*args):
            self.calls+=1;raw.mkdir(exist_ok=True)
            (raw/'boot_id').write_text(boot);(raw/'task_hash').write_text(digest(task))
            return raw
        def acknowledge(self,*args):
            assert (tmp_path/'accepted'/boot/'test_task.json').exists()
        def idle_command(self,*args,**kwargs):return ''
        def job_path(self,*args):return '/fake/job'
    ssh=FakeSSH()
    campaign.accept(ssh,task,tmp_path,'/fake',tmp_path,boot)
    campaign.accept(ssh,task,tmp_path,'/fake',tmp_path,boot)
    assert ssh.calls==1
    (raw/'boot_id').write_text('other_boot')
    with pytest.raises(ValueError):campaign.accept(ssh,task,tmp_path,'/fake',tmp_path,boot)


def test_changed_output_never_gets_an_acceptance_receipt(tmp_path,monkeypatch):
    from phase_profile import campaign
    from cpu_vta_partitioning.common import digest
    task=dict(task_id='test_task');raw=tmp_path/'rawdata';raw.mkdir()
    (raw/'boot_id').write_text('boot');(raw/'task_hash').write_text(digest(task))
    monkeypatch.setattr(campaign,'summary',lambda *args:dict(output_fingerprints=['wrong']))
    class FakeSSH:
        def measure(self,*args):return raw
        def acknowledge(self,*args):raise AssertionError('must not acknowledge invalid outputs')
    with pytest.raises(ValueError,match='qualified'):
        campaign.accept(FakeSSH(),task,tmp_path,'/fake',tmp_path,'boot',['correct'])
    assert not (tmp_path/'accepted/boot/test_task.json').exists()


def test_campaign_creates_network_directories_and_binds_bundles(tmp_path,monkeypatch):
    from phase_profile import campaign
    from yolo_validation import components
    from cpu_vta_partitioning.common import write,read
    output=tmp_path/'output';run=tmp_path/'run';local=tmp_path/'local.json'
    write(local,dict(board={}))
    monkeypatch.setattr(campaign,'specs',lambda *args:[])
    monkeypatch.setattr(campaign,'validate',lambda path:dict(bundle_id=path.parent.name+'_bundle'))
    monkeypatch.setattr(components,'hardware_check',lambda *args:dict(qualified=True))
    class FakeSSH:
        def boot_id(self):return 'boot'
        def deploy(self,bundle,path):
            assert path.is_dir()
            return '/remote/'+path.name
        def qualify_environment(self,*args):pass
        def idle_command(self,*args,**kwargs):return ''
    monkeypatch.setattr(campaign,'matched_ssh',lambda *args,**kwargs:FakeSSH())
    result=campaign.run(output,run,local,execute_board=True)
    assert result['status']=='COMPONENTS_COMPLETE'
    marker=read(run/'run.json')
    assert marker['bundle_ids']==dict(resnet18='resnet18_bundle',yolov3_tiny416='yolov3_tiny416_bundle')
    monkeypatch.setattr(campaign,'validate',lambda path:dict(bundle_id='changed_bundle'))
    with pytest.raises(ValueError,match='identity changed'):
        campaign.run(output,run,local,execute_board=True)
