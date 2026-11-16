from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import run_resource_behavior as campaign

def fake_bundle(tmp_path):
    c=campaign.api();b=tmp_path/'bundle';(b/'plans').mkdir(parents=True)
    for name in ('sources/base_controller/src','sources/cpu_strategy'):(b/name).mkdir(parents=True)
    for name in ('programs.json','inputs.json','executors.json'):
        shutil.copy2(campaign.PARENT/'plans'/name,b/'plans'/name)
    c.write(b/'plans/configurations.json',campaign.configurations())
    c.write(b/'plans/board_execution.json',{'storage_margin_bytes':256*1024**2})
    c.write(b/'manifest.json',{'bundle_id':'fixture'})
    return b

def test_six_cases_are_matched_cpu_work_with_host_and_gate_factorial():
    configs=campaign.configurations();assert len(configs)==len({r['id'] for r in configs.values()})==6
    cpu=lambda r:[s for s in r['stages'] if s['device']=='cpu']
    for name in ('shared_open','shared_serial','host2_open','host2_serial'):
        assert cpu(configs[name])==cpu(configs['host2_serial'])
        assert configs[name]['control_mask']==[0,1,2,3]
        assert configs[name]['execution_mode']==('resource_group' if name.endswith('serial') else 'off')
        assert configs[name]['host_mask']==([2] if name.startswith('host2') else [0,1,2,3])
    assert configs['host2_serial']['id']=='cfg_85c3cc9bfdcdcbae714e259d'

def test_round_identity_reverse_order_no_reboot_and_exact_budget(tmp_path):
    b=fake_bundle(tmp_path);plan=campaign.tasks(b)
    assert len(plan)==len({d['task_id'] for d in plan})==18
    assert all(d['kind']=='qualify_pipeline' for d in plan[:6])
    assert [d['behavior_case'] for d in plan[6:12]]==[d['behavior_case'] for d in plan[12:]][::-1]
    assert sum(sum(d[k] for k in ('warmup','scored','tail')) for d in plan)==342
    assert campaign.PROTOCOL['component_measurements']==0
    assert campaign.PROTOCOL['board_safety']['reboot'] is False
    for d in plan:
        campaign.validate_document(d,b)
        for key in ('behavior_round','executor_sha256','scored','candidate_id'):
            bad=deepcopy(d);bad[key]=999 if isinstance(bad[key],int) else 'wrong'
            with pytest.raises((ValueError,KeyError)):campaign.validate_document(bad,b)

def test_native_commands_preserve_frontiers_and_affinity(tmp_path):
    from cpu_policies.open_resources.runtime import native_command
    b=fake_bundle(tmp_path)
    for d in campaign.tasks(b):
        argv,inputs=native_command(d,b,'/bundle','/job')
        assert '--process-cpu-affinity' not in argv and '--cpu-worker-pinning' in argv
        assert argv[argv.index('--cpu-stage-gate')+1]==d['configuration']['execution_mode']
        assert d['input_indices']==[0] and d['environment']['spin_count']==0
        assert argv[argv.index('--control-cpu-affinity')+1]=='0,1,2,3'

def test_receipt_resume_rejects_hash_and_wrong_boot(tmp_path,monkeypatch):
    b=fake_bundle(tmp_path);root=tmp_path/'run';root.mkdir();doc=campaign.tasks(b)[0]
    state={'bundle':str(b),'boot_id':'boot-A','frozen_files':{}}
    monkeypatch.setattr(campaign,'summary',lambda *a:{'correct':True})
    class Transport:
        calls=0
        def measure(self,d,bundle,remote,root,boot):
            self.calls+=1;p=root/'raw'/boot/d['task_id']/'collected';p.mkdir(parents=True)
            (p/'result.jsonl').write_text('raw\n');return p
        def acknowledge(self,*a):pass
    t=Transport()
    assert campaign.accepted(root,state,doc,t,'/board')==campaign.accepted(root,state,doc,t,'/board')
    assert t.calls==1
    next((root/'raw').rglob('result.jsonl')).write_text('tampered')
    with pytest.raises(ValueError):campaign.accepted(root,state,doc)

def test_wrong_boot_stops_before_any_board_mutation(tmp_path,monkeypatch):
    state={'bundle':str(campaign.PARENT),'boot_id':'old','frozen_files':{},'status':'INTERRUPTED'}
    monkeypatch.setattr(campaign,'validate_state',lambda *a:state)
    monkeypatch.setattr(campaign,'validate_bundle',lambda *a:None)
    class Transport:
        def boot_id(self):return 'new'
        def wait_idle(self,*a):raise AssertionError('must not proceed on wrong boot')
    with pytest.raises(ValueError,match='boot'):campaign.run_current(tmp_path,Transport())
    assert campaign.api().read(tmp_path/'state.json')['status']=='STOPPED_INVALID_RESULT'

def test_disconnect_recovers_completed_remote_session_without_relaunch(tmp_path):
    b=fake_bundle(tmp_path);doc=campaign.tasks(b)[0];root=tmp_path/'run';root.mkdir();c=campaign.api(b)
    from cpu_vta_partitioning.transport import TransportError
    t=object.__new__(campaign._ssh_class(str(b)));t.config={'remote_base':'/board'};t.boot_id=lambda:'boot-A'
    remote_hash='';launched=0;collected=0
    def command(script):
        nonlocal remote_hash,launched
        if 'nohup sh execute.sh' in script:launched+=1;remote_hash=c.digest(doc)
        if 'test -f ' in script and '/task_hash' in script:return remote_hash
        if 'test -f ' in script and '/exit_code' in script:return '0'
        return ''
    t.command=command;t.upload=lambda *a:None;t.check_space=lambda *a:None;t.failure_snapshot=lambda *a:'retained'
    def collect(job,raw):
        nonlocal collected
        collected+=1
        if collected==1:raise TransportError('disconnect')
        p=raw/'collected';p.mkdir()
        for name,text in [('task_hash',c.digest(doc)),('boot_id','boot-A'),('inputs.txt',(raw/'inputs.txt').read_text())]:
            (p/name).write_text(text)
        return p
    t.collect=collect
    with pytest.raises(TransportError):t.measure(doc,b,'/bundle',root,'boot-A')
    t.measure(doc,b,'/bundle',root,'boot-A')
    assert launched==1 and collected==2

@pytest.mark.parametrize('bad',['missing','duplicate','wrong_core','wrong_output'])
def test_raw_frames_affinity_and_outputs_rejected(tmp_path,bad):
    import numpy as np
    b=fake_bundle(tmp_path);c=campaign.api(b)
    old=ROOT/'runs/resnet18_autotvm_diverse_top20_01';state=c.read(old/'state.json')
    kind='qualify_pipeline' if bad=='wrong_output' else 'performance'
    old_task=next(d for d in c.read(old/'tasks.json') if d['kind']==kind and d['candidate_id']=='cfg_b81cf0b8325c03d01989d7a3')
    rec=c.read(old/'accepted'/state['boot_id']/(old_task['task_id']+'.json'))
    raw=tmp_path/'raw';shutil.copytree(old/rec['raw'],raw);shutil.copytree(campaign.PARENT/'references',b/'references')
    doc=campaign._make_task(b,'static_first',kind,-1 if kind.startswith('qualify') else 0)
    campaign.summary(raw,doc,b)
    rows=(raw/'result.jsonl').read_text().splitlines()
    if bad=='missing':rows.pop()
    elif bad=='duplicate':rows[1]=rows[0]
    elif bad=='wrong_core':
        aff=c.read(raw/'affinity.json');aff['stages'][0]['effective_affinities'][0]['cpus']=[9];c.write(raw/'affinity.json',aff,replace=True)
    else:
        row=json.loads(rows[0]);meta=row['stage_raw_outputs'][0]['output_files'][0]
        p=raw/'outputs'/('run_'+str(row['completion_index']))/Path(meta['path']).name
        data=np.fromfile(p,dtype=meta['dtype']);data[0]=123 if data[0]!=123 else 124;data.tofile(p)
    (raw/'result.jsonl').write_text('\n'.join(rows)+'\n')
    with pytest.raises(ValueError):campaign.summary(raw,doc,b)

def test_existing_completed_campaign_identity_is_unchanged():
    from selection_top20.campaign import validate_state
    assert validate_state(ROOT/'runs/resnet18_autotvm_diverse_top20_01')['status']=='COMPLETE'
