import json
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest

from cpu_vta_partitioning.common import read,write
from cpu_strategy.policy import campaign
from cpu_strategy.policy.domain import CUTS,CORES
from cpu_strategy.policy.search import evaluate,Costs
from test_domain_search import programs,synthetic


class FakeBoard:
    def __init__(self):
        self.boot='boot1';self.calls=[];self.acked=[];self.fail_at=None;self.bad_output=False
        self.escape=False;self.duplicate=False;self.bad_control=False;self.drift=set()
        self.mounted=False
        self.change_boot_after_measure=False;self.bridge_error=False
    def boot_id(self):return self.boot
    def wait_idle(self,b):assert b==self.boot
    def mount_storage(self):self.mounted=True
    def deploy(self,*a):return '/fake/bundle'
    def verify_payload(self,r,b):assert b==self.boot
    def qualify_environment(self,*a):pass
    def idle_command(self,*a,**k):pass
    def recover_abandoned(self,*a):assert self.mounted
    def acknowledge(self,d,*a):self.acked.append(d['task_id'])
    def measure(self,d,bundle,remote,root,boot):
        self.calls.append(deepcopy(d))
        if len(self.calls)==self.fail_at:self.fail_at=None;raise RuntimeError('simulated SSH disconnect')
        cfg=d['configuration'];stages=cfg['stages'];raw=Path(root)/'raw'/boot/d['task_id']/'collected';raw.mkdir(parents=True,exist_ok=True)
        if len(stages)==1:ii=synthetic(cfg)['cost']['service_ms']
        else:
            from cpu_strategy.policy.domain import component,component_key
            records={}
            for s in stages:
                c=component(s,cfg['affinity_mode'],cfg['control_mask']);records[component_key(c)]=synthetic(c)
            value=evaluate(cfg,Costs(records),'enhanced')
            ii=value['score_ms']
            if cfg['execution_mode']=='off':ii*=1.15
        if d.get('position')=='after' and (d.get('round'),d.get('attempt')) in self.drift:ii*=1.2
        if self.bridge_error and d['kind']=='executor_bridge' and d['executor']=='policy':ii*=1.1
        affinities=[]
        for j,s in enumerate(stages):
            tids=[1000+j*10+i for i in range(s['threads'])];pin=s['device']=='cpu' and cfg['affinity_mode']=='pinned'
            entries=[{'tid':tid,'cpus':[s['mask'][i]] if pin else s['mask'],'cpu_time_delta_ms':10. if i==0 else 0.} for i,tid in enumerate(tids)]
            if self.escape:entries[0]['cpus']=[99]
            affinities.append({'requested_cpus':s['mask'],'thread_ids':tids,'caller_tid':tids[0],
                'active_runtime_threads':s['threads'],'effective_affinities':entries,
                'worker_slots':[{'slot':i,'tid':tid,'cpus':[s['mask'][i]] if pin else s['mask']} for i,tid in enumerate(tids)]})
        controls=[{'role':role,'tid':900+i,'cpus':([99] if self.bad_control else cfg['control_mask'])} for i,role in enumerate(('collector','producer'))]
        write(raw/'affinity.json',{'stages':affinities,'control_threads':controls,'cpu_worker_pinning':cfg['affinity_mode']=='pinned'})
        rows=[]
        for i in range(d['warmup']+d['scored']+d['tail']):
            row={'frame_id':i,'completion_index':i,'input_index':i%len(d['input_indices']),
                 'completion_ms':10.+i*ii,'completion_process_cpu_ms':100.+i*ii*2,'total_latency_ms':ii*3,
                 'cpu_stage_gate_mode':cfg['execution_mode'],'stage_cpu_time_scope':'exclusive_process_cpu_time' if len(stages)==1 else 'overlapping_process_cpu_time'}
            for j,s in enumerate(stages):
                from cpu_strategy.policy.domain import component
                c=synthetic(component(s,cfg['affinity_mode'],cfg['control_mask']))['cost']
                row.update({f'stage{j}_set_ms':.5,f'stage{j}_run_ms':c['run_ms'],f'stage{j}_get_ms':.25,
                            f'stage{j}_set_process_cpu_ms':.3,f'stage{j}_run_process_cpu_ms':c['run_core_ms'],f'stage{j}_get_process_cpu_ms':.15})
            if d['kind'].startswith('qualify'):
                row['stage_raw_outputs']=[]
                for j,s in enumerate(stages):
                    output=raw/'outputs'/('run_'+str(i))/('stage'+str(j)+'.bin');output.parent.mkdir(parents=True,exist_ok=True)
                    output.write_bytes(bytes([8 if self.bad_output else 7]))
                    row['stage_raw_outputs'].append({'output_files':[{'path':'/remote/'+output.name,'dtype':'int8','shape':[1]}]})
            rows.append(row)
        if self.duplicate:rows[1]['frame_id']=rows[0]['frame_id']
        (raw/'result.jsonl').write_text('\n'.join(json.dumps(v) for v in rows)+'\n')
        if self.change_boot_after_measure:self.boot='unexpected-boot'
        return raw


@pytest.fixture
def setup(tmp_path,monkeypatch):
    from cpu_strategy.policy import prepare
    monkeypatch.setattr(prepare,'identity',lambda:'test-resource-controller')
    monkeypatch.setattr(prepare,'validate',lambda p:read(Path(p)/'manifest.json'))
    bundle=tmp_path/'bundle';bundle.mkdir()
    write(bundle/'manifest.json',{'bundle_id':'test-bundle','controller_identity':'test-resource-controller'})
    ps=programs()
    for name,p in ps.items():p.update(input_names=['data'],input_node_ids=[p['begin']],output_node_ids=[p['end']])
    write(bundle/'plans/programs.json',ps)
    write(bundle/'plans/imported_profiles.json',{'records':{}})
    write(bundle/'plans/executors.json',{k:{'path':'runner/'+k,'sha256':k+'-hash'} for k in ('legacy','policy')})
    (bundle/'references').mkdir();np.save(bundle/'references/value.npy',np.array([7],dtype='int8'))
    write(bundle/'plans/inputs.json',{'records':[{'index':i,'input':'inputs/image.bin','frontiers':{str(n):{'npy':'references/value.npy'} for n in range(22)}} for i in range(67)]})
    root=tmp_path/'run';campaign.create(bundle,root,{'host':'fake'},'preflight')
    return root,bundle,FakeBoard()


def test_six_boot_complete_flow_freezes_profiles_rule_and_rankings(setup):
    root,bundle,b=setup
    for i in range(6):
        b.boot='boot'+str(i+1);s=campaign.run_current(root,b)
        assert s['boot_index']==i+1
        if i==1:ranked=(root/'rankings.json').read_bytes()
        if i>=2:assert (root/'rankings.json').read_bytes()==ranked
        call_count=len(b.calls);again=campaign.run_current(root,b)
        assert again['boot_index']==i+1 and len(b.calls)==call_count
    assert s['status']=='COMPLETE' and len(set(s['completed_boots']))==6
    rounds=campaign.validated_rounds(root)
    assert len(rounds)==12 and len([r for r in rounds if r['role']=='confirmation'])==6
    assert len(read(root/'rankings.json')['enhanced']['rankings'])==20
    assert len(read(root/'search_pool.json'))<=30
    seen=len(b.calls);campaign.run_current(root,b);assert len(b.calls)==seen


def test_disconnect_resumes_without_remeasuring_accepted_signature(setup):
    root,_,b=setup;b.fail_at=7
    with pytest.raises(RuntimeError,match='disconnect'):campaign.run_current(root,b)
    accepted=set(b.acked);n=len(b.calls)
    s=campaign.run_current(root,b)
    assert s['boot_index']==1 and not accepted&{d['task_id'] for d in b.calls[n:]}


@pytest.mark.parametrize('fault',['bad_output','escape','duplicate','change_boot_after_measure'])
def test_invalid_raw_is_never_accepted(setup,fault):
    root,_,b=setup;setattr(b,fault,True)
    with pytest.raises(ValueError):campaign.run_current(root,b)
    assert not list((root/'accepted').glob('*/*.json'))


def test_control_affinity_failure_is_not_accepted(setup):
    from cpu_strategy.policy.domain import cut_candidate
    root,_,b=setup;s=campaign.validate_state(root);s['active_boot']=b.boot;campaign._save(root,s)
    c=cut_candidate('current',(2,2),((0,1),(2,3)),'pinned')
    b.bad_control=True;rec=campaign.Recorder(root,s,b,'/fake')
    with pytest.raises(ValueError,match='control thread'):
        rec.saved('qualification/control',campaign.task(s,'qualify_pipeline',c))
    assert not list((root/'accepted').glob('*/*.json'))


def test_failed_executor_bridge_cannot_be_retried_until_it_passes_by_chance(setup):
    root,_,b=setup;s=campaign.validate_state(root);s['active_boot']=b.boot;campaign._save(root,s)
    rec=campaign.Recorder(root,s,b,'/fake');b.bridge_error=True
    with pytest.raises(RuntimeError,match='bridge exceeds'):campaign.executor_bridge(root,s,rec,8)
    count=len(b.calls);b.bridge_error=False
    with pytest.raises(RuntimeError,match='cannot bypass'):campaign.executor_bridge(root,s,rec,8)
    assert len(b.calls)==count and read(root/'state.json')['status']=='STOPPED_EXECUTOR_DIFFERENCE'


def test_drift_failure_keeps_invalid_rounds_and_marks_stopped(setup):
    root,_,b=setup;b.drift={(0,0),(0,1)}
    with pytest.raises(RuntimeError,match='single replacement'):campaign.run_current(root,b)
    assert campaign.validate_state(root)['status']=='STOPPED_ANCHOR_DRIFT'
    assert len(list((root/'phases').glob('**/invalid*.json')))==2


def test_reboot_restarts_replicate_and_keeps_valid_component_cache(setup):
    root,_,b=setup;b.fail_at=7
    with pytest.raises(RuntimeError):campaign.run_current(root,b)
    keys=set(campaign.load_costs(root,campaign.validate_state(root)))
    b.boot='replacement';b.mounted=False;s=campaign.run_current(root,b)
    assert s['completed_boots']==['replacement'] and s['abandoned_boots']==['boot1']
    assert keys<=set(campaign.load_costs(root,s))


def test_changed_raw_hash_is_rejected_before_analysis(setup):
    root,_,b=setup;campaign.run_current(root,b)
    raw=next((root/'raw').glob('*/*/collected/result.jsonl'));raw.write_text(raw.read_text()+'\n')
    with pytest.raises(ValueError,match='missing or changed'):campaign.validated_rounds(root)
