from copy import deepcopy
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_cpu_exclusion_gap as c


def test_pair_changes_only_execution_mode_and_its_identity():
    for network,rows in c.selection()['selected_configurations'].items():
        a,b=deepcopy(rows[0]),deepcopy(rows[1]);assert a['id']!=b['id']
        for v in (a,b):v.pop('id');v.pop('execution_mode')
        assert a==b
    assert len(c.specs())==8 and sum(1 if s['phase']=='qualify' else 28 for s in c.specs())==116
    with pytest.raises(ValueError,match='execute-board'):c.board()


def fixture():
    doc=dict(cpu_admission='exclusive',warmup=0,scored=1,tail=0,
        stages=[dict(device='cpu',program='a'),dict(device='vta',program='v'),dict(device='cpu',program='b')])
    row=dict(frame_id=0,cpu_stage_gate_mode='exclusive')
    for i,(request,acquire,start,end,release) in enumerate(((0,0,1,2,3),(0,0,0,0,0),(1,3,4,5,6))):
        if i==1:continue
        prefix='stage'+str(i)
        for k,v in [('cpu_gate_request_ms',request),('cpu_gate_acquire_ms',acquire),('cpu_gate_release_ms',release),
            ('cpu_gate_wait_ms',acquire-request),('start_ms',start),('end_ms',end)]:row[prefix+'_'+k]=v
    return row,doc


def test_gate_wait_outside_service_and_all_CPU_leases_are_exclusive():
    row,doc=fixture();r=c.gate_evidence([row],doc)
    assert r['leases']==2 and r['gate_wait_ms']=={'a':0,'b':2}
    assert r['observed_max_CPU_service_concurrency']==1


def test_both_modes_reach_same_frozen_binary_without_observers(monkeypatch):
    original=c.read
    all_configs=c.selection()['selected_configurations']
    for network,configs in all_configs.items():
        bundle=c.ROOT/'artifacts/scheduler_gap_audit_01'/network/'capsule'
        def read(path):
            if Path(path)==bundle/'plans/configurations.json':
                source=original(path)
                return {**source,'pipelines':{**source['pipelines'],**{v['id']:v for v in configs}}}
            return original(path)
        monkeypatch.setattr(c,'read',read)
        binaries=set()
        for cfg in configs:
            spec=next(s for s in c.specs() if s['network']==network and s['case']==cfg['id'] and s['phase']=='clean')
            task=c.make_task(bundle,spec);argv,_=c.native_command(task,bundle,'/bundle','/job')
            assert argv[argv.index('--cpu-stage-gate')+1]==cfg['execution_mode']
            assert '--phase-event-json' not in argv and '--session-profile-json' not in argv
            assert all('libwait_adapter' not in arg for arg in argv)
            binaries.add(task['executor_sha256'])
        assert len(binaries)==1
        monkeypatch.setattr(c,'read',original)


@pytest.mark.parametrize('fault',['overlap','wrong_mode','missed_service','negative','VTA_gate'])
def test_exclusion_errors_rejected(fault):
    row,doc=fixture()
    if fault=='overlap':
        row.update(stage2_cpu_gate_acquire_ms=2,stage2_cpu_gate_wait_ms=1)
    elif fault=='wrong_mode':row['cpu_stage_gate_mode']='off'
    elif fault=='missed_service':row['stage0_cpu_gate_release_ms']=1
    elif fault=='negative':row['stage2_cpu_gate_wait_ms']=-1
    else:row['stage1_cpu_gate_acquire_ms']=3
    with pytest.raises(ValueError):c.gate_evidence([row],doc)
