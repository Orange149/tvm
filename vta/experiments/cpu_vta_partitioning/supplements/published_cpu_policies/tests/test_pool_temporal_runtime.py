"""CPU/VTA observers must not change the requested placement or waiting policy."""
from pathlib import Path
import json
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT.parents[1]/'src')]
from cpu_policies.open_resources.domain import stage,component
from pool_temporal.runtime import make_task,command
from pool_temporal.campaign import run
from pool_temporal.ranking import rank


@pytest.fixture
def bundle(tmp_path):
    parent=ROOT/'artifacts/cross_network_model_01/common_top20/resnet18/capsule'
    if not parent.exists():pytest.skip('frozen local graph metadata absent')
    (tmp_path/'plans').mkdir()
    for f in ('programs.json','inputs.json'):(tmp_path/'plans'/f).write_bytes((parent/'plans'/f).read_bytes())
    (tmp_path/'manifest.json').write_text(json.dumps(dict(network='resnet18',bundle_id='fixture',
        profile_runner_sha256='profile',runtime_sha256='runtime',wait_adapter_sha256='wait',device_adapter_sha256='device')))
    cfg=dict(cpu=component(stage('cpu',0,1,1,[0])),vta=component(stage('vta',3,5,1,[2]),'host_shared_2'))
    (tmp_path/'plans/components.json').write_text(json.dumps(cfg));return tmp_path


@pytest.mark.parametrize('case',('cpu','vta'))
def test_observer_commands_respect_CPU_group_vs_host_only_modes(bundle,case):
    doc=make_task(bundle,dict(network='resnet18',case=case,mode='observed'))
    argv,_=command(doc,bundle,'/bundle','/job')
    assert ('--cpu-worker-pinning' in argv)==(case=='cpu')
    assert '--serial' in argv and '--phase-event-json' in argv
    assert argv[argv.index('--runs')+1]=='10'
    preload=next(s for s in argv if s.startswith('LD_PRELOAD='))
    assert ('libwait_adapter.so' if case=='cpu' else 'libdevice_phase.so') in preload
    if case=='cpu':assert 'VTA_WAIT_POLICY=original' in argv
    else:assert argv[argv.index('--stage0-cpu-affinity')+1]=='2'


def test_component_qualification_and_observation_have_different_identities(bundle):
    a=make_task(bundle,dict(network='resnet18',case='cpu',mode='qualify'))
    b=make_task(bundle,dict(network='resnet18',case='cpu',mode='observed'))
    assert a['task_id']!=b['task_id'] and a['scored']==1 and b['warmup']==2 and b['scored']==8


def test_profiling_requires_explicit_board_flag_before_io(tmp_path):
    with pytest.raises(ValueError,match='execute-board'):run(tmp_path,tmp_path)


def test_incomplete_intrinsics_cannot_create_a_temporal_rank(tmp_path,monkeypatch):
    # The real study may already be measuring. Tests must use their own ledger.
    monkeypatch.setattr('pool_temporal.ranking.ROOT',tmp_path)
    with pytest.raises(FileNotFoundError):rank(tmp_path/'output')
    accepted=tmp_path/'runs/cross_network_common_top20_01/accepted'
    accepted.mkdir(parents=True);(accepted/'measured.json').write_text('{}')
    with pytest.raises(ValueError,match='after formal native timing'):rank(tmp_path/'output')
