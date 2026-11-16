"""Formal task semantics, resource controls, and before-analysis completeness."""
from copy import deepcopy
from pathlib import Path
import json
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT.parents[1]/'src')]
from common_top20.runtime import task,command
from common_top20.analyze import analyze
from common_top20.campaign import run
from common_top20.campaign import qemu_cpu_summary
from common_top20.prepare import runtime_program


@pytest.fixture
def prepared(tmp_path):
    rank=ROOT/'artifacts/cross_network_model_01/common_top20/resnet18/RANKINGS.json'
    parent=ROOT/'artifacts/bundles/8a942b2047c031339168f9d2'
    if not rank.exists():pytest.skip('full-domain local ranking fixtures absent')
    plans=tmp_path/'plans';plans.mkdir()
    (tmp_path/'manifest.json').write_text(json.dumps(dict(network='resnet18',runner_sha256='old',runtime_sha256='runtime',bundle_id='fixture')))
    (plans/'protocol.json').write_text(json.dumps(dict(experiment='test',short_window=dict(warmup=8,scored=16,tail=4))))
    for name in ('programs.json','inputs.json'):(plans/name).write_bytes((parent/'plans'/name).read_bytes())
    return tmp_path,json.loads(rank.read_text())


@pytest.mark.parametrize('method',('current','semantics_revision','fixed_family','mixed','tarnawski'))
def test_task_does_not_encode_scoring_method_and_keeps_all_runtime_controls(prepared,method):
    bundle,ranks=prepared;cfg=ranks['pool'][ranks['lists'][method][0]['id']]
    doc=task(bundle,cfg);argv,_=command(doc,bundle,'/bundle','/job')
    assert doc['warmup']+doc['scored']+doc['tail']==28
    assert 'method' not in doc and doc['executor_sha256']=='old'
    assert argv[argv.index('--cpu-stage-gate')+1]==cfg['execution_mode']
    assert argv[argv.index('--control-cpu-affinity')+1]==','.join(map(str,cfg['control_mask']))
    assert '--resource-affinity-probe' in argv and '--cpu-worker-pinning' in argv
    assert '--process-cpu-affinity' not in argv and '--phase-event-json' not in argv
    assert argv[argv.index('--output-mode')+1]=='raw'


def test_frozen_configuration_cannot_be_modified_after_task_creation(prepared):
    bundle,ranks=prepared;cfg=next(iter(ranks['pool'].values()));doc=task(bundle,cfg)
    changed=deepcopy(doc);changed['scored']=17
    with pytest.raises(ValueError):command(changed,bundle,'/bundle','/job')
    assert task(bundle,cfg,'long')['scored']==64
    assert task(bundle,cfg,'qualify')['scored']==1


def test_incomplete_common_pool_cannot_report_budget20(tmp_path):
    (tmp_path/'state.json').write_text(json.dumps(dict(status='RUNNING')))
    with pytest.raises(ValueError,match='whole frozen short pool'):analyze(tmp_path,tmp_path)


def test_formal_board_access_requires_explicit_flag_before_io(tmp_path):
    with pytest.raises(ValueError,match='execute-board'):run(tmp_path,tmp_path)


def test_only_execution_files_are_required_without_dropping_proof_provenance():
    files={k:dict(sha256=k) for k in ('graph.json','module.so','params.bin','lowering_qir.json')}
    source=dict(files=files,canonical_sha256='graph')
    value=runtime_program(source)
    assert set(value['files'])=={'graph.json','module.so','params.bin'}
    assert value['excluded_nonexecution_proof_files']=={'lowering_qir.json':files['lowering_qir.json']}
    assert 'lowering_qir.json' in source['files']


@pytest.mark.parametrize('wrong_affinity',(False,True))
def test_ARM_CI_checks_outputs_and_actual_slots_without_wait_observer(tmp_path,monkeypatch,wrong_affinity):
    import cpu_vta_partitioning.measurement as measurement
    calls=[];monkeypatch.setattr(measurement,'frames',lambda *a:[{}])
    monkeypatch.setattr(measurement,'correctness',lambda *a:calls.append(True))
    cfg=dict(mask=[0,1],threads=2,worker_core_map=[0,1])
    stage=dict(requested_cpus=[0,1],active_runtime_threads=2,caller_tid=10,
        worker_slots=[dict(slot=0,tid=10,cpus=[0]),dict(slot=1,tid=20,cpus=[1])],
        effective_affinities=[dict(tid=10,cpus=[0],cpu_time_delta_ms=1),
                             dict(tid=20,cpus=[0] if wrong_affinity else [1],cpu_time_delta_ms=1),
                             dict(tid=99,cpus=[0,1],cpu_time_delta_ms=0)])
    (tmp_path/'affinity.json').write_text(json.dumps(dict(cpu_worker_pinning=True,stages=[stage])))
    if wrong_affinity:
        with pytest.raises(ValueError,match='actual logical'):qemu_cpu_summary(tmp_path,{'stages':[cfg]},tmp_path)
        assert not calls
    else:
        result=qemu_cpu_summary(tmp_path,{'stages':[cfg]},tmp_path)
        assert result['correct'] and calls and not result['phase_or_wait_observer_loaded']
