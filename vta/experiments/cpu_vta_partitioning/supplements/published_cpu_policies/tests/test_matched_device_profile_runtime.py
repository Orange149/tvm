from copy import deepcopy
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_matched_device_profile as m
from cpu_vta_partitioning.common import read
from common_top20.runtime import task as old_task,command as old_command


def prepared():
    if not (m.OUT/'PLAN.json').exists():pytest.skip('capsule not yet prepared')
    m.validate()


def test_frozen_scope_and_explicit_board_optin():
    specs=m.specifications()
    assert len(specs)==11
    assert sum(s['scope']=='component' for s in specs)==9
    assert not any(s['scope']=='pipeline' and s['mode']=='clean' for s in specs)
    with pytest.raises(ValueError,match='execute-board'):m.board(execute_board=False)


def test_prepared_task_count_windows_and_original_executor_identity():
    prepared();docs=[m.make_task(s) for s in m.specifications()]
    assert len({d['task_id'] for d in docs})==11
    assert sum(sum(d[k] for k in ('warmup','scored','tail')) for d in docs if d['spec']['scope']=='component')==63
    assert sum(sum(d[k] for k in ('warmup','scored','tail')) for d in docs if d['spec']['scope']=='pipeline')==49
    assert {d['executor_sha256'] for d in docs}=={'25833e7f8b47fc8824ff9460afc0b968f24934dcee41c3b1f28619a99b6b6fe6'}


def test_two_and_three_input_component_frontiers_and_group_host_binding():
    prepared()
    for name,count in [('vta_01_01',1),('vta_05_06',2),('vta_09_11',3)]:
        doc=m.make_task(dict(scope='component',case=name,mode='observed'))
        argv,inputs=m.command(doc,m.BUNDLE,'/frozen','/job')
        assert len(inputs.strip().split(','))==count
        assert '--serial' in argv and '--cpu-worker-pinning' not in argv
        assert doc['configuration']['host_mask']==[3]
        assert 'VTA_DEVICE_PHASE_TRACE=/job/device_trace.jsonl' in argv
        assert '--queue-timing' not in argv and '--process-cpu-affinity' not in argv


def test_pipeline_command_differs_from_accepted_clean_only_by_observer():
    prepared();doc=m.make_task(dict(scope='pipeline',case='static_first',mode='observed'))
    argv,inputs=m.command(doc,m.BUNDLE,'/frozen','/job')
    old_bundle=m.PARENT/'yolov3_tiny416/capsule'
    expected,expected_inputs=old_command(old_task(old_bundle,doc['configuration'],'short'),old_bundle,'/frozen','/job')
    argv=[s for s in argv if not s.startswith('VTA_DEVICE_PHASE_TRACE=')]
    argv=[s if not s.startswith('LD_PRELOAD=') else 'LD_PRELOAD=/frozen/runtime/libvta.so' for s in argv]
    assert argv==expected and inputs==expected_inputs
    wrong=deepcopy(doc);wrong['tail']=4
    with pytest.raises(ValueError,match='identity'):m.command(wrong,m.BUNDLE,'/frozen','/job')


def test_payload_is_self_contained_and_controller_snapshots_are_independent():
    prepared();manifest=read(m.BUNDLE/'manifest.json');payload=read(m.BUNDLE/'plans/board_payload.json')
    assert payload['unpacked_bytes']==sum(v['bytes'] for v in payload['files'].values())
    assert all(manifest['files'][name]==entry for name,entry in payload['files'].items())
    assert 'runner/libdevice_phase.so' in payload['files']
    for i,(source,expected) in enumerate(sorted(m.sources().items())):
        target=m.BUNDLE/'sources'/f'{i:02}_{Path(source).name}'
        assert not Path(source).samefile(target)


def test_controller_provides_an_existing_deployment_directory_before_native_start(tmp_path,monkeypatch):
    prepared()
    class BeforeNative(Exception):pass
    class FakeSSH:
        def boot_id(self):return read(m.OUT/'PLAN.json')['expected_boot']
        def deploy(self,bundle,directory):
            assert Path(directory).is_dir()
            raise BeforeNative('stop before upload or measurement')
    monkeypatch.setattr(m,'RUN',tmp_path/'new_run')
    monkeypatch.setattr(m,'transport',lambda *args,**kwargs:FakeSSH())
    with pytest.raises(BeforeNative):m.board(execute_board=True)
    assert read(m.RUN/'state.json')['accepted_sessions']==0
