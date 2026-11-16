from copy import deepcopy
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_resnet_middle_gap as m
from cpu_vta_partitioning.common import read,write


def test_bounded_observation_reuses_clean_pipeline_instead_of_remeasuring():
    specs=m.specs()
    assert len(specs)==4
    assert [(s['scope'],s['phase']) for s in specs]==[
        ('component','qualify'),('component','clean'),('component','observed'),('pipeline','observed')]
    assert m.protocol()['DNN_inferences']==28 and m.protocol()['CPU_stage_calls']==21
    with pytest.raises(ValueError,match='execute-board'):m.board()


def test_task_and_native_command_keep_exact_layout_and_original_wait():
    bundle=m.clean.POOL/'resnet18/capsule'
    spec=m.specs()[-1];task=m.make_task(bundle,spec)
    argv,_=m.native_command(task,bundle,'/bundle','/job')
    assert task['stages'][2]['program']=='cpu_05_07'
    assert task['stages'][2]['worker_core_map']==[1,2,3]
    assert task['executor_sha256']=='20ca357bbbce14cf41b2ebc7f13bcbd8e082e878af82f043762ce9cf5350d187'
    assert 'VTA_QUEUE_COUNTERS=1' in argv and 'VTA_WAIT_POLICY=original' in argv
    assert not any('blocking' in str(x) or 'taskset' in str(x) for x in argv)
    assert '--cpu-worker-pinning' in argv and '--phase-event-json' in argv


def test_component_identity_keeps_requested_nonprefix_caller(tmp_path):
    parent=m.clean.POOL/'resnet18/capsule'
    manifest=read(parent/'manifest.json');cfg=m.selection()['components']['middle_t3']
    write(tmp_path/'manifest.json',manifest)
    write(tmp_path/'plans/configurations.json',dict(components={'middle_t3':cfg},pipelines={}))
    task=m.make_task(tmp_path,m.specs()[0])
    assert task['stages'][0]['worker_core_map']==[1,2,3] and task['stages'][0]['caller_core']==1
    assert task['kind']=='qualify_component' and task['adapter_sha256'] is not None
    bad=deepcopy(m.specs()[0]);bad['case']='front_t3'
    with pytest.raises(ValueError,match='unplanned'):m.make_task(tmp_path,bad)
