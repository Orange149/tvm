from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'src'))
from cpu_vta_partitioning.common import read,digest
import polling_profile_bridge as bridge


def test_old_batch_is_read_only_and_bridge_controls_are_explicit():
    assert bridge.OUTPUT!=bridge.base.OUTPUT
    assert bridge.base.sources()==read(bridge.base.OUTPUT/'PLAN.json')['sources']
    assert bridge.RUN.name=='polling_profile_02'


def test_command_explicitly_keeps_original_wait_and_disables_queue_data():
    bundle=bridge.base.OUTPUT/'yolov3_tiny416/capsule'
    spec=dict(network='yolov3_tiny416',case='vta_09_12',mode='qualify')
    doc=bridge.task(bundle,spec)
    argv,_=bridge.command(doc,bundle,'/bundle','/job')
    assert 'VTA_WAIT_POLICY=original' in argv and 'VTA_WAIT_PROBE=0' in argv
    assert 'VTA_QUEUE_COUNTERS=0' in argv and 'VTA_WAIT_STATUS=/job/wait_support_status.json' in argv
    assert any(x.startswith('LD_PRELOAD=') and '/runner/libwait_adapter.so' in x for x in argv)
    assert doc['queue_support']['queue_fields_are_not_observations']
    assert doc['configuration']==bridge.base.task(bundle,spec)['configuration']
    assert doc['task_id']!=bridge.base.task(bundle,spec)['task_id']
    clean=bridge.task(bundle,{**spec,'mode':'clean'})
    argv,_=bridge.command(clean,bundle,'/bundle','/job')
    assert not any('libwait_adapter.so' in x for x in argv)
    assert sum(clean[k] for k in ('warmup','scored','tail'))==10
