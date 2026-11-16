from pathlib import Path
import sys,pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'src'))
import completion_holdout_board as w


def test_distinct_batch_keeps_formal_native_protocol_and_requires_optin():
    assert w.RUN!=w.parent.RUN and w.POOL!=w.parent.POOL
    p=w.protocol();assert p['no_reboot'] and p['no_observer']
    assert p['pipeline']==dict(warmup=8,scored=16,tail=4)
    with pytest.raises(ValueError,match='execute-board'):w.board(execute_board=False)


def test_new_task_changes_campaign_identity_but_not_execution_or_waiting():
    bundle=w.parent.POOL/'yolov3_tiny416/capsule'
    seal=w.read(w.parent.POOL/'SELECTION_FREEZE.json')
    cid=seal['selected_configurations']['yolov3_tiny416'][0]['id']
    spec=dict(network='yolov3_tiny416',case=cid,scope='pipeline',phase='clean',runner='phase',probe=False)
    new=w.make_task(bundle,spec);old=w.parent.make_task(bundle,spec)
    assert new['configuration']==old['configuration'] and new['executor_sha256']==old['executor_sha256']
    assert new['task_id']!=old['task_id'] and new['adapter_sha256'] is None
    command,_=w.native_command(new,bundle,'/bundle','/job')
    assert not any('libwait_adapter' in str(x) or 'libdevice_phase' in str(x) for x in command)
    assert '--cpu-worker-pinning' in command
