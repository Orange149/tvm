from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_scope_tail_comparison as m
from cpu_vta_partitioning.common import read
from common_top20.runtime import task,command


def prepared():
    if not (m.OUT/'PREPARATION.json').exists():pytest.skip('requires prepared comparison')


def test_expansion_requires_its_own_concrete_approval():
    with pytest.raises(ValueError,match='execute-board'):m.board(execute_board=False)
    if not (m.OUT/'AUTHORIZATION.json').exists():
        with pytest.raises(ValueError,match='explicit user approval'):m.board(execute_board=True)


def test_exact_frozen_lists_and_88_configuration_scope():
    prepared();plan,pool,order=m.selection()
    assert read(m.OUT/'LISTS.json')==read(m.OLD/'LISTS.json')
    assert sum(map(len,pool.values()))==88 and len(order)==88
    assert plan['native_sessions']==176 and plan['DNN_inferences']==88*49
    assert read(m.OUT/'PROTOCOL.json')['short_window']==dict(warmup=8,scored=16,tail=24)


def test_original_executor_sixteen_timestamps_and_no_observers():
    prepared()
    for network in ('resnet18','yolov3_tiny416'):
        bundle=m.OUT/network/'capsule';manifest=m.validate(bundle)
        assert manifest['runner_sha256']==m.parent.RUNNER_SHA
        cfg=next(iter(read(m.OUT/'POOL.json')[network].values()))
        doc=task(bundle,cfg,'short');argv,_=command(doc,bundle,'/bundle','/job')
        assert (doc['warmup'],doc['scored'],doc['tail'])==(8,16,24)
        assert argv[argv.index('--runs')+1]=='48'
        assert '--phase-event-json' not in argv and '--queue-timing' not in argv


def test_wrong_boot_is_rejected_before_a_run_identity_is_written(tmp_path):
    prepared()
    with pytest.raises(ValueError,match='boot changed'):m.bind_run(tmp_path,'other-boot',{})
    assert not (tmp_path/'run.json').exists()
