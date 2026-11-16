from pathlib import Path
import copy,sys,pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'src'))
from completion_profile import workflow as w


def sample():
    return dict(MMIO_error=False,start_writes=1,auto_writes=3,true_reads=1,
        observed_poll_reads=5,driver_poll_iters=5,false_reads=4,
        wall_start_ns=1,launch_before_ns=10,launch_after_ns=20,
        last_false_before_ns=30,last_false_after_ns=31,
        first_true_before_ns=50,first_true_after_ns=51,wall_end_ns=60)


def test_completion_bounds_include_read_and_launch_uncertainty():
    assert w.completion_bounds(sample())==dict(lower_ns=10,upper_ns=41,width_ns=31,
        first_poll_completed=False,true_completion_not_exactly_observed=True)
    e=sample();e.update(observed_poll_reads=1,driver_poll_iters=1,false_reads=0,
        last_false_before_ns=0,last_false_after_ns=0)
    assert w.completion_bounds(e)['lower_ns']==0


def test_missing_duplicate_or_reordered_mmio_is_not_a_hardware_cost():
    for change in [dict(true_reads=0),dict(true_reads=2),dict(driver_poll_iters=6),
                   dict(MMIO_error=True),dict(last_false_after_ns=55),dict(auto_writes=2),
                   dict(first_true_before_ns=15),dict(launch_after_ns=65)]:
        with pytest.raises(ValueError):w.completion_bounds({**sample(),**change})


def test_bounded_component_only_protocol_and_explicit_board_optin():
    assert w.protocol()['native_sessions']==15 and w.protocol()['VTA_component_calls']==105
    assert w.protocol()['DNN_inferences']==0 and w.protocol()['no_extra_MMIO_reads']
    with pytest.raises(ValueError,match='execute-board'):w.board(execute_board=False)


def test_old_sources_unchanged_and_new_task_keeps_original_wait():
    assert w.parent.sources()==w.read(w.parent.OUTPUT/'PLAN.json')['sources']
    bundle=w.parent.OUTPUT/'yolov3_tiny416/capsule'
    spec=dict(network='yolov3_tiny416',case='vta_09_12',mode='observed');doc=w.task(bundle,spec)
    argv,_=w.command(doc,bundle,'/bundle','/job')
    assert 'VTA_WAIT_POLICY=original' in argv and 'VTA_QUEUE_COUNTERS=0' in argv
    assert doc['task_id']!=w.parent.task(bundle,spec)['task_id']
    assert doc['configuration']==w.parent.task(bundle,spec)['configuration']
