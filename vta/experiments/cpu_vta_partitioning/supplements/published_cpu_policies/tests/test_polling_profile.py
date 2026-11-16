from pathlib import Path
from copy import deepcopy
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'src'))
from polling_profile.workflow import protocol,validate_counters,board


def event():
    return dict(counters_monotonic=True,driver_run_calls=1,driver_run_insns=17,instructions=17,
        driver_timeout_calls=0,driver_poll_iters=10,driver_run_total_us=1000,
        driver_submit_mmio_us=1,driver_post_start_sleep_us=50,driver_poll_wait_us=949,
        wall_start_ns=1000000,wall_end_ns=2010000)


def test_stats_are_per_original_call_and_time_decomposition_is_conservative():
    assert validate_counters([event()])['poll_checks']==10
    e=event();e['driver_run_calls']=2
    with pytest.raises(ValueError,match='scope'):validate_counters([e])
    e=event();e['driver_poll_wait_us']=2000
    with pytest.raises(ValueError,match='clocks'):validate_counters([e])
    e=event();e['driver_poll_wait_us']=float('nan')
    with pytest.raises(ValueError,match='time'):validate_counters([e])


def test_frozen_task_budget_and_board_opt_in():
    p=protocol()
    assert sum(len(v) for v in p['components'].values())==5
    assert p['max_native_sessions']==15 and p['VTA_component_calls']==5*(1+10+10)
    assert p['DNN_inferences']==0 and p['no_per_poll_probe']
    with pytest.raises(ValueError,match='execute-board'):board(execute_board=False)
