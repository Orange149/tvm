from copy import deepcopy
import pytest
from analyze_three_thread_worker_gap import critical_path
from run_three_thread_worker_gap_compat import validate_control_tids


def test_one_last_worker_path_conserves_wall_without_summing_waits():
    k=dict(wall_start_ns=100,wall_end_ns=300,
        worker_wall_start_ns=[110,130,150],worker_wall_end_ns=[180,290,220],worker_cpu_ns=[60,100,30])
    r=critical_path(k)
    assert r['last_worker']==1
    assert (r['entry_delay_ns'],r['callback_CPU_ns'],r['callback_wall_minus_CPU_ns'],r['return_tail_ns'])==(30,100,60,10)
    assert sum(r[f] for f in ('entry_delay_ns','callback_CPU_ns','callback_wall_minus_CPU_ns','return_tail_ns'))==200


def test_other_workers_do_not_get_added_to_critical_path_and_missing_vectors_fail():
    k=dict(wall_start_ns=100,wall_end_ns=300,
        worker_wall_start_ns=[110,130],worker_wall_end_ns=[180,290],worker_cpu_ns=[60,100])
    a=critical_path(k);b=deepcopy(k);b['worker_cpu_ns'][0]=1
    assert critical_path(b)==a
    b['worker_wall_end_ns'].pop()
    with pytest.raises(ValueError,match='incomplete'):critical_path(b)


def test_worker_callback_cannot_escape_launch():
    k=dict(wall_start_ns=100,wall_end_ns=300,
        worker_wall_start_ns=[110,130],worker_wall_end_ns=[180,310],worker_cpu_ns=[60,100])
    with pytest.raises(ValueError,match='escaped'):critical_path(k)


def test_extra_control_records_require_the_actual_role_tid():
    affinity={'control_threads':[dict(role='producer',tid=10),dict(role='collector',tid=11)]}
    rows=[dict(stage='_producer',tid=10),dict(stage='_collector',tid=11)]
    validate_control_tids(rows,affinity)
    rows[1]['tid']=10
    with pytest.raises(ValueError,match='TID'):validate_control_tids(rows,affinity)
