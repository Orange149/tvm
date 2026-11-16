from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_runtime_event_cost_audit as w


def example():
    d={k:0 for k in ('load_buffer_2d_calls','store_buffer_2d_calls','push_gemm_op_calls','push_alu_op_calls',
        'synchronize_calls','mem_copy_from_host_calls','mem_copy_to_host_calls','flush_cache_calls',
        'invalidate_cache_calls','template_capture_calls','template_replay_hits','template_replay_fallbacks',
        'load_buffer_2d_enqueue_us','store_buffer_2d_enqueue_us','push_gemm_op_us','push_alu_op_us')}
    return d


def test_retention_cap_does_not_erase_total_and_nested_copies_are_not_double_DMA():
    d=example();d.update(load_buffer_2d_calls=5000,mem_copy_from_host_calls=1,event_count=4096)
    r=w.summarize_counter(d)
    assert r['emitted_events_per_component']==5002
    assert r['command_API_calls']==5000
    assert r['retained_events']==4096
    with pytest.raises(ValueError):w.summarize_counter({**d,'event_count':200})


def test_board_requires_explicit_flag_without_opening_connection():
    with pytest.raises(ValueError,match='execute-board'):w.board()


def test_native_accounting_rejects_missing_calls_and_wrong_affinity():
    r=dict(status='VALIDATED',DNN_inferences=0,FPGA_access=False,affinity=[3],event_size=152,
        total_events=30000,retained_events=4096,
        blocks=[dict(calls=c,CPU_ns=100) for c in (20000,5000,20000)])
    assert w.check_result(r)==r
    for change in (dict(affinity=[0,1,2,3]),dict(total_events=25000),dict(DNN_inferences=1)):
        with pytest.raises(ValueError):w.check_result({**r,**change})
