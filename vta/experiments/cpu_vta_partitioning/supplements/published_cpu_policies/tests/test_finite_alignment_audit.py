from pathlib import Path
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src'))
from audit_finite_service_alignment import summarize_model


def model():
    joins=[];tasks=[]
    for kernel in range(12):
        ready=kernel*100
        joins.append(dict(stage=0,frame=1,kernel=kernel,generation=0,
            ready_ns=ready,join_ns=ready+20,first_service_skew_ns=3,
            last_slots=[0,2]))
        for slot in range(4):
            tasks.append(dict(role='stage',stage=0,frame=1,kernel=kernel,
                operation='worker_compute',slot=slot,first_service_ns=ready+slot))
        tasks.append(dict(role='stage',stage=0,frame=1,kernel=kernel,
            operation='completion_wait',slot=0,CPU=dict(numerator=7,denominator=2)))
    return dict(kernel_joins=joins,tasks=tasks)


def test_tied_last_workers_get_fractional_credit():
    rows=summarize_model(model(),[1])
    assert len(rows)==12
    assert all(r['caller_last_fraction']==.5 for r in rows)
    assert rows[0]['last_slot_fractional_counts']=={'0':.5,'2':.5}
    assert rows[0]['caller_wait_CPU_ms']==3.5/1_000_000
    assert rows[0]['latest_first_service_from_ready_ms']==3/1_000_000


def test_missing_and_duplicate_callbacks_rejected():
    data=model();data['tasks'].pop(0)
    with pytest.raises(ValueError,match='callback accounting'):summarize_model(data,[1])
    data=model();data['tasks'].append(data['tasks'][0])
    with pytest.raises(ValueError,match='callback accounting'):summarize_model(data,[1])


def test_same_frame_population_is_required():
    data=model();data['kernel_joins'][0]['frame']=0
    with pytest.raises(ValueError,match='populations differ'):summarize_model(data,[1])
