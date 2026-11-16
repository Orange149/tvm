import copy
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_compiled_gap import model_path


def example():
    return dict(tasks=[dict(stage=0, frame=8, kernel=2, generation=0,
        operation="worker_compute", slot=i, ready_ns=100,
        first_service_ns=100 + i, end_ns=110 + 2*i,
        CPU=10, ready_not_running=2*i) for i in range(3)],
        kernel_joins=[dict(stage=0, frame=8, kernel=2, generation=0,
            tasks=3, ready_ns=100, join_ns=114, last_slots=[2])])


def test_one_completion_path_does_not_sum_other_workers():
    p = model_path(example(), 8, 2, native_slot=1)
    assert p["critical_CPU_ns"] + p["critical_ready_ns"] == p["completion_span_ns"] == 14
    assert p["critical_ready_ns"] == 4  # Concurrent worker waits are 0, 2, 4.
    assert p["same_native_slot_ready_ns"] == 2
    assert p["same_native_slot_initial_ready_ns"] == 1
    assert not p["identity_agrees"]


def test_incomplete_or_inconsistent_paths_are_rejected():
    for mutate in (
        lambda r: r["tasks"].pop(),
        lambda r: r["tasks"].append(copy.deepcopy(r["tasks"][0])),
        lambda r: r["tasks"][0].update(end_ns=99),
        lambda r: r["kernel_joins"][0].update(join_ns=113),
        lambda r: r["tasks"][1].update(ready_not_running=20),
        lambda r: r["tasks"][1].update(generation=1),
    ):
        r = example()
        mutate(r)
        with pytest.raises(ValueError):
            model_path(r, 8, 2, native_slot=1)
