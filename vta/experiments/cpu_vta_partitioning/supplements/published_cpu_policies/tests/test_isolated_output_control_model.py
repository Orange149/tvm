from copy import deepcopy
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compiled_sleep_model import CompiledFinite
from isolated_output_control_model import IsolatedOutputFinite, validate_cost


def cost(cpu):
    return dict(schema="isolated-output-control-v1", origin="isolated_output_serialization",
        network="resnet18", ring_count=1, CPU_ns=cpu, warmup=2, scored=8,
        output_bytes=4000, output_count=1, contains_pipeline_data=False,
        measured_scope="ResultJSON_and_deferred_string_insertion",
        **{k:"a"*64 for k in ("receipt_sha256", "plan_sha256", "runtime_sha256", "serializer_source_sha256")})


def stage():
    return dict(device="cpu", mask=[0], worker_core_map=[0], set_cpu_ns=0, get_cpu_ns=0,
        profile=dict(kernels=[], serial_cpu_segments=[10]))


def kwargs():
    return dict(quanta={i:12 for i in range(2,7)}, frames=12, warmup=2, scored=8,
        queue_depth=2, wake=True, sleeper=True, wakeup_granularity_ns=3, sleeper_credit_ns=9)


def test_zero_cost_restores_original_timestamps_work_and_stage_services():
    a = CompiledFinite([stage()], **kwargs()).run()
    b = IsolatedOutputFinite([stage()], isolated_output_cost=cost(0), collector_mask=[1], **kwargs()).run()
    for key in ("timestamps_ns", "stage_service_ns", "useful_CPU_ns", "synthetic_wait_CPU_ns"):
        assert a["replay"][key] == b["replay"][key]


def test_independent_collector_work_is_not_added_to_every_interval():
    r = IsolatedOutputFinite([stage()], isolated_output_cost=cost(3), collector_mask=[1], **kwargs()).run()
    assert r["replay"]["interval_ns"] == 10
    cpu = [t for t in r["actor_totals"] if t["role"] == "_collector"]
    assert len(cpu) == 1 and cpu[0]["CPU"] == dict(numerator=24, denominator=1)
    assert r["input_production_CPU_unmodeled"] and r["output_cost_not_added_directly_to_II"]


def test_collector_backpressure_delays_future_frames_not_current_completion():
    r = IsolatedOutputFinite([stage()], isolated_output_cost=cost(100), collector_mask=[1],
        **{**kwargs(), "frames":8, "warmup":0}).run()
    assert r["replay"]["timestamps_ns"] == [10,20,30,40,120,220,320,420]


def test_pipeline_control_values_or_unknown_fields_are_rejected():
    for change in (dict(FPS=13), dict(origin="observed_pipeline_control_CPU"),
                   dict(contains_pipeline_data=True), dict(CPU_ns=3.5), dict(receipt_sha256="unknown")):
        c = deepcopy(cost(3)); c.update(change)
        with pytest.raises(ValueError):
            validate_cost(c)
    with pytest.raises(ValueError):
        IsolatedOutputFinite([stage()], isolated_output_cost=cost(3), collector_mask=[1],
            controls={"_collector":dict(CPU_ns=1)}, **kwargs())
