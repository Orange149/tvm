"""Regression tests for the scoped force-serial residency boundary."""

from collections import Counter
import json
from pathlib import Path

import tvm
from tvm import autotvm, tir
import vta

import vta.top.vta_conv2d_residency  # pylint: disable=unused-import
from vta.top.vta_conv2d import RESIDENCY_MODES


SELECTED = json.loads(
    (
        Path(__file__).resolve().parent
        / "report_out/stage_tile_cotuning/iteration6_safe_overlay_final/selected_tasks.json"
    ).read_text(encoding="utf-8")
)


def _calls(module):
    result = []

    def visit(node):
        if not isinstance(node, tir.Call):
            return
        if isinstance(node.op, tvm.ir.Op):
            name = node.op.name
        else:
            name = str(node.op)
        if name == "tir.call_extern" and node.args:
            name = node.args[0].value
        result.append((name, list(node.args)))

    tir.stmt_functor.post_order_visit(module["main"].body, visit)
    return result


def test_mode_ten_lowers_scoped_force_serial_and_restores_normal_mode():
    env = vta.get_env()
    workload = SELECTED[0]["workload"]
    task = autotvm.task.create(
        "conv2d_packed_residency.vta",
        args=tuple(workload[1:]) + (10,),
        target=env.target,
        target_host=env.target_host,
    )
    cfg = task.config_space.get(252)
    with task.target:
        schedule, tensors = task.instantiate(cfg)
    with vta.build_config(disabled_pass={"tir.CommonSubexprElimTIR"}):
        module = tvm.lower(schedule, tensors, name="main")
    calls = _calls(module)
    debug_values = [
        int(args[-1]) for name, args in calls if name == "VTASetDebugMode"
    ]
    assert RESIDENCY_MODES[10] == "input_weight_resident_composable"
    assert debug_values == [32, 0]
    names = Counter(name for name, _ in calls)
    assert names["tir.vta.coproc_sync"] >= 1


def test_mode_eleven_adds_one_entry_drain_before_the_same_scoped_segment():
    env = vta.get_env()
    workload = SELECTED[0]["workload"]

    def lower(mode):
        task = autotvm.task.create(
            "conv2d_packed_residency.vta",
            args=tuple(workload[1:]) + (mode,),
            target=env.target,
            target_host=env.target_host,
        )
        cfg = task.config_space.get(252)
        with task.target:
            schedule, tensors = task.instantiate(cfg)
        with vta.build_config(disabled_pass={"tir.CommonSubexprElimTIR"}):
            return tvm.lower(schedule, tensors, name="main")

    mode_ten_calls = _calls(lower(10))
    mode_eleven_calls = _calls(lower(11))
    mode_eleven_debug = [
        int(args[-1]) for name, args in mode_eleven_calls if name == "VTASetDebugMode"
    ]
    assert RESIDENCY_MODES[11] == "input_weight_resident_transition_safe"
    assert mode_eleven_debug == [32, 0]
    mode_ten_names = Counter(name for name, _ in mode_ten_calls)
    mode_eleven_names = Counter(name for name, _ in mode_eleven_calls)
    assert mode_eleven_names["tir.vta.coproc_sync"] == (
        mode_ten_names["tir.vta.coproc_sync"] + 1
    )
