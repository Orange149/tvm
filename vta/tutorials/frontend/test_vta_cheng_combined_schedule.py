"""Local checks for the Cheng Scheme-4 functional reimplementation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import tvm
from tvm import autotvm, rpc, tir
from tvm.contrib import utils
import vta

import vta.top.vta_conv2d_residency  # pylint: disable=unused-import
from extract_static_vta_dma import extract_module_dma
from qualify_vta_residency_fsim import reference_data_from_workload


BASE = Path(__file__).resolve().parent / "report_out" / "stage_tile_cotuning"
SELECTED = json.loads(
    (BASE / "iteration6_safe_overlay_final" / "selected_tasks.json").read_text()
)
REPRESENTATIVES = {0: 252, 2: 177, 9: 83}
EXPECTED = {
    0: {"input": 243712, "weight": 36864},
    2: {"input": 121856, "weight": 147456},
    9: {"input": 43264, "weight": 131072},
}


def _task(workload_index, mode):
    env = vta.get_env()
    return autotvm.task.create(
        "conv2d_packed_residency.vta",
        args=tuple(SELECTED[workload_index]["workload"][1:]) + (mode,),
        target=env.target,
        target_host=env.target_host,
    )


def _lower(workload_index, config_index, mode):
    task = _task(workload_index, mode)
    with task.target:
        schedule, tensors = task.instantiate(task.config_space.get(config_index))
    with vta.build_config(disabled_pass={"tir.CommonSubexprElimTIR"}):
        return tvm.lower(schedule, tensors, name="main")


def _sync_count(module):
    result = []

    def visit(node):
        if (
            isinstance(node, tir.Call)
            and isinstance(node.op, tvm.ir.Op)
            and node.op.name == "tir.vta.coproc_sync"
        ):
            result.append(node)

    tir.stmt_functor.post_order_visit(module["main"].body, visit)
    return len(result)


@pytest.mark.parametrize("workload_index", sorted(REPRESENTATIVES))
def test_combined_mode_reduces_both_input_and_weight_to_expected_values(workload_index):
    env = vta.get_env()
    index = REPRESENTATIVES[workload_index]
    original = extract_module_dma(_lower(workload_index, index, 0), env)["totals"]
    combined_module = _lower(workload_index, index, 5)
    combined = extract_module_dma(combined_module, env)["totals"]
    assert combined["load_buffer_2d_inp_bytes"] == EXPECTED[workload_index]["input"]
    assert combined["load_buffer_2d_wgt_bytes"] == EXPECTED[workload_index]["weight"]
    assert combined["load_buffer_2d_inp_bytes"] <= original["load_buffer_2d_inp_bytes"]
    assert combined["load_buffer_2d_wgt_bytes"] <= original["load_buffer_2d_wgt_bytes"]
    # One explicit end-of-residency barrier plus VTA's function-final sync.
    assert _sync_count(combined_module) == 2


def test_combined_mode_is_not_applicable_when_full_kernel_exceeds_weight_sram():
    with pytest.raises(ValueError, match="not_applicable: full kernel"):
        _task(4, 5)


@pytest.mark.parametrize("workload_index", sorted(REPRESENTATIVES))
def test_combined_mode_matches_reference_for_three_fsim_seeds(workload_index):
    env = vta.get_env()
    if env.TARGET != "sim":
        pytest.skip("Set VTA_HW_PATH to the local FSim configuration")
    from vta.testing import simulator

    assert simulator.enabled()
    task = _task(workload_index, 5)
    with task.target:
        schedule, tensors = task.instantiate(
            task.config_space.get(REPRESENTATIVES[workload_index])
        )
    with vta.build_config(disabled_pass={"tir.CommonSubexprElimTIR"}):
        module = vta.build(
            schedule,
            tensors,
            target=tvm.target.Target(env.target, host=env.target_host),
            name="main",
        )
    temporary = utils.tempdir()
    module_name = "cheng_combined_{}.o".format(workload_index)
    module.save(temporary.relpath(module_name))
    remote = rpc.LocalSession()
    remote.upload(temporary.relpath(module_name))
    function = remote.load_module(module_name)["main"]
    device = remote.ext_dev(0)
    for seed in (0, 20250901, 20260910):
        data, weight, expected = reference_data_from_workload(
            SELECTED[workload_index]["workload"], seed
        )
        output = tvm.nd.empty(expected.shape, "int8", device=device)
        function(tvm.nd.array(data, device), tvm.nd.array(weight, device), output)
        assert (output.numpy() == expected).all()
