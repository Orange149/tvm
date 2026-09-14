"""Local FSim qualification for the explicit-drain weight-residency probe."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
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
REPRESENTATIVES = {
    0: {"name": "W00", "config_index": 252, "sync_executions": 2},
    2: {"name": "W02", "config_index": 177, "sync_executions": 4},
    9: {"name": "W09", "config_index": 83, "sync_executions": 2},
}
EXPECTED_WEIGHT_BYTES = {
    0: {0: 258048, 4: 36864},
    2: {0: 589824, 4: 147456},
    9: {0: 131072, 4: 131072},
}
SEEDS = (0, 20250901, 20260910)


def _task(workload_index, mode, env):
    item = SELECTED[workload_index]
    return autotvm.task.create(
        "conv2d_packed_residency.vta",
        args=tuple(item["workload"][1:]) + (mode,),
        target=env.target,
        target_host=env.target_host,
    )


def _instantiate(task, config_index):
    config = task.config_space.get(config_index)
    assert config["oc_nthread"].val == 1
    assert config["h_nthread"].val == 1
    with task.target:
        return task.instantiate(config)


def _lower(task, config_index):
    schedule, tensors = _instantiate(task, config_index)
    with vta.build_config(disabled_pass={"tir.CommonSubexprElimTIR"}):
        return tvm.lower(schedule, tensors, name="main"), schedule, tensors


def _dependency_pairs(module):
    pairs = []

    def visit(node):
        if not isinstance(node, tir.Call) or not isinstance(node.op, tvm.ir.Op):
            return
        if node.op.name in ("tir.vta.coproc_dep_push", "tir.vta.coproc_dep_pop"):
            pairs.append(
                (node.op.name.rsplit("_", 1)[-1], int(node.args[0]), int(node.args[1]))
            )

    tir.stmt_functor.post_order_visit(module["main"].body, visit)
    return pairs


def _sync_call_multiplicity(module):
    """Count static executions of sync calls using enclosing constant-loop extents."""

    loop_extents = []
    multiplicities = []

    def preorder(node):
        if isinstance(node, tir.For):
            loop_extents.append(int(node.extent))
        if (
            isinstance(node, tir.Call)
            and isinstance(node.op, tvm.ir.Op)
            and node.op.name == "tir.vta.coproc_sync"
        ):
            product = 1
            for extent in loop_extents:
                product *= extent
            multiplicities.append(product)

    def postorder(node):
        if isinstance(node, tir.For):
            loop_extents.pop()

    tir.stmt_functor.ir_transform(module["main"].body, preorder, postorder, None)
    return multiplicities


@pytest.mark.parametrize("workload_index", REPRESENTATIVES)
def test_weight_probe_has_expected_dma_and_legal_dependencies(workload_index):
    env = vta.get_env()
    index = REPRESENTATIVES[workload_index]["config_index"]
    observed = {}
    for mode in (0, 4):
        task = _task(workload_index, mode, env)
        module, _, _ = _lower(task, index)
        observed[mode] = extract_module_dma(module, env)["totals"].get(
            "load_buffer_2d_wgt_bytes", 0
        )
        if mode == 4:
            pairs = _dependency_pairs(module)
            pushes = Counter((src, dst) for kind, src, dst in pairs if kind == "push")
            pops = Counter((src, dst) for kind, src, dst in pairs if kind == "pop")
            assert pushes == pops
            assert not any({src, dst} == {1, 3} for src, dst in pushes)
            # The first call is the probe's per-weight-tile drain.  VTA's existing
            # lowering also retains one function-final synchronization outside it.
            assert _sync_call_multiplicity(module) == [
                REPRESENTATIVES[workload_index]["sync_executions"],
                1,
            ]
    assert observed == EXPECTED_WEIGHT_BYTES[workload_index]


@pytest.mark.parametrize("workload_index", REPRESENTATIVES)
def test_weight_probe_matches_independent_reference_for_three_seeds(workload_index):
    env = vta.get_env()
    if env.TARGET != "sim":
        pytest.skip("Set VTA_HW_PATH to a local fsim config; this test never uses board RPC")
    from vta.testing import simulator

    assert simulator.enabled()
    representative = REPRESENTATIVES[workload_index]
    task = _task(workload_index, 4, env)
    _, schedule, tensors = _lower(task, representative["config_index"])
    with vta.build_config(disabled_pass={"tir.CommonSubexprElimTIR"}):
        module = vta.build(
            schedule,
            tensors,
            target=tvm.target.Target(env.target, host=env.target_host),
            name="main",
        )

    temporary = utils.tempdir()
    module_name = "{}_mode4.o".format(representative["name"])
    module.save(temporary.relpath(module_name))
    remote = rpc.LocalSession()
    remote.upload(temporary.relpath(module_name))
    function = remote.load_module(module_name)["main"]
    device = remote.ext_dev(0)
    workload = SELECTED[workload_index]["workload"]
    for seed in SEEDS:
        data, weight, expected = reference_data_from_workload(workload, seed)
        output = tvm.nd.empty(expected.shape, "int8", device=device)
        function(tvm.nd.array(data, device), tvm.nd.array(weight, device), output)
        np.testing.assert_array_equal(output.numpy(), expected)


if __name__ == "__main__":
    pytest.main([__file__])
