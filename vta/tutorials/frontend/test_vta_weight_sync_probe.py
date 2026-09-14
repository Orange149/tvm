"""Qualification and historical counterfactual for VTA weight/sync probe mode 4."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import pytest
import tvm
from tvm import autotvm, tir
import vta
from vta import transform

import vta.top.vta_conv2d_residency  # pylint: disable=unused-import
from extract_static_vta_dma import extract_module_dma
from vta.top.vta_conv2d import RESIDENCY_MODES


BASE = Path(__file__).resolve().parent / "report_out" / "stage_tile_cotuning"
SELECTED = json.loads(
    (BASE / "iteration6_safe_overlay_final" / "selected_tasks.json").read_text()
)
REPRESENTATIVES = {0: 252, 2: 177, 9: 83}
EXPECTED_COUNTERFACTUAL_WGT_BYTES = {
    0: {0: 258048, 4: 36864},
    2: {0: 589824, 4: 147456},
    9: {0: 131072, 4: 131072},
}


def _task(workload_index, mode, env):
    item = SELECTED[workload_index]
    return autotvm.task.create(
        "conv2d_packed_residency.vta",
        args=tuple(item["workload"][1:]) + (mode,),
        target=env.target,
        target_host=env.target_host,
    )


def _schedule(task, config_index):
    config = task.config_space.get(config_index)
    assert config["oc_nthread"].val == 1
    assert config["h_nthread"].val == 1
    with task.target:
        return task.instantiate(config)


def _raw_sync_multiplicity(module):
    loops = []
    multiplicities = []

    def preorder(node):
        if isinstance(node, tir.For):
            loops.append(int(node.extent))
        if isinstance(node, tir.AttrStmt) and node.attr_key == "pragma_coproc_sync":
            product = 1
            for extent in loops:
                product *= extent
            multiplicities.append(product)

    def postorder(node):
        if isinstance(node, tir.For):
            loops.pop()

    tir.stmt_functor.ir_transform(module["main"].body, preorder, postorder, None)
    return multiplicities


def _counterfactual_lower(task, config_index):
    """Strip the broken explicit call only to inspect downstream DMA/dependencies.

    This is diagnostic and is never an executable implementation of the probe.
    """

    @tvm.tir.transform.prim_func_pass(opt_level=0)
    def strip_explicit_sync(func, *_):
        def postorder(stmt):
            if isinstance(stmt, tir.AttrStmt) and stmt.attr_key == "pragma_coproc_sync":
                return stmt.body
            return None

        return func.with_body(
            tir.stmt_functor.ir_transform(
                func.body, None, postorder, ["tir.AttrStmt"]
            )
        )

    def replacement():
        return tvm.transform.Sequential([strip_explicit_sync, tvm.tir.transform.CoProcSync()])

    schedule, tensors = _schedule(task, config_index)
    with patch.object(transform, "InjectCoProcSync", replacement):
        with vta.build_config(disabled_pass={"tir.CommonSubexprElimTIR"}):
            return tvm.lower(schedule, tensors, name="main")


def _dependency_pairs(module):
    pairs = []

    def visit(node):
        if not isinstance(node, tir.Call) or not isinstance(node.op, tvm.ir.Op):
            return
        if node.op.name in ("tir.vta.coproc_dep_push", "tir.vta.coproc_dep_pop"):
            pairs.append((node.op.name.rsplit("_", 1)[-1], int(node.args[0]), int(node.args[1])))

    tir.stmt_functor.post_order_visit(module["main"].body, visit)
    return pairs


def test_mode4_is_isolated_from_existing_mode_numbers():
    assert {key: RESIDENCY_MODES[key] for key in range(4)} == {
        0: "original",
        1: "input_stationary",
        2: "weight_stationary",
        3: "paper_inspired_hybrid",
    }
    assert RESIDENCY_MODES[4] == "weight_stationary_sync_probe"


@pytest.mark.parametrize("workload_index", REPRESENTATIVES)
def test_probe_requests_one_sync_per_outer_weight_tile(workload_index):
    env = vta.get_env()
    task = _task(workload_index, 4, env)
    schedule, tensors = _schedule(task, REPRESENTATIVES[workload_index])
    raw = tvm.lower(schedule, tensors, name="main")
    tile_co = task.config_space.get(REPRESENTATIVES[workload_index])["tile_co"].size[-1]
    co_outer = SELECTED[workload_index]["workload"][2][1][0] // tile_co
    assert _raw_sync_multiplicity(raw) == [co_outer]


@pytest.mark.parametrize("workload_index", REPRESENTATIVES)
def test_explicit_sync_legalizer_removes_cross_barrier_dependency(workload_index):
    env = vta.get_env()
    task = _task(workload_index, 4, env)
    schedule, tensors = _schedule(task, REPRESENTATIVES[workload_index])
    with vta.build_config(disabled_pass={"tir.CommonSubexprElimTIR"}):
        module = tvm.lower(schedule, tensors, name="main")
    pairs = _dependency_pairs(module)
    pushes = Counter((source, target) for kind, source, target in pairs if kind == "push")
    pops = Counter((source, target) for kind, source, target in pairs if kind == "pop")
    assert pushes == pops
    assert not any({source, target} == {1, 3} for source, target in pushes)


@pytest.mark.parametrize("workload_index", REPRESENTATIVES)
def test_counterfactual_exposes_dma_potential_but_illegal_dependency(workload_index):
    env = vta.get_env()
    results = {}
    for mode in (0, 4):
        task = _task(workload_index, mode, env)
        module = _counterfactual_lower(task, REPRESENTATIVES[workload_index])
        results[mode] = extract_module_dma(module, env)["totals"].get(
            "load_buffer_2d_wgt_bytes", 0
        )
        if mode == 4:
            pairs = _dependency_pairs(module)
            pushes = [(source, target) for kind, source, target in pairs if kind == "push"]
            pops = [(source, target) for kind, source, target in pairs if kind == "pop"]
            assert pushes == pops
            assert (3, 1) in pushes
    assert results == EXPECTED_COUNTERFACTUAL_WGT_BYTES[workload_index]


if __name__ == "__main__":
    pytest.main([__file__])
