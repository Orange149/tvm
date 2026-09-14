"""Tests for the lowered-TIR VTA coprocessor dependency auditor."""

import pytest
import tvm
from tvm import tir

from audit_vta_coproc_dependencies import audit_vta_coproc_dependencies
from test_vta_residency_schedule import MODES, REPRESENTATIVES, _lower, _residency_task
import vta


def _dependency_call(kind, source, target):
    op = tvm.ir.Op.get("tir.vta.coproc_dep_{}".format(kind))
    return tir.Evaluate(tir.Call("int32", op, [source, target]))


def _module(*statements):
    return tvm.IRModule({"main": tir.PrimFunc([], tir.SeqStmt(list(statements)))})


def _pair(source, target):
    return (
        _dependency_call("push", tir.IntImm("int32", source), tir.IntImm("int32", target)),
        _dependency_call("pop", tir.IntImm("int32", source), tir.IntImm("int32", target)),
    )


def test_accepts_balanced_adjacent_stage_edges_and_emits_schema():
    module = _module(*_pair(1, 2), *_pair(2, 1), *_pair(2, 3), *_pair(3, 2))
    result = audit_vta_coproc_dependencies(module)
    assert result["schema"] == "vta_coproc_dependency_audit_v1"
    assert result["valid"]
    assert result["summary"] == {
        "function_count": 1,
        "call_count": 8,
        "push_count": 4,
        "pop_count": 4,
        "edge_count": 4,
        "only_supported_edges": True,
        "push_pop_balanced": True,
        "forbidden_1_3_present": False,
    }
    assert all(edge["allowed"] and edge["balanced"] for edge in result["edges"])


@pytest.mark.parametrize("source_stage,target_stage", [(1, 3), (3, 1)])
def test_rejects_direct_load_store_edges(source_stage, target_stage):
    result = audit_vta_coproc_dependencies(_module(*_pair(source_stage, target_stage)))
    assert not result["valid"]
    assert result["summary"]["forbidden_1_3_present"]
    assert "forbidden_direct_load_store" in {error["code"] for error in result["errors"]}


def test_rejects_push_pop_imbalance_per_directed_edge():
    module = _module(
        _dependency_call("push", tir.IntImm("int32", 1), tir.IntImm("int32", 2)),
        _dependency_call("pop", tir.IntImm("int32", 2), tir.IntImm("int32", 1)),
    )
    result = audit_vta_coproc_dependencies(module)
    assert not result["valid"]
    assert not result["summary"]["push_pop_balanced"]
    assert sum(error["code"] == "push_pop_imbalance" for error in result["errors"]) == 2


def test_traverses_dependencies_nested_in_loop_and_branch():
    body = tir.IfThenElse(
        tir.const(True, "bool"),
        tir.SeqStmt(list(_pair(2, 3))),
        tir.SeqStmt(list(_pair(3, 2))),
    )
    loop = tir.For(tir.Var("i", "int32"), 0, 4, tir.ForKind.SERIAL, body)
    result = audit_vta_coproc_dependencies(tvm.IRModule({"main": tir.PrimFunc([], loop)}))
    assert result["valid"]
    assert result["summary"]["call_count"] == 4
    assert result["semantics"]["loop_multiplicity"] == "not_expanded"
    assert result["semantics"]["path_sensitive"] is False


@pytest.mark.parametrize("mode", MODES)
def test_current_w00_safe_modes_have_supported_balanced_dependencies(mode):
    env = vta.get_env()
    task = _residency_task(0, mode, env)
    config = task.config_space.get(REPRESENTATIVES[0]["config_index"])
    module, _ = _lower(task, config)
    result = audit_vta_coproc_dependencies(module)
    assert result["valid"], result["errors"]
    assert result["summary"]["only_supported_edges"]
    assert result["summary"]["push_pop_balanced"]
    assert not result["summary"]["forbidden_1_3_present"]


def test_rejects_nonconstant_stage_endpoint():
    stage = tir.Var("stage", "int32")
    function = tir.PrimFunc(
        [stage],
        _dependency_call("push", stage, tir.IntImm("int32", 2)),
    )
    result = audit_vta_coproc_dependencies(function)
    assert not result["valid"]
    assert result["errors"][0]["code"] == "nonconstant_stage_endpoint"
