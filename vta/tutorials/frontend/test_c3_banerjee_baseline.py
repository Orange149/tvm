"""No board, TVM import or timing needed for analytical baseline tests."""

import copy
import json

import pytest

from c3_banerjee_baseline import (Tile, paper_cost, enumerate_node24, function_ast,
                                 write_new, verify_protected, digest)

CAP = {"INP": 32768, "WGT": 262144, "ACC": 131072}


def test_old_tile_units_and_non_mutation():
    cap = copy.deepcopy(CAP)
    r = paper_cost(Tile(1, 2, 14, 1, 1, 2, 1), cap)
    assert cap == CAP
    assert r["paper_allocation_bytes"] == {"INP": 2048, "WGT": 4608, "ACC": 3712}
    assert r["paper_payload_bytes"]["WGT"] == 2016 * 1024
    assert not r["full_fusion_qualified"]


def test_enumeration_unique_and_deterministic():
    rows, ranked = enumerate_node24(CAP)
    assert len(rows) == 1280 and len({r["id"] for r in rows}) == 1280
    assert ranked == enumerate_node24(CAP)[1]
    assert all(r["necessary_conditions_passed"] for r in ranked)
    assert ranked == sorted(ranked, key=lambda r: (r["paper_score_bytes"], r["id"]))


def test_both_thread_directions_rejected_and_fractional_cost_not_truncated():
    both = paper_cost(Tile(1, 2, 7, 1, 1, 2, 2), CAP)
    assert "paper_both_threads_two" in both["rejections"]
    odd = paper_cost(Tile(1, 2, 14, 1, 1, 1, 2), CAP)
    assert "local_exact_outer_thread_divisibility" in odd["rejections"]
    assert odd["paper_score_bytes"] is None


def test_capacity_is_inclusive_and_no_unknown_hardware_capacity():
    row = paper_cost(Tile(1, 14, 14, 8, 8, 1, 1), CAP)
    assert row["paper_allocation_bytes"]["INP"] == CAP["INP"]
    assert row["necessary_conditions_passed"]
    with pytest.raises(KeyError):
        paper_cost(Tile(1, 2, 14, 1, 1, 1, 1), {})
    with pytest.raises(ValueError):
        paper_cost(Tile(1, 3, 14, 1, 1, 1, 1), CAP)


def test_ast_equality_ignores_comments_not_semantics(tmp_path):
    a, b = tmp_path / "a.py", tmp_path / "b.py"
    a.write_text("def f():\n    return 1\n")
    b.write_text("# comment\ndef f():\n    return 1\n")
    assert function_ast(a, "f") == function_ast(b, "f")
    b.write_text("def f():\n    return 2\n")
    assert function_ast(a, "f") != function_ast(b, "f")


def test_protected_history_and_exclusive_artifacts(tmp_path):
    old = tmp_path / "old"
    old.write_text("sealed")
    write_new(tmp_path / "protected.json", {str(old): digest(old)})
    assert verify_protected(tmp_path)
    with pytest.raises(FileExistsError):
        write_new(tmp_path / "protected.json", {})
    old.write_text("changed")
    with pytest.raises(ValueError, match="protected"):
        verify_protected(tmp_path)


def test_paper_ties_are_not_claimed_as_performance_ties():
    rows = [paper_cost(Tile(1, 14, 14, ci, 8, 1, 1), CAP) for ci in (1, 2, 4, 8)]
    assert len({r["paper_score_bytes"] for r in rows}) == 1
    assert len({r["paper_allocation_bytes"]["WGT"] for r in rows}) == 4
    assert not any(r["full_fusion_qualified"] for r in rows)


def test_real_tps_anchor_rejects_unbalanced_tail_dependency():
    from c3_banerjee_baseline import GROUPED
    from audit_c3_banerjee_baseline import certificate
    import re
    path = GROUPED / "20261013_c3_banerjee_baseline_completion_run01/tps_single_thread_attempt03.log"
    text = path.read_text()
    part = re.search(r"C3_NATIVE_CALL_BEGIN 1\n(.*?)C3_NATIVE_CALL_END 1\n", text, re.S)[1]
    with pytest.raises(ValueError, match="unmatched native tokens"):
        certificate(part)


def test_fsim_worker_captures_author_environment_inside_sim_context():
    # Regression for the failed attempt02: distinct coprocessor-axis objects.
    import ast
    from c3_banerjee_baseline import HERE
    tree = ast.parse((HERE / "run_c3_banerjee_baseline.py").read_text())
    scopes = [n for n in ast.walk(tree) if isinstance(n, ast.With) and any(
        isinstance(item.context_expr, ast.Call) and isinstance(item.context_expr.func, ast.Attribute)
        and item.context_expr.func.attr == "Environment" for item in n.items)]
    assert len(scopes) == 1
    assert any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "author_namespace"
               for n in ast.walk(scopes[0]))
