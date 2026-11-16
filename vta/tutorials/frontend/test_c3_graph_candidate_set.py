import copy

import pytest

from c3_composability import candidate_certificate, graph_context_certificate, static_set_certificate
from c3_graph_candidate_set import (
    build_graph_candidate_set,
    enumerate_graph_candidate_sets,
    inherit_boot_result_identity,
    inherit_graph_context_certificate,
)
from test_c3_composability import records


GRAPH = {"model": "network", "graph_sha256": "graph", "params_sha256": "params"}


def candidate(candidate_id, workload, node, function):
    return {
        "candidate_id": candidate_id,
        "candidate_mode": "input_weight_resident_transition_safe",
        "execution_task_id": workload,
        "graph_node_ids": [node],
        "scoped_functions": [function],
        "candidate_metrics": {
            "total_request_bytes": 10, "total_request_calls": 2,
            "uop_load_bytes": 3, "instruction_bytes": 4, "submissions": 1,
        },
        "incumbent_metrics": {
            "total_request_bytes": 20, "total_request_calls": 4,
            "uop_load_bytes": 2, "instruction_bytes": 3, "submissions": 1,
        },
    }


def certificates(*items):
    return {item[0]: candidate_certificate(*records(item[0], item[1])) for item in items}


def test_graph_set_identity_covers_all_members_scopes_and_graph():
    rows = [candidate("c1", "w1", "n1", "f1"), candidate("c2", "w2", "n2", "f2")]
    result = build_graph_candidate_set(GRAPH, rows, certificates(("c1", "w1"), ("c2", "w2")))
    assert result["candidate_ids"] == ["c1", "c2"]
    assert result["aggregate_metrics"]["total_request_bytes"] == 20.0
    assert result["incumbent_aggregate_metrics"]["total_request_bytes"] == 40.0
    assert result["graph_correctness_claimed"] is False
    changed = copy.deepcopy(rows)
    changed[1]["graph_node_ids"] = ["another-node"]
    assert build_graph_candidate_set(
        GRAPH, changed, certificates(("c1", "w1"), ("c2", "w2"))
    )["candidate_set_id"] != result["candidate_set_id"]

    no_incumbent = copy.deepcopy(rows)
    del no_incumbent[0]["incumbent_metrics"]
    missing = build_graph_candidate_set(
        GRAPH, no_incumbent, certificates(("c1", "w1"), ("c2", "w2"))
    )
    assert missing["candidate_set_id"] == result["candidate_set_id"]
    assert missing["incumbent_aggregate_metrics"] is None


def test_graph_set_rejects_overlapping_node_function_or_workload():
    certs = certificates(("c1", "w1"), ("c2", "w2"))
    rows = [candidate("c1", "w1", "n", "f1"), candidate("c2", "w2", "n", "f2")]
    with pytest.raises(ValueError, match="graph node"):
        build_graph_candidate_set(GRAPH, rows, certs)
    rows[1]["graph_node_ids"] = ["n2"]
    rows[1]["scoped_functions"] = ["f1"]
    with pytest.raises(ValueError, match="function"):
        build_graph_candidate_set(GRAPH, rows, certs)
    rows[1]["scoped_functions"] = ["f2"]
    rows[1]["execution_task_id"] = "w1"
    with pytest.raises(ValueError, match="workload"):
        build_graph_candidate_set(GRAPH, rows, certs)


def test_graph_set_rejects_legacy_unscoped_completion():
    row = candidate("legacy", "w", "n", "f")
    row["candidate_mode"] = "input_weight_resident_barrier"
    with pytest.raises(ValueError, match="ineligible"):
        build_graph_candidate_set(GRAPH, [row], {})


def test_exact_enumeration_keeps_incumbent_and_all_bounded_combinations():
    rows = {
        "w1": [candidate("c1", "w1", "n1", "f1")],
        "w2": [candidate("c2", "w2", "n2", "f2")],
    }
    result = enumerate_graph_candidate_sets(
        GRAPH, rows, certificates(("c1", "w1"), ("c2", "w2")), max_replacements=2
    )
    assert result["cartesian_product_size_including_incumbent_choices"] == 4
    assert len(result["accepted_sets"]) == 4
    assert result["accepted_sets"][0]["is_incumbent"] is True


def test_exact_enumeration_refuses_silent_product_truncation():
    rows = {
        "w1": [candidate("c1", "w1", "n1", "f1")],
        "w2": [candidate("c2", "w2", "n2", "f2")],
    }
    with pytest.raises(ValueError, match="exceeds exact enumeration limit"):
        enumerate_graph_candidate_sets(
            GRAPH, rows, certificates(("c1", "w1"), ("c2", "w2")),
            maximum_combinations=3,
        )


def test_graph_certificate_identity_migration_requires_exact_program_match():
    cert = candidate_certificate(*records("c1", "w1"))
    row = candidate("c1", "w1", "n1", "f1")
    graph_set = build_graph_candidate_set(GRAPH, [row], {"c1": cert})
    static = static_set_certificate(
        [cert], GRAPH,
        {"status": "passed", "global_vta_debug_flag": 0,
         "calls_by_function": {"f1": 2}},
    )
    graph = graph_context_certificate(static, {
        "candidate_set_id": static["candidate_set_id"], "all_equal": True,
        "boot_id": "boot", "bitstream_reloaded": False,
        "mismatch_count_by_seed_and_output": {"0": [0], "1": [0], "2": [0]},
    })
    migrated = inherit_graph_context_certificate(graph_set, static, graph)
    assert migrated["candidate_set_id"] == graph_set["candidate_set_id"]
    assert migrated["checks"]["identity_migration_exact"] is True
    changed = copy.deepcopy(graph_set)
    changed["identity"]["graph"]["params_sha256"] = "different"
    with pytest.raises(ValueError, match="graph or parameter identity changed"):
        inherit_graph_context_certificate(changed, static, graph)


def test_boot_result_identity_migration_never_changes_latency_values():
    cert = candidate_certificate(*records("c1", "w1"))
    graph_set = build_graph_candidate_set(
        GRAPH, [candidate("c1", "w1", "n1", "f1")], {"c1": cert}
    )
    static = static_set_certificate(
        [cert], GRAPH,
        {"status": "passed", "global_vta_debug_flag": 0,
         "calls_by_function": {"f1": 2}},
    )
    graph = graph_context_certificate(static, {
        "candidate_set_id": static["candidate_set_id"], "all_equal": True,
        "boot_id": "boot", "bitstream_reloaded": False,
        "mismatch_count_by_seed_and_output": {"0": [0], "1": [0], "2": [0]},
    })
    migrated_graph = inherit_graph_context_certificate(graph_set, static, graph)
    boot = {
        "candidate_set_id": static["candidate_set_id"], "candidate_ids": ["c1"],
        "boot_id": "boot", "median_improvement_pct": 7.5, "wins": 6, "rounds": 6,
    }
    migrated = inherit_boot_result_identity(graph_set, migrated_graph, boot)
    assert migrated["candidate_set_id"] == graph_set["candidate_set_id"]
    assert migrated["median_improvement_pct"] == 7.5
    assert migrated["latency_values_modified"] is False
    bad = dict(boot, candidate_ids=["another"])
    with pytest.raises(ValueError, match="membership changed"):
        inherit_boot_result_identity(graph_set, migrated_graph, bad)
