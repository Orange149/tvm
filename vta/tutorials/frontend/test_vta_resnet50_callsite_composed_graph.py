import copy

import pytest

from build_vta_resnet50_callsite_composed_graph import patch_graph_callsite


def graph_with_repeated_function():
    return {
        "nodes": [
            {"op": "null", "name": "data", "attrs": {}},
            {"op": "tvm_op", "attrs": {"func_name": "target"}},
            {"op": "tvm_op", "attrs": {"func_name": "other"}},
            {"op": "tvm_op", "attrs": {"func_name": "target"}},
        ],
        "heads": [[3, 0, 0]],
    }


def test_patch_graph_callsite_changes_exactly_one_occurrence():
    original = graph_with_repeated_function()
    frozen = copy.deepcopy(original)
    composed, matches, selected = patch_graph_callsite(
        original, "target", "target_callsite1", 2, 1
    )

    assert original == frozen
    assert matches == [1, 3]
    assert selected == 3
    assert composed["nodes"][1]["attrs"]["func_name"] == "target"
    assert composed["nodes"][3]["attrs"]["func_name"] == "target_callsite1"
    composed["nodes"][3]["attrs"]["func_name"] = "target"
    assert composed == original


def test_patch_graph_callsite_rejects_changed_instance_count():
    with pytest.raises(RuntimeError, match="instance count changed"):
        patch_graph_callsite(graph_with_repeated_function(), "target", "alias", 3, 0)


def test_patch_graph_callsite_rejects_out_of_range_occurrence():
    with pytest.raises(ValueError, match="outside"):
        patch_graph_callsite(graph_with_repeated_function(), "target", "alias", 2, 2)
