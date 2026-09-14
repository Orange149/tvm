import copy

import pytest

from build_vta_resnet50_callsite_dose_graphs import patch_prefix


def repeated_graph():
    return {"nodes": [
        {"op": "null", "attrs": {}},
        {"op": "tvm_op", "attrs": {"func_name": "same"}},
        {"op": "tvm_op", "attrs": {"func_name": "other"}},
        {"op": "tvm_op", "attrs": {"func_name": "same"}},
    ]}


@pytest.mark.parametrize("count,changed", [(0, []), (1, [1]), (2, [1, 3])])
def test_patch_prefix_changes_only_selected_nodes(count, changed):
    original = repeated_graph()
    frozen = copy.deepcopy(original)
    actual = patch_prefix(original, "same", "alias", [1, 3], count)
    assert original == frozen
    assert [index for index, pair in enumerate(zip(original["nodes"], actual["nodes"]))
            if pair[0] != pair[1]] == changed


def test_patch_prefix_rejects_identity_change():
    with pytest.raises(RuntimeError, match="nodes changed"):
        patch_prefix(repeated_graph(), "same", "alias", [1, 2], 1)


def test_patch_prefix_rejects_count_outside_domain():
    with pytest.raises(ValueError, match="outside"):
        patch_prefix(repeated_graph(), "same", "alias", [1, 3], 3)
