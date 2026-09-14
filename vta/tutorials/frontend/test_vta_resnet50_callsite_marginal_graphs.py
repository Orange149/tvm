import copy

import pytest

from build_vta_resnet50_callsite_marginal_graphs import mask_name, patch_mask


def repeated_graph():
    return {"nodes": [
        {"op": "null", "attrs": {}},
        {"op": "tvm_op", "attrs": {"func_name": "same"}},
        {"op": "tvm_op", "attrs": {"func_name": "other"}},
        {"op": "tvm_op", "attrs": {"func_name": "same"}},
    ]}


def test_patch_mask_changes_exact_arbitrary_subset():
    original = repeated_graph()
    frozen = copy.deepcopy(original)
    actual = patch_mask(original, "same", "alias", [1, 3], (0, 1))
    assert original == frozen
    assert actual["nodes"][1]["attrs"]["func_name"] == "same"
    assert actual["nodes"][3]["attrs"]["func_name"] == "alias"


@pytest.mark.parametrize("mask", [(1,), (0, 2)])
def test_patch_mask_rejects_invalid_mask(mask):
    with pytest.raises(ValueError, match="binary value"):
        patch_mask(repeated_graph(), "same", "alias", [1, 3], mask)


def test_patch_mask_rejects_changed_graph_identity():
    with pytest.raises(RuntimeError, match="nodes changed"):
        patch_mask(repeated_graph(), "same", "alias", [1, 2], (1, 0))


def test_mask_name_is_stable():
    assert mask_name((1, 0, 1, 0)) == "mask_1010"
