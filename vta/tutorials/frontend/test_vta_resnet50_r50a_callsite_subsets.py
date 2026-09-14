from build_vta_resnet50_r50a_callsite_subsets import mask_name, patch_mask


def test_patch_mask_handles_two_symbols_and_shared_source_symbol():
    graph = {
        "nodes": [
            {"op": "null"},
            {"op": "tvm_op", "attrs": {"func_name": "f7"}},
            {"op": "tvm_op", "attrs": {"func_name": "f9"}},
            {"op": "tvm_op", "attrs": {"func_name": "f9"}},
        ]
    }
    routes = [
        {"graph_node": 1, "func_name": "f7", "alias_symbol": "f7_n1"},
        {"graph_node": 2, "func_name": "f9", "alias_symbol": "f9_n2"},
        {"graph_node": 3, "func_name": "f9", "alias_symbol": "f9_n3"},
    ]
    patched = patch_mask(graph, routes, (1, 0, 1))
    assert [node.get("attrs", {}).get("func_name") for node in patched["nodes"]] == [
        None, "f7_n1", "f9", "f9_n3"
    ]
    assert graph["nodes"][1]["attrs"]["func_name"] == "f7"
    assert mask_name((1, 0, 1)) == "mask101"
