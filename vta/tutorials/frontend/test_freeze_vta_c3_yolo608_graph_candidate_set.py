from freeze_vta_c3_yolo608_graph_candidate_set import graph_route_scope


def test_graph_route_scope_uses_both_input_and_kernel_shape():
    graph = {
        "nodes": [
            {"op": "null", "name": "data"},
            {"op": "null", "name": "weight"},
            {"op": "tvm_op", "name": "fused", "inputs": [[0, 0, 0], [1, 0, 0]],
             "attrs": {"func_name": "fused"}},
        ],
        "node_row_ptr": [0, 1, 2, 3],
        "attrs": {"shape": ["list_shape", [[1, 2], [3, 2], [1, 3]]]},
    }
    workload = ["conv2d_packed.vta", ["TENSOR", [1, 2], "int8"],
                ["TENSOR", [3, 2], "int8"]]
    result = graph_route_scope(graph, workload)
    assert result == {"graph_node_id": "2:fused", "scoped_function": "fused_compute_"}
