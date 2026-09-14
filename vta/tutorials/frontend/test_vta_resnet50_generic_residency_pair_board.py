from run_vta_resnet50_generic_residency_pair_board import (
    graph_workload_occurrences,
    static_delta,
)


def test_graph_workload_occurrences_counts_reused_function_nodes():
    graph = {
        "attrs": {"shape": ["list_shape", [[1], [2], [3], [2], [3]]]},
        "node_row_ptr": [0, 1, 2, 3, 4, 5],
        "nodes": [
            {"op": "null"},
            {"op": "null"},
            {"op": "tvm_op", "attrs": {"func_name": "f0"}, "inputs": [[0, 0, 0], [1, 0, 0]]},
            {"op": "null"},
            {"op": "tvm_op", "attrs": {"func_name": "f0"}, "inputs": [[0, 0, 0], [3, 0, 0]]},
        ],
    }
    workload = ["task", ["TENSOR", [1], "int8"], ["TENSOR", [2], "int8"]]
    rows = graph_workload_occurrences(graph, workload)
    assert [row["graph_node"] for row in rows] == [2, 4]
    assert [row["func_name"] for row in rows] == ["f0", "f0"]


def test_static_delta_preserves_weight_bytes_but_reduces_calls():
    original = {
        "static_feature_vector": {
            "load_dma_bytes": 100, "load_dma_calls": 10,
            "input_dma_bytes": 60, "input_dma_calls": 6,
            "weight_dma_bytes": 40, "weight_dma_calls": 4,
        },
        "sync": {"function_final_drain": 1, "residency_drains": 0},
    }
    residency = {
        "static_feature_vector": {
            "load_dma_bytes": 70, "load_dma_calls": 5,
            "input_dma_bytes": 30, "input_dma_calls": 3,
            "weight_dma_bytes": 40, "weight_dma_calls": 2,
        },
        "sync": {"function_final_drain": 1, "residency_drains": 0},
    }
    assert static_delta(original, residency) == {
        "load_buffer_2d_bytes": -30,
        "load_buffer_2d_calls": -5,
        "load_buffer_2d_inp_bytes": -30,
        "load_buffer_2d_inp_calls": -3,
        "load_buffer_2d_wgt_bytes": 0,
        "load_buffer_2d_wgt_calls": -2,
        "synchronize_calls": 0,
        "driver_run_calls": 0,
    }
