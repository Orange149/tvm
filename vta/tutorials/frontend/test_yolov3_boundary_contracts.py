#!/usr/bin/env python3
"""Unit tests for YOLO stage-split boundary contracts."""

from types import SimpleNamespace

import tvm
from tvm import relay
from tvm.relay import transform

from vta.top import graphpack

import search_yolov3_tiny_stage_splits as search


def typed_expr(shape, dtype="float32"):
    var = relay.var("x", shape=shape, dtype=dtype)
    mod = tvm.IRModule.from_expr(relay.Function([var], var))
    mod = transform.InferType()(mod)
    return mod["main"].body


def test_unpack_batch_channel_slices_padded_channels():
    data = relay.var("packed", shape=(1, 16, 26, 26, 1, 16), dtype="int8")
    unpacked = graphpack._unpack_batch_channel(
        data, old_shape=(1, 255, 26, 26), unpack_transpose=True
    )
    mod = tvm.IRModule.from_expr(relay.Function([data], unpacked))
    mod = transform.InferType()(mod)
    ret_type = mod["main"].ret_type
    assert [int(dim) for dim in ret_type.shape] == [1, 255, 26, 26]
    assert ret_type.dtype == "int8"


def test_unpack_batch_channel_keeps_aligned_channels():
    data = relay.var("packed", shape=(1, 16, 26, 26, 1, 16), dtype="int8")
    unpacked = graphpack._unpack_batch_channel(
        data, old_shape=(1, 256, 26, 26), unpack_transpose=True
    )
    mod = tvm.IRModule.from_expr(relay.Function([data], unpacked))
    mod = transform.InferType()(mod)
    ret_type = mod["main"].ret_type
    assert [int(dim) for dim in ret_type.shape] == [1, 256, 26, 26]


def test_boundary_contract_records_padding_adapter():
    env = SimpleNamespace(BLOCK_OUT=16)
    contract = search.boundary_contract_for_expr(
        "small_logits51",
        typed_expr((1, 255, 26, 26)),
        producer_device="vta",
        consumer_device="cpu",
        env=env,
    )
    assert contract["channel_padding"] == 1
    assert "slice_channel_to_255" in contract["adapters"]


def test_boundary_validation_allows_cpu_tail_but_marks_serial_gate_required():
    env = SimpleNamespace(BLOCK_OUT=16)
    named = {
        "pool4_route": typed_expr((1, 256, 13, 13)),
        "route23": typed_expr((1, 256, 26, 26)),
        "shared38": typed_expr((1, 256, 13, 13)),
    }
    candidate = {
        "candidate_id": "yolo_pool4_route_to_shared13",
        "start_name": "pool4_route",
        "end_name": "shared13",
        "end_output_names": ["route23", "shared38"],
        "needs_route_input": True,
    }
    args = SimpleNamespace(split_quantization_mode="full_graph", tail_device="cpu")
    contracts = search.boundary_contracts_for_candidate(candidate, named, env, args.tail_device)
    validation = search.validate_boundary_contracts(candidate, contracts, args)
    assert validation["status"] == "safe"
    assert "cpu_tail_requires_serial_raw_gate" in validation["boundary_warnings"]
    assert "concat_route_cpu_tail_boundary" in validation["boundary_warnings"]


def test_boundary_validation_allows_route_boundary_but_marks_warning():
    env = SimpleNamespace(BLOCK_OUT=16)
    named = {
        "pool4_route": typed_expr((1, 256, 13, 13)),
        "route23": typed_expr((1, 256, 26, 26)),
        "shared38": typed_expr((1, 256, 13, 13)),
    }
    candidate = {
        "candidate_id": "yolo_pool4_route_to_shared13",
        "start_name": "pool4_route",
        "end_name": "shared13",
        "end_output_names": ["route23", "shared38"],
        "needs_route_input": True,
    }
    args = SimpleNamespace(split_quantization_mode="full_graph", tail_device="vta")
    contracts = search.boundary_contracts_for_candidate(candidate, named, env, args.tail_device)
    validation = search.validate_boundary_contracts(candidate, contracts, args)
    assert validation["status"] == "safe"
    assert "route_branch_cpu_to_vta_boundary" in validation["boundary_warnings"]


def tiny_graph():
    return {
        "nodes": [
            {"op": "null", "name": "data", "inputs": []},
            {"op": "null", "name": "p0", "inputs": []},
            {"op": "tvm_op", "name": "conv0", "inputs": [[0, 0, 0], [1, 0, 0]]},
            {"op": "tvm_op", "name": "conv1", "inputs": [[2, 0, 0], [1, 0, 0]]},
            {"op": "tvm_op", "name": "head", "inputs": [[3, 0, 0]]},
        ],
        "arg_nodes": [0, 1],
        "heads": [[4, 0, 0]],
        "attrs": {
            "dltype": ["list_str", ["float32", "int8", "int8", "int8", "float32"]],
            "device_index": ["list_int", [1, 1, 1, 1, 1]],
            "storage_id": ["list_int", [0, 1, 2, 3, 4]],
            "shape": [
                "list_shape",
                [[1, 3, 4, 4], [16], [1, 16, 4, 4], [1, 16, 2, 2], [1, 10]],
            ],
        },
        "node_row_ptr": [0, 1, 2, 3, 4, 5],
    }


def test_packed_subgraph_boundary_input_preserves_physical_schema():
    graph = tiny_graph()
    sub = search.make_subgraph_json(graph, [4], boundary_input_node_ids=[3])
    assert sub["nodes"][0]["op"] == "null"
    assert sub["nodes"][0]["name"] == "conv1"
    assert sub["attrs"]["shape"][1][0] == [1, 16, 2, 2]
    assert sub["attrs"]["dltype"][1][0] == "int8"
    assert sub["nodes"][1]["name"] == "head"
    assert sub["nodes"][1]["inputs"] == [[0, 0, 0]]
    assert sub["heads"] == [[1, 0, 0]]


def test_packed_subgraph_remaps_params_and_storage_ids():
    graph = tiny_graph()
    sub = search.make_subgraph_json(graph, [3], boundary_input_node_ids=[2])
    assert [node["name"] for node in sub["nodes"]] == ["p0", "conv0", "conv1"]
    assert sub["arg_nodes"] == [0, 1]
    assert sub["nodes"][2]["inputs"] == [[1, 0, 0], [0, 0, 0]]
    assert sub["attrs"]["storage_id"][1] == [0, 1, 2]


def test_packed_subgraph_can_forward_boundary_input_as_output():
    graph = tiny_graph()
    sub = search.make_subgraph_json(graph, [2, 3], boundary_input_node_ids=[2])
    assert [node["name"] for node in sub["nodes"]] == ["p0", "conv0", "conv1"]
    assert sub["arg_nodes"] == [0, 1]
    assert sub["heads"] == [[1, 0, 0], [2, 0, 0]]
    assert sub["attrs"]["shape"][1][1] == [1, 16, 4, 4]


def test_packed_graph_candidate_enumeration_returns_requested_top20():
    graph = tiny_graph()
    # Pad tiny graph with the YOLO node ids used by the enumerator.
    max_node = 68
    graph["nodes"] = graph["nodes"] + [
        {"op": "tvm_op", "name": "n{}".format(i), "inputs": [[i - 1, 0, 0]]}
        for i in range(len(graph["nodes"]), max_node)
    ]
    graph["attrs"]["dltype"][1].extend(["int8"] * (max_node - 5))
    graph["attrs"]["device_index"][1].extend([1] * (max_node - 5))
    graph["attrs"]["storage_id"][1].extend(list(range(5, max_node)))
    graph["attrs"]["shape"][1].extend([[1, 16, 2, 2, 1, 16]] * (max_node - 5))
    graph["heads"] = [[52, 0, 0], [64, 0, 0]]
    candidates = search.enumerate_packed_graph_candidates(graph, 20)
    assert len(candidates) == 20
    assert candidates[0]["candidate_id"] == "yolo_packed_n25_40_to_heads"
    assert all(item["split_backend"] == "packed_graph" for item in candidates)


def test_packed_hetero_candidate_enumeration_marks_cpu_vta_devices():
    graph = tiny_graph()
    max_node = 68
    graph["nodes"] = graph["nodes"] + [
        {"op": "tvm_op", "name": "n{}".format(i), "inputs": [[i - 1, 0, 0]]}
        for i in range(len(graph["nodes"]), max_node)
    ]
    graph["attrs"]["dltype"][1].extend(["int8"] * (max_node - 5))
    graph["attrs"]["device_index"][1].extend([1] * (max_node - 5))
    graph["attrs"]["storage_id"][1].extend(list(range(5, max_node)))
    graph["attrs"]["shape"][1].extend([[1, 16, 2, 2, 1, 16]] * (max_node - 5))
    graph["heads"] = [[52, 0, 0], [64, 0, 0]]
    candidates = search.enumerate_packed_hetero_candidates(graph, 5)
    assert candidates
    assert candidates[0]["split_backend"] == "packed_hetero"
    assert candidates[0]["stage_devices"] == "cpu/vta"
    assert candidates[0]["boundary_valid"]
