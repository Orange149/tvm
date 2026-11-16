"""Unit tests for the C3 real-model inventory."""

from pathlib import Path

import build_vta_c3_real_task_inventory as inventory


HERE = Path(__file__).resolve().parent
CFG = Path("/home/orange/.tvm_test_data/darknet/yolov3-tiny.cfg")


def test_darknet_cfg_recovers_real_yolo_convolutions():
    rows = inventory.darknet_call_instances(CFG)
    assert len(rows) == 13
    by_name = {row["source_layer"]: row for row in rows}
    assert by_name["conv2"]["operator_contract"] == {
        "batch": 1,
        "ci": 16,
        "co": 32,
        "input_height": 208,
        "input_width": 208,
        "output_height": 208,
        "output_width": 208,
        "kernel_height": 3,
        "kernel_width": 3,
        "strides": [1, 1],
        "padding": [1, 1, 1, 1],
        "dilation": [1, 1],
        "groups": 1,
        "source_layout": "NCHW/OIHW",
        "input_dtype": "float32_source",
        "weight_dtype": "float32_source",
        "output_dtype": "float32_source",
    }
    assert by_name["conv21"]["operator_contract"]["ci"] == 384
    assert by_name["conv21"]["operator_contract"]["input_height"] == 26
    assert by_name["conv15"]["operator_contract"]["co"] == 255


def test_vta_workload_rejects_nonpacked_channel_counts():
    rows = inventory.annotate_calls(inventory.darknet_call_instances(CFG))
    by_name = {row["source_layer"]: row for row in rows}
    assert by_name["conv0"]["vta_workload"] is None
    assert by_name["conv15"]["vta_workload"] is None
    assert by_name["conv22"]["vta_workload"] is None
    assert by_name["conv2"]["geometry_signature"] == [
        1, 16, 208, 208, 32, 3, 3, 1, 1, 1, 1, 1, 1,
    ]


def test_contexts_and_dispatch_scopes_are_not_collapsed():
    base = {
        "model": "m",
        "source": "test",
        "operator_contract": {
            "batch": 1,
            "ci": 16,
            "co": 16,
            "input_height": 8,
            "input_width": 8,
            "output_height": 8,
            "output_width": 8,
            "kernel_height": 3,
            "kernel_width": 3,
            "strides": [1, 1],
            "padding": [1, 1, 1, 1],
            "dilation": [1, 1],
            "groups": 1,
            "source_layout": "NCHW1n16c/OIHW16o16i",
            "input_dtype": "int8",
            "weight_dtype": "int8",
            "output_dtype": "int32",
        },
    }
    rows = []
    for ordinal, consumer in enumerate(("add", "cast")):
        row = dict(base)
        row.update(
            source_layer="c{}".format(ordinal),
            call_ordinal=ordinal,
            call_instance_id="call{}".format(ordinal),
            context_contract={"consumer": consumer},
        )
        rows.append(row)
    annotated = inventory.annotate_calls(rows)
    assert len({row["dispatch_scope_id"] for row in annotated}) == 1
    assert len({row["execution_contract_id"] for row in annotated}) == 2
    dispatch = inventory.group_rows(annotated, "dispatch_scope_id", "dispatch")
    assert dispatch[0]["call_count"] == 2
    assert dispatch[0]["fusion_context_status"] == "multiple_observed_contexts"


def test_status_does_not_claim_latency_or_incumbent():
    counts = {
        "call_instances": 2,
        "vta_representable_calls": 2,
        "unique_execution_contracts": 2,
        "unique_dispatch_scopes": 1,
        "strict_latency_unseen_pending_review": 1,
    }
    dispatch = [{
        "dispatch_scope_id": "d",
        "model": "m",
        "call_count": 2,
        "execution_contract_ids": ["a", "b"],
        "source_layers": ["x", "y"],
    }]
    exposure = [{
        "dispatch_scope_id": "d",
        "strict_latency_unseen_status": "eligible_pending_manual_evidence_review",
    }]
    text = inventory.render_status(counts, dispatch, exposure)
    assert "No board was contacted" in text
    assert "Incumbents are intentionally unbound" in text
    assert "not presented as lowered fusion certificates" in text
