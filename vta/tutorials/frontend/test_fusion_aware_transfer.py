"""Unit tests for fixed-tile fusion-aware transfer accounting."""

import json

from fusion_aware_transfer import (
    add_vectors,
    contract_bytes,
    estimate_segment,
    is_conversion_helper,
    pareto_layers,
    zero_dma,
)


def minimal_model():
    dma = zero_dma()
    dma["load_buffer_2d_inp_bytes"] = 100
    residual = zero_dma()
    residual["load_buffer_2d_acc_bytes"] = 8
    return {
        "environment": {"sha256": "unused"},
        "runtime_boundary_policy": "set_get_copy_once_per_standalone_segment_contract",
        "workloads": {"known": {"base_dma": dma}},
        "fusion_classes": {
            "fusion": {
                "supported": True,
                "fusion_residual_dma": residual,
                "host": {"cpu_read_bytes": 4, "cpu_write_bytes": 8, "cpu_helper_calls": 1, "convert_calls": 0},
            }
        },
    }


def test_contract_bytes_supports_tuple_slots():
    contract = {"slots": [
        {"shape": [1, 4, 2, 2], "dtype": "int8"},
        {"shape": [1, 2], "dtype": "float32"},
    ]}
    assert contract_bytes(contract) == 24


def test_conversion_helper_excludes_cpu_residual_tail():
    assert is_conversion_helper(["cast"])
    assert is_conversion_helper(["reshape", "transpose"])
    assert not is_conversion_helper(["cast", "add", "nn.relu", "cast"])


def test_estimate_is_fail_closed_for_unknown_fusion(tmp_path, monkeypatch):
    import fusion_aware_transfer

    monkeypatch.setattr(fusion_aware_transfer, "workload_key", lambda workload, config: "known")
    segment_dir = tmp_path / "vta_01_01"
    segment_dir.mkdir()
    (segment_dir / "primitives.json").write_text(json.dumps([
        {"signature": "fusion"}, {"signature": "unknown"}
    ]))
    segment = {
        "segment_id": "vta:01:01",
        "convolutions": [{"workload": ["w"], "config": {"index": 1}, "is_fallback": False}],
        "input_contract": {"slots": [{"shape": [1, 2], "dtype": "int8"}]},
        "output_contract": {"slots": [{"shape": [1, 2], "dtype": "int8"}]},
    }
    row = estimate_segment(model=minimal_model(), scan_dir=tmp_path, segment=segment)
    assert row["status"] == "new_fusion_class"
    assert not row["usable_for_ranking"]
    assert row["full_build_required"]
    assert row["total_dma"]["load_buffer_2d_inp_bytes"] == 100
    assert row["total_dma"]["load_buffer_2d_acc_bytes"] == 8


def test_estimate_is_fail_closed_for_unknown_workload(tmp_path, monkeypatch):
    import fusion_aware_transfer

    monkeypatch.setattr(fusion_aware_transfer, "workload_key", lambda workload, config: "missing")
    segment_dir = tmp_path / "vta_01_01"
    segment_dir.mkdir()
    (segment_dir / "primitives.json").write_text(json.dumps([{"signature": "fusion"}]))
    segment = {
        "segment_id": "vta:01:01",
        "convolutions": [{"workload": ["w"], "config": {"index": 1}, "is_fallback": False}],
        "input_contract": {"slots": []},
        "output_contract": {"slots": []},
    }
    row = estimate_segment(model=minimal_model(), scan_dir=tmp_path, segment=segment)
    assert row["status"] == "new_workload"
    assert row["full_build_required"] and not row["usable_for_ranking"]


def test_estimate_is_fail_closed_for_environment_mismatch(tmp_path, monkeypatch):
    import fusion_aware_transfer

    monkeypatch.setattr(fusion_aware_transfer, "workload_key", lambda workload, config: "known")
    segment_dir = tmp_path / "vta_01_01"
    segment_dir.mkdir()
    (segment_dir / "primitives.json").write_text(json.dumps([{"signature": "fusion"}]))
    segment = {
        "segment_id": "vta:01:01",
        "convolutions": [{"workload": ["w"], "config": {"index": 1}, "is_fallback": False}],
        "input_contract": {"slots": []},
        "output_contract": {"slots": []},
    }
    row = estimate_segment(
        model=minimal_model(), scan_dir=tmp_path, segment=segment, candidate_provenance={}
    )
    assert row["status"] == "environment_mismatch"
    assert row["full_build_required"] and not row["usable_for_ranking"]


def test_estimate_is_fail_closed_for_dynamic_dma(tmp_path, monkeypatch):
    import fusion_aware_transfer

    model = minimal_model()
    model["fusion_classes"]["fusion"]["supported"] = False
    monkeypatch.setattr(fusion_aware_transfer, "workload_key", lambda workload, config: "known")
    segment_dir = tmp_path / "vta_01_01"
    segment_dir.mkdir()
    (segment_dir / "primitives.json").write_text(json.dumps([{"signature": "fusion"}]))
    segment = {
        "segment_id": "vta:01:01",
        "convolutions": [{"workload": ["w"], "config": {"index": 1}, "is_fallback": False}],
        "input_contract": {"slots": []},
        "output_contract": {"slots": []},
    }
    row = estimate_segment(model=model, scan_dir=tmp_path, segment=segment)
    assert row["status"] == "unsupported_dynamic_dma"
    assert row["full_build_required"] and not row["usable_for_ranking"]


def test_pareto_layers_order_without_pruning():
    rows = [
        {"candidate_id": "a", "m1_score_ms": 1.0, "x": 1, "y": 2},
        {"candidate_id": "b", "m1_score_ms": 1.0, "x": 2, "y": 1},
        {"candidate_id": "c", "m1_score_ms": 1.0, "x": 3, "y": 3},
    ]
    layers = pareto_layers(rows, ("x", "y"))
    assert [[row["candidate_id"] for row in layer] for layer in layers] == [["a", "b"], ["c"]]
    assert sum(map(len, layers)) == len(rows)


def test_vector_addition_keeps_dimensions_separate():
    assert add_vectors({"bytes": 4}, {"calls": 2}) == {"bytes": 4, "calls": 2}
