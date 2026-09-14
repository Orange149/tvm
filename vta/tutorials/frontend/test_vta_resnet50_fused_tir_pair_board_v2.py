import json
import hashlib

from run_vta_resnet50_fused_tir_pair_board_v2 import load_fused_contract, profile_delta


def test_profile_delta_includes_fused_acc_and_store_fields():
    modes = ["original", "input_stationary"]
    base = {
        "load_buffer_2d_bytes": 100,
        "load_buffer_2d_calls": 10,
        "load_buffer_2d_acc_bytes": 10,
        "load_buffer_2d_acc_calls": 2,
        "load_buffer_2d_inp_bytes": 50,
        "load_buffer_2d_inp_calls": 4,
        "load_buffer_2d_wgt_bytes": 40,
        "load_buffer_2d_wgt_calls": 4,
        "store_buffer_2d_bytes": 20,
        "store_buffer_2d_calls": 2,
    }
    changed = dict(base)
    changed["load_buffer_2d_bytes"] -= 30
    changed["load_buffer_2d_calls"] -= 3
    changed["load_buffer_2d_acc_calls"] -= 1
    rows = [
        {"public_mode": modes[0], "runtime_profile_complete": base},
        {"public_mode": modes[1], "runtime_profile_complete": changed},
    ]
    result = profile_delta(rows, modes)
    assert result["load_buffer_2d_bytes"] == -30
    assert result["load_buffer_2d_calls"] == -3
    assert result["load_buffer_2d_acc_calls"] == -1
    assert result["store_buffer_2d_calls"] == 0


def test_load_fused_contract_rejects_tampered_artifact(tmp_path):
    analysis = {
        "status": "compiler_fused_tir_prediction_matches_fpga_profile_exactly",
        "full_graph_fused_tir_delta": {},
    }
    analysis_path = tmp_path / "fused_tir_occurrence_delta.json"
    analysis_path.write_text(json.dumps(analysis), encoding="utf-8")
    (tmp_path / "artifact_hashes.json").write_text(
        json.dumps({"artifacts": {"fused_tir_occurrence_delta.json": "bad"}}),
        encoding="utf-8",
    )
    try:
        load_fused_contract(tmp_path)
    except RuntimeError as error:
        assert "hash mismatch" in str(error)
    else:
        raise AssertionError("tampered contract was accepted")


def test_load_fused_contract_accepts_prospective_non_three_occurrence_contract(tmp_path):
    expected = {
        "load_buffer_2d_bytes": -10,
        "load_buffer_2d_calls": -5,
        "load_buffer_2d_acc_bytes": 0,
        "load_buffer_2d_acc_calls": -1,
        "load_buffer_2d_inp_bytes": -10,
        "load_buffer_2d_inp_calls": -2,
        "load_buffer_2d_wgt_bytes": 0,
        "load_buffer_2d_wgt_calls": -2,
        "store_buffer_2d_bytes": 0,
        "store_buffer_2d_calls": 0,
    }
    analysis = {
        "status": "compiler_fused_tir_prediction_frozen_before_fpga",
        "full_graph_latency_observed": False,
        "partial_mask_latency_observed": False,
        "full_graph_fused_tir_delta": expected,
        "graph_occurrence_count": 5,
        "graph_occurrences": [{"graph_node": index} for index in range(5)],
    }
    payload = json.dumps(analysis)
    analysis_path = tmp_path / "fused_tir_occurrence_delta.json"
    analysis_path.write_text(payload, encoding="utf-8")
    digest = hashlib.sha256(analysis_path.read_bytes()).hexdigest()
    (tmp_path / "artifact_hashes.json").write_text(
        json.dumps({"artifacts": {"fused_tir_occurrence_delta.json": digest}}),
        encoding="utf-8",
    )
    loaded, loaded_expected, artifact_count = load_fused_contract(tmp_path)
    assert loaded["graph_occurrence_count"] == 5
    assert loaded_expected == expected
    assert artifact_count == 1
