"""Read-only input, pairing, normalization and missing-data tests for Node A."""

import argparse
import json
from pathlib import Path

import pytest

import audit_vta_c3_reuse_evidence as audit


def profile(factor=1):
    result = {key: 0 for keys in audit.COUNTERS.values() for key in keys}
    result.update(load_buffer_2d_inp_bytes=40, load_buffer_2d_wgt_bytes=60,
                  store_buffer_2d_out_bytes=10, load_buffer_2d_bytes=100,
                  store_buffer_2d_bytes=10, load_buffer_2d_inp_calls=2,
                  load_buffer_2d_wgt_calls=3, load_buffer_2d_calls=5,
                  store_buffer_2d_out_calls=1, store_buffer_2d_calls=1,
                  synchronize_calls=1, driver_run_calls=1, driver_run_total_us=900)
    return {k: v * factor for k, v in result.items()}


def candidate(mode="original", key="o"):
    return {"candidate_id": key, "residence_mode": mode,
            "complete_config_entity": {"entity": [["tile_co", "sp", [-1, 1]]]},
            "hardware_fingerprint": {"wgt_buff_size": 262144},
            "identity": {"workload": ["conv", "test"]},
            "lowered_tir_sha256": None, "tir_sha256": "a" * 64}


def checked():
    return {"correct": True, "mismatch_count": 0,
            "actual_sha256": "a" * 64, "expected_sha256": "a" * 64}


def certificate():
    return {"status": "passed", "seeds": [
        dict(checked(), seed=seed, runtime_profile_complete=profile()) for seed in (0, 1, 2)]}


def timing():
    return [dict(checked(), round=r, latency_ms=2.0, runtime_profile_complete=profile(2),
                 time_evaluator_total_kernel_invocations=2) for r in (0, 1)]


def measured(mode="original", key="o", binary="b"):
    return audit.measured_candidate(candidate(mode, key), certificate(), timing(), 2, binary,
                                    {"instruction_peak_bytes": 120, "uop_peak_bytes": 16,
                                     "explicit_residency_drains": 0})


def frozen_directory(path, files):
    path.mkdir()
    for name, value in files.items():
        (path / name).write_text(json.dumps(value))
    manifest = {"artifacts": {name: audit.sha256(path / name) for name in files}}
    (path / "artifact_hashes.json").write_text(json.dumps(manifest))
    return path


def test_timing_profile_is_normalized_by_whole_function_not_submission():
    rows = timing()
    for row in rows:
        row["runtime_profile_complete"]["driver_run_calls"] = 34
    value = audit.measured_candidate(candidate(), certificate(), rows, 2)
    assert value["metrics"]["input_dma_bytes"] == 40
    assert value["metrics"]["submissions"] == 17
    assert value["metrics"]["driver_execution_ms"] == pytest.approx(0.9)
    assert value["lowered_tir_sha256"] == "a" * 64


def test_legacy_partial_direct_profile_can_certify_two_complete_calls():
    cert = certificate()
    fields = ("load_buffer_2d_inp_bytes", "load_buffer_2d_wgt_bytes", "load_buffer_2d_bytes",
              "store_buffer_2d_bytes", "load_buffer_2d_calls", "store_buffer_2d_calls")
    for seed in cert["seeds"]:
        seed["runtime_profile"] = {k: profile()[k] for k in fields}
        del seed["runtime_profile_complete"]
    rows = timing()
    for row in rows:
        del row["time_evaluator_total_kernel_invocations"]
    value = audit.measured_candidate(candidate(), cert, rows, 2)
    assert value["profile_divisors"] == [2, 2]
    assert value["metrics"]["total_dma_bytes"] == 110


def test_missing_profile_multiplicity_is_not_assumed():
    rows = timing()
    for row in rows:
        del row["time_evaluator_total_kernel_invocations"]
    cert = certificate()
    for seed in cert["seeds"]:
        del seed["runtime_profile_complete"]
    with pytest.raises(ValueError, match="uncertified"):
        audit.measured_candidate(candidate(), cert, rows, 2)


def test_profile_disagreement_is_not_hidden():
    rows = timing()
    rows[0]["runtime_profile_complete"]["load_buffer_2d_wgt_bytes"] += 1
    with pytest.raises(ValueError, match="profile mismatch"):
        audit.measured_candidate(candidate(), certificate(), rows, 2)


def test_wrong_or_unverified_outputs_cannot_enter_performance_pairs():
    cert = certificate()
    del cert["seeds"][0]["actual_sha256"]
    value = audit.measured_candidate(candidate(), cert, [], 2)
    assert value["status"] == "fpga_invalid"
    assert value["metrics"]["operator_latency_ms"] is None
    with pytest.raises(ValueError, match="invalid identity has timing"):
        audit.measured_candidate(candidate(), cert, timing(), 2)


def test_incomplete_or_wrong_timing_rejected():
    with pytest.raises(ValueError, match="timing rounds"):
        audit.measured_candidate(candidate(), certificate(), timing()[:1], 2)
    rows = timing()
    rows[1]["correct"] = False
    with pytest.raises(ValueError, match="wrong or invalid"):
        audit.measured_candidate(candidate(), certificate(), rows, 2)


def test_fallback_does_not_count_as_a_measured_mode_or_pair():
    coverage, pairs = audit.family_result("W", "F", [measured(), measured("input_stationary", "i")],
        {"weight_resident_barrier": "not_applicable_fallback_original",
         "input_weight_resident_barrier": "not_applicable_fallback_original"})
    assert coverage["four_resolved_with_fallback"]
    assert not coverage["four_measured"]
    assert len(pairs) == 1
    assert coverage["effective_dual_reuse"] is None


def test_failed_original_is_not_a_qualified_fallback():
    original = measured()
    original["status"] = "fpga_invalid"
    coverage, pairs = audit.family_result("W", "F", [original], {
        m: "not_applicable_fallback_original" for m in audit.MODES[1:]})
    assert not coverage["four_resolved_with_fallback"]
    assert coverage["weight_resident_barrier_status"].endswith("unqualified")
    assert not pairs


def test_four_flags_are_not_four_programs_or_effective_dual_reuse():
    coverage, _ = audit.family_result("W", "F", [measured(m, m) for m in audit.MODES])
    assert coverage["four_measured"]
    assert not coverage["four_distinct_programs"]
    assert not coverage["effective_dual_reuse"]


def test_both_operand_reductions_required_for_effective_combination():
    rows = [measured(m, m, m) for m in audit.MODES]
    rows[-1]["metrics"]["input_dma_bytes"] = 20
    coverage, _ = audit.family_result("W", "F", rows)
    assert not coverage["effective_dual_reuse"]
    rows[-1]["metrics"]["weight_dma_bytes"] = 30
    coverage, _ = audit.family_result("W", "F", rows)
    assert coverage["effective_dual_reuse"]
    assert coverage["four_distinct_programs"]


@pytest.mark.parametrize("changed", ["config", "context"])
def test_same_tile_requires_same_config_and_context(changed):
    rows = [measured(), measured("input_stationary", "i")]
    rows[1][changed] = {"different": True}
    with pytest.raises(ValueError, match="different ConfigEntity or execution context"):
        audit.family_result("W", "F", rows)


def test_zero_missing_and_nan_are_not_interchangeable():
    assert audit.number(0) == 0
    assert audit.number(None) is None
    assert audit.number(True) is None
    assert audit.number(float("nan")) is None
    assert audit.delta(0, 0) is None
    value = measured()
    assert value["metrics"]["acc_dma_bytes"] == 0
    assert value["sram_allocation_span_bytes"] is None


def test_empty_lowered_allocation_list_is_not_zero_sram_and_null_aliases_are_supported():
    c = candidate()
    c["static_features"] = {"instruction_peak_bytes": None}
    c["command_features"] = {"values": {"peaks": {"insn_bytes": 128, "uop_bytes": 16}}}
    features = audit.frozen_features(c)
    assert features["instruction_peak_bytes"] == 128
    value = audit.measured_candidate(c, certificate(), timing(), 2, features=features)
    assert value["sram_allocation_span_bytes"] is None


def test_input_hash_mismatch_and_mid_audit_change_fail_closed(tmp_path):
    directory = frozen_directory(tmp_path / "input", {"contract.json": {"value": 1}})
    reader = audit.FrozenReader(tmp_path)
    reader.read(directory, "contract.json")
    (directory / "contract.json").write_text("{}")
    with pytest.raises(ValueError, match="changed during audit"):
        reader.recheck()
    with pytest.raises(ValueError, match="hash mismatch"):
        audit.FrozenReader(tmp_path).directory(directory)


def test_manifest_link_and_undeclared_files_are_rejected(tmp_path):
    directory = frozen_directory(tmp_path / "input", {"contract.json": {}})
    (directory / "undeclared.json").write_text("{}")
    reader = audit.FrozenReader(tmp_path)
    with pytest.raises(ValueError, match="linked manifest"):
        reader.directory(directory, "a" * 64)
    with pytest.raises(ValueError, match="not covered"):
        reader.read(directory, "undeclared.json")


def test_nested_retained_binaries_verified_and_path_escape_rejected(tmp_path):
    directory = frozen_directory(tmp_path / "input", {"contract.json": {}})
    (directory / "binaries").mkdir()
    binary = directory / "binaries/o.so"
    binary.write_bytes(b"example")
    manifest = {"artifacts": {"binaries": {"o.so": audit.sha256(binary)}}}
    (directory / "artifact_hashes.json").write_text(json.dumps(manifest))
    audit.FrozenReader(tmp_path).directory(directory)
    manifest["artifacts"]["../other.json"] = "a" * 64
    (directory / "artifact_hashes.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="escapes"):
        audit.FrozenReader(tmp_path).directory(directory)


def test_output_is_immutable_even_before_inputs_are_loaded(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    sentinel = output / "user.json"
    sentinel.write_text("user evidence")
    with pytest.raises(FileExistsError):
        audit.run(argparse.Namespace(c3_root=tmp_path, output_dir=output))
    assert sentinel.read_text() == "user evidence"


def test_independent_input_region_check_handles_padding_and_partial_tile():
    assert audit.input_rectangle_check() == 11288
    assert audit.input_rectangle_check(5, 7, 3, 2, 3) > 35


def test_historical_snapshot_end_to_end(tmp_path):
    root = Path(audit.__file__).parent / "report_out/stage_tile_cotuning/c3_dma_residency_autotune"
    if not (root / "07_grouped_holdout" / audit.SOURCES["r18_analysis"]).is_dir():
        pytest.skip("historical evidence not installed")
    output = tmp_path / "audit"
    result = audit.run(argparse.Namespace(c3_root=root, output_dir=output))
    assert result["status"] == "NODE_A_COMPLETE_PAUSED"
    assert result["pair_session_count"] == 54
    assert result["new_fpga_dispatches"] == result["new_compiler_attempts"] == 0
    assert result["coverage"]["R18-H2"]["four_resolved_with_fallback"] == 18
    assert result["coverage"]["R18-H3"]["four_measured"] == 1
    assert result["coverage"]["R18-H3"]["four_distinct_programs"] == 0
    assert result["coverage"]["Y02"]["unique_tile_context_count"] == 1
    assert result["unknown_metrics"]["instruction_peak_bytes"] == 4
    manifest = json.loads((output / "artifact_hashes.json").read_text())
    for name, expected in manifest["artifacts"].items():
        assert audit.sha256(output / name) == expected
