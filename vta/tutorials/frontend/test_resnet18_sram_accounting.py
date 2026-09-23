"""Regression tests for byte capacities and evidence-qualified SRAM reporting."""
import json
from types import SimpleNamespace

import pytest
import tvm

import audit_resnet18_sram_accounting as audit
import calibrate_resource_cost_model as calibrate
import search_resnet18_stage_splits as search
import split_resnet18_stages as split


CAPACITIES = dict(inp_buff_size=32768, wgt_buff_size=262144, acc_buff_size=131072)


def test_capacity_sizes_are_already_bytes(monkeypatch):
    env = SimpleNamespace(INP_BUFF_SIZE=32768, WGT_BUFF_SIZE=262144,
                          ACC_BUFF_SIZE=131072, OUT_BUFF_SIZE=32768,
                          INP_ELEM_BYTES=16, WGT_ELEM_BYTES=256,
                          ACC_ELEM_BYTES=64, OUT_ELEM_BYTES=16)
    monkeypatch.setattr(split.vta, "get_env", lambda: env)
    assert split.build_vta_capacity_spec() == dict(
        inp_cap_bytes=32768, wgt_cap_bytes=262144,
        acc_cap_bytes=131072, out_cap_bytes=32768)


def test_calibration_out_capacity_uses_output_width(tmp_path, monkeypatch):
    (tmp_path / "config").mkdir()
    (tmp_path / "config/vta_config.json").write_text(json.dumps(dict(
        LOG_INP_BUFF_SIZE=15, LOG_WGT_BUFF_SIZE=18, LOG_ACC_BUFF_SIZE=17,
        LOG_INP_WIDTH=3, LOG_ACC_WIDTH=5)))
    monkeypatch.setenv("VTA_HW_PATH", str(tmp_path))
    result = calibrate.vta_sram_capacities()
    assert result["acc_bytes"] == 131072
    assert result["out_bytes"] == 32768


def test_legacy_ratio_is_reconstructed_not_relabelled_as_utilization():
    best = dict(onchip_inp_util_pct=38.28125, onchip_wgt_util_pct=7.037353515625,
                onchip_acc_util_pct=9.5703125, onchip_out_util_pct=38.28125)
    rows = {row["domain"]: row for row in audit.reconstruct_legacy(best)}
    assert rows["inp"]["inferred_whole_unit_bytes"] == 200704
    assert rows["inp"]["legacy_capacity_bytes"] == 524288
    assert rows["inp"]["whole_unit_capacity_ratio_pct"] == 612.5
    assert rows["wgt"]["inferred_whole_unit_bytes"] == 4722688
    assert all(row["compiled_sram_utilization_pct"] is None for row in rows.values())


def test_allocation_counts_dtype_bytes_not_loop_iterations():
    pointer = tvm.ir.PointerType(tvm.ir.PrimType("int32"), "local.acc_buffer")
    buffer = tvm.tir.Var("acc", pointer)
    allocate = tvm.tir.Allocate(buffer, "int32", [128], tvm.tir.const(True, "bool"),
                                tvm.tir.Evaluate(0))
    loop = tvm.tir.For(tvm.tir.Var("i", "int32"), 0, 1000,
                       tvm.tir.ForKind.SERIAL, allocate)
    module = tvm.IRModule({"main": tvm.tir.PrimFunc([], loop)})
    rows = audit.storage_allocations(module)
    assert rows == [dict(buffer="acc", scope="local.acc_buffer", bytes=512, extents=[128])]
    assert audit.domain_metrics(rows, CAPACITIES)["acc_arena_capacity_pct"] == 0.390625


def test_missing_and_multiple_arenas_are_not_claimed_as_exact_occupancy():
    assert audit.domain_metrics([], CAPACITIES)["inp_arena_capacity_pct"] is None
    allocations = [dict(scope="local.inp_buffer", bytes=128)] * 2
    metrics = audit.domain_metrics(allocations, CAPACITIES)
    assert metrics["inp_arena_span_bytes"] == 256
    assert metrics["inp_arena_capacity_pct"] is None
    assert metrics["inp_status"] == "span_sum_upper_bound"
    with pytest.raises(ValueError, match="capacity"):
        audit.domain_metrics([], {**CAPACITIES, "inp_buff_size": 0})


@pytest.mark.parametrize("status", [None, "unknown_not_lowered", "unknown_legacy_proxy"])
def test_report_does_not_republish_legacy_occupancy(status):
    row = dict(onchip_peak_util_pct=38.28125, sram_utilization_status=status)
    assert search.reported_sram_utilization(row, "onchip_peak_util_pct") is None
    assert search.theory_candidate_row(row)["onchip_peak_util_pct"] is None


def test_whole_tensor_exceeding_capacity_does_not_prove_illegal_schedule():
    name = "layer1_block1_main_preadd"
    scheme = [dict(name="vta_test", device="vta", kind="units", unit_names=[name])]
    meta = dict(compute_ops_est=1000000, vta_inp_bytes_est=200704,
                vta_wgt_bytes_est=4722688, vta_acc_bytes_est=802816,
                vta_out_bytes_est=200704, conv_count=2, shape_change=False,
                residual_live_bytes_est=0, output_bytes=200704)
    capacity = dict(inp_cap_bytes=32768, wgt_cap_bytes=262144,
                    acc_cap_bytes=131072, out_cap_bytes=32768)
    args = SimpleNamespace(cpu_peak_gops=8, vta_peak_gops=16, ps_pl_bandwidth_gbps=12)
    result = search.score_scheme(scheme, {name: meta}, capacity, args,
                                 search.default_cost_model(args))
    assert result["unit_tensor_capacity_ratio_pct"]["inp"] == 612.5
    assert result["unit_tensor_capacity_ratio_pct"]["wgt"] > 100
    assert not result["hard_reject"]
    assert result["onchip_peak_util_pct"] is None
    assert result["stage_infos"][0]["onchip_inp_util_pct"] is None
    assert result["sram_risk_penalty_ms_est"] == 0
    result.update(scheme_cfg=scheme, vta_islands=[dict(start_idx=1, end_idx=4)],
                  onchip_peak_util_pct=1801.5625, static_tile_spill_penalty=999)
    theory = search.theory_base_metrics(result)
    assert not theory["theory_hard_reject"]
    assert theory["theory_sram_spill_penalty_ms"] == 0


def test_archived_tir_and_board_artifact_chain_are_consistent():
    result = audit.audit(audit.QUALIFICATION, audit.BOARD, audit.LEGACY)
    passed = [r for r in result["candidates"] if r["lower_status"] == "passed"]
    assert len(result["candidates"]) == 21
    assert len(passed) == 17
    baseline = next(r for r in passed if r["candidate_id"] == result["baseline_candidate_id"])
    assert [baseline[d + "_arena_span_bytes"] for d in audit.SCOPES] == [18560, 9216, 114688]
    assert [baseline[d + "_arena_capacity_pct"] for d in audit.SCOPES] == [56.640625, 3.515625, 87.5]
    assert result["input_files_unchanged"]
    assert len(result["input_sha256"]) == 57
    assert result["whole_graph_sram_peak_pct"] is None
    assert result["dynamic_sram_occupancy_pct"] is None
    assert result["board_contacted"] is False
