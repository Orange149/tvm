"""Fixed-protocol analysis fixtures; no hardware or historical labels required."""

import copy
import math

import pytest

from c3_input_early_release_analysis import (
    DMA_RESERVATION_BYTES,
    EXPECTED_LEDGER,
    INPUT_BYTES,
    ORDERS,
    analyze_blocks,
    block_metrics,
    percentile,
    render_report,
)


def blocks_fixture(fps=None, latency_ms=451.75, evidence=False):
    fps = fps or {"L1": 9.0, "L2": 10.0, "C1": 10.0, "E1": 10.0}
    blocks = []
    for round_id, order in enumerate(ORDERS, 1):
        for variant in order:
            rate = fps[variant]
            if isinstance(rate, list):
                rate = rate[round_id - 1]
            interval = 1000.0 / rate
            frames = []
            for position in range(105):
                # All five warmups drain before starting the formal sequence.
                enqueue = position * interval if position < 5 else 2000 + (position - 5) * interval
                frames.append({
                    "frame_id": position,
                    "input_index": position if position < 5 else position - 5,
                    "warmup": position < 5,
                    "enqueue_ms": enqueue,
                    "completion_ms": enqueue + latency_ms,
                })
            block = {"round": round_id, "variant": variant, "status": "passed", "frames": frames}
            if evidence:
                external = INPUT_BYTES if variant in ("L2", "C1") else 0
                high_water = 10000000 + external
                block.update({
                    "boot_id": "one-boot",
                    "outputs_verified": True,
                    "call_ledger_verified": True,
                    "lease_trace_verified": True,
                    "environment_verified": True,
                    "early_release_overlap_verified": variant == "E1",
                    "formal_copy_count": 100 if variant == "C1" else 0,
                    "formal_copy_bytes": 100 * INPUT_BYTES if variant == "C1" else 0,
                    "allocation_snapshots": [{
                        "phase": phase,
                        "initialized": True,
                        "capacity": DMA_RESERVATION_BYTES,
                        "high_water_bytes": high_water,
                        "target_external_input_bytes": external,
                    } for phase in ("post_warmup", "post_measure")],
                })
            blocks.append(block)
    return blocks


def find(blocks, variant, round_id=1):
    return next(block for block in blocks if block["variant"] == variant and block["round"] == round_id)


def test_complete_protocol_and_unknown_evidence_do_not_fake_success():
    blocks = blocks_fixture()
    original = copy.deepcopy(blocks)
    result = analyze_blocks(blocks)
    assert blocks == original
    assert result["observed_formal_frames"] == 1600
    assert result["observed_formal_warmups"] == 80
    assert result["performance_gates"] == {"E1_versus_L1": True, "E1_versus_L2": True, "E1_versus_C1": True}
    assert result["evidence_gates"]["outputs"] is None
    assert result["evidence_gates"]["E1_saves_one_actual_external_slot"] is None
    assert result["single_boot_continue_gate"] is None


def test_fps_11_414_can_coexist_with_latency_451_75_ms():
    fps = {variant: 11.414 for variant in ("L1", "L2", "C1", "E1")}
    result = analyze_blocks(blocks_fixture(fps=fps))
    row = result["variants"]["E1"]
    assert row["median_round_fps"] == pytest.approx(11.414)
    assert row["median_round_latency_median_ms"] == pytest.approx(451.75)
    assert row["median_round_initiation_interval_ms"] == pytest.approx(1000 / 11.414)
    assert row["median_round_fps"] != pytest.approx(1000 / 451.75)


def test_warmups_do_not_enter_fps_or_latency_statistics():
    blocks = blocks_fixture()
    for block in blocks:
        for position, frame in enumerate(block["frames"][:5]):
            frame["enqueue_ms"] = position
            frame["completion_ms"] = position + 1500
    result = analyze_blocks(blocks)
    assert result["variants"]["E1"]["median_round_fps"] == pytest.approx(10)
    assert result["variants"]["E1"]["median_round_latency_median_ms"] == pytest.approx(451.75)


def test_proxy_calibration_block_uses_16_frames_and_15_intervals():
    block = blocks_fixture()[0]
    block.pop("round")
    block["frames"] = block["frames"][:21]
    metrics = block_metrics(block, expected_frames=16)
    assert metrics["fps"] == pytest.approx(9)
    assert metrics["formal_frames"] == 16
    assert metrics["completion_window_ms"] == pytest.approx(15 * 1000 / 9)
    block["status"] = "failed"
    with pytest.raises(ValueError):
        block_metrics(block, expected_frames=16)


@pytest.mark.parametrize("mutate", [
    lambda rows: rows.pop(),
    lambda rows: rows.append(copy.deepcopy(rows[0])),
    lambda rows: rows.__setitem__(1, copy.deepcopy(rows[0])),
    lambda rows: rows.reverse(),
    lambda rows: rows[0].__setitem__("round", True),
    lambda rows: rows[0].__setitem__("status", "failed"),
    lambda rows: rows[0].__setitem__("global_stop", True),
])
def test_partial_duplicate_failed_or_reordered_attempt_has_no_result(mutate):
    blocks = blocks_fixture()
    mutate(blocks)
    with pytest.raises(ValueError):
        analyze_blocks(blocks)


@pytest.mark.parametrize("mutate", [
    lambda frames: frames.pop(),
    lambda frames: frames[7].__setitem__("frame_id", frames[6]["frame_id"]),
    lambda frames: frames[7].__setitem__("input_index", 0),
    lambda frames: frames[5].__setitem__("input_index", False),
    lambda frames: frames[0].__setitem__("warmup", 1),
    lambda frames: frames[0].__setitem__("warmup", False),
    lambda frames: frames[8].__setitem__("completion_ms", float("nan")),
    lambda frames: frames[8].__setitem__("enqueue_ms", "2000"),
    lambda frames: frames[8].__setitem__("completion_ms", frames[7]["completion_ms"]),
    lambda frames: frames[8].__setitem__("enqueue_ms", frames[8]["completion_ms"] + 1),
    lambda frames: frames[8].__setitem__("output_match", False),
    lambda frames: frames[8].__setitem__("complete", False),
    lambda frames: frames[8].__setitem__("status", "failed"),
    lambda frames: frames[5].__setitem__("enqueue_ms", 700),
])
def test_missing_duplicate_wrong_unit_failed_or_nonfifo_frame_rejected(mutate):
    blocks = blocks_fixture()
    mutate(blocks[0]["frames"])
    with pytest.raises(ValueError):
        analyze_blocks(blocks)


def test_percentile_is_linear_interpolation():
    assert percentile(list(range(100)), 0.95) == pytest.approx(94.05)
    assert percentile([3], 0.95) == 3
    with pytest.raises(ValueError):
        percentile([], 0.95)


def test_pair_threshold_uses_fps_ratio_not_inverse_interval_improvement():
    # 5.1% FPS gain is only 4.85% II reduction; frozen criterion is the former.
    result = analyze_blocks(blocks_fixture(fps={"L1": 10, "L2": 10.51, "C1": 10.51, "E1": 10.51}))
    assert result["comparisons"]["L1"]["median_paired_fps_gain_pct"] == pytest.approx(5.1)
    assert result["performance_gates"]["E1_versus_L1"] is True


def test_near_dual_slot_threshold_uses_exact_paired_98percent_ratio():
    result = analyze_blocks(blocks_fixture(fps={"L1": 9, "L2": 10, "C1": 10, "E1": 9.81}))
    assert result["performance_gates"]["E1_versus_L2"] is True
    assert result["comparisons"]["L2"]["strict_wins"] == 0
    result = analyze_blocks(blocks_fixture(fps={"L1": 9, "L2": 10, "C1": 10, "E1": 9.79}))
    assert result["performance_gates"]["E1_versus_L2"] is False


def test_four_round_count_required_even_with_favorable_median():
    result = analyze_blocks(blocks_fixture(fps={"L1": [9, 9, 11, 11], "L2": 10, "C1": 10, "E1": 10}))
    assert result["comparisons"]["L1"]["strict_wins"] == 2
    assert result["performance_gates"]["E1_versus_L1"] is False


def test_perfect_timing_with_real_evidence_passes_only_single_boot_gate():
    result = analyze_blocks(blocks_fixture(evidence=True))
    assert result["single_boot_continue_gate"] is True
    assert result["variants"]["E1"]["round_actual_dma_peak_bytes"] == [10000000] * 4
    assert result["variants"]["L2"]["round_actual_dma_peak_bytes"] == [10000000 + INPUT_BYTES] * 4
    assert result["independent_boot_replication_claimed"] is False
    assert result["statistical_significance_claimed"] is False


def test_reservation_does_not_substitute_for_allocator_highwater():
    blocks = blocks_fixture(evidence=True)
    for block in blocks:
        for row in block["allocation_snapshots"]:
            row["high_water_bytes"] = DMA_RESERVATION_BYTES
    result = analyze_blocks(blocks)
    assert result["evidence_gates"]["E1_saves_one_actual_external_slot"] is False
    assert result["single_boot_continue_gate"] is False


def test_incomplete_snapshot_evidence_unknown_and_highwater_growth_fails_gate():
    blocks = blocks_fixture(evidence=True)
    find(blocks, "E1")["allocation_snapshots"].pop()
    result = analyze_blocks(blocks)
    assert result["evidence_gates"]["E1_saves_one_actual_external_slot"] is None
    blocks = blocks_fixture(evidence=True)
    find(blocks, "E1")["allocation_snapshots"][1]["high_water_bytes"] += 64
    result = analyze_blocks(blocks)
    assert result["evidence_gates"]["E1_saves_one_actual_external_slot"] is False


def test_raw_native_snapshot_phases_integer_initialized_and_capacity_bytes():
    blocks = blocks_fixture(evidence=True)
    for block in blocks:
        block["target_external_input_bytes"] = block["allocation_snapshots"][0]["target_external_input_bytes"]
        for row in block["allocation_snapshots"]:
            row["capacity_bytes"] = row.pop("capacity")
            row["initialized"] = 1
            row.pop("target_external_input_bytes")
        block["allocation_snapshots"].insert(0, {"phase": "before_graph_load", "initialized": 0})
    assert analyze_blocks(blocks)["evidence_gates"]["E1_saves_one_actual_external_slot"] is True
    for block in blocks:
        block.pop("target_external_input_bytes")
    assert analyze_blocks(blocks)["evidence_gates"]["E1_saves_one_actual_external_slot"] is None


@pytest.mark.parametrize("field", ["outputs_verified", "call_ledger_verified", "lease_trace_verified", "environment_verified"])
def test_failure_certificate_cannot_be_hidden_by_status_passed(field):
    blocks = blocks_fixture(evidence=True)
    blocks[0][field] = False
    with pytest.raises(ValueError):
        analyze_blocks(blocks)


def test_runtime_identity_boot_changes_reject_and_no_boot_remains_unknown():
    blocks = blocks_fixture(evidence=True)
    blocks[1]["boot_id"] = "another-boot"
    with pytest.raises(ValueError):
        analyze_blocks(blocks)
    blocks[1].pop("boot_id")
    with pytest.raises(ValueError):
        analyze_blocks(blocks)
    assert analyze_blocks(blocks_fixture())["boot_id"] is None


def test_copy_ledgers_are_not_inferred_from_variant_name():
    blocks = blocks_fixture(evidence=True)
    find(blocks, "E1")["formal_copy_bytes"] = 1
    result = analyze_blocks(blocks)
    assert result["evidence_gates"]["E1_no_added_target_copy"] is False
    blocks = blocks_fixture(evidence=True)
    find(blocks, "C1")["formal_copy_count"] = 99
    with pytest.raises(ValueError):
        analyze_blocks(blocks)


def test_total_trial_ledger_includes_proxy_calibration_and_three_stage_calls():
    counts = EXPECTED_LEDGER
    assert sum(counts[name] for name in ("serial_reference", "qualification", "event_tests",
               "proxy_calibration_including_warmups", "formal_warmups", "formal_scored")) == 1956
    assert counts["total_complete_inferences"] * 3 == counts["total_stage_calls"] == 5868
    assert counts["C1_formal_copy_bytes"] == 321126400


def test_report_keeps_fps_latency_and_memory_claims_distinct():
    report = render_report(analyze_blocks(blocks_fixture()))
    assert "FPS" in report and "时延" in report
    assert "内存减半" in report and "不是其倒数" in report


def test_nonfinite_percentile_and_impossible_memory_are_rejected():
    with pytest.raises(ValueError):
        percentile([math.inf], 0.95)
    blocks = blocks_fixture(evidence=True)
    blocks[0]["allocation_snapshots"][0]["high_water_bytes"] = DMA_RESERVATION_BYTES + 1
    with pytest.raises(ValueError):
        analyze_blocks(blocks)
