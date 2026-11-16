#!/usr/bin/env python3
"""Pure analysis for the fixed four-arm, four-round early-release trial.

All timestamps are milliseconds. Percentiles use deterministic linear interpolation
at (n-1)*p (the NumPy default convention). Missing safety/allocation evidence stays
unknown; a complete timing table is not itself a certificate of safe early release.
"""

import argparse
import json
import math
from statistics import median


VARIANTS = ("L1", "L2", "C1", "E1")
ORDERS = (
    ("L1", "L2", "E1", "C1"),
    ("L2", "C1", "L1", "E1"),
    ("C1", "E1", "L2", "L1"),
    ("E1", "L1", "C1", "L2"),
)
INPUT_BYTES = 802816
DMA_RESERVATION_BYTES = 201326592
EXPECTED_LEDGER = {
    "serial_reference": 100,
    "qualification": 128,
    "event_tests": 6,
    "proxy_calibration_including_warmups": 42,
    "formal_warmups": 80,
    "formal_scored": 1600,
    "total_complete_inferences": 1956,
    "total_stage_calls": 5868,
    "C1_formal_copy_count": 400,
    "C1_formal_copy_bytes": 400 * INPUT_BYTES,
}


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(label + " must be a finite number, in milliseconds")
    if not math.isfinite(value):
        raise ValueError(label + " must be finite")
    return value


def _bytes(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(label + " must be a nonnegative integer byte count")
    return value


def percentile(values, fraction):
    """Linearly interpolate sorted observations; do not treat them as replicates."""
    if not values or not 0 <= fraction <= 1:
        raise ValueError("percentile requires observations and fraction in [0, 1]")
    values = sorted(_number(value, "percentile observation") for value in values)
    position = (len(values) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    return values[lower] + (position - lower) * (values[upper] - values[lower])


def _all_verified(rows, name):
    """Three-valued conjunction: an absent certificate is not a successful gate."""
    values = [row.get(name) for row in rows]
    if any(value is False for value in values):
        return False
    if any(value is not True for value in values):
        return None
    return True


def _conjunction(values):
    if any(value is False for value in values):
        return False
    return None if any(value is not True for value in values) else True


def _frame_metrics(block, expected_frames=100):
    frames = block.get("frames")
    if type(expected_frames) is not int or expected_frames < 2:
        raise ValueError("expected_frames must be an integer of at least two")
    if not isinstance(frames, list) or len(frames) != expected_frames + 5:
        raise ValueError("each block requires exactly five warmup and {} scored frames".format(expected_frames))
    if any(type(frame.get("warmup")) is not bool for frame in frames) or [frame.get("warmup") for frame in frames] != [True] * 5 + [False] * expected_frames:
        raise ValueError("warmups must be five explicit frames before scored frames")
    identifiers = [frame.get("frame_id") for frame in frames]
    if any(isinstance(value, bool) or not isinstance(value, int) for value in identifiers):
        raise ValueError("frame_id must be an integer")
    if identifiers != sorted(set(identifiers)):
        raise ValueError("frame IDs must be unique and FIFO")
    scored = frames[5:]
    if any(type(frame.get("input_index")) is not int for frame in scored) or [frame.get("input_index") for frame in scored] != list(range(expected_frames)):
        raise ValueError("scored input indices must be exactly 0..{}, in FIFO order".format(expected_frames - 1))
    previous_enqueue = previous_completion = None
    latencies = []
    for frame in frames:
        if frame.get("status", "passed") != "passed" or frame.get("output_match") is False or frame.get("complete", True) is not True:
            raise ValueError("failed frame invalidates the complete performance attempt")
        enqueue = _number(frame.get("enqueue_ms"), "enqueue_ms")
        completion = _number(frame.get("completion_ms"), "completion_ms")
        if enqueue < 0 or completion < enqueue:
            raise ValueError("invalid frame timestamp interval")
        if previous_enqueue is not None and enqueue < previous_enqueue:
            raise ValueError("frame enqueue order is not FIFO")
        if previous_completion is not None and completion <= previous_completion:
            raise ValueError("frame completion order is not strictly FIFO")
        previous_enqueue, previous_completion = enqueue, completion
        if not frame["warmup"]:
            latencies.append(completion - enqueue)
    if scored[0]["enqueue_ms"] < frames[4]["completion_ms"]:
        raise ValueError("warmup pipeline must drain before scored enqueue")
    window_ms = scored[-1]["completion_ms"] - scored[0]["completion_ms"]
    return {
        "round": block.get("round"),
        "variant": block["variant"],
        "completion_window_ms": window_ms,
        "fps": 1000.0 * (expected_frames - 1) / window_ms,
        "initiation_interval_ms": window_ms / (expected_frames - 1),
        "latency_median_ms": median(latencies),
        "latency_p95_ms": percentile(latencies, 0.95),
        "formal_frames": expected_frames,
        "warmup_frames": 5,
    }


def block_metrics(block, expected_frames=100):
    """Metrics for one block, including the fixed 5+16 proxy calibration blocks.

    Unlike analyze_blocks this is not a complete-trial result and does not grant
    any safety gate. A failed block is rejected before computing its FPS.
    """
    if block.get("status") != "passed" or block.get("global_stop", False):
        raise ValueError("failed/global-stop block cannot produce valid metrics")
    return _frame_metrics(block, expected_frames=expected_frames)


def _memory(block):
    """Normalize actual allocator bump high-water, never call it live memory.

    Native allocator snapshots use high_water_bytes/initialized/capacity. The
    coordinator may normalize the high-water name to dma_peak_bytes. A supplied
    dma_current_bytes is accepted only as a separate, actually measured counter.
    """
    snapshots = block.get("allocation_snapshots")
    if snapshots is None:
        return None
    if not isinstance(snapshots, list) or not snapshots:
        raise ValueError("allocation_snapshots must be a nonempty list when supplied")
    normalized = {}
    for row in snapshots:
        phase = row.get("phase")
        if phase in ("before_graph_load", "graphs_loaded", "slots_allocated"):
            continue
        if phase not in ("post_warmup", "post_measure") or phase in normalized:
            raise ValueError("allocation phases must be unique post_warmup/post_measure")
        if "valid" in row:
            initialized = row["valid"] is True
        else:
            initialized = type(row.get("initialized")) in (int, float, bool) and row["initialized"] == 1
        if not initialized:
            raise ValueError("invalid/uninitialized allocation evidence")
        current = row.get("dma_current_bytes")
        if current is not None:
            current = _bytes(current, "dma_current_bytes")
        peak = _bytes(row.get("dma_peak_bytes", row.get("high_water_bytes")), "dma_peak_bytes")
        external = row.get("target_external_input_bytes", block.get("target_external_input_bytes"))
        if external is None:
            return None
        external = _bytes(external, "target_external_input_bytes")
        capacity = row.get("capacity", row.get("capacity_bytes"))
        if capacity != DMA_RESERVATION_BYTES or peak > capacity or external > peak:
            raise ValueError("allocator snapshot is inconsistent with its actual DMA counts")
        if current is not None and (current > peak or external > current):
            raise ValueError("actual current DMA counter is inconsistent with high-water")
        normalized[phase] = {
            "dma_current_bytes": current,
            "dma_peak_bytes": peak,
            "target_external_input_bytes": external,
        }
    if set(normalized) != {"post_warmup", "post_measure"}:
        return None
    if normalized["post_measure"]["dma_peak_bytes"] < normalized["post_warmup"]["dma_peak_bytes"]:
        raise ValueError("allocator high-water mark cannot decrease within a process")
    return normalized


def analyze_blocks(blocks):
    """Analyze complete blocks without mutating input or accepting partial rankings.

    Required: ordered 16 blocks, round/variant/status, and frames with integer
    frame_id, input_index, bool warmup, enqueue_ms/completion_ms. Frames contain
    five warmups followed by 100 scored inputs 0..99. Optional actual allocator
    snapshots use the schema in _memory; absent evidence produces unknown gates.

    Optional externally certified block booleans are outputs_verified,
    call_ledger_verified, lease_trace_verified, environment_verified, and
    early_release_overlap_verified (E1 only). E1 formal_copy_bytes must be zero
    to certify no additional copy. These must be backed by runner records, not
    inferred by setting status='passed'. This function does not certify machine
    code, thread affinity, references, or the proxy calibration itself.
    """
    if not isinstance(blocks, list) or len(blocks) != 16:
        raise ValueError("complete analysis requires exactly 4 rounds x 4 variants")
    expected = [(round_id, variant) for round_id, order in enumerate(ORDERS, 1) for variant in order]
    if any(type(block.get("round")) is not int for block in blocks):
        raise ValueError("round must be an integer in 1..4")
    actual = [(block.get("round"), block.get("variant")) for block in blocks]
    if actual != expected:
        raise ValueError("blocks must follow the frozen four-round acquisition order")
    if any(block.get("status") != "passed" or block.get("global_stop", False) for block in blocks):
        raise ValueError("failed/global-stop attempt has no valid formal ranking")
    boot_ids = [block.get("boot_id") for block in blocks]
    if any(value is not None for value in boot_ids) and (None in boot_ids or len(set(boot_ids)) != 1):
        raise ValueError("all formal blocks must belong to the same attested Linux boot")
    cells = [_frame_metrics(block) for block in blocks]
    by_key = {(row["round"], row["variant"]): row for row in cells}
    raw_by_key = {(row["round"], row["variant"]): row for row in blocks}
    memory = {(row["round"], row["variant"]): _memory(row) for row in blocks}
    variants = {}
    for variant in VARIANTS:
        rows = [by_key[(round_id, variant)] for round_id in range(1, 5)]
        actual_memory = [memory[(round_id, variant)] for round_id in range(1, 5)]
        variants[variant] = {
            "round_fps": [row["fps"] for row in rows],
            "median_round_fps": median(row["fps"] for row in rows),
            "median_round_initiation_interval_ms": median(row["initiation_interval_ms"] for row in rows),
            "median_round_latency_median_ms": median(row["latency_median_ms"] for row in rows),
            "median_round_latency_p95_ms": median(row["latency_p95_ms"] for row in rows),
            "round_actual_dma_peak_bytes": [None if row is None else row["post_measure"]["dma_peak_bytes"] for row in actual_memory],
            "round_target_external_input_bytes": [None if row is None else row["post_measure"]["target_external_input_bytes"] for row in actual_memory],
        }
    comparisons = {}
    for baseline in ("L1", "L2", "C1"):
        ratios = [by_key[(r, "E1")]["fps"] / by_key[(r, baseline)]["fps"] for r in range(1, 5)]
        comparisons[baseline] = {
            "round_fps_ratios": ratios,
            "median_fps_ratio": median(ratios),
            "median_paired_fps_gain_pct": median(100 * (ratio - 1) for ratio in ratios),
            "strict_wins": sum(ratio > 1 for ratio in ratios),
            "ties": sum(ratio == 1 for ratio in ratios),
            "rounds_at_least_98pct": sum(ratio >= 0.98 for ratio in ratios),
        }
    performance = {
        "E1_versus_L1": comparisons["L1"]["median_paired_fps_gain_pct"] >= 5 and comparisons["L1"]["strict_wins"] >= 3,
        "E1_versus_L2": comparisons["L2"]["median_fps_ratio"] >= 0.98 and comparisons["L2"]["rounds_at_least_98pct"] >= 3,
        "E1_versus_C1": comparisons["C1"]["median_fps_ratio"] >= 0.98 and comparisons["C1"]["rounds_at_least_98pct"] >= 3,
    }
    saving_rounds = []
    for round_id in range(1, 5):
        e1, l2 = memory[(round_id, "E1")], memory[(round_id, "L2")]
        if e1 is None or l2 is None:
            saving_rounds.append(None)
            continue
        phase_checks = []
        for phase in ("post_warmup", "post_measure"):
            e, l = e1[phase], l2[phase]
            saved = l["target_external_input_bytes"] - e["target_external_input_bytes"]
            phase_checks.append(e["target_external_input_bytes"] == 0 and saved >= INPUT_BYTES
                                and l["dma_peak_bytes"] - e["dma_peak_bytes"] == saved)
        # Allocator bump high-water must remain stable during the formal block;
        # this does not claim that all bytes below the watermark remain live.
        phase_checks.extend(value["post_warmup"]["dma_peak_bytes"] == value["post_measure"]["dma_peak_bytes"]
                            for value in (e1, l2))
        saving_rounds.append(all(phase_checks))
    e1_blocks = [raw_by_key[(r, "E1")] for r in range(1, 5)]
    copies = [row.get("formal_copy_bytes") for row in e1_blocks]
    if any(value is not None for value in copies):
        for value in copies:
            if value is not None:
                _bytes(value, "formal_copy_bytes")
    no_copy = None if any(value is None for value in copies) else all(value == 0 for value in copies)
    c1_blocks = [raw_by_key[(r, "C1")] for r in range(1, 5)]
    c1_copy_ledger = []
    for row in c1_blocks:
        count, byte_count = row.get("formal_copy_count"), row.get("formal_copy_bytes")
        if count is None or byte_count is None:
            c1_copy_ledger.append(None)
        else:
            _bytes(count, "formal_copy_count")
            _bytes(byte_count, "formal_copy_bytes")
            if count != 100 or byte_count != 100 * INPUT_BYTES:
                raise ValueError("C1 must copy the complete target input exactly once per scored frame")
            c1_copy_ledger.append(True)
    evidence = {
        "outputs": _all_verified(blocks, "outputs_verified"),
        "call_ledger": _all_verified(blocks, "call_ledger_verified"),
        "lease_trace": _all_verified(blocks, "lease_trace_verified"),
        "environment": _all_verified(blocks, "environment_verified"),
        "E1_no_added_target_copy": no_copy,
        "C1_exact_formal_copy_ledger": _conjunction(c1_copy_ledger),
        "E1_saves_one_actual_external_slot": _conjunction(saving_rounds),
        "E1_overlap_before_consumer_stage_end": _all_verified(e1_blocks, "early_release_overlap_verified"),
    }
    if any(block.get("outputs_verified") is False or block.get("call_ledger_verified") is False
           or block.get("lease_trace_verified") is False or block.get("environment_verified") is False
           for block in blocks):
        raise ValueError("failed correctness/call/lease/environment certificate invalidates formal analysis")
    return {
        "schema": "c3_input_early_release_analysis_v1_1",
        "status": "COMPLETE_SINGLE_BOOT_TIMING_TABLE",
        "boot_id": boot_ids[0],
        "time_unit": "ms",
        "dma_peak_definition": "allocator bump high-water; not dynamic live SRAM/DDR utilization",
        "percentile_method": "linear interpolation at (n-1)*p",
        "blocks": cells,
        "variants": variants,
        "comparisons": comparisons,
        "performance_gates": performance,
        "evidence_gates": evidence,
        "memory_saving_rounds": saving_rounds,
        "single_boot_continue_gate": _conjunction(list(performance.values()) + list(evidence.values())),
        "continue_gate_excludes_unread_external_prerequisites": True,
        "expected_full_trial_ledger": dict(EXPECTED_LEDGER),
        "observed_formal_frames": sum(row["formal_frames"] for row in cells),
        "observed_formal_warmups": sum(row["warmup_frames"] for row in cells),
        "statistical_significance_claimed": False,
        "independent_boot_replication_claimed": False,
        "pause_after_trial": True,
    }


def render_report(result):
    """A short ms/FPS-first report; no latency reciprocal is called throughput."""
    lines = [
        "# C3 共享输入提前归还：单启动结果", "",
        "FPS按首末完成帧间隔计算；每帧时延不是其倒数。", "",
        "| 版本 | 四轮FPS中位数 | 每帧时延中位数 ms | 每帧P95时延 ms |",
        "|---|---:|---:|---:|",
    ]
    for name in VARIANTS:
        row = result["variants"][name]
        lines.append("| {} | {:.3f} | {:.3f} | {:.3f} |".format(
            name, row["median_round_fps"], row["median_round_latency_median_ms"],
            row["median_round_latency_p95_ms"]))
    lines.extend(["", "表中时延为各块相应统计量的四轮中位数；图片不视作独立硬件实验。",
                  "", "单启动继续门槛：`{}`；缺失证据为 unknown，不按通过处理。".format(
                      result["single_boot_continue_gate"]),
                  "外部参考、资格、事件测试和代理校准仍须由主控单独验收。",
                  "不宣称统计非劣效、独立启动稳定性或整图内存减半。", ""])
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="JSON list of blocks or an object containing blocks")
    args = parser.parse_args()
    with open(args.input, encoding="utf-8") as stream:
        payload = json.load(stream)
    if isinstance(payload, dict) and payload.get("global_stop", False):
        raise ValueError("globally stopped trial cannot produce a formal result")
    print(json.dumps(analyze_blocks(payload["blocks"] if isinstance(payload, dict) else payload), indent=2, sort_keys=True))
