#!/usr/bin/env python3
"""CPU admission diagnostics; waits are outside executed stage intervals."""
import math
import statistics


def overlap(intervals_a, intervals_b):
    a, b = sorted(intervals_a), sorted(intervals_b)
    i = j = 0
    total = 0.0
    while i < len(a) and j < len(b):
        total += max(0.0, min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return total


def analyze_gate_rows(rows, mode, discard=7):
    if mode not in ("off", "observe", "exclusive"):
        raise ValueError("Unknown CPU gate mode")
    if [r.get("frame_id") for r in rows] != list(range(len(rows))):
        raise ValueError("Missing or reordered frame")
    if len(rows) <= discard:
        raise ValueError("Insufficient rows")
    for row in rows:
        if row.get("cpu_stage_gate_mode", "off") != mode:
            raise ValueError("Gate identity drift")
        if row.get("mode") != "pipeline" or row.get("stage_count") != 3:
            raise ValueError("Expected three-stage ordinary pipeline")
        for i in range(3):
            if row.get(f"stage{i}_process_cpu_ms") is not None:
                raise ValueError("Concurrent process CPU attribution")
            start, end = row[f"stage{i}_start_ms"], row[f"stage{i}_end_ms"]
            if not all(math.isfinite(x) for x in (start, end)) or end < start:
                raise ValueError("Invalid stage interval")
            if mode != "off":
                request, acquire, release, wait = [row.get(f"stage{i}_cpu_gate_{f}_ms")
                                                 for f in ("request", "acquire", "release", "wait")]
                if i == 1:
                    if any(x is not None for x in (request, acquire, release, wait)):
                        raise ValueError("VTA unexpectedly gated")
                elif not all(isinstance(x, (int, float)) and math.isfinite(x)
                             for x in (request, acquire, release, wait)):
                    raise ValueError("Missing gate timing")
                elif not (request <= acquire <= start <= end <= release):
                    raise ValueError("Gate wait included in stage or invalid order")
                elif abs(acquire - request - wait) > 0.00001:
                    raise ValueError("Gate wait mismatch")
    # Verify exclusivity on all frames, including warmup and drain. Comparing
    # release/acquire additionally covers the sub-microsecond tracing margins.
    if mode == "exclusive":
        leases = sorted((r[f"stage{i}_cpu_gate_acquire_ms"], r[f"stage{i}_cpu_gate_release_ms"])
                        for r in rows for i in (0, 2))
        if any(b[0] < a[1] for a, b in zip(leases, leases[1:])):
            raise ValueError("Exclusive CPU leases overlap")
    window = rows[discard:]
    span = window[-1]["completion_ms"] - window[0]["completion_ms"]
    gaps = [b["completion_ms"] - a["completion_ms"] for a, b in zip(window, window[1:])]
    if span <= 0 or min(gaps) <= 0:
        raise ValueError("Invalid completion order")
    intervals = [[(r[f"stage{i}_start_ms"], r[f"stage{i}_end_ms"]) for r in window]
                 for i in (0, 2)]
    run_intervals = [[(r[f"stage{i}_run_start_ms"], r[f"stage{i}_run_end_ms"]) for r in window]
                     for i in (0, 2)]
    stage_overlap = overlap(*intervals)
    run_overlap = overlap(*run_intervals)
    return {"gate_mode": mode, "scored_rows": len(window), "completion_gaps": len(gaps),
            "ii_ms": statistics.mean(gaps), "cpu_stage_overlap_ms": stage_overlap,
            "cpu_run_overlap_ms": run_overlap,
            "cpu_stage_overlap_fraction_of_front_busy": stage_overlap / sum(b-a for a, b in intervals[0]),
            "stages": [{"index": i,
                        "run_mean_ms": statistics.mean(r[f"stage{i}_run_ms"] for r in window),
                        "executed_service_mean_ms": statistics.mean(r[f"stage{i}_ms"] for r in window),
                        "gate_wait_mean_ms": statistics.mean(r[f"stage{i}_cpu_gate_wait_ms"] for r in window)
                        if mode != "off" and i != 1 else None}
                       for i in range(3)],
            "interpretation": "Gate wait is excluded from both run and executed service; no additive II correction",
            "ddr_and_scheduler_split": "unknown"}


def pair_effect(a, b):
    for key in ("boot_id", "runner_sha256", "runtime_bindings", "program_identity", "numeric_input_sha256",
                "threads", "affinity", "spin_count", "queue_depth", "cpu_frequency_khz"):
        if a.get(key) is None or a.get(key) != b.get(key):
            raise ValueError("Pair identity mismatch: " + key)
    if a["gate_metrics"]["gate_mode"] not in ("off", "observe") or b["gate_metrics"]["gate_mode"] != "exclusive":
        raise ValueError("Wrong paired gate modes")
    am, bm = a["gate_metrics"], b["gate_metrics"]
    return {"boot_id": a["boot_id"], "block": a["block"],
            "front_run_reduction_fraction": 1 - bm["stages"][0]["run_mean_ms"] / am["stages"][0]["run_mean_ms"],
            "tail_run_reduction_fraction": 1 - bm["stages"][2]["run_mean_ms"] / am["stages"][2]["run_mean_ms"],
            "ii_reduction_fraction": 1 - bm["ii_ms"] / am["ii_ms"],
            "control": am, "exclusive": bm,
            "identity_verified": True}


def repeated_pair_inference(pairs):
    grouped = {}
    identity = None
    for pair in pairs:
        a, b = pair["control"], pair["exclusive"]
        pair_effect(a, b)
        current = {k: a[k] for k in ("runner_sha256", "runtime_bindings", "program_identity", "numeric_input_sha256",
                                    "threads", "affinity", "spin_count", "queue_depth", "cpu_frequency_khz")}
        if identity is None:
            identity = current
        elif current != identity:
            raise ValueError("Cross-boot intervention identity drift")
        boot = pair["boot_id"]
        if boot != a["boot_id"] or boot != b["boot_id"]:
            raise ValueError("Paired boot identity drift")
        support = grouped.setdefault(boot, {})
        if pair["block"] in support:
            raise ValueError("Duplicate boot/block pair")
        support[pair["block"]] = pair
    if len(grouped) != 3 or any(set(x) != {0, 1} for x in grouped.values()):
        raise ValueError("Exactly three boots and two balanced pairs per boot required")
    rows = []
    for boot, blocks in sorted(grouped.items()):
        am = [r["control"]["gate_metrics"] for r in blocks.values()]
        bm = [r["exclusive"]["gate_metrics"] for r in blocks.values()]
        a = statistics.mean(m["stages"][0]["run_mean_ms"] for m in am)
        b = statistics.mean(m["stages"][0]["run_mean_ms"] for m in bm)
        rows.append({"boot_id": boot, "control_front_run_ms": a, "exclusive_front_run_ms": b,
                     "front_delta_ms": a-b, "front_reduction_fraction": 1-b/a,
                     "ii_delta_ms": statistics.mean(m["ii_ms"] for m in am)-statistics.mean(m["ii_ms"] for m in bm),
                     "trace_verified": all(m["cpu_run_overlap_ms"] > 0 for m in am) and
                                       all(m["cpu_stage_overlap_ms"] == 0 for m in bm)})
    def interval(field):
        values = [r[field] for r in rows]
        mean = statistics.mean(values)
        half = 4.3026527299 * statistics.stdev(values) / math.sqrt(3)
        return {"mean_ms": mean, "lower_ms": mean-half, "upper_ms": mean+half,
                "degrees_of_freedom": 2, "statistical_unit": "three paired boot means"}
    ci = interval("front_delta_ms")
    direction = all(r["front_delta_ms"] > 0 for r in rows)
    magnitude = statistics.median(r["front_reduction_fraction"] for r in rows) >= 0.05
    return {"boot_rows": rows, "front_run_ci95": ci, "ii_ci95": interval("ii_delta_ms"),
            "primary_gate_passed": direction and magnitude and ci["lower_ms"] > 0 and all(r["trace_verified"] for r in rows),
            "ddr_and_scheduler_split": "unknown"}
