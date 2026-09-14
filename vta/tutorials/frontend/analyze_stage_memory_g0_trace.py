#!/usr/bin/env python3
"""Audit passive managed-slot traces. Encounters are not contention/benefit labels."""

import argparse
import hashlib
import json
from pathlib import Path

from run_stage_memory_g0_discovery import percentile, save, union_ms


def intersect(left, right):
    return [(max(s, x), min(e, y)) for s, e in left for x, y in right
            if max(s, x) < min(e, y)]


def audit(rows, manifest, discard=2):
    stages = manifest["stages"]
    first, last = rows[discard]["completion_ms"], rows[-1]["completion_ms"]
    wall = last - first
    intervals, clipped, holds = {}, {}, []
    tolerance = 0.00001  # Six-decimal ms output plus subtraction rounding.
    previous_slots, ranges = {}, {}
    for row in rows:
        assert row["p8_framework_materialization_bytes"] == 0
        assert row["p8_managed_slot_count"] == 2
        for stage in stages:
            i = stage["index"]
            names = ["queue_pop", "slots_acquired", "start", "binding_done", "run_start",
                     "run_end", "end"]
            values = [row[f"stage{i}_{name}_ms"] for name in names]
            assert all(v is not None for v in values), (i, "missing timestamp")
            assert all(a <= b + tolerance for a, b in zip(values, values[1:])), (i, values)
            assert abs(row[f"stage{i}_run_ms"] - (values[-2] - values[-3])) <= tolerance
            if stage["device"] == "vta":
                holds.append((row[f"stage{i}_vta_mutex_acquire_ms"],
                              row[f"stage{i}_vta_mutex_release_ms"], i, row["frame_id"]))
        assert len(row["p8_boundaries"]) == len(stages) - 1
        for boundary in row["p8_boundaries"]:
            edge = boundary["edge_index"]
            key = (edge, boundary["slot_id"])
            assert 0 <= key[1] < 2
            assert boundary["tensor_count"] == len(boundary["physical_addresses"])
            assert boundary["tensor_count"] == len(boundary["tensor_bytes"])
            assert sum(boundary["tensor_bytes"]) == boundary["boundary_bytes"]
            shape = tuple((int(addr, 16), size) for addr, size in
                          zip(boundary["physical_addresses"], boundary["tensor_bytes"]))
            if key in previous_slots:
                generation, consumer_run_end = previous_slots[key]
                assert boundary["generation"] == generation + 1, (key, "generation")
                assert row[f"stage{edge}_slots_acquired_ms"] + tolerance >= consumer_run_end
                assert ranges[key] == shape, (key, "physical range changed")
            else:
                assert boundary["generation"] == 1
                ranges[key] = shape
            previous_slots[key] = (boundary["generation"], row[f"stage{edge+1}_run_end_ms"])
    physical = sorted((addr, addr + size) for group in ranges.values() for addr, size in group)
    assert all(a[1] <= b[0] for a, b in zip(physical, physical[1:])), "slot range alias"
    ordered_holds = sorted(holds)
    assert all(a[1] <= b[0] + tolerance for a, b in zip(ordered_holds, ordered_holds[1:])), \
        "single VTA mutex ownership overlap"
    for stage in stages:
        i = stage["index"]
        intervals[i] = [(r[f"stage{i}_run_start_ms"], r[f"stage{i}_run_end_ms"])
                        for r in rows]
        clipped[i] = [(max(first, s), min(last, e)) for s, e in intervals[i]
                      if max(first, s) < min(last, e)]
    overlaps, encounters, all_cross = [], [], []
    for a in stages:
        for b in stages:
            i, j = a["index"], b["index"]
            if i >= j:
                continue
            spans = intersect(clipped[i], clipped[j])
            overlap = union_ms(spans)
            item = {"stages": [i, j], "devices": [a["device"], b["device"]],
                    "overlap_union_ms": overlap, "fraction_wall": overlap / wall}
            if a["device"] != b["device"]:
                all_cross.extend(spans)
                vta = i if a["device"] == "vta" else j
                item["fraction_vta_graph_run"] = overlap / union_ms(clipped[vta])
            overlaps.append(item)
    for active in stages:
        for pending in stages:
            if active["device"] == pending["device"]:
                continue
            i, j = active["index"], pending["index"]
            ready_rows = [r for r in rows[discard:]
                          if first <= r[f"stage{j}_binding_done_ms"] < last]
            samples = []
            for row in ready_rows:
                ready = row[f"stage{j}_binding_done_ms"]
                for (start, end), active_row in zip(intervals[i], rows):
                    if start <= ready < end:
                        owners = [{"stage": k, "frame": f} for s, e, k, f in holds
                                  if s <= ready < e]
                        samples.append({"active_frame": active_row["frame_id"],
                                        "ready_frame": row["frame_id"],
                                        "ready_ms": ready, "active_age_ms": ready - start,
                                        "active_remaining_ms": end - ready,
                                        "vta_owners_at_ready": owners,
                                        "not_blocked_by_vta_owner":
                                            pending["device"] != "vta" or not owners})
            encounters.append({"active_stage": i, "ready_stage": j,
                               "eligible_ready_events_in_window": len(ready_rows),
                               "encounters": len(samples),
                               "encounter_fraction": len(samples) / max(1, len(ready_rows)),
                               "not_blocked_by_vta_owner":
                                   sum(s["not_blocked_by_vta_owner"] for s in samples),
                               "samples": samples})
    service = []
    for stage in stages:
        i = stage["index"]
        steady = rows[discard:]
        waits = [r[f"stage{i}_vta_mutex_wait_ms"] for r in steady]
        service.append({"stage": i, "device": stage["device"],
                        "run_median_ms": percentile([r[f"stage{i}_run_ms"] for r in steady], .5),
                        "run_p95_ms": percentile([r[f"stage{i}_run_ms"] for r in steady], .95),
                        "mutex_wait_p95_ms": percentile(waits, .95) if waits[0] is not None else None})
    return {"frames_audited": len(rows), "steady_completion_window_ms": wall,
            "timestamp_and_slot_audit": "pass",
            "audit_limit": "Repeated frozen input; not unique-frame corruption or abort stress coverage.",
            "all_cpu_vta_overlap_union_ms": union_ms(all_cross),
            "all_cpu_vta_overlap_wall_fraction": union_ms(all_cross) / wall,
            "overlap_pairs": overlaps, "directed_encounters": encounters, "stage_service": service,
            "slot_wait_p95_ms": percentile([r["p8_total_slot_wait_ms"] for r in rows[discard:]], .95),
            "confirmed_protect_cells": None, "confirmed_allow_cells": None,
            "opportunity_ceiling": None,
            "interpretation": "Graph-run host intervals, not hardware compute or DDR stall; "
                              "readiness establishes scheduling opportunity only. No action labels "
                              "or avoidable penalty have been measured."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_dir", type=Path)
    args = parser.parse_args()
    summary_path = args.trace_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    assert summary["completed"]
    results = []
    for result in summary["results"]:
        if result["mode"] != "shared":
            continue
        rank = {"A": 1, "B": 5, "C": 7, "D": 13}[result["topology"]]
        local = args.trace_dir / f"rank{rank:02d}"
        raw = (local / "shared.jsonl").read_bytes()
        assert hashlib.sha256(raw).hexdigest() == result["result_sha256"]
        rows = [json.loads(line) for line in raw.splitlines()]
        entry = audit(rows, json.loads((local / "manifest.json").read_text()))
        entry.update(topology=result["topology"], result_sha256=result["result_sha256"])
        results.append(entry)
        print(result["topology"], "FPS", round(result["fps"], 3), "overlap fraction",
              round(entry["all_cpu_vta_overlap_wall_fraction"], 3), "directions",
              [(e["active_stage"], e["ready_stage"], e["encounters"],
                e["not_blocked_by_vta_owner"]) for e in entry["directed_encounters"]])
    save(args.trace_dir / "readiness_audit.json", {
        "summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        "analyzer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "results": results})


if __name__ == "__main__":
    main()
