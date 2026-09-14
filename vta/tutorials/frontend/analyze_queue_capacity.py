#!/usr/bin/env python3
"""Summarize queue diagnostics without treating profiled timing as pipeline FPS."""
import argparse
from collections import Counter
import json
from pathlib import Path
import tarfile


def profile_counts(path):
    with tarfile.open(path) as archive:
        profile = json.load(archive.extractfile("./benchmark_totals_status.json"))
    # Timings/signature presentation are not invariants; aggregate traffic/call counts are.
    return {k: v for k, v in profile.items() if k.endswith(("_calls", "_bytes")) or k == "synchronize_insns"}


def summarize(directory):
    summary = directory / "reconciled_summary.json"
    if not summary.exists():
        summary = directory / "summary.json"
    data = json.loads(summary.read_text())
    out = {"boot": data["boot"], "scope": data["scope"], "results": []}
    for result in data["results"]:
        topology = result["topology"]
        modes = result["modes"]
        reference = profile_counts(directory / f"{topology}_default_diag_profile.tar")
        for mode in modes:
            assert profile_counts(directory / f"{topology}_{mode}_profile.tar") == reference
        item = {"topology": topology, "traffic_counts_equal": True, "modes": {}}
        for mode, values in modes.items():
            rows = values["queue_records"]
            record = {"high_water_bytes": values["high_water"], "byte_exact": values["byte_exact"]}
            if rows:
                record.update(submits_including_warmup=len(rows),
                    submits_per_frame=len(rows) / (data["runs"] + data["warmups"]),
                    insn_peak=max(r["insn_bytes"] for r in rows),
                    uop_peak=max(r["uop_bytes"] for r in rows),
                    reasons=dict(Counter(r["reason"] for r in rows)),
                    mean_us_per_frame_including_warmup={key: sum(r[key] for r in rows) /
                        (data["runs"] + data["warmups"]) for key in rows[0] if key.endswith("_us")})
            item["modes"][mode] = record
        if "small_diag" in modes:
            item["saved_bytes"] = modes["default_diag"]["high_water"] - modes["small_diag"]["high_water"]
            item["high_water_reduction_pct"] = item["saved_bytes"] / modes["default_diag"]["high_water"] * 100
        out["results"].append(item)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.directory), indent=2))
