#!/usr/bin/env python3
"""Revalidate all saved Q1 cases without rerunning or replacing raw measurements.

The initial collector incorrectly required global submission order to be identical
across D's two islands. Require multiplicities/traffic and per-frame outputs instead.
This does not prove byte-identical instruction streams or per-stage ordering.
"""
from collections import Counter
import argparse
import json
from pathlib import Path

from audit_shared_buffer_storage import sha
from analyze_queue_capacity import profile_counts
from run_queue_capacity import queue_records
from run_shared_buffer_qualification import archive_outputs


def reconcile(directory):
    data = json.loads((directory / "preregistered.json").read_text())
    reference_dir = directory.parents[1] / "c3s_buffer_reuse/s1_board_run1"
    assert sha(reference_dir / "summary.json") == data["reference_sha256"]
    data["reconciliation"] = {"script_sha256": sha(__file__),
        "reason": "D permits different global island interleavings; use multiplicities not global order",
        "raw_measurements_unchanged": True, "original_partial_summary_sha256": sha(directory / "summary.json")}
    data["results"] = []
    saved = (1 << 26) - sum(data["capacities"].values())
    for name in "ABCD":
        references = archive_outputs((reference_dir / f"{name}_external_outputs.tar").read_bytes())
        modes = {}
        profile = profile_counts(directory / f"{name}_default_diag_profile.tar")
        for mode in ("default_off", "default_diag", "small_diag"):
            stem = name + "_" + mode
            assert archive_outputs((directory / f"{stem}_outputs.tar").read_bytes()) == references
            rows = [json.loads(r) for r in (directory / f"{stem}.jsonl").read_text().splitlines()]
            assert len(rows) == 32 and [r["frame_id"] for r in rows] == list(range(32))
            assert all(r["input_index"] == i % 8 for i,r in enumerate(rows))
            assert profile_counts(directory / f"{stem}_profile.tar") == profile
            phases = {r["phase"]: r for r in map(json.loads, (directory / f"{stem}_allocation.jsonl").read_text().splitlines())}
            high = phases["after_steady_state"]["high_water_bytes"]
            assert high == phases["after_warmup"]["high_water_bytes"]
            records = queue_records((directory / f"{stem}.stderr").read_bytes())
            assert bool(records) == (mode != "default_off")
            assert all(r["timeout"] == 0 and r["reason"] == "explicit_sync" for r in records)
            if records:
                assert len({r["queue_id"] for r in records}) == 1
                assert [r["submit"] for r in records] == list(range(1, len(records)+1))
                for r in records:
                    for kind in ("insn", "uop"):
                        assert r[kind+"_capacity"] == (data["capacities"][kind] if mode == "small_diag" else (1 << 25))
            assert b"source=pool-anchor" in (directory / f"{stem}.stdout").read_bytes()
            modes[mode] = {"byte_exact": True, "frames": 32, "high_water": high,
                "submits_including_warmup": len(records), "queue_records": records}
        signature = lambda rs: Counter((r["insn_bytes"], r["uop_bytes"], r["load_bytes"], r["store_bytes"], r["reason"]) for r in rs)
        assert signature(modes["default_diag"]["queue_records"]) == signature(modes["small_diag"]["queue_records"])
        assert modes["default_off"]["high_water"] == modes["default_diag"]["high_water"]
        assert modes["default_diag"]["high_water"] - modes["small_diag"]["high_water"] == saved
        data["results"].append({"topology": name, "modes": modes})
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = reconcile(args.directory)
    with (args.directory / "reconciled_summary.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print("PASS: 12 saved cases; byte-exact, traffic and batch multiplicities equal; allocation savings exact")
