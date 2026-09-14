#!/usr/bin/env python3
"""Pre-drain screen and post-drain balance, not a submission safety proof or FPS."""
import argparse
from collections import defaultdict
import json
from pathlib import Path


def summarize(directory):
    manifest = json.loads((directory / "summary.json").read_text())
    result = {"boot": manifest["boot"], "scope": "pre-drain screen and post-drain balance only",
              "pre_drain_closure_is_not_necessary_for_normal_sync": True,
              "runtime_qualified": False, "topologies": []}
    for item in manifest["results"]:
        name = item["topology"]
        bins = defaultdict(lambda: defaultdict(int))
        for line in (directory / f"{name}_default_diag.stderr").read_text().splitlines():
            if not line.startswith("[VTA_BOUNDARY] "):
                continue
            row = json.loads(line.split(" ", 1)[1])
            b = bins[row["threshold"]]
            b["batches"] += 1
            for key in ("visited", "pending_pop", "open_tokens", "pending_uop", "closure_candidates", "drain_closed_candidates"):
                if key in row:
                    b[key] += row[key]
            if "full_batch_drain_closed" in row:
                b["full_batch_drain_closed_count"] += row["full_batch_drain_closed"]
            candidate = row.get("minimum_finalized_candidate_bytes", 0)
            if candidate:
                old = b.get("minimum_finalized_candidate_bytes", candidate)
                b["minimum_finalized_candidate_bytes"] = min(old, candidate)
                b["batches_with_smaller_closed_prefix"] += candidate < row["full_finalized_bytes"]
            b["max_raw_batch_bytes"] = max(b["max_raw_batch_bytes"], row["raw_batch_bytes"])
        assert set(bins) == {2656, 1328, 656, 320}
        result["topologies"].append({"topology": name, "thresholds": dict(bins)})
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = json.dumps(summarize(args.directory), indent=2) + "\n"
    if args.output:
        with args.output.open("x") as stream:
            stream.write(result)
    print(result, end="")
