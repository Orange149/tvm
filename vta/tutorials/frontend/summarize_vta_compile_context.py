#!/usr/bin/env python3
"""Summarize completed E1 level 1 and freeze a proposed level-2 coverage set."""
import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    summary = read(args.scan / "summary.json")
    assert summary["complete"] and summary["successful_segments"] == 87
    mapping = read(args.scan / "occurrence_contexts.json")
    assert not mapping["unresolved"]
    rows = summary["rows"]
    contexts = {r["segment_id"]: {p["signature"] for p in
                read(args.scan/r["segment_id"].replace(":", "_")/"primitives.json")} for r in rows}
    all_contexts = set().union(*contexts.values())
    changed = [o for o in mapping["occurrences"] if o["unique_primitive_contexts"] > 1]
    # Conservative: all placements of varying occurrences, not just a chosen baseline.
    difference_segments = {p["segment"] for o in changed for p in o["placements"]}
    endpoints = {f"vta:{start:02d}:{end:02d}" for start in (5, 10, 15)
                 for end in range(start, start+3)}
    negative_controls = {"vta:01:02", "vta:03:04", "vta:18:19"}
    selected = difference_segments | endpoints | negative_controls
    assert selected <= contexts.keys()
    covered = set().union(*(contexts[s] for s in selected))
    greedy = []
    while covered != all_contexts:
        candidate = min((s for s in contexts if s not in selected),
                        key=lambda s: (-len(contexts[s]-covered), s))
        assert contexts[candidate]-covered
        selected.add(candidate)
        greedy.append(candidate)
        covered |= contexts[candidate]
    occurrences = [{k:v for k,v in o.items() if k != "placements"} |
                   {"placement_count":len(o["placements"])} for o in mapping["occurrences"]]
    result = {"source_summary_sha256": hashlib.sha256((args.scan/"summary.json").read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "segments": len(rows), "convolution_placements": sum(len(r["convolutions"]) for r in rows),
              "primitive_invocations_precodegen": sum(r["primitive_count"] for r in rows),
              "semantic_occurrences": len(occurrences), "unresolved": len(mapping["unresolved"]),
              "unique_workloads": len({json.dumps(c["workload"],sort_keys=True) for r in rows for c in r["convolutions"]}),
              "unique_primitive_contexts_all_ops": len(all_contexts),
              "varying_convolution_occurrences": [o["occurrence"] for o in changed],
              "occurrences": occurrences,
              "level2_proposal": {"status":"selection only; builds not executed by this tool",
                  "coverage_definition":"all unique typed primitive DAGs including non-convolution adapters",
                  "all_difference_segments":sorted(difference_segments),
                  "layered_endpoints":sorted(endpoints), "negative_controls":sorted(negative_controls),
                  "greedy_additions":greedy, "selected_segments":sorted(selected),
                  "covered_contexts":len(covered), "total_contexts":len(all_contexts)},
              "limitations":["Pre-codegen signatures are not full lowered TIR equality",
                             "Incumbent reuse is not proof of optimal tile invariance",
                             "Complete-stage adapter and invocation changes remain relevant"]}
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k != "occurrences"},indent=2))


if __name__ == "__main__":
    main()
