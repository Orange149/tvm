#!/usr/bin/env python3
"""Compare archived PrimFuncs structurally, retaining graph invocation multiplicity."""
import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path

import tvm
import vta  # Register VTA intrinsics before loading archived TIR.


def read(path):
    return json.loads(path.read_text())


def normalize(function):
    # No algebraic rewriting: only normalize the generated exported symbol.
    return function.with_attr("global_symbol", "normalized")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    summary = read(args.archive / "summary.json")
    assert summary["complete"]
    stages, functions = [], {}
    for row in summary["rows"]:
        sid = row["segment_id"]
        local = args.archive / sid.replace(":", "_")
        graph = read(local / "graph.json")
        calls = Counter(node["attrs"]["func_name"] for node in graph["nodes"]
                        if node["op"] == "tvm_op")
        funcs = {}
        for snapshot in row["snapshots"]:
            data = (local / snapshot["file"]).read_bytes()
            assert hashlib.sha256(data).hexdigest() == snapshot["sha256"]
            mod = tvm.ir.load_json(data.decode())
            for gv, func in mod.functions.items():
                assert gv.name_hint not in funcs, "Repeated snapshot requires explicit pass disambiguation"
                assert isinstance(func, tvm.tir.PrimFunc)
                funcs[gv.name_hint] = normalize(func)
        assert set(calls) == set(funcs), (sid, set(calls) ^ set(funcs))
        functions[sid] = funcs
        stages.append({"segment_id": sid, "graph_invocations": dict(calls),
                       "unique_lowered_functions": len(funcs), "coverage_complete": True})
    pairs = []
    for left, right in itertools.combinations(functions, 2):
        matches = [{"left_function": ln, "right_function": rn}
                   for ln, lf in functions[left].items()
                   for rn, rf in functions[right].items()
                   if tvm.ir.structural_equal(lf, rf, map_free_vars=True)]
        pairs.append({"left": left, "right": right, "structural_matches": matches})
    result = {"scope": "PrimFunc structural equality at CPUAccessRewrite; not board correctness, physical traffic, or all-domain TIR qualification",
              "normalization": "Normalize only global_symbol; TVM structural_equal(map_free_vars=True)",
              "archive_summary_sha256": hashlib.sha256((args.archive / "summary.json").read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "stages": stages, "pairs": pairs}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
