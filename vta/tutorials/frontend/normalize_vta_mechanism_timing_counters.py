#!/usr/bin/env python3
"""Normalize already captured P6d counters for time_evaluator's extra invocation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = Path(args.input)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("refusing to overwrite normalized samples")
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    normalized = []
    for row in rows:
        number = int(row["timer_number"])
        invocations = 1 + number
        raw = row.pop("runtime_profile")
        row["runtime_profile_raw"] = raw
        row["profiler_invocations"] = invocations
        row["runtime_profile"] = {key: value / invocations for key, value in raw.items()}
        row["counter_normalization"] = (
            "TVM time_evaluator invokes the function 1 + number * repeat times; "
            "repeat=1 in P6d, so counters are divided by 1 + number"
        )
        normalized.append(row)
    with output.open("x", encoding="utf-8") as stream:
        for row in normalized:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    metadata = {
        "schema": "c3_p6d_counter_normalization_v1",
        "source": str(source),
        "source_sha256": sha256(source),
        "output": str(output),
        "output_sha256": sha256(output),
        "records": len(normalized),
        "latency_values_changed": False,
        "derivation": "runtime_profile / (1 + timer_number * repeat), with repeat=1",
        "local_evidence": {
            "python/tvm/runtime/module.py": "time_evaluator documentation states total invocations are 1 + number x repeat",
            "src/runtime/profiling.cc": "WrapTimeEvaluator performs one warmup call before its repeat loop"
        }
    }
    Path(str(output) + ".metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("normalized={} records={} latency_unchanged=true".format(output, len(normalized)))


if __name__ == "__main__":
    main()
