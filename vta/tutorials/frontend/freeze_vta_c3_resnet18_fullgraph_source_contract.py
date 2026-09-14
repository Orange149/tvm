#!/usr/bin/env python3
"""Bind the C3 operator workloads to exact Relay Testing ResNet18 call sites."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import tvm
from tvm import autotvm, relay
import vta

from analyze_vta_fused_program_service_proxy_norefit import verify_artifacts_compatible
from build_vta_resnet50_relay_residency_dispatch_pair import (
    make_relay_testing_resnet_program,
    sha256,
    write_json,
)
from tune_resnet18_vta import register_vta_conv2d_template


WORKLOAD_IDS = ("R18-H1", "R18-H2", "R18-H3")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def normalized(value):
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if hasattr(value, "items"):
        return {str(key): normalized(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)) or hasattr(value, "__iter__"):
        return [normalized(item) for item in value]
    if hasattr(value, "value"):
        return normalized(value.value)
    raise TypeError("unsupported workload value: {}".format(type(value).__name__))


def key(value):
    return json.dumps(normalized(value), sort_keys=True, separators=(",", ":"))


def finish(output):
    write_json(
        output / "artifact_hashes.json",
        {
            "artifacts": {
                str(path.relative_to(output)): sha256(path)
                for path in sorted(output.rglob("*"))
                if path.is_file() and path.name != "artifact_hashes.json"
            },
            "source_sha256": sha256(Path(__file__).resolve()),
        },
    )


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite {}".format(output))
    pool_dir = Path(args.board_pool_dir).resolve()
    verified_count = len(verify_artifacts_compatible(pool_dir))
    candidates = read_jsonl(pool_dir / "candidates.jsonl")
    representative = {}
    for workload_id in WORKLOAD_IDS:
        rows = [row for row in candidates if row["workload_id"] == workload_id]
        if not rows:
            raise ValueError("merged pool lacks {}".format(workload_id))
        workload_keys = {key(row["identity"]["workload"]) for row in rows}
        if len(workload_keys) != 1:
            raise ValueError("{} candidates disagree on workload".format(workload_id))
        representative[workload_id] = normalized(rows[0]["identity"]["workload"])

    register_vta_conv2d_template()
    env = vta.get_env()
    # Task extraction needs an unannotated graph and a single VTA target.  The
    # deployed source uses the same quantized graph with device annotations.
    relay_program, params = make_relay_testing_resnet_program(
        env, 18, device_annot=False
    )
    tasks = autotvm.task.extract_from_program(
        tvm.IRModule.from_expr(relay_program),
        params=params,
        ops=(relay.op.get("nn.conv2d"),),
        target=env.target,
        target_host=env.target_host,
    )
    extracted = Counter(key(task.workload) for task in tasks)
    exact = {}
    for workload_id, workload in representative.items():
        exact[workload_id] = {
            "workload": workload,
            "exact_extracted_task_matches": extracted[key(workload)],
            "merged_pool_candidate_count": sum(
                row["workload_id"] == workload_id for row in candidates
            ),
        }
        if exact[workload_id]["exact_extracted_task_matches"] != 1:
            raise RuntimeError(
                "{} is not an exact unique extracted ResNet18 task".format(workload_id)
            )

    repo = Path(__file__).resolve().parents[3]
    source_path = repo / "python/tvm/relay/testing/resnet.py"
    original_counts = {"R18-H1": 2304, "R18-H2": 1600, "R18-H3": 480}
    h1 = exact["R18-H1"]
    contract = {
        "schema": "c3_resnet18_fullgraph_source_contract_v1",
        "status": "frozen_before_resnet18_fullgraph_build_fpga_or_latency",
        "workload_id": "R18-H1",
        "workload": h1["workload"],
        "complete_original_domain_count": original_counts["R18-H1"],
        "source_model": "relay_testing_resnet18",
        "source_num_layers": 18,
        "source_model_implementation": {
            "path": "python/tvm/relay/testing/resnet.py",
            "sha256": sha256(source_path),
            "num_layers": 18,
        },
        "input_spec": {
            "shape": [env.BATCH, 3, 224, 224],
            "dtype": "float32",
            "deterministic_seeds": [0, 20250901, 20260910],
            "all_outputs": True,
        },
        "verified_target_workloads": exact,
        "extracted_conv_task_count": len(tasks),
        "extracted_unique_workload_count": len(extracted),
        "board_pool_manifest_sha256": sha256(pool_dir / "artifact_hashes.json"),
        "verified_board_pool_artifact_count": verified_count,
        "model_hash_scope": (
            "exact Relay source file, quantize configuration, graph_pack boundary, "
            "num_layers and input shape; no MXNet model is substituted"
        ),
        "board_contacted": False,
        "performance_labels_read": False,
        "claim_boundary": (
            "Compiler task-extraction identity only. One extracted task can occur at "
            "multiple graph call sites; graph occurrence counts require the frozen full-graph build."
        ),
    }
    output.mkdir(parents=True)
    write_json(output / "contract.json", contract)
    write_json(
        output / "summary.json",
        {
            "status": "three_operator_workloads_exactly_bound_to_relay_testing_resnet18",
            "target_workloads": exact,
            "extracted_conv_task_count": len(tasks),
            "extracted_unique_workload_count": len(extracted),
            "board_contacted": False,
            "performance_labels_read": False,
        },
    )
    (output / "timeline.jsonl").write_text("", encoding="utf-8")
    (output / "results.jsonl").write_text("", encoding="utf-8")
    (output / "command.txt").write_text(
        " ".join([sys.executable, str(Path(__file__).resolve())] + sys.argv[1:]) + "\n",
        encoding="utf-8",
    )
    finish(output)
    print(json.dumps(read_json(output / "summary.json"), indent=2, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board-pool-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
