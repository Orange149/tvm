#!/usr/bin/env python3
"""Bind a selected route plan to exact graph artifacts and board evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resolve(path, workspace_root):
    path = Path(path)
    return path if path.is_absolute() else workspace_root / path


def candidate_from_source(route, workspace_root):
    source = resolve(route["candidate_source"], workspace_root)
    if sha256(source) != route["candidate_source_sha256"]:
        raise RuntimeError("candidate source hash mismatch for " + route["candidate_id"])
    document = read_json(source)
    matches = [
        row for row in document.get("candidates", [])
        if row.get("candidate_id") == route["candidate_id"]
    ]
    if len(matches) != 1:
        raise RuntimeError("candidate source did not resolve exactly one route")
    row = matches[0]
    for key in ("public_mode", "implementation_mode"):
        if row.get(key) != route.get(key):
            raise RuntimeError("candidate source {} mismatch".format(key))
    return {
        "candidate_id": route["candidate_id"],
        "workload_id": route.get("workload_id"),
        "public_mode": route["public_mode"],
        "implementation_mode": route["implementation_mode"],
        "identity": row["identity"],
        "source": {"path": str(source.resolve()), "sha256": sha256(source)},
    }


def run(args):
    workspace_root = Path(args.workspace_root).resolve()
    plan_path = Path(args.selected_route_plan).resolve()
    plan = read_json(plan_path)
    if plan.get("schema") != "c3_incumbent_protected_route_plan_v1":
        raise ValueError("unexpected route-plan schema")
    if plan.get("status") != "route_composition_complete":
        raise ValueError("route plan is incomplete")
    selected_ids = plan["selected_route_candidate_ids"]
    accepted = [row for row in plan["decisions"] if row["accepted"]]
    if [row["route"]["candidate_id"] for row in accepted] != selected_ids:
        raise RuntimeError("selected route list differs from accepted decisions")

    build_dir = Path(args.build_dir).resolve()
    build_path = build_dir / "summary.json"
    build = read_json(build_path)
    if build.get("status") != "yolov3_tiny_two_route_factorial_cross_build_verified":
        raise ValueError("unsupported graph build status")
    variants = [
        row for row in build["variants"] if row.get("candidate_ids") == selected_ids
    ]
    if len(variants) != 1:
        raise RuntimeError("selected route set did not resolve one graph variant")
    variant = variants[0]
    local = build_dir / variant["deployment_variant"]
    artifacts = {}
    for name, field in (
        ("graph.json", "graph_sha256"),
        ("params.bin", "params_sha256"),
        ("graphlib.so", "graphlib_sha256"),
    ):
        path = local / name
        actual = sha256(path)
        if actual != variant[field]:
            raise RuntimeError("selected graph artifact hash mismatch: " + name)
        artifacts[name] = {"path": str(path.resolve()), "sha256": actual}
    audit = variant.get("dispatch_audit")
    if not audit or not audit.get("all_routes_scheduled"):
        raise RuntimeError("selected graph lacks complete route scheduling audit")
    if {row["candidate_id"] for row in audit["schedule_hits"]} != set(selected_ids):
        raise RuntimeError("selected graph schedule hits differ from route plan")

    board_path = Path(args.board_summary).resolve()
    board = read_json(board_path)
    if board.get("status") != "yolov3_tiny_multi_residency_board_correctness_and_timing_complete":
        raise ValueError("unsupported board evidence status")
    if not board.get("all_paired_outputs_equal") or not board.get("all_correctness_outputs_nonzero"):
        raise RuntimeError("board evidence lacks nonzero exact output equivalence")
    board_variant = [
        row for row in board["variants"]
        if row.get("deployment_variant") == variant["deployment_variant"]
    ]
    if len(board_variant) != 1 or board_variant[0]["graphlib_sha256"] != artifacts["graphlib.so"]["sha256"]:
        raise RuntimeError("board evidence does not bind the selected graph binary")
    state = board["board_after"]
    if state.get("FPGA") != "operating" or state.get("UDMABUF") != "201326592":
        raise RuntimeError("board state does not match deployment hardware")
    if state.get("STORAGE_ERRORS"):
        raise RuntimeError("board evidence contains storage errors")

    clean_path = Path(args.clean_start).resolve()
    clean = read_json(clean_path)
    if clean.get("fresh_rpc", {}).get("CWD") != "/var/volatile/vta_c3_ram/runtime":
        raise RuntimeError("board evidence did not use the tmpfs RPC runtime")
    bitstream = clean.get("bitstream_reload", {})
    if bitstream.get("state") != "operating" or not bitstream.get("sha256"):
        raise RuntimeError("clean-start evidence lacks a frozen operating bitstream")

    routes = [candidate_from_source(row["route"], workspace_root) for row in accepted]
    rejected = [
        {
            "route": row["route"],
            "checks": row["checks"],
            "median_paired_delta_ms": row["evaluation"]["median_paired_delta_ms"],
        }
        for row in plan["decisions"] if not row["accepted"]
    ]
    payload = {
        "schema": "c3_vta_graph_route_deployment_manifest_v1",
        "status": "ready_for_exact_graph_deployment_on_bound_hardware",
        "model": build["model"],
        "selected_deployment_variant": variant["deployment_variant"],
        "selected_route_candidate_ids": selected_ids,
        "routes": routes,
        "rejected_routes": rejected,
        "graph_artifacts": artifacts,
        "hardware_contract": {
            "target": "axu5evb",
            "boot_id_at_qualification": board["boot_id"],
            "bitstream": bitstream,
            "udmabuf_bytes": int(state["UDMABUF"]),
            "rpc_runtime_directory": clean["fresh_rpc"]["CWD"],
            "on_mismatch": "reject before graph execution and rebuild/requalify",
        },
        "evidence": {
            "selected_route_plan": {"path": str(plan_path), "sha256": sha256(plan_path)},
            "graph_build_summary": {"path": str(build_path), "sha256": sha256(build_path)},
            "board_summary": {"path": str(board_path), "sha256": sha256(board_path)},
            "clean_start": {"path": str(clean_path), "sha256": sha256(clean_path)},
        },
        "command_capacity_status": (
            "not derived for this whole graph; the separate single-workload command certificate "
            "must not be reused as a whole-model bound"
        ),
        "claim_boundary": (
            "Exact current-hardware graph route manifest; workload-level routes, one selected "
            "graph variant, no dynamic shapes, cross-boot guarantee, physical AXI or mAP claim"
        ),
    }
    payload["manifest_id"] = canonical_hash(payload)
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite " + str(output))
    output.mkdir(parents=True)
    write_json(output / "graph_route_manifest.json", payload)
    (output / "STATUS.md").write_text(
        "# Exact graph route deployment manifest\n\n"
        "- Status: `{}`\n"
        "- Model: `{}`\n"
        "- Selected variant: `{}`\n"
        "- Accepted routes: {}\n"
        "- Rejected routes: {}\n"
        "- u-dma-buf: {} B\n"
        "- Whole-graph command-capacity claim: none\n".format(
            payload["status"], payload["model"], payload["selected_deployment_variant"],
            len(routes), len(rejected), payload["hardware_contract"]["udmabuf_bytes"],
        ),
        encoding="utf-8",
    )
    (output / "command.txt").write_text(" ".join(__import__("sys").argv) + "\n", encoding="utf-8")
    write_json(output / "artifact_hashes.json", {"artifacts": {
        path.name: sha256(path) for path in sorted(output.iterdir())
        if path.is_file() and path.name != "artifact_hashes.json"
    }})
    print(json.dumps({
        "status": payload["status"],
        "manifest_id": payload["manifest_id"],
        "selected_deployment_variant": payload["selected_deployment_variant"],
        "selected_route_candidate_ids": selected_ids,
        "rejected_route_candidate_ids": [row["route"]["candidate_id"] for row in rejected],
        "graphlib_sha256": artifacts["graphlib.so"]["sha256"],
    }, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-route-plan", required=True)
    parser.add_argument("--build-dir", required=True)
    parser.add_argument("--board-summary", required=True)
    parser.add_argument("--clean-start", required=True)
    parser.add_argument("--workspace-root", default=".")
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
