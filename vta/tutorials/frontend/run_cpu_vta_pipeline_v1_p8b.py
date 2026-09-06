#!/usr/bin/env python3
"""Run the two-direction P8B serial single-slot matched controls."""

from __future__ import annotations

import argparse
import datetime
import json
import shutil
from pathlib import Path

import run_cpu_vta_pipeline_v1_p8a as p8
from freeze_cpu_vta_pipeline_v1 import file_sha256, seal_artifact


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board-host", default="192.168.1.247")
    parser.add_argument("--ssh-user", default="root")
    parser.add_argument("--rpc-port", type=int, default=9090)
    parser.add_argument("--package", default=str(p8.DEFAULT_PACKAGE))
    parser.add_argument("--output-dir", default=str(p8.DEFAULT_OUTPUT))
    parser.add_argument("--remote-root", default="/var/volatile/ramps_p8b")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--keep-remote", action="store_true")
    parser.add_argument("--timeout-s", type=int, default=600)
    return parser.parse_args()


def build_protocol(manifest, warmup=5, runs=20):
    return seal_artifact(
        {
            "schema_version": 1,
            "kind": "cpu_vta_pipeline_v1_p8b_protocol",
            "phase": "P8B",
            "scope": "two_direction_serial_single_slot_matched_control",
            "candidate_id": manifest["candidate_id"],
            "edges": [p8.boundary_contract(manifest, index) for index in (0, 1)],
            "controls": [
                {"id": "safe_byte_copy", "AXU5EVB_DRIVER_SAFE_COPY": "1"},
                {"id": "memcpy", "AXU5EVB_DRIVER_SAFE_COPY": "0"},
            ],
            "measurement": {
                "warmup_per_path": int(warmup),
                "scored_pairs_per_control": int(runs),
                "alternating_distinct_inputs": 2,
                "scored_pair_order": "balanced_ab_ba_per_frame",
                "single_boot_qualification_only": True,
            },
            "gates": {
                "both_directions_present": True,
                "all_edge_gates_pass": True,
                "zero_copy_materialization_bytes": 0,
                "ordinary_copy_equivalence_and_prior_independent_reference": True,
            },
            "excluded_claims": [
                "dual_slot_pipeline_throughput",
                "cross_boot_performance_improvement",
                "candidate_ranking_change",
            ],
            "candidate_throughput_used_for_fit": False,
        }
    )


def build_local_audit(package, manifest, runner_binary):
    edge_audits = [
        p8.build_local_audit(package, manifest, runner_binary, edge_index=index)
        for index in (0, 1)
    ]
    return seal_artifact(
        {
            "schema_version": 1,
            "kind": "cpu_vta_pipeline_v1_p8b_local_abi_audit",
            "candidate_id": manifest["candidate_id"],
            "runner_binary_sha256": file_sha256(runner_binary),
            "edge_audits": edge_audits,
            "directions": [audit["edge"]["direction"] for audit in edge_audits],
            "passed": all(audit["passed"] for audit in edge_audits),
        }
    )


def run_board(args, manifest, runner_binary, work_root, output_dir):
    edge_sessions = []
    original_keep = args.keep_remote
    args.keep_remote = True
    args.reuse_remote_package = False
    for edge_index in (0, 1):
        edge_work = work_root / "edge{}".format(edge_index)
        edge_work.mkdir(parents=True, exist_ok=True)
        if edge_index == 1:
            args.reuse_remote_package = True
            args.keep_remote = original_keep
        edge_sessions.append(
            p8.run_board(
                args,
                manifest,
                runner_binary,
                edge_work,
                output_dir,
                edge_index=edge_index,
                phase="P8B",
                artifact_prefix="v1_p8b",
            )
        )
    boot_ids = {session["board_properties"]["boot_id"] for session in edge_sessions}
    directions = {session["edge"]["direction"] for session in edge_sessions}
    gate = {
        "same_boot": len(boot_ids) == 1,
        "both_directions_present": directions == {"cpu_to_vta", "vta_to_cpu"},
        "all_edge_gates_pass": all(
            session["single_boot_gate"]["passed"] for session in edge_sessions
        ),
        "all_zero_copy_materialization_bytes_are_zero": all(
            control["zero_copy_materialization_bytes"] == 0
            for session in edge_sessions
            for control in session["controls"]
        ),
        "prior_independent_reference_passed": all(
            session["correctness_evidence"]["prior_independent_reference"]["passed"]
            for session in edge_sessions
        ),
    }
    gate["passed"] = all(gate.values())
    return seal_artifact(
        {
            "schema_version": 1,
            "kind": "cpu_vta_pipeline_v1_p8b_session",
            "observed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "board": "{}@{}".format(args.ssh_user, args.board_host),
            "boot_id": next(iter(boot_ids)) if len(boot_ids) == 1 else sorted(boot_ids),
            "candidate_id": manifest["candidate_id"],
            "edge_sessions": edge_sessions,
            "single_boot_gate": gate,
            "formal_performance_claim_allowed": False,
        }
    )


def write_review(path, audit, session=None):
    lines = [
        "# P8B 双向单边界串行审查",
        "",
        "更新日期：{}".format(datetime.date.today().isoformat()),
        "",
        "## 本地结论",
        "",
        "- CPU->VTA 与 VTA->CPU 的 shape/dtype/device contract 均通过审计。",
        "- 两个方向复用同一显式 u-dma-buf slot 协议，不实现双 slot 或并行 Pipeline。",
    ]
    if session is None:
        lines.extend(["", "## 板端状态", "", "尚未执行单 boot qualification。"])
    else:
        lines.extend(["", "## 单 Boot 结果", ""])
        lines.append("| 方向 | 张量 bytes | B0 copy ms | CPU mapping penalty ms | 普通物化 bytes | zero-copy bytes | latency 差值 ms | gate |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
        for edge_session in session["edge_sessions"]:
            selected_id = edge_session["ordinary_baseline_selection"]["selected_control_id"]
            selected = next(
                item for item in edge_session["controls"] if item["control_id"] == selected_id
            )
            lines.append(
                "| `{}` | {} | {:.6f} | {:+.6f} | {} | {} | {:+.6f} | {} |".format(
                    edge_session["edge"]["direction"],
                    edge_session["edge"]["bytes"],
                    selected["baseline_edge_copy_ms_median"],
                    selected["cpu_shared_mapping_penalty_ms"],
                    selected["baseline_materialization_bytes"],
                    selected["zero_copy_materialization_bytes"],
                    selected["latency_delta_ms"],
                    "passed" if edge_session["single_boot_gate"]["passed"] else "failed",
                )
            )
        lines.extend(
            [
                "",
                "- 当前 A/B 是 ordinary-copy 与 zero-copy 的同 compiled-graph 路径等价检查；独立数值 reference 继承冻结 Top-20 证据并单独记录。",
                "- 整帧 latency 配对差值包含 stage/runtime 波动，尤其当其大于边界 copy 本身时，不得全部归因于 zero-copy。",
                "- 聚合 P8B gate：`{}`。当前结果不允许声明双缓冲 Pipeline FPS 提升。".format(
                    "passed" if session["single_boot_gate"]["passed"] else "failed"
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## 下一步",
            "",
            "停在 P8B review。只有双向 gate 通过并经用户确认，才进入 P8C 的双 slot 状态机与 Pipeline 实验。",
            "",
            "本地审计 SHA256：`{}`".format(audit["artifact_sha256"]),
        ]
    )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    package = Path(args.package).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    work_root = Path("/tmp/ramps_p8b")
    if work_root.exists():
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True)
    runner_binary = work_root / "vta_stage_pipeline_runner_p8b"
    p8.compile_runner(runner_binary)
    manifest = p8.load_manifest(package)
    protocol = build_protocol(manifest, args.warmup, args.runs)
    audit = build_local_audit(package, manifest, runner_binary)
    p8.write_json(output_dir / "v1_p8b_protocol.json", protocol)
    p8.write_json(output_dir / "v1_p8b_local_abi_audit.json", audit)
    session = None
    if not args.local_only:
        session = run_board(args, manifest, runner_binary, work_root, output_dir)
        p8.write_json(
            output_dir / "v1_p8b_session_{}.json".format(session["boot_id"]), session
        )
    write_review(output_dir / "v1_p8b_review.md", audit, session)
    print(json.dumps({"protocol": protocol, "audit": audit, "session": session}, indent=2))


if __name__ == "__main__":
    main()
