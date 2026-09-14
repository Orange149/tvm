#!/usr/bin/env python3
"""Freeze the parent-aware RPC restart amendment after two fail-closed runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from analyze_vta_fused_program_service_proxy_norefit import verify_artifacts_compatible
from run_vta_c3_resnet18_from_scratch_ours_fullgraph import (
    ParentAwareCleanStartBoard,
)
from run_vta_resnet50_relay_residency_dispatch_pair_board import sha256, write_json


EXPECTED_FIRST_FAILURE = "fresh default RPC failed to start"
EXPECTED_SECOND_FAILURE_PREFIX = "RPC did not become an idle single-parent server:"
REMOTE_LOGS = (
    "r18_h1_ours_candidate_001.log",
    "r18_h1_ours_candidate_002.log",
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def finish(output):
    write_json(
        output / "artifact_hashes.json",
        {
            "artifacts": {
                path.name: sha256(path)
                for path in sorted(output.iterdir())
                if path.is_file() and path.name != "artifact_hashes.json"
            },
            "source_sha256": sha256(Path(__file__).resolve()),
        },
    )


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite {}".format(output))
    failures = []
    for value in args.failed_session_dir:
        directory = Path(value).resolve()
        verify_artifacts_compatible(directory)
        summary = read_json(directory / "summary.json")
        invalid = read_json(directory / "invalid_session.json")
        if summary.get("status") != "invalid_entire_session_fail_closed":
            raise RuntimeError("expected fail-closed session")
        message = invalid.get("message", "")
        if not (
            message == EXPECTED_FIRST_FAILURE
            or message.startswith(EXPECTED_SECOND_FAILURE_PREFIX)
        ):
            raise RuntimeError("failure does not match the repeated RPC restart fault")
        if invalid.get("may_splice_with_other_session") is not False:
            raise RuntimeError("failed session was not sealed against splicing")
        failures.append(
            {
                "path": str(directory),
                "artifact_ledger_sha256": sha256(directory / "artifact_hashes.json"),
                "W0_to_failure_seconds": summary["W0_to_failure_seconds"],
                "T0_to_failure_seconds": summary["T0_to_failure_seconds"],
                "failure": summary["failure"],
            }
        )
    if len(failures) < 2:
        raise RuntimeError("at least two repeated fail-closed sessions are required")

    protocol_dir = Path(args.protocol_dir).resolve()
    verify_artifacts_compatible(protocol_dir)
    prior_amendment_sha256 = None
    if args.prior_amendment_dir:
        prior_amendment_dir = Path(args.prior_amendment_dir).resolve()
        verify_artifacts_compatible(prior_amendment_dir)
        prior_amendment_sha256 = sha256(prior_amendment_dir / "artifact_hashes.json")
    board = ParentAwareCleanStartBoard(args.host, args.ssh_known_hosts)
    processes = board.rpc_processes()
    parent = board.rpc_state()
    runtime = args.default_runtime.rstrip("/")
    logs = {}
    for name in REMOTE_LOGS:
        path = runtime + "/" + name
        content = board.run("test -f {0}; sha256sum {0}; cat {0}".format(path)).stdout
        first, body = content.split("\n", 1)
        logs[name] = {
            "remote_path": path,
            "sha256": first.split()[0],
            "content": body,
        }
    if "Bind failed to 0.0.0.0:9090" not in logs[REMOTE_LOGS[1]]["content"]:
        raise RuntimeError("second-candidate log does not prove the bind conflict")
    if not any(row["PPID"] == "1" and row["PID"] == parent["PID"] for row in processes):
        raise RuntimeError("persistent RPC parent not identified")

    here = Path(__file__).resolve().parent
    source_hashes = {
        name: sha256(here / name)
        for name in (
            "run_vta_c3_resnet18_from_scratch_ours_fullgraph.py",
            "run_vta_p7r132_y00_search_confirmation.py",
            "test_vta_c3_resnet18_fullgraph_protocol.py",
        )
    }
    source_hashes["apps/vta_rpc/start_axu5evb_cpp_rpc.sh"] = sha256(
        here.parents[2] / "apps/vta_rpc/start_axu5evb_cpp_rpc.sh"
    )
    contract = {
        "schema": "c3_resnet18_rpc_parent_identity_amendment_v1",
        "status": "frozen_after_rpc_parent_and_retained_child_failures_before_retry",
        "prior_protocol_manifest_sha256": sha256(protocol_dir / "artifact_hashes.json"),
        "prior_rpc_amendment_manifest_sha256": prior_amendment_sha256,
        "failed_sessions": failures,
        "board_rpc_processes": processes,
        "persistent_parent": parent,
        "remote_logs": logs,
        "root_cause_first_order": (
            "The C++ RPC parent forks a per-connection child. pidof ordering exposed the child "
            "first, so the old control path killed that child and left the parent listening on "
            "9090; the replacement server then failed to bind."
        ),
        "root_cause_second_order": (
            "After parent selection was corrected, device, module, packed-function and NDArray "
            "locals still retained two RPC connection children after the remote handle alone "
            "was deleted. The idle-parent gate rejected the restart without killing the parent."
        ),
        "amendment": (
            "Select the unique tvm_rpc process with PPid=1 as the persistent server and wait "
            "until every per-connection child exits before stopping the parent. Explicitly "
            "release buffers, module, packed functions, device and remote before garbage collection."
        ),
        "unchanged": [
            "H1 workload and complete ConfigSpace",
            "deterministic max-min batches and local threshold",
            "DMA ordering and candidate budget 13",
            "correctness seeds, timing rounds and full-graph gate",
            "P7R517--P7R522 serialized method order",
        ],
        "target_candidate_latency_read_for_amendment": False,
        "fullgraph_performance_labels_read_for_amendment": False,
        "board_contact_scope": "RPC process table and server logs only; no VTA candidate execution",
        "source_hashes": source_hashes,
    }
    output.mkdir(parents=True)
    write_json(output / "contract.json", contract)
    write_json(
        output / "summary.json",
        {
            "status": contract["status"],
            "repeated_failure_count": len(failures),
            "persistent_parent": parent,
            "bind_conflict_confirmed": True,
            "algorithm_or_candidate_order_changed": False,
            "rpc_derived_objects_explicitly_released": True,
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
    parser.add_argument("--protocol-dir", required=True)
    parser.add_argument("--prior-amendment-dir")
    parser.add_argument("--failed-session-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument("--ssh-known-hosts", required=True)
    parser.add_argument("--default-runtime", default="/var/volatile/vta_c3_ram/runtime")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
