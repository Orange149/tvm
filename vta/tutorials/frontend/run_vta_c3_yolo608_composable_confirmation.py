#!/usr/bin/env python3
"""One-boot formal confirmation for the composable YOLO-608 candidate set.

This runner deliberately measures only the frozen production incumbent and the
already graph-qualified scoped-completion deployment.  One invocation produces
one boot result.  Cross-boot acceptance is performed separately by selector v3.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import statistics
import tempfile
import time
import traceback

from tvm import rpc

import run_vta_c3_reuse_board as reuse_board
from c3_thread_sync_board import ExperimentBoard
from run_vta_c3_holdout_board_boot_v2 import sha256_file, verify_flat
from run_vta_c3_yolo608_graph_boot import (
    SEEDS,
    invoke_once,
    load_executor,
    mismatch_counts,
    random_input,
    read,
    seal,
    write,
)
from run_vta_p7r120_yolo_rpc_board import runtime_inventory


HERE = Path(__file__).resolve().parent
GROUPED = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout"
DEFAULT_REFERENCE = GROUPED / "20260924_c3_yolo608_graph_deployments_build_run02"
DEFAULT_SCOPED = GROUPED / "20260924_c3_yolo608_composable_build_run01"
DEFAULT_RUNTIME = GROUPED / "20260922_c3_thread_sync_freeze_run04/contract.json"
DEFAULT_CERTIFICATES = GROUPED / "20260924_c3_composability_certificates_run01"
VARIANTS = ("stock", "scoped_composable")
ROUND_ORDERS = (
    ("stock", "scoped_composable"),
    ("scoped_composable", "stock"),
    ("stock", "scoped_composable"),
    ("scoped_composable", "stock"),
    ("stock", "scoped_composable"),
    ("scoped_composable", "stock"),
)


def append(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")
        stream.flush()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def connect(host, port):
    for _ in range(30):
        try:
            return rpc.connect(host, port, session_timeout=1200)
        except Exception:
            time.sleep(0.25)
    raise RuntimeError("RPC unavailable on port {}".format(port))


def summarize_rounds(rows):
    """Reduce three seed calls to one paired observation per frozen round."""

    by_key = {}
    for row in rows:
        by_key.setdefault((int(row["round"]), str(row["variant"])), []).append(
            float(row["latency_ms"])
        )
    round_rows = []
    for round_index in range(len(ROUND_ORDERS)):
        baseline = by_key.get((round_index, "stock"), [])
        candidate = by_key.get((round_index, "scoped_composable"), [])
        if len(baseline) != len(SEEDS) or len(candidate) != len(SEEDS):
            raise ValueError("incomplete seed coverage in round {}".format(round_index))
        baseline_median = float(statistics.median(baseline))
        candidate_median = float(statistics.median(candidate))
        gain = 100.0 * (1.0 - candidate_median / baseline_median)
        round_rows.append({
            "round": round_index,
            "baseline_median_ms": baseline_median,
            "candidate_median_ms": candidate_median,
            "improvement_pct": gain,
            "candidate_wins": gain > 0.0,
        })
    gains = [row["improvement_pct"] for row in round_rows]
    return {
        "round_results": round_rows,
        "median_improvement_pct": float(statistics.median(gains)),
        "wins": sum(bool(row["candidate_wins"]) for row in round_rows),
        "rounds": len(round_rows),
    }


def _yolo_graph_certificate(certificate_root):
    rows = read_jsonl(Path(certificate_root) / "graph_context_certificates.jsonl")
    matches = [row for row in rows if len(row.get("candidate_ids") or []) == 2]
    if len(matches) != 1:
        raise ValueError("expected exactly one two-candidate YOLO graph certificate")
    return matches[0]


def run(args):
    roots = {
        "stock": (Path(args.reference_build).resolve(), "stock_production"),
        "scoped_composable": (Path(args.scoped_build).resolve(), "selector_v3_composable"),
    }
    manifests = {name: verify_flat(root) for name, (root, _) in roots.items()}
    scoped = read(roots["scoped_composable"][0] / "summary.json")
    if scoped["status"] != "COMPOSABLE_TWO_ROUTE_GRAPH_BUILT_AND_SCOPED":
        raise ValueError("scoped composable build identity mismatch")
    graph_certificate = _yolo_graph_certificate(args.certificate_root)
    runtime = read(args.runtime_contract)
    for name, path in runtime["runtime_paths"].items():
        if sha256_file(path) != runtime["runtime_sha256"][name]:
            raise ValueError("runtime changed: " + name)

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "calls.jsonl").write_text("", encoding="utf-8")
    contract = {
        "schema": "c3_yolo608_composable_confirmation_boot_v1",
        "status": "frozen_before_this_confirmation_session",
        "latency_exposure": "development latency was already observed; this is confirmation, not holdout",
        "variants": list(VARIANTS),
        "round_orders": [list(row) for row in ROUND_ORDERS],
        "seeds_per_session": list(SEEDS),
        "fresh_rpc_per_variant_per_round": True,
        "explicit_warmups_per_session": 1,
        "timed_calls_per_session": len(SEEDS),
        "expected_sessions": len(ROUND_ORDERS) * len(VARIANTS),
        "expected_timing_calls": len(ROUND_ORDERS) * len(VARIANTS) * len(SEEDS),
        "expected_complete_graph_calls": len(ROUND_ORDERS) * len(VARIANTS) * (1 + len(SEEDS)),
        "timing_boundary": "host steady_clock around one synchronous GraphExecutor run RPC",
        "candidate_set_id": graph_certificate["candidate_set_id"],
        "candidate_ids": graph_certificate["candidate_ids"],
        "graph_certificate_id": graph_certificate["graph_certificate_id"],
        "build_manifests": manifests,
        "board_reboot_requested": False,
        "bitstream_reload_forbidden": True,
        "acceptance_owner": "c3_selector_v3.deployment_gate",
        "claim_boundary": "one boot result only; never sufficient for two-boot deployment acceptance",
    }
    write(output / "contract.json", contract)

    board = ExperimentBoard(args, "/var/volatile/" + output.name + "_runtime", args.port)
    rows, references = [], {}
    session_index = 0
    started = time.monotonic()
    initial = None
    try:
        initial = board.state()
        reuse_board.validate_state(initial, exclusive=True)
        environment = board.environment()
        reuse_board.RUNTIME = runtime["runtime_sha256"]
        write(output / "runtime_attestation.json", board.deploy(runtime["runtime_paths"]))
        with tempfile.TemporaryDirectory(prefix="c3_yolo608_confirm_") as temporary:
            for round_index, order in enumerate(ROUND_ORDERS):
                for position, variant in enumerate(order):
                    root, artifact_variant = roots[variant]
                    board.port = args.port + 2 * session_index
                    board.pid = None
                    remote = None
                    try:
                        state = board.state()
                        reuse_board.validate_state(state, exclusive=True)
                        if state["boot_id"] != initial["boot_id"]:
                            raise RuntimeError("boot changed before session")
                        board.start()
                        remote = connect(args.host, board.port)
                        _, funcs = runtime_inventory(remote)
                        functions = {
                            "clear": funcs["vta.runtime.profiler_clear"],
                            "status": funcs["vta.runtime.profiler_status"],
                        }
                        executor = load_executor(
                            remote, root, artifact_variant,
                            [remote.ext_dev(0), remote.cpu(0)], temporary,
                        )
                        current = board.environment(require_affinity=True)
                        if current["governor"] != environment["governor"]:
                            raise RuntimeError("governor changed")
                        _, warm = invoke_once(
                            executor, random_input(SEEDS[0]), functions, timed=False
                        )
                        warm.update(phase="explicit_warmup", round=round_index,
                                    position=position, variant=variant,
                                    session=session_index, seed=SEEDS[0])
                        append(output / "calls.jsonl", warm)
                        for seed in SEEDS:
                            outputs, row = invoke_once(
                                executor, random_input(seed), functions, timed=True
                            )
                            if not math.isfinite(row["latency_ms"]) or row["latency_ms"] <= 0:
                                raise RuntimeError("invalid latency")
                            profile = row["runtime_profile_complete"]
                            if profile.get("driver_timeout_calls") != 0 or profile.get("driver_run_calls", 0) <= 0:
                                raise RuntimeError("invalid runtime completion profile")
                            if variant == "stock" and seed not in references:
                                references[seed] = outputs
                            elif seed in references and any(mismatch_counts(references[seed], outputs)):
                                raise RuntimeError("full graph output mismatch")
                            row.update(phase="timing", round=round_index,
                                       position=position, variant=variant,
                                       session=session_index, seed=seed)
                            rows.append(row)
                            append(output / "calls.jsonl", row)
                        del executor
                        del remote
                        remote = None
                        gc.collect()
                    finally:
                        if remote is not None:
                            del remote
                            gc.collect()
                        board.stop()
                        board.pid = None
                    print("round {} position {} {} complete".format(
                        round_index, position, variant), flush=True)
                    session_index += 1

        if set(references) != set(SEEDS):
            raise RuntimeError("stock reference coverage incomplete")
        formal = summarize_rounds(rows)
        final = board.state()
        reuse_board.validate_state(final, exclusive=True)
        if final["boot_id"] != initial["boot_id"]:
            raise RuntimeError("boot changed during experiment")
        summary = {
            "schema": contract["schema"],
            "status": "ONE_BOOT_CONFIRMATION_COMPLETE",
            "candidate_set_id": graph_certificate["candidate_set_id"],
            "candidate_ids": graph_certificate["candidate_ids"],
            "boot_id": initial["boot_id"],
            **formal,
            "all_outputs_equal": True,
            "sessions": session_index,
            "timing_calls": len(rows),
            "complete_graph_run_calls": session_index * (1 + len(SEEDS)),
            "implicit_extra_calls": 0,
            "board_reboot_requested": False,
            "bitstream_reloaded": False,
            "claim_boundary": contract["claim_boundary"],
            "wall_seconds": time.monotonic() - started,
        }
        write(output / "summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    except BaseException as error:
        write(output / "invalid_session.json", {
            "message": str(error), "type": type(error).__name__,
            "traceback": traceback.format_exc()[-10000:],
            "sessions_completed": session_index,
            "timing_calls_completed": len(rows),
            "all_timing_invalid": True,
        })
        raise
    finally:
        try:
            board.stop()
        except Exception:
            pass
        seal(output)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-build", default=str(DEFAULT_REFERENCE))
    parser.add_argument("--scoped-build", default=str(DEFAULT_SCOPED))
    parser.add_argument("--runtime-contract", default=str(DEFAULT_RUNTIME))
    parser.add_argument("--certificate-root", default=str(DEFAULT_CERTIFICATES))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument("--port", type=int, default=9110)
    parser.add_argument("--known-hosts", default="/home/orange/.ssh/known_hosts")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
