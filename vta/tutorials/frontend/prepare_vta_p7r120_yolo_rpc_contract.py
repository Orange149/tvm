#!/usr/bin/env python3
"""Freeze the RPC-only Y02 barrier-board contract after local AXU qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import tvm
from tvm import autotvm
from tvm.contrib import cc
import vta

import vta.top.vta_conv2d_residency  # pylint: disable=unused-import
from c3_candidate_identity import canonical_json_bytes, normalize_config_entity


SCHEMA = "c3_p7r120_y02_rpc_board_contract_v2"
MODES = {
    "original": 0,
    "input_stationary": 1,
    "weight_resident_barrier": 4,
}
FAMILIES = ("Y02B00", "Y02B01")
SEEDS = (0, 20250901, 20260910)
TIMING_ROUNDS = 7
TIMING_ORDER_SEED = 20260911

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
C3 = HERE / "report_out" / "stage_tile_cotuning" / "c3_dma_residency_autotune"
P7R115 = C3 / "07_grouped_holdout" / "20260911_p7r115_yolo_confirmation_protocol_run01"
P7R119 = (
    C3
    / "07_grouped_holdout"
    / "20260911_p7r119_y02_weight_resident_barrier_pilot_v2_run01"
)
DEFAULT_OUTPUT = (
    C3 / "07_grouped_holdout" / "20260911_p7r120_y02_rpc_contract_v2_run01"
)
GUARD_PATHS = (
    REPO / "vta/python/vta/top/vta_conv2d.py",
    REPO / "vta/python/vta/top/vta_conv2d_residency.py",
    REPO / "vta/python/vta/transform.py",
    REPO / "src/tir/transforms/coproc_sync.cc",
    REPO / "vta/runtime/runtime.cc",
    REPO / "3rdparty/vta-hw/config/vta_config.json",
    REPO / "build/libtvm.so",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value):
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def verify_run(run_dir):
    run_dir = Path(run_dir)
    ledger = load_json(run_dir / "artifact_hashes.json")["artifacts"]
    verified = {}
    for name, expected in ledger.items():
        path = run_dir / name
        observed = sha256_file(path)
        if observed != expected:
            raise ValueError("artifact hash mismatch: {}".format(path))
        verified[name] = observed
    return verified


def select_six(candidates, static_rows, fsim_rows):
    static = {row["candidate_id"]: row for row in static_rows}
    fsim = {row["candidate_id"]: row for row in fsim_rows}
    selected = [row for row in candidates if row["family_id"] in FAMILIES]
    expected = {(family, mode) for family in FAMILIES for mode in MODES}
    observed = {(row["family_id"], row["public_mode"]) for row in selected}
    if len(selected) != 6 or observed != expected:
        raise ValueError("P7R120 requires exactly two complete three-mode families")
    for row in selected:
        candidate_id = row["candidate_id"]
        if canonical_sha256(row["identity"]) != candidate_id:
            raise ValueError("candidate v2 identity mismatch")
        if row["identity"]["implementation_mode"] != MODES[row["public_mode"]]:
            raise ValueError("public/implementation mode mismatch")
        if static.get(candidate_id, {}).get("status") != "ok":
            raise ValueError("candidate lacks local lowering pass")
        qualification = fsim.get(candidate_id)
        if not qualification or qualification.get("status") != "passed":
            raise ValueError("candidate lacks local FSim pass")
        if len(qualification.get("seeds", ())) != 3 or not all(
            seed.get("correct") for seed in qualification["seeds"]
        ):
            raise ValueError("candidate lacks three exact local FSim seeds")
        row = dict(row)
    mode_order = {mode: position for position, mode in enumerate(MODES)}
    return sorted(selected, key=lambda row: (row["family_id"], mode_order[row["public_mode"]]))


def make_timing_orders(candidate_ids, rounds=TIMING_ROUNDS, seed=TIMING_ORDER_SEED):
    if len(candidate_ids) != len(set(candidate_ids)) or not candidate_ids:
        raise ValueError("timing identities must be nonempty and unique")
    base = list(candidate_ids)
    random.Random(seed).shuffle(base)
    return [base[offset % len(base) :] + base[: offset % len(base)] for offset in range(rounds)]


def cross_options():
    sysroot = os.environ.get("SDKTARGETSYSROOT")
    if not sysroot:
        raise RuntimeError("SDKTARGETSYSROOT is unset; source the AXU SDK environment")
    return [
        "--sysroot=" + sysroot,
        "-Wl,-rpath-link," + sysroot + "/lib",
        "-Wl,-rpath-link," + sysroot + "/usr/lib",
        "-L" + sysroot + "/lib",
        "-L" + sysroot + "/usr/lib",
    ]


def make_task(workload, implementation_mode, env):
    return autotvm.task.create(
        "conv2d_packed_residency.vta",
        args=tuple(workload[1:]) + (int(implementation_mode),),
        target=env.target,
        target_host=env.target_host,
    )


def resolve_exact_tophub(workload, env, audit):
    """Reveal the already-authorized exact incumbent after candidate freeze."""

    if not audit.get("all_exact_hits") or not audit["workloads"]["Y02"]["exact_hit"]:
        raise ValueError("P7R115 did not authorize an exact Y02 TopHub reference")
    task = make_task(workload, 0, env)
    original_workload = ("conv2d_packed.vta",) + tuple(task.workload[1:-1])
    with autotvm.tophub.context(env.target):
        config = autotvm.task.DispatchContext.current.query(task.target, original_workload)
    if config.is_fallback:
        raise RuntimeError("exact Y02 TopHub lookup unexpectedly fell back")
    package = Path(audit["package_path"])
    if sha256_file(package) != audit["package_sha256"]:
        raise ValueError("TopHub package differs from post-commit audit")
    def knob_values(value):
        return {
            name: int(item[-1] if kind == "sp" else item)
            for name, kind, item in value["entity"]
        }

    source_entity = normalize_config_entity(config.to_json_dict())
    semantic_knobs = knob_values(source_entity)
    adapter_indices = [
        index
        for index in range(len(task.config_space))
        if knob_values(normalize_config_entity(task.config_space.get(index).to_json_dict()))
        == semantic_knobs
    ]
    if len(adapter_indices) != 1:
        raise RuntimeError(
            "exact TopHub entity maps to {} residency-adapter indices".format(
                len(adapter_indices)
            )
        )
    adapter_entity = normalize_config_entity(
        task.config_space.get(adapter_indices[0]).to_json_dict()
    )
    identity = {
        "schema": "c3_sealed_tophub_canary_identity_v1",
        "workload": workload,
        "source_workload": list(original_workload),
        "source_tophub_complete_config_entity": source_entity,
        "complete_config_entity": adapter_entity,
        "semantic_knobs": semantic_knobs,
        "config_serialization_note": (
            "TopHub stores resolved outer split factors while the isolated adapter "
            "stores -1 for inferred factors; semantic knob sizes are identical"
        ),
        "public_mode": "sealed_tophub_canary",
        "execution_adapter": "conv2d_packed_residency.vta mode0 original schedule branch",
        "selection_role": "independent pre/post canary and deployment fallback only",
        "package_sha256": audit["package_sha256"],
    }
    return {
        "candidate_id": canonical_sha256(identity),
        "identity": identity,
        "debug": {
            "source_tophub_config_index": int(config.index),
            "config_index": int(adapter_indices[0]),
            "config_index_semantics": "residency-template adapter only; not TopHub identity",
        },
        "implementation_mode": 0,
        "public_mode": "sealed_tophub_canary",
        "included_in_search_candidates": False,
        "included_in_training_labels": False,
    }


def build_and_export(entry, env, directory, expected_tir=None):
    workload = entry["identity"]["workload"]
    mode = int(entry["implementation_mode"])
    task = make_task(workload, mode, env)
    config = task.config_space.get(int(entry["debug"]["config_index"]))
    if canonical_json_bytes(normalize_config_entity(config.to_json_dict())) != canonical_json_bytes(
        entry["identity"]["complete_config_entity"]
    ):
        raise RuntimeError("complete ConfigEntity mismatch: {}".format(entry["candidate_id"]))
    with task.target:
        schedule, tensors = task.instantiate(config)
    with vta.build_config(disabled_pass={"tir.CommonSubexprElimTIR"}):
        lowered = tvm.lower(schedule, tensors, name="main")
    tir_hash = hashlib.sha256(tvm.ir.save_json(lowered).encode("utf-8")).hexdigest()
    if expected_tir is not None and tir_hash != expected_tir:
        raise RuntimeError("re-lowered TIR differs from P7R119")
    with vta.build_config(disabled_pass={"tir.CommonSubexprElimTIR"}):
        module = vta.build(
            schedule,
            tensors,
            target=tvm.target.Target(env.target, host=env.target_host),
            name="main",
        )
    binary = Path(directory) / (entry["candidate_id"] + ".so")
    module.export_library(
        str(binary),
        fcompile=cc.cross_compiler("aarch64-xilinx-linux-g++"),
        options=cross_options(),
    )
    return {
        "status": "passed",
        "tir_sha256": tir_hash,
        "binary_sha256": sha256_file(binary),
        "binary_size_bytes": binary.stat().st_size,
        "binary_retained": False,
    }


def env_fingerprint(env):
    return {
        "TARGET": env.TARGET,
        "BATCH": env.BATCH,
        "BLOCK_IN": env.BLOCK_IN,
        "BLOCK_OUT": env.BLOCK_OUT,
        "INP_BUFF_SIZE": env.INP_BUFF_SIZE,
        "WGT_BUFF_SIZE": env.WGT_BUFF_SIZE,
        "ACC_BUFF_SIZE": env.ACC_BUFF_SIZE,
        "UOP_BUFF_SIZE": env.UOP_BUFF_SIZE,
        "target": str(env.target),
        "target_host": str(env.target_host),
    }


def run(args):
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable output {}".format(output))
    p7r119_hashes = verify_run(args.p7r119_dir)
    candidates = load_jsonl(Path(args.p7r119_dir) / "candidates_v2.jsonl")
    static_rows = load_jsonl(Path(args.p7r119_dir) / "static_results.jsonl")
    fsim_rows = load_jsonl(Path(args.p7r119_dir) / "fsim_results.jsonl")
    selected = select_six(candidates, static_rows, fsim_rows)
    static = {row["candidate_id"]: row for row in static_rows}

    env = vta.get_env()
    if env.TARGET != "axu5evb":
        raise RuntimeError("P7R120 contract requires VTA TARGET=axu5evb")
    p7r115_contract = load_json(args.p7r115_contract)
    audit = p7r115_contract["post_commit_exact_tophub_audit"]
    workload = selected[0]["identity"]["workload"]
    canary = resolve_exact_tophub(workload, env, audit)
    guards_before = {str(path.relative_to(REPO)): sha256_file(path) for path in GUARD_PATHS}
    compiler = subprocess.run(
        ["aarch64-xilinx-linux-g++", "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()[0]

    qualification = []
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="c3_p7r120_cross_") as directory:
        for entry in selected:
            certificate = build_and_export(
                entry, env, directory, static[entry["candidate_id"]]["tir_sha256"]
            )
            qualification.append({
                "candidate_id": entry["candidate_id"],
                "family_id": entry["family_id"],
                "public_mode": entry["public_mode"],
                **certificate,
            })
        canary_certificate = build_and_export(canary, env, directory)
    guards_after = {str(path.relative_to(REPO)): sha256_file(path) for path in GUARD_PATHS}
    if guards_before != guards_after:
        raise RuntimeError("guarded sources changed during cross qualification")
    if len(qualification) != 6 or not all(row["status"] == "passed" for row in qualification):
        raise RuntimeError("not all six candidates cross-qualified")

    q_by_id = {row["candidate_id"]: row for row in qualification}
    public_candidates = []
    for entry in selected:
        public_candidates.append(
            {
                **entry,
                "fresh_axu_cross_certificate": q_by_id[entry["candidate_id"]],
                "board_status": "not_dispatched",
            }
        )
    canary = {**canary, "fresh_axu_cross_certificate": canary_certificate}
    candidate_ids = [entry["candidate_id"] for entry in public_candidates]
    timing_orders = make_timing_orders(candidate_ids)
    source = {
        "p7r119_dir": str(Path(args.p7r119_dir).resolve()),
        "p7r119_artifacts": p7r119_hashes,
        "p7r119_artifact_ledger_sha256": sha256_file(
            Path(args.p7r119_dir) / "artifact_hashes.json"
        ),
        "p7r115_contract": {
            "path": str(Path(args.p7r115_contract).resolve()),
            "sha256": sha256_file(args.p7r115_contract),
        },
    }
    contract = {
        "schema": SCHEMA,
        "status": "frozen_after_local_cross_qualification_before_rpc_labels",
        "study_tier": "six_candidate_y02_mode4_board_pilot",
        "source": source,
        "fresh_hardware_certificate_v2": {
            "hardware_fingerprint": env_fingerprint(env),
            "source_guards_sha256": guards_before,
            "cross_compiler": compiler,
            "sysroot": os.environ.get("SDKTARGETSYSROOT"),
            "qualified_candidates": qualification,
            "sealed_canary": canary_certificate,
            "elapsed_seconds": time.monotonic() - started,
            "claim": "local AXU build/export only; board correctness is not inherited",
        },
        "candidate_pool": {
            "gross_candidates": 6,
            "families": list(FAMILIES),
            "public_modes": MODES,
            "candidates": public_candidates,
            "pool_commitment_sha256": canonical_sha256(public_candidates),
            "equal_dispatch_budget": "two same-tile identities per mode; no replacements",
        },
        "sealed_reference": {
            **canary,
            "reveal_phase": "after P7R119 pool freeze and local FSim measurements",
            "execution_role": "pre/post timing canary and protected fallback",
            "prohibited_roles": ["candidate generation", "search dispatch", "training label"],
        },
        "rpc_contract": {
            "host": args.host,
            "port": args.port,
            "transport": "direct TVM RPC only",
            "ssh_forbidden": True,
            "board_restart_forbidden": True,
            "persistent_board_write_forbidden": True,
            "module_transport": "ephemeral RPC upload, load_module, then immediate remote.remove",
            "boot_id": "unknown_not_exposed_by_rpc",
            "boot_id_semantics": "must remain unknown unless a read-only RPC runtime API exposes it",
            "required_runtime_functions": ["vta.runtime.profiler_status"],
            "optional_runtime_functions": [
                "vta.runtime.profiler_clear",
                "vta.runtime.queue_capacity_status",
                "vta.runtime.replay_status",
            ],
        },
        "correctness": {
            "seeds": list(SEEDS),
            "candidate_order": candidate_ids,
            "checks": "elementwise exact equality against independent NumPy reference",
            "timing_gate": "all six identities must pass all three seeds",
            "failure_policy": "record and stop failed identity; no candidate is timed unless global 6/6 gate passes",
        },
        "timing": {
            "rounds": TIMING_ROUNDS,
            "candidate_invocations_per_round": 1,
            "candidate_samples": 6 * TIMING_ROUNDS,
            "order_seed": TIMING_ORDER_SEED,
            "candidate_orders": timing_orders,
            "balance": "one sample per candidate per round; cyclic rotations balance positions over first six rounds",
            "canary_bracketing": "sealed TopHub once before and once after candidate order in every round",
            "canary_samples": 2 * TIMING_ROUNDS,
            "output_check": "verify the output produced by every timed invocation",
        },
        "analysis": {
            "primary": "within-family mode versus same-tile original median",
            "secondary": "all six candidates versus sealed TopHub bracket midpoint",
            "promotion_rule": (
                "retain TopHub unless a correctness-passed candidate is at least 2% faster "
                "than pooled TopHub median and wins at least 5 of 7 round-midpoint comparisons"
            ),
            "incumbent_protection": True,
            "minimum_access_role": "post-measure deployment policy only; never a seventh schedule",
        },
        "claim_limits": [
            "single geometry and current RPC session only",
            "boot ID is unknown because SSH is unavailable and RPC exposes no boot identity",
            "no stage/FPS, cross-boot, full-pool-oracle, or exact-paper-reproduction claim",
        ],
    }
    output.mkdir(parents=True)
    write_json(output / "contract.json", contract)
    (output / "cross_qualification.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in qualification),
        encoding="utf-8",
    )
    (output / "STATUS.md").write_text(
        "# P7R120 Y02 RPC-only board contract\n\n"
        "- Status: `frozen_after_local_cross_qualification_before_rpc_labels`\n"
        "- Candidate pool: 6 (2 families x 3 real modes)\n"
        "- Fresh AXU cross qualification: 6/6; sealed TopHub canary: passed\n"
        "- Board path: direct RPC {}:{}; SSH/restart/persistent writes forbidden\n"
        "- Correctness gate: all 18 elementwise seed checks before timing\n"
        "- Timing: 7 balanced rounds, TopHub bracketed but excluded from search/training\n"
        "- Boot ID: `unknown_not_exposed_by_rpc`\n".format(args.host, args.port),
        encoding="utf-8",
    )
    (output / "command.txt").write_text(
        " ".join([sys.executable, str(Path(__file__).resolve())] + sys.argv[1:]) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema": "c3_p7r120_local_cross_manifest_v2",
        "python": sys.executable,
        "python_version": platform.python_version(),
        "tvm_version": tvm.__version__,
        "source_sha256": sha256_file(__file__),
        "board_contacted": False,
        "ssh_used": False,
        "rpc_used": False,
    }
    write_json(output / "manifest.json", manifest)
    hashes = {
        path.name: sha256_file(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    write_json(
        output / "artifact_hashes.json",
        {"artifacts": hashes, "source_sha256": {str(Path(__file__).resolve()): sha256_file(__file__)}},
    )
    print(json.dumps({"output": str(output), "cross_passed": 6, "canary": canary["debug"]}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p7r119-dir", type=Path, default=P7R119)
    parser.add_argument("--p7r115-contract", type=Path, default=P7R115 / "contract.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument("--port", type=int, default=9090)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
