#!/usr/bin/env python3
"""Replay a fail-fast FPGA-correctness gate on frozen P7R122/125/126 evidence.

This analysis is retrospective and never contacts the board.  It preserves the
frozen seed order and stops an identity after its first wrong answer.  A passing
identity still requires all three seeds, so fail-fast cannot promote a candidate
that the original three-seed policy rejected.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


SCHEMA = "c3_p7r129_fail_fast_correctness_replay_v1"
SEED_ORDER = (0, 20250901, 20260910)
FIELDS = (
    "load_buffer_2d_bytes",
    "load_buffer_2d_calls",
    "store_buffer_2d_bytes",
    "store_buffer_2d_calls",
    "driver_run_insns",
)
HERE = Path(__file__).resolve().parent
C3 = HERE / "report_out" / "stage_tile_cotuning" / "c3_dma_residency_autotune"
P7 = C3 / "07_grouped_holdout"
DEFAULT_INPUTS = (
    (
        "Y02",
        P7 / "20260912_p7r122_y02_health_gated_correctness_run01",
        "candidate_correctness.jsonl",
    ),
    (
        "Y01",
        P7 / "20260912_p7r125_y01_complete_board_pool_run01",
        "correctness.jsonl",
    ),
    (
        "Y01",
        P7 / "20260912_p7r126_y01_remaining_correctness_run01",
        "correctness.jsonl",
    ),
)
DEFAULT_OUTPUT = P7 / "20260912_p7r129_fail_fast_correctness_replay_run01"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def verify_run_file(run_dir, name):
    run_dir = Path(run_dir)
    ledger = json.loads((run_dir / "artifact_hashes.json").read_text(encoding="utf-8"))
    expected = ledger["artifacts"].get(name)
    observed = sha256_file(run_dir / name)
    if expected != observed:
        raise ValueError("frozen input hash mismatch: {}".format(run_dir / name))
    return observed, sha256_file(run_dir / "artifact_hashes.json")


def load_rows(inputs=DEFAULT_INPUTS):
    rows = []
    bindings = []
    for geometry, run_dir, name in inputs:
        artifact_hash, ledger_hash = verify_run_file(run_dir, name)
        bindings.append(
            {
                "geometry": geometry,
                "path": str((Path(run_dir) / name).resolve()),
                "sha256": artifact_hash,
                "artifact_ledger_sha256": ledger_hash,
            }
        )
        for line in (Path(run_dir) / name).read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append({"geometry": geometry, **json.loads(line)})
    return rows, bindings


def seed_profile(seed):
    profile = seed.get("runtime_profile") or seed.get("runtime_profile_complete")
    if not isinstance(profile, dict):
        raise ValueError("seed lacks runtime profile")
    return profile


def replay_identity(row):
    seeds = row["seeds"]
    if tuple(int(seed["seed"]) for seed in seeds) != SEED_ORDER:
        raise ValueError("seed order differs from frozen fail-fast contract")
    used = []
    for seed in seeds:
        used.append(seed)
        if not seed.get("correct"):
            break
    decision = "passed" if len(used) == len(seeds) and all(seed["correct"] for seed in used) else "failed"
    if row["status"] not in ("passed", "failed"):
        raise ValueError("full policy has no binary classification")
    totals_full = {field: 0 for field in FIELDS}
    totals_fail_fast = {field: 0 for field in FIELDS}
    for position, seed in enumerate(seeds):
        profile = seed_profile(seed)
        for field in FIELDS:
            totals_full[field] += int(profile.get(field, 0))
            if position < len(used):
                totals_fail_fast[field] += int(profile.get(field, 0))
    return {
        "geometry": row["geometry"],
        "candidate_id": row["candidate_id"],
        "family_id": row["family_id"],
        "public_mode": row["public_mode"],
        "full_decision": row["status"],
        "fail_fast_decision": decision,
        "decision_matches": decision == row["status"],
        "full_invocations": len(seeds),
        "fail_fast_invocations": len(used),
        "first_failure_seed": next(
            (int(seed["seed"]) for seed in seeds if not seed.get("correct")), None
        ),
        "full": totals_full,
        "fail_fast": totals_fail_fast,
    }


def aggregate(records):
    result = {
        "identities": len(records),
        "passed": sum(row["full_decision"] == "passed" for row in records),
        "failed": sum(row["full_decision"] == "failed" for row in records),
        "decision_matches": sum(row["decision_matches"] for row in records),
        "full_invocations": sum(row["full_invocations"] for row in records),
        "fail_fast_invocations": sum(row["fail_fast_invocations"] for row in records),
        "full": {field: sum(row["full"][field] for row in records) for field in FIELDS},
        "fail_fast": {
            field: sum(row["fail_fast"][field] for row in records) for field in FIELDS
        },
    }
    result["invocation_reduction_fraction"] = 1.0 - (
        result["fail_fast_invocations"] / result["full_invocations"]
    )
    result["reduction_fraction"] = {
        field: 1.0 - result["fail_fast"][field] / result["full"][field]
        for field in FIELDS
    }
    return result


def analyze(rows):
    records = [replay_identity(row) for row in rows]
    if len({row["candidate_id"] for row in records}) != len(records):
        raise ValueError("duplicate candidate identities across frozen inputs")
    geometries = {
        geometry: aggregate([row for row in records if row["geometry"] == geometry])
        for geometry in sorted({row["geometry"] for row in records})
    }
    result = {
        "schema": SCHEMA,
        "status": "retrospective_replay_complete",
        "seed_order": list(SEED_ORDER),
        "policy": (
            "run seeds in frozen order; reject immediately on first mismatch; "
            "require all three seeds before passing"
        ),
        "overall": aggregate(records),
        "by_geometry": geometries,
        "records": records,
        "claim_boundary": (
            "retrospective on 29 completed identities from Y01/Y02; exact decision preservation "
            "is not a prospective false-negative guarantee"
        ),
    }
    return result


def render_results(result):
    overall = result["overall"]
    return """# P7R129 FPGA 正确性 fail-fast 回放

- 身份：{identities}（通过 {passed}，失败 {failed}）
- 完整三 seed 调用：{full_invocations}
- fail-fast 调用：{fail_fast_invocations}
- 调用减少：{invocation_reduction:.2%}
- 分类一致：{decision_matches}/{identities}
- LOAD bytes：{full_load:,} → {fast_load:,}（-{load_reduction:.2%}）
- LOAD calls：{full_calls:,} → {fast_calls:,}（-{call_reduction:.2%}）
- STORE bytes：{full_store:,} → {fast_store:,}（-{store_reduction:.2%}）

该策略不会用一个种子批准候选：失败时首错即停，只有三个种子全部正确才允许进入 timing。当前
29 个冻结身份中，所有最终失败都在首个种子暴露，因此在保持 29/29 分类一致的同时避免了 54 次
无效 FPGA 执行及其共享内存访问。

这是回顾性结果，不是未来候选零漏判的证明。正式调优时仍需记录 gross candidate dispatch，报告
第一种子未捕获、后续种子才失败的数量，并把所有通过身份跑满三 seed。
""".format(
        identities=overall["identities"],
        passed=overall["passed"],
        failed=overall["failed"],
        full_invocations=overall["full_invocations"],
        fail_fast_invocations=overall["fail_fast_invocations"],
        invocation_reduction=overall["invocation_reduction_fraction"],
        decision_matches=overall["decision_matches"],
        full_load=overall["full"]["load_buffer_2d_bytes"],
        fast_load=overall["fail_fast"]["load_buffer_2d_bytes"],
        load_reduction=overall["reduction_fraction"]["load_buffer_2d_bytes"],
        full_calls=overall["full"]["load_buffer_2d_calls"],
        fast_calls=overall["fail_fast"]["load_buffer_2d_calls"],
        call_reduction=overall["reduction_fraction"]["load_buffer_2d_calls"],
        full_store=overall["full"]["store_buffer_2d_bytes"],
        fast_store=overall["fail_fast"]["store_buffer_2d_bytes"],
        store_reduction=overall["reduction_fraction"]["store_buffer_2d_bytes"],
    )


def run(output_dir=DEFAULT_OUTPUT):
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable output {}".format(output))
    rows, bindings = load_rows()
    result = analyze(rows)
    output.mkdir(parents=True)
    write_json(output / "summary.json", result)
    write_json(output / "input_bindings.json", {"inputs": bindings})
    (output / "RESULTS.md").write_text(render_results(result), encoding="utf-8")
    with (output / "identity_replay.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "geometry",
                "candidate_id",
                "family_id",
                "public_mode",
                "full_decision",
                "fail_fast_decision",
                "full_invocations",
                "fail_fast_invocations",
                "first_failure_seed",
            ),
        )
        writer.writeheader()
        for row in result["records"]:
            writer.writerow({name: row[name] for name in writer.fieldnames})
    hashes = {
        path.name: sha256_file(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    write_json(output / "artifact_hashes.json", {"schema": SCHEMA, "artifacts": hashes})
    print(json.dumps(result["overall"], indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    run(parser.parse_args().output_dir)


if __name__ == "__main__":
    main()
