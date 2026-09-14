#!/usr/bin/env python3
"""One independent boot of final T2656 E/P/F ABBA performance validation."""
import argparse
import json
import math
from pathlib import Path
import shlex
import subprocess

from audit_shared_buffer_storage import sha
from run_c3s_allocation_baseline import SSH, remote
from run_queue_threshold_abba import metrics, preflight


F = {"threshold": 2656, "insn": 5376, "uop": 5632}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--final-qualification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host", default="root@192.168.1.247")
    args = parser.parse_args()

    build = json.loads((args.build / "build_manifest.json").read_text())
    for name, digest in build["artifacts"].items():
        assert sha(args.build / name) == digest
    for path, digest in build["source_sha256"].items():
        assert sha(path) == digest
    q1 = json.loads((args.reference / "reconciled_summary.json").read_text())
    threshold = json.loads((args.qualification / "summary.json").read_text())
    assert threshold["status"] == "complete_correctness_screen"
    final = json.loads((args.final_qualification / "summary.json").read_text())
    assert final["status"] == "complete_final_candidate_qualification"
    assert final["candidate"] == {**F, "total": F["insn"] + F["uop"]}
    assert len(final["results"]) == 4
    assert all(
        item["cases"]["allocation"]["byte_exact"]
        and item["cases"]["stress"]["byte_exact"]
        and item["cases"]["stress"]["frames"] >= 1000
        for item in final["results"]
    )

    boot, initial = preflight(args.host)
    args.output.mkdir(parents=True, exist_ok=False)
    board = "/tmp/queue_" + args.output.name
    remote(args.host, "mkdir " + shlex.quote(board))
    (args.output / "board_preflight.txt").write_bytes(initial)

    def upload(path):
        subprocess.run(
            ["scp", *SSH[1:], str(path), args.host + ":" + board + "/"],
            check=True,
            capture_output=True,
        )

    for name in build["artifacts"]:
        upload(args.build / name)
    s1 = args.reference.parents[1] / "c3s_buffer_reuse/s1_board_run1"
    upload(s1 / "inputs.txt")
    for item in q1["inputs"]:
        assert sha(s1 / item["file"]) == item["sha256"]
        upload(s1 / item["file"])
    for name in "ABCD":
        upload(args.qualification / f"{name}_plan.json")

    report = {
        "boot": boot,
        "build": build,
        "build_manifest_sha256": sha(args.build / "build_manifest.json"),
        "reference_sha256": sha(args.reference / "reconciled_summary.json"),
        "threshold_qualification_sha256": sha(args.qualification / "summary.json"),
        "final_qualification_sha256": sha(args.final_qualification / "summary.json"),
        "script_sha256": sha(__file__),
        "imported_prescreen_script_sha256": sha(Path(__file__).with_name("run_queue_threshold_abba.py")),
        "board": board,
        "F": {**F, "buffer": "pool-anchor"},
        "modes": {
            "E": "external-K2/default-64MiB/no-threshold",
            "P": "pool-anchor-K2/default-64MiB/no-threshold",
            "F": "pool-anchor-K2/T2656/11008B",
        },
        "blocks_per_topology_per_contrast": 5,
        "runs": 302,
        "discard_first": 2,
        "extra_warmup_runs": 0,
        "contrasts": ["EP", "PF"],
        "order": "ABBA within block; topology and contrast rotation copied from preregistered Q1 design",
        "scope": "one independent boot; no inference until three complete distinct boots",
        "status": "running",
        "runs_completed": [],
        "blocks_completed": [],
    }
    (args.output / "preregistered.json").write_text(json.dumps(report, indent=2) + "\n")

    prepared = {}
    for name in "ABCD":
        stored = (args.reference / f"{name}_small_diag_command.sh").read_text().strip()
        cd, env_command = stored.split(" && env ", 1)
        words = shlex.split(env_command)
        runner = next(i for i, word in enumerate(words) if word.endswith("/vta_stage_pipeline_runner"))
        words[runner] = board + "/vta_stage_pipeline_runner"
        words[words.index("--input-list") + 1] = board + "/inputs.txt"
        words[words.index("--shared-buffer-plan") + 1] = board + f"/{name}_plan.json"
        for flag in ("--allocation-snapshot", "--output-dump-dir"):
            index = words.index(flag)
            del words[index:index + 2]
        words[words.index("--runs") + 1] = "302"
        words[words.index("--warmup-runs") + 1] = "0"
        words[words.index("--vta-runtime-profile-dir") + 1] = ""
        words = [word for word in words if not word.startswith((
            "VTA_QUEUE_DIAGNOSTICS=", "VTA_QUEUE_BOUNDARY_AUDIT=",
            "VTA_INSN_BUFFER_BYTES=", "VTA_UOP_BUFFER_BYTES=",
            "VTA_INSN_SUBMIT_THRESHOLD_BYTES=",
        ))]
        words[0:0] = [
            "VTA_QUEUE_DIAGNOSTICS=0",
            "VTA_QUEUE_BOUNDARY_AUDIT=0",
            "VTA_INSN_BUFFER_BYTES=33554432",
            "VTA_UOP_BUFFER_BYTES=33554432",
        ]
        preload = next(i for i, word in enumerate(words) if word.startswith("LD_PRELOAD="))
        tvm_runtime = words[preload].split("=", 1)[1].split(":", 1)[0]
        words[preload] = "LD_PRELOAD=" + tvm_runtime + ":" + board + "/libvta.so"
        package_dir = shlex.split(cd)[1]
        package = {
            line.split()[1]: line.split()[0]
            for line in (args.reference / f"{name}_package_sha256.txt").read_text().splitlines()
        }
        checks = {package_dir + "/" + path: digest for path, digest in package.items()}
        checks.update({board + "/" + n: d for n, d in build["artifacts"].items()})
        checks[board + f"/{name}_plan.json"] = sha(args.qualification / f"{name}_plan.json")
        checks[board + "/inputs.txt"] = sha(s1 / "inputs.txt")
        for item in q1["inputs"]:
            checks[board + "/" + item["file"]] = item["sha256"]
        reference_rows = [
            json.loads(row)
            for row in (args.reference / f"{name}_default_off.jsonl").read_text().splitlines()
        ]
        expected = {
            row["input_index"]: [output["fnv1a64"] for output in row["raw_outputs"]]
            for row in reference_rows
        }
        prepared[name] = (cd, words, checks, expected)

    try:
        for block in range(5):
            names = "ABCD"[block % 4:] + "ABCD"[:block % 4]
            for name in names:
                cd, original, checks, expected = prepared[name]
                contrasts = ("EP", "PF") if block % 2 == 0 else ("PF", "EP")
                for contrast in contrasts:
                    paired = []
                    for position, mode in enumerate((contrast[0], contrast[1], contrast[1], contrast[0])):
                        stem = f"block{block}_{name}_{contrast}_{position}_{mode}"
                        observed_boot, state = preflight(args.host)
                        assert observed_boot == boot, "boot changed"
                        hashes = remote(
                            args.host, "sha256sum " + " ".join(map(shlex.quote, checks))
                        ).decode()
                        assert {line.split()[1]: line.split()[0] for line in hashes.splitlines()} == checks
                        (args.output / f"{stem}_preflight.txt").write_bytes(state + hashes.encode())
                        current = list(original)
                        if mode == "E":
                            index = current.index("--shared-buffer-plan")
                            del current[index:index + 2]
                        if mode == "F":
                            for prefix, value in (
                                ("VTA_INSN_BUFFER_BYTES=", F["insn"]),
                                ("VTA_UOP_BUFFER_BYTES=", F["uop"]),
                            ):
                                index = next(i for i, word in enumerate(current) if word.startswith(prefix))
                                current[index] = prefix + str(value)
                            current.insert(0, "VTA_INSN_SUBMIT_THRESHOLD_BYTES=" + str(F["threshold"]))
                        current[current.index("--output-jsonl") + 1] = board + "/" + stem + ".jsonl"
                        shell = cd + " && env " + shlex.join(current)
                        (args.output / f"{stem}_command.sh").write_text(shell + "\n")
                        print("[RUN]", stem, flush=True)
                        proc = subprocess.run(SSH + [args.host, shell], capture_output=True, timeout=300)
                        (args.output / f"{stem}.stdout").write_bytes(proc.stdout)
                        (args.output / f"{stem}.stderr").write_bytes(proc.stderr)
                        proc.check_returncode()
                        assert b"[VTA_QUEUE]" not in proc.stderr and b"[VTA_BOUNDARY]" not in proc.stderr
                        if mode != "E":
                            assert b"source=pool-anchor" in proc.stdout and b"fallback=" not in proc.stdout
                        raw = remote(args.host, "cat " + shlex.quote(board + "/" + stem + ".jsonl"))
                        path = args.output / f"{stem}.jsonl"
                        path.write_bytes(raw)
                        rows = [json.loads(row) for row in raw.splitlines()]
                        assert all(
                            row["input_index"] == i % 8
                            and [output["fnv1a64"] for output in row["raw_outputs"]] == expected[i % 8]
                            for i, row in enumerate(rows)
                        )
                        item = {"block": block, "topology": name, "contrast": contrast,
                                "position": position, "mode": mode,
                                "result_sha256": sha(path), **metrics(rows)}
                        paired.append(item)
                        report["runs_completed"].append(item)
                        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
                    log_ratio = (
                        math.log(paired[1]["ii_ms"]) + math.log(paired[2]["ii_ms"])
                        - math.log(paired[0]["ii_ms"]) - math.log(paired[3]["ii_ms"])
                    ) / 2
                    item = {"block": block, "topology": name, "contrast": contrast,
                            "log_ii_ratio": log_ratio,
                            "ii_change_pct": math.expm1(log_ratio) * 100}
                    report["blocks_completed"].append(item)
                    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
                    print("[BLOCK]", name, contrast, block, round(item["ii_change_pct"], 3), "%", flush=True)
        report["status"] = "complete_one_boot"
    except Exception as error:
        report["status"] = "failed_no_retry"
        report["error"] = repr(error)
        raise
    finally:
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    observed_boot, final_state = preflight(args.host)
    assert observed_boot == boot
    (args.output / "board_after.txt").write_bytes(final_state)


if __name__ == "__main__":
    main()
