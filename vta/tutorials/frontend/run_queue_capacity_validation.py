#!/usr/bin/env python3
"""Negative/recovery and natural stress of a Q1-qualified queue configuration.

Does not qualify early submission, forced slow consumers or general fault handling.
Uses fresh processes and never overwrites a prior run's output or frozen artifact.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

from audit_shared_buffer_storage import sha
from run_c3s_allocation_baseline import SSH, remote
from run_shared_buffer_qualification import archive_outputs, interleavings


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--host", default="root@192.168.1.247")
    p.add_argument("--runs", type=int, default=1000)
    args = p.parse_args()
    assert args.runs >= 32
    summary_path = args.reference / "reconciled_summary.json"
    if not summary_path.exists():
        summary_path = args.reference / "summary.json"
    summary = json.loads(summary_path.read_text())
    assert len(summary["results"]) == 4 and summary["capacities"]
    boot = remote(args.host, "cat /proc/sys/kernel/random/boot_id").decode().strip()
    assert boot == summary["boot"]
    args.output.mkdir(parents=True, exist_ok=False)
    board = "/media/sd-mmcblk1p2/queue_" + args.output.name
    remote(args.host, "mkdir " + shlex.quote(board))
    report = {"boot": boot, "reference_sha256": sha(summary_path),
              "script_sha256": sha(__file__), "runs": args.runs,
              "scope": "capacity rejection/recovery and natural varied-input stress; NOT FPS or forced fault qualification",
              "results": [], "negative": []}
    (args.output / "preregistered.json").write_text(json.dumps(report, indent=2) + "\n")
    for item in summary["results"]:
        name = item["topology"]
        stored = (args.reference / f"{name}_small_diag_command.sh").read_text().strip()
        cd, env_command = stored.split(" && env ", 1)
        words = shlex.split(env_command)
        ri = next(i for i, word in enumerate(words) if word.endswith("/vta_stage_pipeline_runner"))
        old_board = str(Path(words[ri]).parent)
        checks = {old_board + "/" + file: value for file, value in summary["build"]["artifacts"].items()}
        checks.update({old_board + "/" + i["file"]: i["sha256"] for i in summary["inputs"]})
        checks[old_board + f"/{name}_plan.json"] = sha(args.reference / f"{name}_plan.json")
        # inputs.txt was copied from frozen S1 and is retained on the board.
        ref_inputs = Path(__file__).resolve().parent / "report_out/stage_tile_cotuning/c3s_buffer_reuse/s1_board_run1/inputs.txt"
        checks[old_board + "/inputs.txt"] = sha(ref_inputs)
        actual = remote(args.host, "sha256sum " + " ".join(map(shlex.quote, checks))).decode()
        assert {r.split()[1]: r.split()[0] for r in actual.splitlines()} == checks
        expected_package = {r.split()[1]: r.split()[0] for r in (args.reference / f"{name}_package_sha256.txt").read_text().splitlines()}
        package_hashes = remote(args.host, cd + " && sha256sum " + " ".join(map(shlex.quote, expected_package))).decode()
        assert {r.split()[1]: r.split()[0] for r in package_hashes.splitlines()} == expected_package
        (args.output / f"{name}_hashes.txt").write_text(actual + package_hashes)
        cases = ["bad_insn", "bad_uop", "bad_alignment", "unqualified_threshold", "stress"] if name == "A" else ["stress"]
        for case in cases:
            assert remote(args.host, "cat /proc/sys/kernel/random/boot_id").decode().strip() == boot
            stem = name + "_" + case
            command = list(words)
            command[command.index("VTA_QUEUE_DIAGNOSTICS=1")] = "VTA_QUEUE_DIAGNOSTICS=0"
            for flag, value in (("--runs", str(args.runs) if case == "stress" else "32"),
                    ("--output-jsonl", board + "/" + stem + ".jsonl"),
                    ("--allocation-snapshot", board + "/" + stem + "_allocation.jsonl"),
                    ("--output-dump-dir", board + "/" + stem + "_outputs"),
                    ("--vta-runtime-profile-dir", "")):
                command[command.index(flag)+1] = value
            if case in ("bad_insn", "bad_alignment", "bad_uop"):
                env_name = "VTA_UOP_BUFFER_BYTES=" if case == "bad_uop" else "VTA_INSN_BUFFER_BYTES="
                index = next(i for i, w in enumerate(command) if w.startswith(env_name))
                command[index] = env_name + ("257" if case == "bad_alignment" else "256")
            if case == "unqualified_threshold":
                command.insert(0, "VTA_INSN_SUBMIT_THRESHOLD_BYTES=1024")
            shell = cd + " && env " + shlex.join(command)
            (args.output / f"{stem}_command.sh").write_text(shell + "\n")
            print("[RUN]", stem, flush=True)
            proc = subprocess.run(SSH + [args.host, shell], capture_output=True, timeout=max(180, args.runs))
            (args.output / f"{stem}.stdout").write_bytes(proc.stdout)
            (args.output / f"{stem}.stderr").write_bytes(proc.stderr)
            if case != "stress":
                expected = b"capacity exceeded before submission" if case in ("bad_insn", "bad_uop") else (
                    b"positive and aligned" if case == "bad_alignment" else b"early submission is not qualified")
                assert proc.returncode != 0 and expected in proc.stderr, (case, proc.stderr)
                report["negative"].append({"case": case, "rejected": True, "exit_code": proc.returncode})
                (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
                continue
            proc.check_returncode()
            assert b"source=pool-anchor" in proc.stdout and b"fallback=" not in proc.stdout
            for suffix in (".jsonl", "_allocation.jsonl"):
                (args.output / (stem + suffix)).write_bytes(remote(args.host, "cat " + shlex.quote(board + "/" + stem + suffix)))
            rows = [json.loads(r) for r in (args.output / (stem + ".jsonl")).read_text().splitlines()]
            assert len(rows) == args.runs and [r["frame_id"] for r in rows] == list(range(args.runs))
            assert all(r["input_index"] == i % 8 for i, r in enumerate(rows))
            archive = remote(args.host, "tar -C " + shlex.quote(board + "/" + stem + "_outputs") + " -cf - .")
            (args.output / f"{stem}_outputs.tar").write_bytes(archive)
            outputs = archive_outputs(archive)
            references = archive_outputs((args.reference / f"{name}_default_off_outputs.tar").read_bytes())
            assert len(outputs) == args.runs
            for path, value in outputs.items():
                pieces = path.split("/")
                run_index = int(pieces[-2].removeprefix("run_"))
                ref = "/".join(pieces[:-2] + ["run_" + str(run_index % 8), pieces[-1]])
                assert value == references[ref], (name, run_index)
            phases = {r["phase"]: r for r in map(json.loads, (args.output / (stem + "_allocation.jsonl")).read_text().splitlines())}
            high = phases["after_steady_state"]["high_water_bytes"]
            assert high == phases["after_warmup"]["high_water_bytes"] == item["modes"]["small_diag"]["high_water"]
            plan = json.loads((args.reference / f"{name}_plan.json").read_text())
            report["results"].append({"topology": name, "frames": len(rows), "byte_exact": True,
                "compared_bytes": sum(map(len, outputs.values())), "high_water": high, "steady_growth": 0,
                "overlaps": {str(e["producer_stage"]): interleavings(rows, e["producer_stage"])
                             for e in plan["edges"] if e["source"] == "pool-anchor"}})
            (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
            print("[PASS]", stem, flush=True)
    (args.output / "board_after.txt").write_bytes(remote(args.host, "cat /proc/sys/kernel/random/boot_id; ps -eo pid,args"))


if __name__ == "__main__":
    main()
