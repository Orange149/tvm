#!/usr/bin/env python3
"""One-boot Q2 threshold ABBA pre-screen; not formal performance evidence.

Compare the threshold-capable runtime with threshold disabled (P) against the
two non-dominated qualified threshold/capacity candidates.  Diagnostics,
allocation snapshots and output dumps are disabled during timing.
"""
import argparse
import json
import math
from pathlib import Path
import shlex
import subprocess

import numpy as np

from audit_shared_buffer_storage import sha
from run_c3s_allocation_baseline import SSH, remote


CANDIDATES = {
    "T2656": {"threshold": 2656, "insn": 5376, "uop": 5632},
    "T1328": {"threshold": 1328, "insn": 4096, "uop": 5632},
}


def metrics(rows):
    assert len(rows) == 302
    assert [r["frame_id"] for r in rows] == list(range(302))
    steady = rows[2:]
    completion = np.asarray([r["completion_ms"] for r in steady])
    assert np.all(np.diff(completion) > 0)
    ii = float(np.mean(np.diff(completion)))
    return {
        "ii_ms": ii,
        "fps": 1000 / ii,
        "submit_to_complete_p95_ms": float(
            np.percentile([r["total_latency_ms"] for r in steady], 95)
        ),
        "frames": 302,
        "scored_frames": 300,
        "scored_intervals": 299,
    }


def preflight(host):
    raw = remote(
        host,
        "cat /proc/sys/kernel/random/boot_id; cat /sys/class/fpga_manager/fpga0/state; "
        "cat /sys/class/u-dma-buf/udmabuf0/size; "
        "cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor; "
        "cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq; "
        "sha256sum /lib/firmware/vta_hpc.bit; ps -eo pid,args",
    )
    lines = raw.decode().splitlines()
    assert lines[1:5] == ["operating", "201326592", "userspace", "1066666"]
    assert lines[5].split()[0] == (
        "7bf1ac95b1182c670cd25241111f33725b4ea37ed6d66f2df37b0b2e7df528d6"
    )
    for line in lines[6:]:
        fields = line.split()
        if len(fields) > 1:
            assert Path(fields[1]).name not in (
                "vta_stage_pipeline_runner",
                "vta_stage_pair_runner",
                "tvm_rpc",
            ), line
    return lines[0], raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host", default="root@192.168.1.247")
    args = parser.parse_args()

    build = json.loads((args.build / "build_manifest.json").read_text())
    for name, digest in build["artifacts"].items():
        assert sha(args.build / name) == digest
    for path, digest in build["source_sha256"].items():
        assert sha(path) == digest
    qualified = json.loads((args.qualification / "summary.json").read_text())
    assert qualified["status"] == "complete_correctness_screen"
    for candidate in CANDIDATES.values():
        rows = [r for r in qualified["results"] if r["threshold"] == candidate["threshold"]]
        assert len(rows) == 4 and all(r["byte_exact"] for r in rows)
        assert max(r["insn_peak"] for r in rows) * 2 <= candidate["insn"]
        assert max(r["uop_peak"] for r in rows) * 2 <= candidate["uop"]

    q1 = json.loads((args.reference / "reconciled_summary.json").read_text())
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
    input_list = args.reference.parents[1] / "c3s_buffer_reuse/s1_board_run1/inputs.txt"
    upload(input_list)
    for item in q1["inputs"]:
        upload(args.reference.parents[1] / "c3s_buffer_reuse/s1_board_run1" / item["file"])
    for name in "ABCD":
        upload(args.qualification / f"{name}_plan.json")

    report = {
        "boot": boot,
        "build": build,
        "build_manifest_sha256": sha(args.build / "build_manifest.json"),
        "qualification_sha256": sha(args.qualification / "summary.json"),
        "reference_sha256": sha(args.reference / "reconciled_summary.json"),
        "script_sha256": sha(__file__),
        "board": board,
        "baseline": {"mode": "P", "insn": 33554432, "uop": 33554432,
                     "threshold": None, "buffer": "pool-anchor"},
        "candidates": CANDIDATES,
        "dominated_without_timing": {
            "T656": "same 9728 B total capacity as T1328, more submissions",
            "T320": "same 9728 B total capacity as T1328, more submissions",
        },
        "blocks_per_topology_per_candidate": 1,
        "order": "P,T,T,P",
        "runs": 302,
        "discard_first": 2,
        "scope": "one-boot pre-screen only; not non-inferiority evidence",
        "status": "running",
        "runs_completed": [],
        "blocks_completed": [],
    }
    (args.output / "preregistered.json").write_text(json.dumps(report, indent=2) + "\n")

    prepared = {}
    for name in "ABCD":
        command = (args.reference / f"{name}_small_diag_command.sh").read_text().strip()
        cd, env_command = command.split(" && env ", 1)
        words = shlex.split(env_command)
        runner_index = next(i for i, word in enumerate(words) if word.endswith("/vta_stage_pipeline_runner"))
        words[runner_index] = board + "/vta_stage_pipeline_runner"
        words[words.index("--input-list") + 1] = board + "/inputs.txt"
        words[words.index("--shared-buffer-plan") + 1] = board + f"/{name}_plan.json"
        for flag in ("--allocation-snapshot", "--output-dump-dir"):
            index = words.index(flag)
            del words[index:index + 2]
        words[words.index("--runs") + 1] = "302"
        words[words.index("--warmup-runs") + 1] = "0"
        words[words.index("--vta-runtime-profile-dir") + 1] = ""
        words = [w for w in words if not w.startswith((
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
        preload = next(i for i, w in enumerate(words) if w.startswith("LD_PRELOAD="))
        runtime = words[preload].split("=", 1)[1].split(":", 1)[0]
        words[preload] = "LD_PRELOAD=" + runtime + ":" + board + "/libvta.so"
        package_dir = shlex.split(cd)[1]
        package = {
            line.split()[1]: line.split()[0]
            for line in (args.reference / f"{name}_package_sha256.txt").read_text().splitlines()
        }
        checks = {package_dir + "/" + path: digest for path, digest in package.items()}
        checks.update({board + "/" + n: d for n, d in build["artifacts"].items()})
        checks[board + f"/{name}_plan.json"] = sha(args.qualification / f"{name}_plan.json")
        checks[board + "/inputs.txt"] = sha(input_list)
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
        for candidate_name, candidate in CANDIDATES.items():
            for name in "ABCD":
                cd, original, checks, expected = prepared[name]
                paired = []
                for position, mode in enumerate(("P", candidate_name, candidate_name, "P")):
                    stem = f"{candidate_name}_{name}_{position}_{mode}"
                    observed_boot, state = preflight(args.host)
                    assert observed_boot == boot
                    hashes = remote(
                        args.host,
                        "sha256sum " + " ".join(map(shlex.quote, checks)),
                    ).decode()
                    assert {line.split()[1]: line.split()[0] for line in hashes.splitlines()} == checks
                    (args.output / f"{stem}_preflight.txt").write_bytes(state + hashes.encode())
                    words = list(original)
                    if mode != "P":
                        for prefix, value in (
                            ("VTA_INSN_BUFFER_BYTES=", candidate["insn"]),
                            ("VTA_UOP_BUFFER_BYTES=", candidate["uop"]),
                        ):
                            index = next(i for i, word in enumerate(words) if word.startswith(prefix))
                            words[index] = prefix + str(value)
                        words.insert(0, "VTA_INSN_SUBMIT_THRESHOLD_BYTES=" + str(candidate["threshold"]))
                    words[words.index("--output-jsonl") + 1] = board + "/" + stem + ".jsonl"
                    shell = cd + " && env " + shlex.join(words)
                    (args.output / f"{stem}_command.sh").write_text(shell + "\n")
                    print("[RUN]", stem, flush=True)
                    result = subprocess.run(SSH + [args.host, shell], capture_output=True, timeout=300)
                    (args.output / f"{stem}.stdout").write_bytes(result.stdout)
                    (args.output / f"{stem}.stderr").write_bytes(result.stderr)
                    result.check_returncode()
                    assert b"[VTA_QUEUE]" not in result.stderr and b"[VTA_BOUNDARY]" not in result.stderr
                    assert b"source=pool-anchor" in result.stdout and b"fallback=" not in result.stdout
                    raw = remote(args.host, "cat " + shlex.quote(board + "/" + stem + ".jsonl"))
                    path = args.output / f"{stem}.jsonl"
                    path.write_bytes(raw)
                    rows = [json.loads(row) for row in raw.splitlines()]
                    assert all(
                        row["input_index"] == i % 8
                        and [output["fnv1a64"] for output in row["raw_outputs"]] == expected[i % 8]
                        for i, row in enumerate(rows)
                    )
                    item = {"candidate": candidate_name, "topology": name,
                            "position": position, "mode": mode,
                            "result_sha256": sha(path), **metrics(rows)}
                    paired.append(item)
                    report["runs_completed"].append(item)
                    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
                log_ratio = (
                    math.log(paired[1]["ii_ms"]) + math.log(paired[2]["ii_ms"])
                    - math.log(paired[0]["ii_ms"]) - math.log(paired[3]["ii_ms"])
                ) / 2
                block = {"candidate": candidate_name, "topology": name,
                         "log_ii_ratio": log_ratio,
                         "ii_change_pct": math.expm1(log_ratio) * 100}
                report["blocks_completed"].append(block)
                (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
                print("[BLOCK]", candidate_name, name, round(block["ii_change_pct"], 3), "%", flush=True)
        report["status"] = "complete_one_boot_prescreen"
    except Exception as error:
        report["status"] = "failed_stop_no_retry"
        report["error"] = repr(error)
        raise
    finally:
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    observed_boot, final = preflight(args.host)
    assert observed_boot == boot
    (args.output / "board_after.txt").write_bytes(final)


if __name__ == "__main__":
    main()
