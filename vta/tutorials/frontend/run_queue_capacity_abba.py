#!/usr/bin/env python3
"""One independent boot of preregistered E/P/F performance protection.

Frozen Q1 binary, no queue diagnostic/dumps/allocation profiler. Five ABBA blocks
per topology per contrast. No reboot, deletion, retry, or parameter search.
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


def metrics(rows):
    assert len(rows) == 302
    assert [r["frame_id"] for r in rows] == list(range(302))
    steady = rows[2:]
    completion = np.asarray([r["completion_ms"] for r in steady])
    assert np.all(np.diff(completion) > 0)
    ii = float(np.mean(np.diff(completion)))
    return {"ii_ms": ii, "fps": 1000 / ii,
            "submit_to_complete_p95_ms": float(np.percentile([r["total_latency_ms"] for r in steady], 95)),
            "frames": 302, "scored_frames": 300, "scored_intervals": 299}


def preflight(host):
    raw = remote(host, "cat /proc/sys/kernel/random/boot_id; cat /sys/class/fpga_manager/fpga0/state; "
        "cat /sys/class/u-dma-buf/udmabuf0/size; "
        "cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor; "
        "cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq; "
        "sha256sum /lib/firmware/vta_hpc.bit; ps -eo pid,args")
    lines = raw.decode().splitlines()
    assert lines[1:5] == ["operating", "201326592", "userspace", "1066666"]
    assert lines[5].split()[0] == "7bf1ac95b1182c670cd25241111f33725b4ea37ed6d66f2df37b0b2e7df528d6"
    for line in lines[6:]:
        fields = line.split()
        if len(fields) > 1:
            assert Path(fields[1]).name not in ("vta_stage_pipeline_runner", "vta_stage_pair_runner", "tvm_rpc"), line
    return lines[0], raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host", default="root@192.168.1.247")
    parser.add_argument("--blocks", type=int, default=5, choices=[5])
    args = parser.parse_args()
    q1 = json.loads((args.reference / "reconciled_summary.json").read_text())
    assert q1["capacities"] == {"insn": 10752, "uop": 5632}
    assert q1["build"]["artifacts"]["libvta.so"] == "8950c9ef85d88f6c1c5ca8b2304d0d874288c24bdcbb1583fa60709629dd2689"
    boot, initial = preflight(args.host)
    args.output.mkdir(parents=True, exist_ok=False)
    board = "/tmp/queue_" + args.output.name
    remote(args.host, "mkdir " + shlex.quote(board))
    (args.output / "board_preflight.txt").write_bytes(initial)
    report = {"boot": boot, "reference_sha256": sha(args.reference / "reconciled_summary.json"),
        "script_sha256": sha(__file__), "board": board, "build": q1["build"],
        "F": {"insn": 10752, "uop": 5632, "submission_policy": "unchanged"},
        "modes": {"E": "external/default", "P": "anchor/default", "F": "anchor/16KiB"},
        "blocks_per_topology_per_contrast": 5, "runs": 302, "discard_first": 2,
        "extra_warmup_runs": 0, "contrasts": ["EP", "PF"], "log_destination": "tmpfs (all modes)",
        "scope": "one boot only; no cross-boot confidence conclusion", "status": "running",
        "runs_completed": [], "blocks_completed": []}
    (args.output / "preregistered.json").write_text(json.dumps(report, indent=2) + "\n")
    prepared = {}
    for name in "ABCD":
        stored = (args.reference / f"{name}_small_diag_command.sh").read_text().strip()
        cd, command = stored.split(" && env ", 1)
        words = shlex.split(command)
        runner_index = next(i for i, w in enumerate(words) if w.endswith("/vta_stage_pipeline_runner"))
        old_board = str(Path(words[runner_index]).parent)
        checks = {old_board + "/" + f: s for f,s in q1["build"]["artifacts"].items()}
        checks.update({old_board + "/" + i["file"]: i["sha256"] for i in q1["inputs"]})
        checks[old_board + f"/{name}_plan.json"] = sha(args.reference / f"{name}_plan.json")
        input_list = args.reference.parents[1] / "c3s_buffer_reuse/s1_board_run1/inputs.txt"
        checks[old_board + "/inputs.txt"] = sha(input_list)
        package = {line.split()[1]: line.split()[0] for line in
                   (args.reference / f"{name}_package_sha256.txt").read_text().splitlines()}
        # Absolute paths allow checking actual libraries/graphs/parameters before every run.
        checks.update({shlex.split(cd)[1] + "/" + path: digest for path,digest in package.items()})
        for flag in ("--allocation-snapshot", "--output-dump-dir"):
            i = words.index(flag)
            del words[i:i+2]
        for flag, value in (("--runs", "302"), ("--warmup-runs", "0"), ("--vta-runtime-profile-dir", "")):
            words[words.index(flag)+1] = value
        words[words.index("VTA_QUEUE_DIAGNOSTICS=1")] = "VTA_QUEUE_DIAGNOSTICS=0"
        words.insert(0, "VTA_QUEUE_BOUNDARY_AUDIT=0")
        reference_rows = [json.loads(r) for r in (args.reference / f"{name}_default_off.jsonl").read_text().splitlines()]
        expected = {r["input_index"]: [o["fnv1a64"] for o in r["raw_outputs"]] for r in reference_rows}
        prepared[name] = (cd, words, checks, expected)
    try:
        for block in range(5):
            names = "ABCD"[block % 4:] + "ABCD"[:block % 4]
            for name in names:
                cd, original, checks, expected = prepared[name]
                for contrast in (["EP", "PF"] if block % 2 == 0 else ["PF", "EP"]):
                    paired = []
                    for position, mode in enumerate((contrast[0], contrast[1], contrast[1], contrast[0])):
                        stem = f"block{block}_{name}_{contrast}_{position}_{mode}"
                        observed_boot, state = preflight(args.host)
                        assert observed_boot == boot, "boot changed; do not merge independent boots"
                        hashes = remote(args.host, "sha256sum " + " ".join(map(shlex.quote, checks))).decode()
                        assert {r.split()[1]: r.split()[0] for r in hashes.splitlines()} == checks
                        (args.output / f"{stem}_preflight.txt").write_bytes(state + hashes.encode())
                        words = list(original)
                        if mode == "E":
                            i = words.index("--shared-buffer-plan")
                            del words[i:i+2]
                        if mode != "F":
                            words = [w for w in words if not w.startswith(("VTA_INSN_BUFFER_BYTES=", "VTA_UOP_BUFFER_BYTES="))]
                            words[0:0] = ["VTA_INSN_BUFFER_BYTES=33554432", "VTA_UOP_BUFFER_BYTES=33554432"]
                        words[words.index("--output-jsonl")+1] = board + "/" + stem + ".jsonl"
                        shell = cd + " && env " + shlex.join(words)
                        (args.output / f"{stem}_command.sh").write_text(shell + "\n")
                        print("[RUN]", stem, flush=True)
                        result = subprocess.run(SSH + [args.host, shell], capture_output=True, timeout=300)
                        (args.output / f"{stem}.stdout").write_bytes(result.stdout)
                        (args.output / f"{stem}.stderr").write_bytes(result.stderr)
                        result.check_returncode()
                        assert b"[VTA_QUEUE]" not in result.stderr and b"[VTA_BOUNDARY]" not in result.stderr
                        if mode != "E":
                            assert b"source=pool-anchor" in result.stdout and b"fallback=" not in result.stdout
                        raw = remote(args.host, "cat " + shlex.quote(board + "/" + stem + ".jsonl"))
                        path = args.output / (stem + ".jsonl")
                        path.write_bytes(raw)
                        rows = [json.loads(r) for r in raw.splitlines()]
                        assert all(r["input_index"] == i % 8 and [o["fnv1a64"] for o in r["raw_outputs"]] == expected[i % 8]
                                   for i,r in enumerate(rows))
                        item = {"block": block, "topology": name, "contrast": contrast, "position": position,
                                "mode": mode, "result_sha256": sha(path), **metrics(rows)}
                        paired.append(item)
                        report["runs_completed"].append(item)
                        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
                    log_ratio = (math.log(paired[1]["ii_ms"]) + math.log(paired[2]["ii_ms"]) -
                                 math.log(paired[0]["ii_ms"]) - math.log(paired[3]["ii_ms"])) / 2
                    report["blocks_completed"].append({"block": block, "topology": name, "contrast": contrast,
                        "log_ii_ratio": log_ratio, "ii_change_pct": math.expm1(log_ratio) * 100})
                    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
                    print("[BLOCK]", name, contrast, block, round(math.expm1(log_ratio)*100, 3), "%", flush=True)
        final_boot, final_state = preflight(args.host)
        assert final_boot == boot
        (args.output / "board_after.txt").write_bytes(final_state)
        report["status"] = "complete_one_boot"
    except Exception as error:
        report["status"] = "failed_no_retry"
        report["error"] = repr(error)
        raise
    finally:
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
