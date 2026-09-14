#!/usr/bin/env python3
"""Qualify frozen T2656 allocation and 1000-frame correctness, not performance."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

from audit_shared_buffer_storage import sha
from run_c3s_allocation_baseline import SSH, remote
from run_shared_buffer_qualification import interleavings


INSN_BYTES = 5376
UOP_BYTES = 5632
THRESHOLD_BYTES = 2656


def preflight(host):
    command = (
        "cat /proc/sys/kernel/random/boot_id; cat /sys/class/fpga_manager/fpga0/state; "
        "cat /sys/class/u-dma-buf/udmabuf0/size; "
        "cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor; "
        "cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq; "
        "sha256sum /lib/firmware/vta_hpc.bit; ps -eo pid,args"
    )
    raw = remote(host, command)
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
    parser.add_argument("--reference", type=Path, required=True,
                        help="Frozen Q1 directory with allocation/reference outputs")
    parser.add_argument("--qualification", type=Path, required=True,
                        help="Q2 threshold qualification directory with regenerated plans")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host", default="root@192.168.1.247")
    parser.add_argument("--runs", type=int, default=1000)
    args = parser.parse_args()
    assert args.runs >= 1000

    build = json.loads((args.build / "build_manifest.json").read_text())
    for name, digest in build["artifacts"].items():
        assert sha(args.build / name) == digest
    for path, digest in build["source_sha256"].items():
        assert sha(path) == digest
    q1 = json.loads((args.reference / "reconciled_summary.json").read_text())
    threshold = json.loads((args.qualification / "summary.json").read_text())
    assert threshold["status"] == "complete_correctness_screen"
    qualified = [r for r in threshold["results"] if r["threshold"] == THRESHOLD_BYTES]
    assert len(qualified) == 4 and all(r["byte_exact"] for r in qualified)
    assert max(r["insn_peak"] for r in qualified) * 2 <= INSN_BYTES
    assert max(r["uop_peak"] for r in qualified) * 2 <= UOP_BYTES

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
        "script_sha256": sha(__file__),
        "board": board,
        "candidate": {"threshold": THRESHOLD_BYTES, "insn": INSN_BYTES,
                      "uop": UOP_BYTES, "total": INSN_BYTES + UOP_BYTES},
        "allocation_runs": 32,
        "stress_runs": args.runs,
        "warmup_runs": 2,
        "scope": "allocation and natural varied-input correctness; NOT performance",
        "status": "running",
        "results": [],
    }
    (args.output / "preregistered.json").write_text(json.dumps(report, indent=2) + "\n")

    try:
        for name, rank in zip("ABCD", (1, 5, 7, 13)):
            command = (args.reference / f"{name}_small_diag_command.sh").read_text().strip()
            cd, env_command = command.split(" && env ", 1)
            words = shlex.split(env_command)
            runner = next(i for i, word in enumerate(words) if word.endswith("/vta_stage_pipeline_runner"))
            words[runner] = board + "/vta_stage_pipeline_runner"
            words[words.index("--input-list") + 1] = board + "/inputs.txt"
            words[words.index("--shared-buffer-plan") + 1] = board + f"/{name}_plan.json"
            words[words.index("--vta-runtime-profile-dir") + 1] = ""
            for flag in ("--allocation-snapshot", "--output-dump-dir"):
                index = words.index(flag)
                del words[index:index + 2]
            words = [word for word in words if not word.startswith((
                "VTA_QUEUE_DIAGNOSTICS=", "VTA_QUEUE_BOUNDARY_AUDIT=",
                "VTA_INSN_BUFFER_BYTES=", "VTA_UOP_BUFFER_BYTES=",
                "VTA_INSN_SUBMIT_THRESHOLD_BYTES=",
            ))]
            words[0:0] = [
                "VTA_QUEUE_DIAGNOSTICS=0",
                "VTA_QUEUE_BOUNDARY_AUDIT=0",
                f"VTA_INSN_BUFFER_BYTES={INSN_BYTES}",
                f"VTA_UOP_BUFFER_BYTES={UOP_BYTES}",
                f"VTA_INSN_SUBMIT_THRESHOLD_BYTES={THRESHOLD_BYTES}",
            ]
            preload = next(i for i, word in enumerate(words) if word.startswith("LD_PRELOAD="))
            tvm_runtime = words[preload].split("=", 1)[1].split(":", 1)[0]
            words[preload] = "LD_PRELOAD=" + tvm_runtime + ":" + board + "/libvta.so"

            package_dir = shlex.split(cd)[1]
            package_hashes = {
                line.split()[1]: line.split()[0]
                for line in (args.reference / f"{name}_package_sha256.txt").read_text().splitlines()
            }
            checks = {package_dir + "/" + path: digest for path, digest in package_hashes.items()}
            checks.update({board + "/" + n: d for n, d in build["artifacts"].items()})
            checks[board + f"/{name}_plan.json"] = sha(args.qualification / f"{name}_plan.json")
            checks[board + "/inputs.txt"] = sha(s1 / "inputs.txt")
            for item in q1["inputs"]:
                checks[board + "/" + item["file"]] = item["sha256"]
            actual = remote(args.host, "sha256sum " + " ".join(map(shlex.quote, checks))).decode()
            assert {line.split()[1]: line.split()[0] for line in actual.splitlines()} == checks
            expected_rows = [
                json.loads(row)
                for row in (args.reference / f"{name}_default_off.jsonl").read_text().splitlines()
            ]
            expected = {
                row["input_index"]: [output["fnv1a64"] for output in row["raw_outputs"]]
                for row in expected_rows
            }
            q1_item = next(item for item in q1["results"] if item["topology"] == name)

            topology_result = {"topology": name, "cases": {}}
            report["results"].append(topology_result)
            for case, runs in (("allocation", 32), ("stress", args.runs)):
                observed_boot, state = preflight(args.host)
                assert observed_boot == boot
                stem = name + "_" + case
                current = list(words)
                current[current.index("--runs") + 1] = str(runs)
                current[current.index("--warmup-runs") + 1] = "2"
                current[current.index("--output-jsonl") + 1] = board + "/" + stem + ".jsonl"
                if case == "allocation":
                    current += ["--allocation-snapshot", board + "/" + stem + "_allocation.jsonl"]
                shell = cd + " && env " + shlex.join(current)
                (args.output / f"{stem}_command.sh").write_text(shell + "\n")
                (args.output / f"{stem}_preflight.txt").write_bytes(state + actual.encode())
                print("[RUN]", stem, flush=True)
                proc = subprocess.run(
                    SSH + [args.host, shell], capture_output=True,
                    timeout=max(300, runs),
                )
                (args.output / f"{stem}.stdout").write_bytes(proc.stdout)
                (args.output / f"{stem}.stderr").write_bytes(proc.stderr)
                proc.check_returncode()
                assert b"[VTA_QUEUE]" not in proc.stderr and b"[VTA_BOUNDARY]" not in proc.stderr
                assert b"source=pool-anchor" in proc.stdout and b"fallback=" not in proc.stdout
                raw = remote(args.host, "cat " + shlex.quote(board + "/" + stem + ".jsonl"))
                (args.output / f"{stem}.jsonl").write_bytes(raw)
                rows = [json.loads(row) for row in raw.splitlines()]
                assert len(rows) == runs
                assert [row["frame_id"] for row in rows] == list(range(runs))
                assert all(
                    row["input_index"] == i % 8
                    and [output["fnv1a64"] for output in row["raw_outputs"]] == expected[i % 8]
                    for i, row in enumerate(rows)
                )
                item = {"frames": runs, "byte_exact": True}
                if case == "allocation":
                    allocation = remote(
                        args.host,
                        "cat " + shlex.quote(board + "/" + stem + "_allocation.jsonl"),
                    )
                    (args.output / f"{stem}_allocation.jsonl").write_bytes(allocation)
                    phases = {row["phase"]: row for row in map(json.loads, allocation.splitlines())}
                    high = phases["after_steady_state"]["high_water_bytes"]
                    assert high == phases["after_warmup"]["high_water_bytes"]
                    expected_high = q1_item["modes"]["small_diag"]["high_water"] - 5376
                    assert high == expected_high, (name, high, expected_high)
                    item.update({"high_water": high, "expected_high_water": expected_high,
                                 "steady_growth": 0,
                                 "additional_saved_vs_q1_bytes": 5376})
                else:
                    plan = json.loads((args.qualification / f"{name}_plan.json").read_text())
                    item["overlaps"] = {
                        str(edge["producer_stage"]): interleavings(rows, edge["producer_stage"])
                        for edge in plan["edges"] if edge["source"] == "pool-anchor"
                    }
                topology_result["cases"][case] = item
                (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
                print("[PASS]", stem, flush=True)
            (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        report["status"] = "complete_final_candidate_qualification"
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
