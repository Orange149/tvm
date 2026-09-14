#!/usr/bin/env python3
"""Q0/Q1 varied-input, fixed-submission board qualification. NOT a performance run.

Uses frozen S1 inputs/reference outputs, new isolated runtime and regenerated plans.
Never changes the bitstream or original packages; each case is a fresh process.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import shlex
import subprocess

from audit_shared_buffer_storage import sha
from make_shared_buffer_plan import make_plan
from run_c3s_allocation_baseline import SSH, TRACE, remote
from run_shared_buffer_qualification import archive_outputs


def queue_records(stderr):
    return [json.loads(line.split("[VTA_QUEUE] ", 1)[1])
            for line in stderr.decode().splitlines() if line.startswith("[VTA_QUEUE] ")]


def suggested_capacity(peak):
    return min(1 << 25, ((max(4096, peak * 2) + 255) // 256) * 256)


def batch_signature(records):
    return Counter((r["insn_bytes"], r["uop_bytes"], r["load_bytes"], r["store_bytes"], r["reason"])
                   for r in records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capacities", type=Path)
    parser.add_argument("--boundary-audit", action="store_true",
                        help="Read-only necessary-closure audit; never enables early submission")
    parser.add_argument("--host", default="root@192.168.1.247")
    parser.add_argument("--board-root", default="/media/sd-mmcblk1p2",
                        help="Fresh experiment directory parent; /tmp avoids a full SD card")
    args = parser.parse_args()
    base = Path(__file__).resolve().parent / "report_out/stage_tile_cotuning"
    ref = base / "c3s_buffer_reuse/s1_board_run1"
    reference_manifest = json.loads((ref / "summary.json").read_text())
    build = json.loads((args.build / "build_manifest.json").read_text())
    for name, expected in build["artifacts"].items():
        assert sha(args.build / name) == expected
    for path, expected in build["source_sha256"].items():
        assert sha(path) == expected, path
    capacities = json.loads(args.capacities.read_text()) if args.capacities else None
    args.output.mkdir(parents=True, exist_ok=False)
    preflight_command = ("cat /proc/sys/kernel/random/boot_id; cat /sys/class/fpga_manager/fpga0/state; "
        "cat /sys/class/u-dma-buf/udmabuf0/size; "
        "cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq; "
        "sha256sum /lib/firmware/vta_hpc.bit; ps -eo pid,args")
    preflight = remote(args.host, preflight_command)
    (args.output / "board_preflight.txt").write_bytes(preflight)
    lines = preflight.decode().splitlines()
    assert lines[1:4] == ["operating", "201326592", "1066666"]
    assert lines[4].split()[0] == "7bf1ac95b1182c670cd25241111f33725b4ea37ed6d66f2df37b0b2e7df528d6"
    assert not any(any(s in line for s in ("tvm_rpc", "vta_stage_pipeline_runner", "vta_stage_pair_runner"))
                   for line in lines[5:])
    board = args.board_root.rstrip("/") + "/queue_" + args.output.name
    remote(args.host, "mkdir " + shlex.quote(board))

    def upload(path):
        subprocess.run(["scp", *SSH[1:], str(path), args.host + ":" + board + "/"],
                       check=True, capture_output=True)

    for name in build["artifacts"]:
        upload(args.build / name)
    for item in reference_manifest["inputs"]:
        assert sha(ref / item["file"]) == item["sha256"]
        upload(ref / item["file"])
    upload(ref / "inputs.txt")
    summary = {"boot": lines[0], "build": build, "script_sha256": sha(__file__), "board": board,
               "planner_sha256": sha(Path(__file__).with_name("make_shared_buffer_plan.py")),
               "reference_sha256": sha(ref / "summary.json"), "capacities": capacities,
               "boundary_audit": args.boundary_audit,
               "inputs": reference_manifest["inputs"], "runs": 32, "warmups": 2,
               "scope": "byte-exact fixed-submission qualification; NOT performance", "results": []}
    (args.output / "preregistered.json").write_text(json.dumps(summary, indent=2) + "\n")
    peaks = {"insn": 0, "uop": 0}
    for topology, rank in zip("ABCD", (1, 5, 7, 13)):
        audit = json.loads((base / f"c3s_buffer_reuse/s0_static_run1/{topology}.json").read_text())
        plan = make_plan(audit, build)
        plan_path = args.output / f"{topology}_plan.json"
        plan_path.write_text(json.dumps(plan, indent=2) + "\n")
        upload(plan_path)
        pkg = f"/media/sd-mmcblk1p2/v1_p7_top20/legacy_profile_rank{rank:02d}"
        expected = {k: v for k, v in audit["package_sha256"].items() if k != "manifest.json"}
        actual = remote(args.host, "cd " + shlex.quote(pkg) + " && sha256sum " +
                        " ".join(shlex.quote(x) for x in expected)).decode()
        assert {r.split()[1]: r.split()[0] for r in actual.splitlines()} == expected
        (args.output / f"{topology}_package_sha256.txt").write_text(actual)
        original = shlex.split(next(line for line in
            (TRACE / f"rank{rank:02d}/shared_command.sh").read_text().splitlines() if line.startswith("exec ")))[1:]
        original[0] = board + "/vta_stage_pipeline_runner"
        original[original.index("--runs") + 1] = "32"
        index = original.index("--input")
        original[index:index+2] = ["--input-list", board + "/inputs.txt"]
        reference = archive_outputs((ref / f"{topology}_external_outputs.tar").read_bytes())
        results = {}
        for mode in ("default_off", "default_diag", "small_diag") if capacities else ("default_off", "default_diag"):
            stem = topology + "_" + mode
            assert remote(args.host, "cat /proc/sys/kernel/random/boot_id").decode().strip() == lines[0]
            env = {"LD_LIBRARY_PATH": pkg, "LD_PRELOAD": pkg + "/libtvm_runtime.so:" + board + "/libvta.so",
                   "TVM_NUM_THREADS": "4", "TVM_THREAD_POOL_SPIN_COUNT": "0",
                   "AXU5EVB_DRIVER_POST_START_SLEEP_NS": "1000", "AXU5EVB_DRIVER_POLL_SLEEP_NS": "1000",
                   "VTA_QUEUE_DIAGNOSTICS": "0" if mode == "default_off" else "1"}
            if args.boundary_audit:
                env["VTA_QUEUE_BOUNDARY_AUDIT"] = "0" if mode == "default_off" else "1"
            if mode == "small_diag":
                env.update(VTA_INSN_BUFFER_BYTES=str(capacities["insn"]),
                           VTA_UOP_BUFFER_BYTES=str(capacities["uop"]))
            command = list(original)
            command[command.index("--output-jsonl")+1] = board + "/" + stem + ".jsonl"
            command += ["--warmup-runs", "2", "--shared-buffer-plan", board + "/" + plan_path.name,
                        "--allocation-snapshot", board + "/" + stem + "_allocation.jsonl",
                        "--output-dump-dir", board + "/" + stem + "_outputs"]
            command[command.index("--vta-runtime-profile-dir")+1] = board + "/" + stem + "_profile"
            shell = "cd " + shlex.quote(pkg) + " && env " + " ".join(shlex.quote(k+"="+v) for k,v in env.items()) + " " + shlex.join(command)
            (args.output / f"{stem}_command.sh").write_text(shell + "\n")
            print("[RUN]", stem, flush=True)
            result = subprocess.run(SSH + [args.host, shell], capture_output=True, timeout=180)
            (args.output / f"{stem}.stdout").write_bytes(result.stdout)
            (args.output / f"{stem}.stderr").write_bytes(result.stderr)
            result.check_returncode()
            archive = remote(args.host, "tar -C " + shlex.quote(board + "/" + stem + "_outputs") + " -cf - .")
            (args.output / f"{stem}_outputs.tar").write_bytes(archive)
            assert archive_outputs(archive) == reference, stem + " byte mismatch"
            for suffix in (".jsonl", "_allocation.jsonl"):
                (args.output / (stem + suffix)).write_bytes(remote(args.host, "cat " + shlex.quote(board + "/" + stem + suffix)))
            profile = remote(args.host, "tar -C " + shlex.quote(board + "/" + stem + "_profile") + " -cf - .")
            (args.output / f"{stem}_profile.tar").write_bytes(profile)
            rows = [json.loads(r) for r in (args.output / f"{stem}.jsonl").read_text().splitlines()]
            assert len(rows) == 32 and all(r["input_index"] == i % 8 for i, r in enumerate(rows))
            snapshots = {r["phase"]: r for r in map(json.loads, (args.output / f"{stem}_allocation.jsonl").read_text().splitlines())}
            high = snapshots["after_steady_state"]["high_water_bytes"]
            assert high == snapshots["after_warmup"]["high_water_bytes"]
            records = queue_records(result.stderr)
            assert bool(records) == (mode != "default_off")
            assert all(r["timeout"] == 0 and r["reason"] == "explicit_sync" for r in records)
            assert not records or len({r["queue_id"] for r in records}) == 1
            for kind in peaks:
                peaks[kind] = max([peaks[kind]] + [r[kind + "_bytes"] for r in records])
            results[mode] = {"byte_exact": True, "frames": len(rows), "high_water": high,
                             "submits_including_warmup": len(records), "queue_records": records}
            if mode == "small_diag":
                # D has two islands; legal cross-frame scheduling changes global order.
                # Check multiplicities, plus per-frame exact outputs and profile counters.
                assert batch_signature(records) == batch_signature(results["default_diag"]["queue_records"])
                saved = 2 * (1 << 25) - capacities["insn"] - capacities["uop"]
                assert results["default_diag"]["high_water"] - high == saved
        assert results["default_off"]["high_water"] == results["default_diag"]["high_water"]
        summary["results"].append({"topology": topology, "modes": results})
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "suggested_capacities.json").write_text(json.dumps({k: suggested_capacity(v) for k,v in peaks.items()}, indent=2) + "\n")
    (args.output / "board_after.txt").write_bytes(remote(args.host, preflight_command))
    print("PASS", json.dumps(peaks), flush=True)


if __name__ == "__main__":
    main()
