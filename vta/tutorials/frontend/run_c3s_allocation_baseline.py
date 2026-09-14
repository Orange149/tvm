#!/usr/bin/env python3
"""Freeze/check A-D, deploy isolated diagnostics, qualify external-K2 allocation.

This is NOT performance confirmation. All modes are fresh native processes.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

from audit_shared_buffer_storage import sha

SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-o",
       "HostKeyAlgorithms=+ssh-rsa", "-o", "PubkeyAcceptedAlgorithms=+ssh-rsa"]
BASE = Path(__file__).resolve().parent / "report_out/stage_tile_cotuning"
TRACE = BASE / "g0_discovery/boot_a5220e22_trace_run1"


def remote(target, command):
    return subprocess.run(SSH + [target, command], check=True, capture_output=True).stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host", default="root@192.168.1.247")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    build = json.loads((args.build / "build_manifest.json").read_text())
    for name, expected in build["artifacts"].items():
        assert sha(args.build / name) == expected
    preflight = remote(args.host, "cat /proc/sys/kernel/random/boot_id; "
                       "cat /sys/class/fpga_manager/fpga0/state; "
                       "cat /sys/class/u-dma-buf/udmabuf0/size; "
                       "cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq; "
                       "sha256sum /lib/firmware/vta_hpc.bit; ps -eo pid,args")
    (args.output / "board_preflight.txt").write_bytes(preflight)
    lines = preflight.decode().splitlines()
    assert lines[1:4] == ["operating", "201326592", "1066666"]
    assert lines[4].split()[0] == "7bf1ac95b1182c670cd25241111f33725b4ea37ed6d66f2df37b0b2e7df528d6"
    for line in lines[5:]:
        assert not any(x in line for x in ("tvm_rpc", "vta_stage_pipeline_runner", "vta_stage_pair_runner")), line
    remote_dir = "/media/sd-mmcblk1p2/c3s_" + args.output.name
    remote(args.host, "mkdir " + shlex.quote(remote_dir))  # refuse reuse of a prior result directory
    for name in build["artifacts"]:
        subprocess.run(["scp", *SSH[1:], str(args.build / name), args.host + ":" + remote_dir + "/"], check=True)
    summary = {"boot": lines[0], "scope": "external-K2 allocation/compatibility qualification, not FPS",
               "build": build, "script_sha256": sha(__file__), "results": []}
    (args.output / "preregistered.json").write_text(json.dumps({**summary,
        "topologies": {"A": 1, "B": 5, "C": 7, "D": 13}, "runs": 22,
        "warmups": 2, "modes": ["old", "new_frozen_driver", "new_diagnostic_driver"]}, indent=2))
    for topology, rank in (("A", 1), ("B", 5), ("C", 7), ("D", 13)):
        audit = json.loads((args.audit_dir / (topology + ".json")).read_text())
        pkg = f"/media/sd-mmcblk1p2/v1_p7_top20/legacy_profile_rank{rank:02d}"
        # Manifest may differ in deployment-only paths. Computation/runtime hashes must match.
        expected = {k: v for k, v in audit["package_sha256"].items() if k != "manifest.json"}
        actual = remote(args.host, "cd " + shlex.quote(pkg) + " && sha256sum " +
                        " ".join(shlex.quote(x) for x in expected)).decode()
        got = {line.split()[1]: line.split()[0] for line in actual.splitlines()}
        assert got == expected
        (args.output / (topology + "_package_sha256.txt")).write_text(actual)
        command_text = (TRACE / f"rank{rank:02d}/shared_command.sh").read_text()
        original = shlex.split(next(x for x in command_text.splitlines() if x.startswith("exec ")))[1:]
        original[original.index("--runs") + 1] = "22"
        expected_output = json.loads((TRACE / f"rank{rank:02d}/shared.jsonl").read_text().splitlines()[0])["raw_outputs"]
        expected_hashes = [o["fnv1a64"] for o in expected_output]
        for mode in ("old", "new_frozen_driver", "new_diagnostic_driver"):
            command = list(original)
            if mode != "old":
                command[0] = remote_dir + "/vta_stage_pipeline_runner"
            stem = topology + "_" + mode
            result_path = remote_dir + "/" + stem + ".jsonl"
            command[command.index("--output-jsonl") + 1] = result_path
            command += ["--warmup-runs", "2"]
            driver = remote_dir + "/libvta.so" if mode == "new_diagnostic_driver" else pkg + "/libvta.so"
            if mode == "new_diagnostic_driver":
                command += ["--allocation-snapshot", remote_dir + "/" + stem + "_allocation.jsonl"]
            env = {"LD_LIBRARY_PATH": pkg, "LD_PRELOAD": pkg + "/libtvm_runtime.so:" + driver,
                   "TVM_NUM_THREADS": "4", "TVM_THREAD_POOL_SPIN_COUNT": "0",
                   "AXU5EVB_DRIVER_POST_START_SLEEP_NS": "1000", "AXU5EVB_DRIVER_POLL_SLEEP_NS": "1000"}
            shell = "cd " + shlex.quote(pkg) + " && env " + " ".join(
                shlex.quote(k + "=" + v) for k, v in env.items()) + " " + shlex.join(command)
            (args.output / (stem + "_command.sh")).write_text(shell + "\n")
            print("[RUN]", topology, mode, flush=True)
            completed = subprocess.run(SSH + [args.host, shell], capture_output=True)
            (args.output / (stem + ".stdout")).write_bytes(completed.stdout)
            (args.output / (stem + ".stderr")).write_bytes(completed.stderr)
            completed.check_returncode()
            raw = remote(args.host, "cat " + shlex.quote(result_path))
            (args.output / (stem + ".jsonl")).write_bytes(raw)
            rows = [json.loads(x) for x in raw.splitlines()]
            assert len(rows) == 22
            assert all([o["fnv1a64"] for o in r["raw_outputs"]] == expected_hashes for r in rows)
            result = {"topology": topology, "mode": mode, "frames": len(rows), "outputs_exact_hash_match": True}
            if mode == "new_diagnostic_driver":
                raw = remote(args.host, "cat " + shlex.quote(remote_dir + "/" + stem + "_allocation.jsonl"))
                (args.output / (stem + "_allocation.jsonl")).write_bytes(raw)
                snapshots = [json.loads(x) for x in raw.splitlines()]
                phases = {s["phase"]: s for s in snapshots}
                assert len(phases) == 5
                assert phases["before_graph_load"]["high_water_bytes"] == 0
                assert phases["before_graph_load"]["initialized"] == 0
                graph = phases["graph_and_params_loaded"]["high_water_bytes"]
                slot = phases["slots_allocated"]["high_water_bytes"] - graph
                assert slot == audit["external_k2_bytes"], (slot, audit["external_k2_bytes"])
                result.update(snapshots=snapshots, graph_actual=graph,
                              graph_static=audit["graph_pool_aligned_bytes"], slot_actual=slot,
                              steady_growth=phases["after_steady_state"]["high_water_bytes"] - phases["after_warmup"]["high_water_bytes"])
            summary["results"].append(result)
            (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    assert remote(args.host, "cat /proc/sys/kernel/random/boot_id").decode().strip() == lines[0]
    print("[DONE] all outputs match; allocation results saved", flush=True)


if __name__ == "__main__":
    main()
