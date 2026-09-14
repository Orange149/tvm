#!/usr/bin/env python3
"""C2 buffer reuse: same-binary on/off, actual allocation and per-frame byte checks.

This is qualification, not an ABBA performance experiment. No board reboot or
replacement of frozen runtime/packages. Each mode is a fresh native process.
"""
import argparse
import copy
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

import numpy as np
from audit_shared_buffer_storage import sha
from make_shared_buffer_plan import make_plan
from run_c3s_allocation_baseline import SSH, TRACE, remote


def archive_outputs(raw):
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        return {m.name: archive.extractfile(m).read() for m in archive.getmembers() if m.isfile()}


def interleavings(rows, edge):
    """Count observed producer writes slot0 overlapping consumer reads slot1."""
    consumers = [r for r in rows if r["p8_boundaries"][edge]["slot_id"] == 1]
    count = 0
    for p in rows:
        if p["p8_boundaries"][edge]["slot_id"] != 0:
            continue
        for c in consumers:
            if max(p[f"stage{edge}_run_start_ms"], c[f"stage{edge+1}_run_start_ms"]) < min(
                    p[f"stage{edge}_run_end_ms"], c[f"stage{edge+1}_run_end_ms"]):
                count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host", default="root@192.168.1.247")
    parser.add_argument("--runs", type=int, default=32)
    parser.add_argument("--topologies", default="ABCD")
    args = parser.parse_args()
    assert args.runs >= 8 and args.topologies and set(args.topologies) <= set("ABCD")
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
    assert not any(any(x in line for x in ("tvm_rpc", "vta_stage_pipeline_runner", "vta_stage_pair_runner"))
                   for line in lines[5:])
    board = "/media/sd-mmcblk1p2/c2buf_" + args.output.name
    remote(args.host, "mkdir " + shlex.quote(board))

    def upload(path):
        subprocess.run(["scp", *SSH[1:], str(path), args.host + ":" + board + "/"], check=True,
                       capture_output=True)
    for name in build["artifacts"]:
        upload(args.build / name)
    shape = (1, 3, 224, 224)
    cases = [np.zeros(shape, dtype="float32"),
             np.random.default_rng(260908).uniform(-1, 1, shape).astype("float32"),
             np.random.default_rng(260909).normal(0, 2, shape).astype("float32")]
    cases += [np.random.default_rng(9910+i).normal(0, 1, shape).astype("float32") for i in range(5)]
    inputs = []
    for i, case in enumerate(cases):
        path = args.output / f"input{i}.bin"
        path.write_bytes(case.astype("<f4").tobytes())
        upload(path)
        inputs.append({"file": path.name, "sha256": sha(path)})
    input_list = args.output / "inputs.txt"
    input_list.write_text("".join(i["file"] + "\n" for i in inputs))
    upload(input_list)
    summary = {"boot": lines[0], "build": build, "inputs": inputs,
               "runs": args.runs, "warmups": 2, "topologies": args.topologies,
               "script_sha256": sha(__file__), "planner_sha256": sha(Path(__file__).with_name("make_shared_buffer_plan.py")),
               "scope": "allocation and varied-input output qualification; not FPS or full stress qualification",
               "results": []}
    (args.output / "preregistered.json").write_text(json.dumps(summary, indent=2) + "\n")
    for name in args.topologies:
        rank = {"A": 1, "B": 5, "C": 7, "D": 13}[name]
        audit = json.loads((args.audit_dir / f"{name}.json").read_text())
        plan = make_plan(audit, build)
        pkg = f"/media/sd-mmcblk1p2/v1_p7_top20/legacy_profile_rank{rank:02d}"
        expected = {k: v for k, v in audit["package_sha256"].items() if k != "manifest.json"}
        actual = remote(args.host, "cd " + shlex.quote(pkg) + " && sha256sum " +
                        " ".join(shlex.quote(x) for x in expected)).decode()
        assert {r.split()[1]: r.split()[0] for r in actual.splitlines()} == expected
        (args.output / f"{name}_package_sha256.txt").write_text(actual)
        plan_path = args.output / f"{name}_plan.json"
        plan_path.write_text(json.dumps(plan, indent=2) + "\n")
        upload(plan_path)
        original = shlex.split(next(line for line in
            (TRACE / f"rank{rank:02d}/shared_command.sh").read_text().splitlines() if line.startswith("exec ")))[1:]
        original[0] = board + "/vta_stage_pipeline_runner"
        original[original.index("--runs") + 1] = str(args.runs)
        ii = original.index("--input")
        original[ii:ii+2] = ["--input-list", board + "/inputs.txt"]
        env = {"LD_LIBRARY_PATH": pkg, "LD_PRELOAD": pkg + "/libtvm_runtime.so:" + board + "/libvta.so",
               "TVM_NUM_THREADS": "4", "TVM_THREAD_POOL_SPIN_COUNT": "0",
               "AXU5EVB_DRIVER_POST_START_SLEEP_NS": "1000", "AXU5EVB_DRIVER_POLL_SLEEP_NS": "1000"}
        prefix = "cd " + shlex.quote(pkg) + " && env " + " ".join(shlex.quote(k+"="+v) for k, v in env.items())
        records, reference = {}, None
        # One topology also tests runtime rule fallback and fatal hash mismatch.
        modes = ["external", "anchor"] + (["bad_entry", "bad_hash"] if name == "A" else [])
        for mode in modes:
            stem = name + "_" + mode
            command = list(original)
            command[command.index("--output-jsonl") + 1] = board + "/" + stem + ".jsonl"
            command += ["--warmup-runs", "2", "--allocation-snapshot", board + "/" + stem + "_allocation.jsonl",
                        "--output-dump-dir", board + "/" + stem + "_outputs"]
            if mode != "external":
                target_plan = plan_path.name
                if mode in ("bad_entry", "bad_hash"):
                    bad = copy.deepcopy(plan)
                    if mode == "bad_entry": bad["edges"][0]["tensors"][0]["entry"] = 999999
                    else: bad["stages"][0]["graph_sha256"] = "0" * 64
                    path = args.output / f"{stem}_plan.json"
                    path.write_text(json.dumps(bad, indent=2) + "\n")
                    upload(path)
                    target_plan = path.name
                command += ["--shared-buffer-plan", board + "/" + target_plan]
            shell = prefix + " " + shlex.join(command)
            (args.output / f"{stem}_command.sh").write_text(shell + "\n")
            print("[RUN]", stem, flush=True)
            result = subprocess.run(SSH + [args.host, shell], capture_output=True, timeout=max(180, args.runs))
            (args.output / f"{stem}.stdout").write_bytes(result.stdout)
            (args.output / f"{stem}.stderr").write_bytes(result.stderr)
            if mode == "bad_hash":
                assert result.returncode != 0 and b"hash mismatch" in result.stderr and b"[LOAD]" not in result.stdout
                records[mode] = {"rejected_before_graph_load": True}
                continue
            result.check_returncode()
            raw = remote(args.host, "cat " + shlex.quote(board + "/" + stem + ".jsonl"))
            (args.output / f"{stem}.jsonl").write_bytes(raw)
            rows = [json.loads(r) for r in raw.splitlines()]
            assert len(rows) == args.runs and [r["frame_id"] for r in rows] == list(range(args.runs))
            assert all(r["input_index"] == i % len(cases) for i, r in enumerate(rows))
            archive = remote(args.host, "tar -C " + shlex.quote(board + "/" + stem + "_outputs") + " -cf - .")
            (args.output / f"{stem}_outputs.tar").write_bytes(archive)
            values = archive_outputs(archive)
            assert len(values) == args.runs
            if reference is None: reference = values
            else: assert values == reference, "per-frame byte comparison failed"
            raw = remote(args.host, "cat " + shlex.quote(board + "/" + stem + "_allocation.jsonl"))
            (args.output / f"{stem}_allocation.jsonl").write_bytes(raw)
            phases = {r["phase"]: r for r in map(json.loads, raw.splitlines())}
            assert len(phases) == 5
            assert phases["graph_and_params_loaded"]["high_water_bytes"] == audit["graph_pool_aligned_bytes"]
            saved = plan["predicted_saved_bytes"] if mode == "anchor" else 0
            expected_slot = audit["external_k2_bytes"] - saved
            slot_actual = phases["slots_allocated"]["high_water_bytes"] - phases["graph_and_params_loaded"]["high_water_bytes"]
            assert slot_actual == expected_slot, (name, mode, slot_actual, expected_slot, result.stdout.decode())
            assert phases["after_warmup"]["high_water_bytes"] == phases["after_steady_state"]["high_water_bytes"]
            if mode == "anchor": assert b"source=pool-anchor" in result.stdout and b"fallback=" not in result.stdout
            if mode == "bad_entry": assert b"fallback=" in result.stdout
            records[mode] = {"frames": len(rows), "byte_exact": True, "compared_bytes": sum(map(len, values.values())),
                             "slot_actual": slot_actual, "high_water": phases["after_steady_state"]["high_water_bytes"],
                             "steady_growth": 0, "slot0_write_slot1_read_overlap_counts": {
                                 str(e["producer_stage"]): interleavings(rows, e["producer_stage"])
                                 for e in plan["edges"] if e["source"] == "pool-anchor"}}
        assert records["external"]["high_water"] - records["anchor"]["high_water"] == plan["predicted_saved_bytes"]
        summary["results"].append({"topology": name, "predicted_saved_bytes": plan["predicted_saved_bytes"], "modes": records})
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    post = remote(args.host, "cat /proc/sys/kernel/random/boot_id; ps -eo pid,args")
    (args.output / "board_after.txt").write_bytes(post)
    assert post.decode().splitlines()[0] == lines[0]
    print("[DONE] qualification passed; no performance claim", flush=True)


if __name__ == "__main__":
    main()
