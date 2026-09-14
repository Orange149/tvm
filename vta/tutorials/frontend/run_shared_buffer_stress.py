#!/usr/bin/env python3
"""Repeat qualified inputs for 1000 frames/topology; reuse frozen S1 binaries.

Byte references are the S1 external-K2 results for the same eight deterministic
inputs. This does not test 1000 distinct inputs, forced delays or abnormal exits.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

from audit_shared_buffer_storage import sha
from run_shared_buffer_qualification import archive_outputs, interleavings
from run_c3s_allocation_baseline import SSH, remote


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host", default="root@192.168.1.247")
    parser.add_argument("--runs", type=int, default=1000)
    args = parser.parse_args()
    assert args.runs >= 1000
    args.output.mkdir(parents=True, exist_ok=False)
    summary = json.loads((args.reference / "summary.json").read_text())
    preflight = remote(args.host, "cat /proc/sys/kernel/random/boot_id; "
                       "cat /sys/class/fpga_manager/fpga0/state; "
                       "cat /sys/class/u-dma-buf/udmabuf0/size; "
                       "cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq; "
                       "sha256sum /lib/firmware/vta_hpc.bit; ps -eo pid,args")
    (args.output / "board_preflight.txt").write_bytes(preflight)
    lines = preflight.decode().splitlines()
    assert lines[0] == summary["boot"]
    assert lines[1:4] == ["operating", "201326592", "1066666"]
    assert lines[4].split()[0] == "7bf1ac95b1182c670cd25241111f33725b4ea37ed6d66f2df37b0b2e7df528d6"
    assert not any(any(x in line for x in ("tvm_rpc", "vta_stage_pipeline_runner", "vta_stage_pair_runner"))
                   for line in lines[5:])
    board = "/media/sd-mmcblk1p2/c2buf_" + args.output.name
    remote(args.host, "mkdir " + shlex.quote(board))
    report = {"reference_sha256": sha(args.reference / "summary.json"), "boot": lines[0],
              "runs_per_topology": args.runs, "distinct_inputs": len(summary["inputs"]),
              "script_sha256": sha(__file__), "helper_sha256": sha(Path(__file__).with_name("run_shared_buffer_qualification.py")),
              "scope": "natural K2 stress, byte comparison; not forced-delay/exception or performance qualification",
              "results": []}
    (args.output / "preregistered.json").write_text(json.dumps(report, indent=2) + "\n")
    for item in summary["results"]:
        name = item["topology"]
        stored = (args.reference / f"{name}_anchor_command.sh").read_text().strip()
        cd, env_command = stored.split(" && env ", 1)
        words = shlex.split(env_command)
        runner_index = next(i for i, word in enumerate(words) if word.endswith("/vta_stage_pipeline_runner"))
        runner = words[runner_index]
        old_board = str(Path(runner).parent)
        package = shlex.split(cd)[1]
        expected = {row.split()[1]: row.split()[0] for row in
                    (args.reference / f"{name}_package_sha256.txt").read_text().splitlines()}
        actual = remote(args.host, cd + " && sha256sum " + " ".join(map(shlex.quote, expected))).decode()
        assert {r.split()[1]: r.split()[0] for r in actual.splitlines()} == expected
        checks = {old_board + "/" + file: digest for file, digest in summary["build"]["artifacts"].items()}
        checks.update({old_board + "/" + i["file"]: i["sha256"] for i in summary["inputs"]})
        for file in ("inputs.txt", f"{name}_plan.json"):
            checks[old_board + "/" + file] = sha(args.reference / file)
        hashes = remote(args.host, "sha256sum " + " ".join(map(shlex.quote, checks))).decode()
        assert {r.split()[1]: r.split()[0] for r in hashes.splitlines()} == checks
        (args.output / f"{name}_artifact_sha256.txt").write_text(actual + hashes)
        result_path, allocation_path, output_path = (board + f"/{name}.jsonl", board + f"/{name}_allocation.jsonl", board + f"/{name}_outputs")
        for flag, value in (("--runs", str(args.runs)), ("--output-jsonl", result_path),
                            ("--allocation-snapshot", allocation_path), ("--output-dump-dir", output_path)):
            words[words.index(flag) + 1] = value
        command = cd + " && env " + shlex.join(words)
        (args.output / f"{name}_command.sh").write_text(command + "\n")
        print("[RUN]", name, args.runs, "frames", flush=True)
        proc = subprocess.run(SSH + [args.host, command], capture_output=True, timeout=args.runs)
        (args.output / f"{name}.stdout").write_bytes(proc.stdout)
        (args.output / f"{name}.stderr").write_bytes(proc.stderr)
        proc.check_returncode()
        assert b"source=pool-anchor" in proc.stdout and b"fallback=" not in proc.stdout
        raw = remote(args.host, "cat " + shlex.quote(result_path))
        (args.output / f"{name}.jsonl").write_bytes(raw)
        rows = [json.loads(r) for r in raw.splitlines()]
        assert len(rows) == args.runs and [r["frame_id"] for r in rows] == list(range(args.runs))
        assert all(r["input_index"] == i % report["distinct_inputs"] for i, r in enumerate(rows))
        archive = remote(args.host, "tar -C " + shlex.quote(output_path) + " -cf - .")
        (args.output / f"{name}_outputs.tar").write_bytes(archive)
        outputs = archive_outputs(archive)
        references = archive_outputs((args.reference / f"{name}_external_outputs.tar").read_bytes())
        assert len(outputs) == args.runs
        total_bytes = 0
        for path, value in outputs.items():
            pieces = path.split("/")
            run_index = int(pieces[-2].removeprefix("run_"))
            reference_path = "/".join(pieces[:-2] + ["run_" + str(run_index % report["distinct_inputs"]), pieces[-1]])
            assert value == references[reference_path], (name, run_index)
            total_bytes += len(value)
        raw = remote(args.host, "cat " + shlex.quote(allocation_path))
        (args.output / f"{name}_allocation.jsonl").write_bytes(raw)
        phases = {r["phase"]: r for r in map(json.loads, raw.splitlines())}
        assert phases["after_steady_state"]["high_water_bytes"] == item["modes"]["anchor"]["high_water"]
        assert phases["after_warmup"]["high_water_bytes"] == phases["after_steady_state"]["high_water_bytes"]
        anchor_edges = [int(x) for x in item["modes"]["anchor"]["slot0_write_slot1_read_overlap_counts"]]
        record = {"topology": name, "frames": len(rows), "byte_exact": True, "compared_bytes": total_bytes,
                  "high_water": phases["after_steady_state"]["high_water_bytes"], "steady_growth": 0,
                  "slot0_write_slot1_read_overlap_counts": {str(e): interleavings(rows, e) for e in anchor_edges}}
        report["results"].append(record)
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print("[PASS]", name, record["compared_bytes"], "bytes exact", flush=True)
    post = remote(args.host, "cat /proc/sys/kernel/random/boot_id; ps -eo pid,args")
    (args.output / "board_after.txt").write_bytes(post)
    assert post.decode().splitlines()[0] == report["boot"]


if __name__ == "__main__":
    main()
