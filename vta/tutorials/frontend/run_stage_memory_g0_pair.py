#!/usr/bin/env python3
"""G0 harness qualification on A stage0/stage1, not the 9-pair confirmation study."""
import argparse
import datetime
import json
from pathlib import Path
import shlex
import statistics

from run_stage_memory_g0_discovery import BASE, percentile, remote, save, sha


def analyze(rows):
    assert [r["sample"] for r in rows] == list(range(len(rows)))
    scored = [r for r in rows if not r["warmup"]]
    blocks = []
    for i in range(0, len(scored), 4):
        group = scored[i:i+4]
        assert [r["policy"] for r in group] == ["allow", "wait", "wait", "allow"]
        allow = statistics.mean([group[0]["makespan_ms"], group[3]["makespan_ms"]])
        wait = statistics.mean([group[1]["makespan_ms"], group[2]["makespan_ms"]])
        blocks.append({"block": group[0]["block"], "allow_ms": allow, "wait_ms": wait,
                       "wait_relative_improvement": 1 - wait / allow})
    for row in rows:
        a, b = row["sides"]
        assert row["outputs_match_reference"]
        for side in (a, b):
            assert side["binding_ms"] <= side["release_ms"] <= side["start_ms"] <= side["end_ms"]
        assert max(a["binding_ms"], b["binding_ms"]) <= a["release_ms"]
        assert row["eligible_ms"] >= a["start_ms"] + row["offset_ms"] - .00001
        assert row["eligible_ms"] <= b["release_ms"]
        if row["policy"] == "wait":
            assert a["end_ms"] <= b["release_ms"]
        expected = max(a["end_ms"], b["end_ms"]) - a["start_ms"]
        assert abs(expected - row["makespan_ms"]) < .00001
    return {"samples": len(rows), "scored_samples": len(scored), "blocks": blocks,
            "allow_makespan_median_ms": statistics.median(r["makespan_ms"] for r in scored if r["policy"] == "allow"),
            "wait_makespan_median_ms": statistics.median(r["makespan_ms"] for r in scored if r["policy"] == "wait"),
            "paired_wait_improvement_median": statistics.median(b["wait_relative_improvement"] for b in blocks),
            "active_at_eligible_count": sum(r["active_at_eligible"] for r in scored),
            "actual_active_age_median_ms": statistics.median(r["eligible_ms"]-r["sides"][0]["start_ms"] for r in scored),
            "release_to_start_p95_ms": percentile([s["start_ms"]-s["release_ms"] for r in scored for s in r["sides"]], .95),
            "hook_protocol_audit": "pass", "action_label": "unclassified",
            "scope": "single-boot harness qualification; not protect/allow confirmation or pipeline FPS"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host", default="192.168.1.247")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    target = "root@" + args.host
    runner = "/media/sd-mmcblk1p2/vta_stage_pair_runner_g0"
    boot = remote(target, "cat /proc/sys/kernel/random/boot_id").decode().strip()
    assert not remote(target, "pidof tvm_rpc vta_stage_pipeline_runner "
                      "vta_stage_pipeline_runner_g0_a5220e22 vta_stage_pair_runner_g0 || true").strip()
    probe = remote(target, "cat /sys/class/fpga_manager/fpga0/state; "
                   "cat /sys/class/u-dma-buf/udmabuf0/size; "
                   "cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq; "
                   "sha256sum /lib/firmware/vta_hpc.bit " + runner)
    (args.output / "board_preflight.txt").write_bytes(probe)
    assert probe.decode().startswith("operating\n201326592\n1066666\n")
    assert "7bf1ac95b1182c670cd25241111f33725b4ea37ed6d66f2df37b0b2e7df528d6" in probe.decode()
    pkg = "/media/sd-mmcblk1p2/v1_p7_top20/legacy_profile_rank01"
    frozen = BASE / "g0_discovery/boot_a5220e22_trace_run1/rank01"
    checks = (frozen / "artifact_sha256.txt").read_text().splitlines()
    paths = [line.split()[1] for line in checks]
    actual = remote(target, "cd " + pkg + "; sha256sum " + shlex.join(paths)).decode().splitlines()
    assert actual == checks
    (args.output / "artifact_sha256.txt").write_text("\n".join(actual) + "\n")
    root = "/media/sd-mmcblk1p2/" + args.output.name
    remote(target, "mkdir " + shlex.quote(root))
    trace_path = BASE / "g0_discovery/boot_a5220e22_trace_run1/readiness_audit.json"
    cells = [{"active": 0, "ready": 1, "offset_ms": 24.78},
             {"active": 1, "ready": 0, "offset_ms": .38}]
    prereg = {"purpose": "qualify pre-run hooks using one real pair in both natural directions",
              "selection": "A stage0/stage1; first upstream CPU and first VTA, no concurrent outcome selection",
              "offset_source": "rounded median active age from existing A natural readiness trace",
              "trace_sha256": sha(trace_path.read_bytes()), "cells": cells,
              "blocks": 10, "warmups_per_action": 5, "order": "ABBA = allow,wait,wait,allow",
              "host_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "boot_id": boot,
              "script_sha256": sha(Path(__file__).read_bytes()),
              "runner_sha256": probe.decode().splitlines()[-1].split()[0],
              "vta_host_affinity": "3", "cpu_threads_affinity": "frozen stage0: t4, 0,1,2,3",
              "protocol": "both inputs bound while idle; release active; mark pending eligible at frozen offset; "
                          "allow now or wait for active graph completion; defer all get-output until both graph runs end",
              "limits": "not slot-handoff experiment; no new tuning or production controller; "
                        "not isolated-service/dose-response/9-pair or multi-boot confirmation"}
    save(args.output / "preregistered.json", prereg)
    report = {"prereg_sha256": sha((args.output / "preregistered.json").read_bytes()), "results": []}
    source = (frozen / "copy_command.sh").read_text()
    prefix, invocation = source.split("exec ", 1)
    for cell in cells:
        name = f"active{cell['active']}_ready{cell['ready']}"
        argv = shlex.split(invocation)
        argv[0] = runner
        output = root + "/" + name + ".jsonl"
        for flag, value in [("--runs", "10"), ("--output-jsonl", output)]:
            argv[argv.index(flag)+1] = value
        argv += ["--serial", "--warmup-runs", "5", "--pair-active", str(cell["active"]),
                 "--pair-ready", str(cell["ready"]), "--pair-offset-ms", str(cell["offset_ms"])]
        command = prefix + "exec " + shlex.join(argv)
        (args.output / (name + "_command.sh")).write_text(command + "\n")
        sensor = "cat /sys/bus/iio/devices/iio:device0/in_temp0_ps_temp_raw " \
                 "/sys/bus/iio/devices/iio:device0/in_temp2_pl_temp_raw"
        (args.output / (name + "_temp_before.txt")).write_bytes(remote(target, sensor))
        print("[RUN]", name, "50 samples", flush=True)
        try:
            stdout = remote(target, command, timeout=240)
        except Exception as error:
            save(args.output / (name + "_failure.json"), {"error": str(error), "remote_output": output})
            raise
        (args.output / (name + ".stdout")).write_bytes(stdout)
        reference_line = next(line for line in stdout.decode().splitlines() if line.startswith("[REFERENCE] "))
        assert json.loads(reference_line[len("[REFERENCE] "):])[0]["fnv1a64"] == "00b1f37535ac3647"
        (args.output / (name + "_temp_after.txt")).write_bytes(remote(target, sensor))
        raw = remote(target, "cat " + shlex.quote(output))
        (args.output / (name + ".jsonl")).write_bytes(raw)
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        assert len(rows) == 50
        result = analyze(rows)
        result.update(cell=cell, result_sha256=sha(raw))
        report["results"].append(result)
        save(args.output / "summary.json", report)
        print("[OK]", name, "allow", result["allow_makespan_median_ms"],
              "wait", result["wait_makespan_median_ms"], flush=True)
    assert remote(target, "cat /proc/sys/kernel/random/boot_id").decode().strip() == boot
    report["completed"] = True
    save(args.output / "summary.json", report)


if __name__ == "__main__":
    main()
