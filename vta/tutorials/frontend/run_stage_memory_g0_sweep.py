#!/usr/bin/env python3
"""Preregister and replay every observed A-D CPU/VTA direction, without action labels."""
import argparse
import datetime
import json
import math
from pathlib import Path
import random
import re
import shlex

from run_stage_memory_g0_discovery import BASE, percentile, remote, save, sha
from run_stage_memory_g0_pair import analyze

TRACE = BASE / "g0_discovery/boot_a5220e22_trace_run1"
RANKS = {"A": 1, "B": 5, "C": 7, "D": 13}
RUNNER = "/media/sd-mmcblk1p2/vta_stage_pair_runner_g0"
RUNNER_SHA = "e0e36ad3bbf7a9f0cc9680673aec9752e73331bfba060ab68e23d53663055f01"


def make_plan(trace, dma):
    cells, excluded, inventory = [], [], []
    for topology in trace["results"]:
        name = topology["topology"]
        rank = RANKS[name]
        manifest = json.loads((TRACE / f"rank{rank:02d}/manifest.json").read_text())
        for stage in manifest["stages"]:
            item = {"topology": name, "stage": stage["index"], "device": stage["device"],
                    "units": stage["unit_names"], "output_bytes": stage["output_bytes"],
                    "artifact_sha256": stage["artifact_sha256"],
                    "cpu_threads": manifest["pipeline_stage_runtime_threads"][f"stage{stage['index']}"]}
            item["contract_input_bytes"] = sum(math.prod(s["shape"]) *
                int(re.search(r"\d+", s["dtype"]).group()) // 8 for s in stage["input_schema"]["slots"])
            if stage["device"] == "vta":
                matches = [d for d in dma["experiment_b"]["stages"] if d["stage_units"] == stage["unit_names"]]
                assert len(matches) == 1
                item["historical_logical_dma_not_physical_ddr"] = matches[0]["comparison"]
            inventory.append(item)
        for direction in topology["directed_encounters"]:
            samples = [s for s in direction["samples"] if s["not_blocked_by_vta_owner"]]
            info = {"topology": name, "rank": rank, "active": direction["active_stage"],
                    "ready": direction["ready_stage"], "natural_eligible_encounters": len(samples)}
            if not samples:
                excluded.append(dict(info, reason="no owner-eligible encounter in frozen trace; not universally impossible"))
                continue
            ages = [s["active_age_ms"] for s in samples]
            quantiles = [.25, .5, .75] if len(samples) >= 10 else [.5]
            for quantile in quantiles:
                offset = round(percentile(ages, quantile), 3)
                cells.append(dict(info, quantile=quantile, offset_ms=offset,
                                  natural_age_min_ms=min(ages), natural_age_max_ms=max(ages),
                                  sparse_natural_support=len(samples) < 10,
                                  name=f"{name}_a{info['active']}_r{info['ready']}_q{int(100*quantile)}"))
    random.Random(908).shuffle(cells)
    return {"cells": cells, "excluded_directions": excluded, "stage_inventory": inventory,
            "selection": "All A-D observed directions, filtered only by existing VTA ownership; "
                         "q25/50/75 when n>=10, median otherwise; no pair outcome selection",
            "seed": 908, "warmups_per_action": 5, "abba_blocks": 10,
            "prior_outcomes_known": "A stage0/stage1 median pilot already observed; sweep is exploratory, not held-out",
            "state_qualification": "at least 36/40 scored samples active at eligible; do not drop failed samples",
            "screen_only": "state-qualified median paired wait improvement >=5% and >=8/10 positive blocks; "
                           "candidate only, not protect/allow label or noise-qualified confirmation",
            "scope": "natural-reachable A-D census with historical memory descriptors; not the planned isolated-intensity 3x3 or 5-boot confirmation"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    raw_trace = (TRACE / "readiness_audit.json").read_bytes()
    raw_dma = (BASE / "stage_memory_experiments/experiment_ab.json").read_bytes()
    plan = make_plan(json.loads(raw_trace), json.loads(raw_dma))
    plan.update(trace_sha256=sha(raw_trace), historical_dma_sha256=sha(raw_dma),
                host_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                script_sha256=sha(Path(__file__).read_bytes()), runner_sha256=RUNNER_SHA,
                helper_sha256=sha(Path(__file__).with_name("run_stage_memory_g0_pair.py").read_bytes()),
                vta_host_affinity="3", cpu_affinity="frozen per-stage settings, not disjoint from VTA host")
    save(args.output / "preregistered.json", plan)
    print("[PLAN]", len(plan["cells"]), "cells;", len(plan["excluded_directions"]), "unobserved directions", flush=True)
    if args.plan_only:
        return
    target = "root@" + args.host
    boot = remote(target, "cat /proc/sys/kernel/random/boot_id").decode().strip()
    assert not remote(target, "pidof tvm_rpc vta_stage_pipeline_runner "
                      "vta_stage_pipeline_runner_g0_a5220e22 vta_stage_pair_runner_g0 || true").strip()
    probe_command = "cat /sys/class/fpga_manager/fpga0/state; " \
                    "cat /sys/class/u-dma-buf/udmabuf0/size; " \
                    "cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq; " \
                    "sha256sum /lib/firmware/vta_hpc.bit " + RUNNER
    probe = remote(target, probe_command)
    (args.output / "board_preflight.txt").write_bytes(probe)
    assert probe.decode().startswith("operating\n201326592\n1066666\n")
    assert "7bf1ac95b1182c670cd25241111f33725b4ea37ed6d66f2df37b0b2e7df528d6" in probe.decode()
    assert probe.decode().splitlines()[-1].split()[0] == RUNNER_SHA
    for name, rank in RANKS.items():
        pkg = f"/media/sd-mmcblk1p2/v1_p7_top20/legacy_profile_rank{rank:02d}"
        checks = (TRACE / f"rank{rank:02d}/artifact_sha256.txt").read_text().splitlines()
        paths = [line.split()[1] for line in checks]
        actual = remote(target, "cd " + pkg + "; sha256sum " + shlex.join(paths)).decode().splitlines()
        assert actual == checks
        (args.output / f"{name}_artifact_sha256.txt").write_text("\n".join(actual) + "\n")
    root = "/media/sd-mmcblk1p2/" + args.output.name
    remote(target, "mkdir " + shlex.quote(root))
    report = {"boot_id": boot, "prereg_sha256": sha((args.output / "preregistered.json").read_bytes()),
              "planned_cells": len(plan["cells"]), "remote_root": root, "results": []}
    sensor = "cat /sys/bus/iio/devices/iio:device0/in_temp0_ps_temp_raw " \
             "/sys/bus/iio/devices/iio:device0/in_temp0_ps_temp_offset " \
             "/sys/bus/iio/devices/iio:device0/in_temp0_ps_temp_scale " \
             "/sys/bus/iio/devices/iio:device0/in_temp2_pl_temp_raw " \
             "/sys/bus/iio/devices/iio:device0/in_temp2_pl_temp_offset " \
             "/sys/bus/iio/devices/iio:device0/in_temp2_pl_temp_scale"
    for number, cell in enumerate(plan["cells"]):
        name = cell["name"]
        local = args.output / name
        local.mkdir()
        frozen = TRACE / f"rank{cell['rank']:02d}"
        prefix, invocation = (frozen / "copy_command.sh").read_text().split("exec ", 1)
        argv = shlex.split(invocation)
        argv[0] = RUNNER
        output = root + "/" + name + ".jsonl"
        for flag, value in [("--runs", "10"), ("--output-jsonl", output)]:
            argv[argv.index(flag)+1] = value
        argv += ["--serial", "--warmup-runs", "5", "--pair-active", str(cell["active"]),
                 "--pair-ready", str(cell["ready"]), "--pair-offset-ms", str(cell["offset_ms"])]
        command = prefix + "exec " + shlex.join(argv)
        (local / "command.sh").write_text(command + "\n")
        (local / "temp_before.txt").write_bytes(remote(target, sensor))
        print(f"[RUN {number+1}/{len(plan['cells'])}] {name} offset={cell['offset_ms']}", flush=True)
        try:
            stdout = remote(target, command, timeout=240)
        except Exception as error:
            save(local / "failure.json", {"error": str(error), "remote_output": output})
            raise
        (local / "stdout.txt").write_bytes(stdout)
        reference = next(s for s in stdout.decode().splitlines() if s.startswith("[REFERENCE] "))
        expected = json.loads((frozen / "shared.jsonl").read_text().splitlines()[0])["raw_outputs"]
        assert [r["fnv1a64"] for r in json.loads(reference[len("[REFERENCE] "):])] == [r["fnv1a64"] for r in expected]
        (local / "temp_after.txt").write_bytes(remote(target, sensor))
        raw = remote(target, "cat " + shlex.quote(output))
        (local / "samples.jsonl").write_bytes(raw)
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        assert len(rows) == 50
        result = analyze(rows)
        qualified = result["active_at_eligible_count"] >= 36
        positive = sum(b["wait_relative_improvement"] > 0 for b in result["blocks"])
        result.update(cell=cell, result_sha256=sha(raw), state_qualified=qualified,
                      positive_wait_blocks=positive,
                      exploratory_candidate=qualified and positive >= 8 and result["paired_wait_improvement_median"] >= .05,
                      scope="single-boot natural-direction census, not 3x3/confirmation/pipeline FPS")
        report["results"].append(result)
        save(args.output / "summary.json", report)
        print(f"[OK] allow={result['allow_makespan_median_ms']:.3f} wait={result['wait_makespan_median_ms']:.3f} "
              f"paired_wait_gain={result['paired_wait_improvement_median']:+.2%} active={result['active_at_eligible_count']}/40", flush=True)
    assert remote(target, "cat /proc/sys/kernel/random/boot_id").decode().strip() == boot
    (args.output / "board_postflight.txt").write_bytes(remote(target, probe_command))
    report["completed"] = True
    save(args.output / "summary.json", report)


if __name__ == "__main__":
    main()
