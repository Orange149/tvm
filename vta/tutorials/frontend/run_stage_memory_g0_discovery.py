#!/usr/bin/env python3
"""Replay frozen stages and audit natural overlap; this does not classify protect/allow."""

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess


SSH_OPTIONS = ["-o", "HostKeyAlgorithms=+ssh-rsa", "-o",
               "PubkeyAcceptedAlgorithms=+ssh-rsa", "-o", "BatchMode=yes",
               "-o", "ConnectTimeout=10"]
TOPOLOGIES = {1: "A", 5: "B", 7: "C", 13: "D"}
BASE = Path(__file__).parent / "report_out/stage_tile_cotuning"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def remote(target, script, timeout=180):
    result = subprocess.run(["ssh", *SSH_OPTIONS, target, script],
                            capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors="replace") +
                           result.stdout.decode(errors="replace"))
    return result.stdout


def union_ms(intervals):
    end, total = float("-inf"), 0.0
    for start, stop in sorted(intervals):
        total += max(0.0, stop - max(start, end))
        end = max(end, stop)
    return total


def percentile(values, q):
    values = sorted(values)
    pos = (len(values) - 1) * q
    low = int(pos)
    return values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (pos - low)


def summarize(rows, manifest, expected, discard):
    hashes = {tuple(x["fnv1a64"] for x in row["raw_outputs"]) for row in rows}
    assert hashes == {expected}, ("output mismatch", hashes, expected)
    assert [r["frame_id"] for r in rows] == list(range(len(rows))), "frame order"
    steady = rows[discard:]
    first, last = steady[0]["completion_ms"], steady[-1]["completion_ms"]
    ii = (last - first) / (len(steady) - 1)
    intervals, full_intervals = {}, {}
    for stage in manifest["stages"]:
        i = stage["index"]
        # Older runners expose run duration and set_end, not exact run boundaries.
        # New readiness instrumentation supplies actual invocation timestamps.
        spans, full_spans = [], []
        for row in rows:
            start = row.get(f"stage{i}_run_start_ms")
            stop = row.get(f"stage{i}_run_end_ms")
            if start is None:
                start = row[f"stage{i}_set_end_ms"]
                stop = start + row[f"stage{i}_run_ms"]
            full_spans.append((start, stop, row["frame_id"]))
            if stop > first and start < last:
                spans.append((max(first, start), min(last, stop), row["frame_id"]))
        intervals[i] = spans
        full_intervals[i] = full_spans
    overlaps, encounters = [], []
    for a in manifest["stages"]:
        for b in manifest["stages"]:
            if a["device"] == b["device"]:
                continue
            i, j = a["index"], b["index"]
            if i < j:
                intersections = [(max(s, x), min(e, y))
                                 for s, e, _ in intervals[i]
                                 for x, y, _ in intervals[j] if max(s, x) < min(e, y)]
                overlaps.append({"stage_pair": [i, j], "union_ms": union_ms(intersections),
                                 "wall_fraction": union_ms(intersections) / (last - first)})
            field = f"stage{j}_binding_done_ms"
            observed = []
            for row in steady:
                ready = row.get(field)
                if ready is not None and first <= ready <= last:
                    for start, end, frame in full_intervals[i]:
                        if start <= ready < end:
                            observed.append({"active_frame": frame, "ready_frame": row["frame_id"],
                                             "active_age_ms": ready - start,
                                             "overlap_after_ready_ms": end - ready})
            encounters.append({"active_stage": i, "newly_ready_stage": j,
                               "readiness_observed": all(row.get(field) is not None for row in steady),
                               "count": len(observed), "samples": observed})
    return {"frames": len(rows), "discard": discard, "ii_ms": ii, "fps": 1000 / ii,
            "submit_to_complete_p95_ms": percentile([r["total_latency_ms"] for r in steady], .95),
            "output_hashes": sorted(hashes), "correctness": "exact_frozen_output_hash_and_frame_order",
            "overlap": overlaps, "directed_encounters": encounters,
            "protect_allow_classified": False,
            "scope": "single-boot natural overlap discovery; no memory-causality or benefit claim"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=102)
    parser.add_argument("--runner", default="", help="Optional absolute remote runner path")
    parser.add_argument("--modes", nargs="+", default=["serial", "copy", "shared"])
    args = parser.parse_args()
    assert args.frames >= 12
    assert set(args.modes) <= {"serial", "copy", "shared"}
    target = "root@" + args.host
    args.output.mkdir(parents=True, exist_ok=False)
    boot = remote(target, "cat /proc/sys/kernel/random/boot_id").decode().strip()
    remote_root = "/media/sd-mmcblk1p2/g0_" + boot[:8] + "_" + args.output.name
    probe = remote(target, "cat /sys/class/fpga_manager/fpga0/state; "
                   "cat /sys/class/u-dma-buf/udmabuf0/size; "
                   "cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq; "
                   "sha256sum /lib/firmware/vta_hpc.bit; "
                   "ps | grep -E '[t]vm_rpc|[v]ta_stage_pipeline_runner' || true")
    (args.output / "board_preflight.txt").write_bytes(probe)
    assert "operating\n201326592\n1066666\n" in probe.decode()
    assert "7bf1ac95b1182c670cd25241111f33725b4ea37ed6d66f2df37b0b2e7df528d6" in probe.decode()
    # Require no active board measurement/RPC before taking the exclusive native lease.
    assert not remote(target, "pidof tvm_rpc vta_stage_pipeline_runner || true").strip()
    remote(target, "mkdir " + shlex.quote(remote_root))
    report = {"boot_id": boot, "host_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "runner_override": args.runner, "script_sha256": sha(Path(__file__).read_bytes()),
              "remote_root": remote_root, "results": []}
    for rank, topology in TOPOLOGIES.items():
        pkg = f"/media/sd-mmcblk1p2/v1_p7_top20/legacy_profile_rank{rank:02d}"
        manifest = json.loads(remote(target, f"cat {pkg}/manifest.json"))
        local = args.output / f"rank{rank:02d}"
        local.mkdir()
        save(local / "manifest.json", manifest)
        old = BASE / f"legacy_topology_profiles/rank{rank:02d}/native_result.jsonl"
        expected = tuple(x["fnv1a64"] for x in json.loads(old.read_text().splitlines()[0])["raw_outputs"])
        # Verify stage graph, parameters, code and runtime against frozen manifests.
        checks = []
        for stage in manifest["stages"]:
            for key, hash_key in [("graph", "graph_sha256"), ("lib", "lib_sha256"),
                                  ("params", "params_sha256")]:
                checks.append((stage[key], stage["artifact_sha256"][hash_key]))
        checks += [("libvta.so", "eedfabb0630d58bf2eabaef9b4f2d2e4bfcdfa52a70503090f5608e79404ee5d"),
                   ("libtvm_runtime.so", "04e894baf305311315fd0c79d03ec2a6375b9591916e9c6f429cf3697c4457b2")]
        hashes = remote(target, "cd " + shlex.quote(pkg) + "; sha256sum " +
                        " ".join(shlex.quote(p) for p, _ in checks)).decode().splitlines()
        for (path, expected_sha), line in zip(checks, hashes):
            assert line.split()[0] == expected_sha, path
        assert len(hashes) == len(checks)
        (local / "artifact_sha256.txt").write_text("\n".join(hashes) + "\n")
        for mode in args.modes:
            script_name = "run_stage_serial.sh" if mode == "serial" else "run_stage_pipeline.sh"
            source = remote(target, f"cat {pkg}/{script_name}").decode()
            prefix, invocation = source.split("exec ./vta_stage_pipeline_runner", 1)
            # POSIX shells remove escaped newlines before tokenization. shlex.split
            # alone instead leaves newline tokens, which the runner rejects.
            argv = [args.runner or "./vta_stage_pipeline_runner",
                    *shlex.split(invocation.replace("\\\n", ""))]
            count = 8 if mode == "serial" else args.frames
            output = f"{remote_root}/rank{rank:02d}_{mode}.jsonl"
            for flag, value in [("--runs", str(count)), ("--output-jsonl", output),
                                ("--vta-runtime-profile-dir", "")]:
                argv[argv.index(flag) + 1] = value
            if mode == "shared":
                argv += ["--p8-managed-slots", "2"]
            prefix = prefix.replace('cd "$(dirname "$0")"', "cd " + shlex.quote(pkg))
            command = prefix + "exec " + shlex.join(argv)
            (local / f"{mode}_command.sh").write_text(command + "\n")
            print(f"[RUN] {topology} {mode} {count} frames", flush=True)
            runner_hash = remote(target, "cd " + shlex.quote(pkg) + "; sha256sum " +
                                 shlex.quote(argv[0])).decode().split()[0]
            try:
                stdout = remote(target, command, timeout=240)
            except (RuntimeError, subprocess.TimeoutExpired) as error:
                save(local / f"{mode}_failure.json", {"error": str(error),
                     "remote_output": output, "runner_sha256": runner_hash})
                raise
            (local / f"{mode}.stdout").write_bytes(stdout)
            raw = remote(target, "cat " + shlex.quote(output))
            (local / f"{mode}.jsonl").write_bytes(raw)
            rows = [json.loads(line) for line in raw.splitlines() if line]
            assert len(rows) == count
            result = summarize(rows, manifest, expected, 2)
            result.update(topology=topology, mode=mode, result_sha256=sha(raw),
                          runner_sha256=runner_hash,
                          readiness_is_managed_launch_boundary=(mode == "shared"))
            report["results"].append(result)
            save(args.output / "summary.json", report)
            print(f"[OK] {topology} {mode}: {result['fps']:.3f} FPS, exact output", flush=True)
    assert remote(target, "cat /proc/sys/kernel/random/boot_id").decode().strip() == boot
    report["completed"] = True
    save(args.output / "summary.json", report)


if __name__ == "__main__":
    main()
