#!/usr/bin/env python3
"""Terminate frozen F mid-pipeline, then verify a clean same-boot recovery."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time

from audit_shared_buffer_storage import sha
from run_c3s_allocation_baseline import SSH, remote
from run_queue_threshold_final_qualification import preflight


def line_count(host, path):
    raw = remote(host, "test -f " + shlex.quote(path) + " && wc -l < " + shlex.quote(path) + " || echo 0")
    return int(raw.decode().strip())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--final-qualification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host", default="root@192.168.1.247")
    parser.add_argument("--recovery-runs", type=int, default=128)
    args = parser.parse_args()

    summary = json.loads((args.final_qualification / "summary.json").read_text())
    assert summary["status"] == "complete_final_candidate_qualification"
    boot, initial = preflight(args.host)
    args.output.mkdir(parents=True, exist_ok=False)
    board = "/tmp/fault_recovery_" + args.output.name
    remote(args.host, "mkdir " + shlex.quote(board))

    report = {
        "boot": boot,
        "scope": "forced SIGTERM and same-boot clean recovery; not performance",
        "signal": "SIGTERM",
        "recovery_policy": "reload the verified vta_hpc.bit before restarting the process",
        "kill_after_completed_frames": 8,
        "requested_interrupted_frames": 10000,
        "recovery_runs": args.recovery_runs,
        "final_qualification_sha256": sha(args.final_qualification / "summary.json"),
        "script_sha256": sha(__file__),
        "status": "running",
        "results": [],
    }
    (args.output / "board_preflight.txt").write_bytes(initial)
    (args.output / "preregistered.json").write_text(json.dumps(report, indent=2) + "\n")

    try:
        for name in "ABCD":
            source = (args.final_qualification / f"{name}_stress_command.sh").read_text().strip()
            expected_rows = [
                json.loads(row)
                for row in (args.reference / f"{name}_default_off.jsonl").read_text().splitlines()
            ]
            expected = {
                row["input_index"]: [output["fnv1a64"] for output in row["raw_outputs"]]
                for row in expected_rows
            }
            partial = board + f"/{name}_interrupted.jsonl"
            recovery = board + f"/{name}_recovery.jsonl"
            pidfile = board + f"/{name}.pid"
            words = shlex.split(source.split(" && env ", 1)[1])
            words[words.index("--runs") + 1] = "10000"
            words[words.index("--warmup-runs") + 1] = "0"
            words[words.index("--output-jsonl") + 1] = partial
            cd = source.split(" && env ", 1)[0]
            interrupted_shell = (
                cd + " && echo $$ > " + shlex.quote(pidfile) + " && exec env " + shlex.join(words)
            )
            (args.output / f"{name}_interrupted_command.sh").write_text(interrupted_shell + "\n")
            proc = subprocess.Popen(
                SSH + [args.host, interrupted_shell], stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            deadline = time.monotonic() + 60
            observed = 0
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    break
                observed = line_count(args.host, partial)
                if observed >= report["kill_after_completed_frames"]:
                    break
                time.sleep(0.1)
            assert proc.poll() is None, f"{name}: runner exited before fault injection"
            assert observed >= report["kill_after_completed_frames"], f"{name}: no progress before timeout"
            pid = remote(args.host, "cat " + shlex.quote(pidfile)).decode().strip()
            assert pid.isdigit()
            remote(args.host, "kill -TERM " + pid)
            stdout, stderr = proc.communicate(timeout=30)
            (args.output / f"{name}_interrupted.stdout").write_bytes(stdout)
            (args.output / f"{name}_interrupted.stderr").write_bytes(stderr)
            assert proc.returncode != 0, f"{name}: interrupted runner returned success"
            remaining = remote(
                args.host,
                "ps -eo pid,args | grep vta_stage_pipeline_runner | grep -v grep || true",
            ).decode().strip()
            assert not remaining, f"{name}: runner remains after SIGTERM: {remaining}"
            raw_partial = remote(args.host, "cat " + shlex.quote(partial))
            (args.output / f"{name}_interrupted.jsonl").write_bytes(raw_partial)
            partial_rows = [json.loads(row) for row in raw_partial.splitlines()]
            assert 0 < len(partial_rows) < 10000
            assert all(
                row["input_index"] == i % 8
                and [output["fnv1a64"] for output in row["raw_outputs"]] == expected[i % 8]
                for i, row in enumerate(partial_rows)
            )

            state = remote(
                args.host,
                "echo vta_hpc.bit > /sys/class/fpga_manager/fpga0/firmware; "
                "cat /sys/class/fpga_manager/fpga0/state",
            ).decode().strip()
            assert state == "operating", f"{name}: FPGA reload failed: {state}"
            observed_boot, post_reload = preflight(args.host)
            assert observed_boot == boot
            (args.output / f"{name}_post_reload_preflight.txt").write_bytes(post_reload)

            words[words.index("--runs") + 1] = str(args.recovery_runs)
            words[words.index("--warmup-runs") + 1] = "2"
            words[words.index("--output-jsonl") + 1] = recovery
            recovery_shell = cd + " && env " + shlex.join(words)
            (args.output / f"{name}_recovery_command.sh").write_text(recovery_shell + "\n")
            recovered = subprocess.run(
                SSH + [args.host, recovery_shell], capture_output=True, timeout=300,
            )
            (args.output / f"{name}_recovery.stdout").write_bytes(recovered.stdout)
            (args.output / f"{name}_recovery.stderr").write_bytes(recovered.stderr)
            recovered.check_returncode()
            raw_recovery = remote(args.host, "cat " + shlex.quote(recovery))
            (args.output / f"{name}_recovery.jsonl").write_bytes(raw_recovery)
            recovery_rows = [json.loads(row) for row in raw_recovery.splitlines()]
            assert len(recovery_rows) == args.recovery_runs
            assert all(
                row["frame_id"] == i
                and row["input_index"] == i % 8
                and [output["fnv1a64"] for output in row["raw_outputs"]] == expected[i % 8]
                for i, row in enumerate(recovery_rows)
            )
            result = {
                "topology": name,
                "interrupted_returncode": proc.returncode,
                "completed_before_sigterm": len(partial_rows),
                "partial_outputs_byte_exact": True,
                "runner_remaining_after_sigterm": False,
                "bitstream_reloaded_before_recovery": True,
                "recovery_frames": len(recovery_rows),
                "recovery_outputs_byte_exact": True,
            }
            report["results"].append(result)
            (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
            print("[PASS]", name, result, flush=True)
        observed_boot, final = preflight(args.host)
        assert observed_boot == boot
        (args.output / "board_after.txt").write_bytes(final)
        report["status"] = "complete_fault_recovery"
    except Exception as error:
        report["status"] = "failed_stop_no_retry"
        report["error"] = repr(error)
        raise
    finally:
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
