#!/usr/bin/env python3
"""Immutable run contracts and measured, append-only actions for thesis stages."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


class RunStore:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.contract = json.loads((self.directory / "contract.json").read_text())
        self.verify_contract()

    @classmethod
    def create(cls, directory, contract):
        directory = Path(directory).resolve()
        directory.mkdir(parents=True, exist_ok=False)
        for name in ("raw", "analysis", "figures"):
            (directory / name).mkdir()
        write_new_json(directory / "contract.json", contract)
        (directory / "contract.sha256").write_text(sha256(directory / "contract.json") + "\n")
        (directory / "timeline.jsonl").touch(exist_ok=False)
        result = cls(directory)
        result.event("run_created", contract_sha256=sha256(directory / "contract.json"))
        return result

    def verify_contract(self):
        expected = (self.directory / "contract.sha256").read_text().strip()
        if sha256(self.directory / "contract.json") != expected:
            raise ValueError("Frozen contract changed; use a new run")

    def event(self, event, **fields):
        self.verify_contract()
        item = {
            "event_id": uuid.uuid4().hex,
            "event": event,
            "run_id": self.contract["run_id"],
            "stage": self.contract["stage"],
            "campaign_id": self.contract["campaign_id"],
            "utc": utc(),
            "monotonic_ns": time.monotonic_ns(),
            "clock_domain": f"host:{os.getpid()}",
            **fields,
        }
        with (self.directory / "timeline.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        return item["event_id"]

    def execute(self, argv, action, cost_class, timeout, stdin_path=None):
        if timeout <= 0:
            raise ValueError("Positive timeout required")
        action_id = uuid.uuid4().hex
        stem = self.directory / "raw" / (action + "_" + action_id)
        argv = list(argv)
        input_bytes = Path(stdin_path).read_bytes() if stdin_path else None
        begin = time.monotonic_ns()
        parent = self.event(
            "action_start", action_id=action_id, action=action,
            argv=argv, cost_class=cost_class, timeout_seconds=timeout,
            stdin_sha256=sha256(stdin_path) if stdin_path else None,
        )
        process = None
        status = "ok"
        try:
            process = subprocess.Popen(
                argv, stdin=subprocess.PIPE if input_bytes is not None else subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
            )
            try:
                stdout, stderr = process.communicate(input=input_bytes, timeout=timeout)
            except subprocess.TimeoutExpired:
                status = "timeout"
                self.event("action_cancel", action_id=action_id, action=action,
                           cost_class="failed_recovery", parent_event_id=parent)
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    stdout, stderr = process.communicate(timeout=2)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    stdout, stderr = process.communicate(timeout=2)
            code = process.returncode
            if status == "ok" and code != 0:
                status = "failed"
        except OSError as error:
            stdout, stderr, code, status = b"", str(error).encode(), None, "launch_failed"
        end = time.monotonic_ns()
        stdout_path = stem.with_suffix(".stdout")
        stderr_path = stem.with_suffix(".stderr")
        stdout_path.write_bytes(stdout)
        stderr_path.write_bytes(stderr)
        result = {
            "action_id": action_id, "action": action, "argv": argv,
            "status": status, "returncode": code, "cost_class": cost_class,
            "wall_seconds": (end - begin) / 1e9,
            "begin_monotonic_ns": begin, "end_monotonic_ns": end,
            "stdout": str(stdout_path.relative_to(self.directory)),
            "stderr": str(stderr_path.relative_to(self.directory)),
            "stdout_sha256": sha256(stdout_path), "stderr_sha256": sha256(stderr_path),
        }
        write_new_json(stem.with_suffix(".json"), result)
        self.event("action_end", parent_event_id=parent, **result)
        return result

    def events(self):
        return [json.loads(row) for row in (self.directory / "timeline.jsonl").read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--action", required=True)
    parser.add_argument("--cost-class", required=True)
    parser.add_argument("--timeout", type=float, required=True)
    parser.add_argument("--stdin-file")
    parser.add_argument("argv", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    argv = args.argv[1:] if args.argv[:1] == ["--"] else args.argv
    if not argv:
        parser.error("Command argv required after --")
    result = RunStore(args.run_dir).execute(argv, args.action, args.cost_class, args.timeout, args.stdin_file)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
