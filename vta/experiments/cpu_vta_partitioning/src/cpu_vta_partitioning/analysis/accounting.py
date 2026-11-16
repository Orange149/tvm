"""Count physical work separately from samples accepted for inference.

A remote task is launched at most once per (boot, task ID). Retransfers and
resume attempts therefore contribute the union of completed frame IDs, not a
sum of snapshot lengths. Unrecovered work is unknown, never assumed to be zero.
"""
import json
from pathlib import Path

from cpu_vta_partitioning.common import read


def cost_accounting(root):
    root = Path(root)
    sessions, boots = {}, set()

    def session(boot, task_id):
        boots.add(boot)
        return sessions.setdefault((boot, task_id), {
            "expected": None, "accepted": False, "frame_ids": set(), "complete": False})

    def expect(item, count):
        if type(count) is not int or count < 0:
            raise ValueError("invalid planned native frame count")
        if item["expected"] is not None and item["expected"] != count:
            raise ValueError("cost accounting task frame count changed")
        item["expected"] = count

    def task_count(document):
        return document["warmup"] + document["scored"] + document["tail"]

    failed_events = malformed_events = 0
    events_path = root / "events.jsonl"
    if events_path.exists():
        for line in events_path.read_text(errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("event is not an object")
            except ValueError:
                malformed_events += 1
                continue
            failed_events += record.get("event") == "session_failed"
            if record.get("boot_id"):
                boots.add(record["boot_id"])
                if record.get("task_id"):
                    item = session(record["boot_id"], record["task_id"])
                    if "planned_native_frames" in record:
                        expect(item, record["planned_native_frames"])

    for path in sorted((root / "accepted").glob("*/*.json")):
        receipt = read(path)
        item = session(receipt["boot_id"], receipt["task"]["task_id"])
        expect(item, task_count(receipt["task"]))
        # Acceptance already required the entire native frame sequence. Keep
        # accepted counts distinct even if later round-level drift rejects it.
        item["accepted"] = item["complete"] = True
        item["frame_ids"].update(range(item["expected"]))

    for directory in sorted((root / "raw").glob("*/*")):
        if not directory.is_dir():
            continue
        item = session(directory.parent.name, directory.name)
        if (directory / "task.json").exists():
            expect(item, task_count(read(directory / "task.json")))
        for path in sorted(directory.rglob("result.jsonl")):
            ids, valid = set(), True
            try:
                with path.open(errors="replace") as stream:
                    for line in stream:
                        if not line.strip():
                            continue
                        try:
                            row = json.loads(line)
                            frame = row["frame_id"]
                            if type(frame) is not int or frame < 0:
                                raise ValueError("invalid frame ID")
                        except (ValueError, KeyError, TypeError):
                            valid = False
                            continue
                        if frame in ids:
                            valid = False
                        ids.add(frame)
            except OSError:
                valid = False
            item["frame_ids"].update(ids)
            if valid and item["expected"] is not None and ids == set(range(item["expected"])):
                item["complete"] = True

    unresolved = []
    for (boot, task_id), item in sorted(sessions.items()):
        expected = item["expected"]
        if not item["complete"] or expected is None or item["frame_ids"] != set(range(expected)):
            unresolved.append({"boot_id": boot, "task_id": task_id, "planned_native_frames": expected,
                               "observed_completed_frames": len(item["frame_ids"]),
                               "reason": "complete_frame_count_not_established"})
    observed = sum(len(s["frame_ids"]) for s in sessions.values())
    exact = not unresolved and not malformed_events
    return {
        "accepted_sessions": sum(s["accepted"] for s in sessions.values()),
        "accepted_native_frames": sum(s["expected"] for s in sessions.values() if s["accepted"]),
        "observed_sessions": len(sessions), "observed_completed_frames": observed,
        "actual_native_frames": observed if exact else None,
        "native_frames_status": "complete" if exact else "lower_bound",
        "sessions_with_unknown_frame_count": len(unresolved), "unresolved_sessions": unresolved,
        "malformed_event_lines": malformed_events,
        "distinct_observed_boots": len(boots),
        "distinct_accepted_boots": len({boot for (boot, _), s in sessions.items() if s["accepted"]}),
        "failed_session_events": failed_events,
        "raw_bytes_retained": sum(p.stat().st_size for p in (root / "raw").rglob("*") if p.is_file()),
    }
