#!/usr/bin/env python3
"""Analyze VTA runtime profiler status/events snapshots."""

from __future__ import absolute_import, print_function

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-dir", required=True, help="Directory containing *_status/events/meta.json")
    parser.add_argument("--snapshot-prefix", default=None, help="Snapshot prefix before _status.json")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def detect_prefix(profile_dir, explicit_prefix=None):
    if explicit_prefix:
        return explicit_prefix
    matches = sorted(profile_dir.glob("*_status.json"))
    if len(matches) != 1:
        raise RuntimeError(
            "Could not infer unique snapshot prefix in {} (found {}).".format(
                profile_dir, len(matches)
            )
        )
    return matches[0].name[: -len("_status.json")]


def top_counter(counter, top_k):
    rows = []
    for key, value in counter.most_common(top_k):
        rows.append({"key": key, "count": int(value)})
    return rows


def summarize_events(events, top_k):
    by_kind = Counter()
    by_memory_type = Counter()
    by_signature = Counter()
    by_kind_duration = defaultdict(float)
    padded_true = 0
    load_count = 0
    store_count = 0

    for event in events:
        kind = str(event.get("kind", "unknown"))
        by_kind[kind] += 1
        by_kind_duration[kind] += float(event.get("duration_us", 0.0) or 0.0)

        memory_type = int(event.get("memory_type", 0) or 0)
        by_memory_type[memory_type] += 1

        if bool(event.get("is_padded", False)):
            padded_true += 1

        signature = "x={x},y={y},stride={s}".format(
            x=int(event.get("x_size", 0) or 0),
            y=int(event.get("y_size", 0) or 0),
            s=int(event.get("x_stride", 0) or 0),
        )
        if kind in ("load_buffer_2d", "store_buffer_2d"):
            by_signature["{}:{}".format(kind, signature)] += 1
            if kind == "load_buffer_2d":
                load_count += 1
            else:
                store_count += 1

    kind_rows = []
    for kind, count in by_kind.most_common(top_k):
        total_us = by_kind_duration[kind]
        kind_rows.append(
            {
                "kind": kind,
                "count": int(count),
                "total_us": round(total_us, 3),
                "avg_us": round(total_us / count, 3) if count else 0.0,
            }
        )

    return {
        "events_count": int(len(events)),
        "kinds": kind_rows,
        "memory_types": top_counter(by_memory_type, top_k),
        "padded_true": int(padded_true),
        "padded_ratio": round(float(padded_true) / len(events), 4) if events else 0.0,
        "dma_signature_top": top_counter(by_signature, top_k),
        "load_event_count": int(load_count),
        "store_event_count": int(store_count),
    }


def summarize_status(status):
    return {
        "event_count": status.get("event_count"),
        "device_run_wait_us": status.get("device_run_wait_us"),
        "driver_poll_wait_us": status.get("driver_poll_wait_us"),
        "driver_run_insns": status.get("driver_run_insns"),
        "load_buffer_2d_calls": status.get("load_buffer_2d_calls"),
        "load_buffer_2d_bytes": status.get("load_buffer_2d_bytes"),
        "load_buffer_2d_small_calls": status.get("load_buffer_2d_small_calls"),
        "load_buffer_2d_strided_calls": status.get("load_buffer_2d_strided_calls"),
        "load_buffer_2d_padded_calls": status.get("load_buffer_2d_padded_calls"),
        "load_buffer_2d_top_signatures": status.get("load_buffer_2d_top_signatures", []),
        "store_buffer_2d_calls": status.get("store_buffer_2d_calls"),
        "store_buffer_2d_bytes": status.get("store_buffer_2d_bytes"),
        "store_buffer_2d_top_signatures": status.get("store_buffer_2d_top_signatures", []),
    }


def print_summary(prefix, meta, status_summary, event_summary):
    print("snapshot: {}".format(prefix))
    if isinstance(meta, dict):
        print("scheme: {}".format(meta.get("scheme", "unknown")))
        print("phase: {}".format(meta.get("phase", "unknown")))
    print("events_count: {}".format(event_summary["events_count"]))
    print("status.event_count: {}".format(status_summary["event_count"]))
    print("has_events: {}".format("yes" if event_summary["events_count"] else "no"))
    print()

    print("status_summary:")
    for key in [
        "device_run_wait_us",
        "driver_poll_wait_us",
        "driver_run_insns",
        "load_buffer_2d_calls",
        "load_buffer_2d_bytes",
        "load_buffer_2d_small_calls",
        "load_buffer_2d_strided_calls",
        "load_buffer_2d_padded_calls",
        "store_buffer_2d_calls",
        "store_buffer_2d_bytes",
    ]:
        print("  {}: {}".format(key, status_summary.get(key)))
    print("  load_buffer_2d_top_signatures:")
    for item in status_summary.get("load_buffer_2d_top_signatures", []):
        print("    - {}".format(item))
    print("  store_buffer_2d_top_signatures:")
    for item in status_summary.get("store_buffer_2d_top_signatures", []):
        print("    - {}".format(item))
    print()

    if event_summary["events_count"] == 0:
        print("event_summary: no event records")
        return

    print("event_summary:")
    print("  padded_ratio: {}".format(event_summary["padded_ratio"]))
    print("  load_event_count: {}".format(event_summary["load_event_count"]))
    print("  store_event_count: {}".format(event_summary["store_event_count"]))
    print("  kinds:")
    for row in event_summary["kinds"]:
        print(
            "    - {kind}: count={count} total_us={total_us} avg_us={avg_us}".format(**row)
        )
    print("  memory_types:")
    for row in event_summary["memory_types"]:
        print("    - {}: {}".format(row["key"], row["count"]))
    print("  dma_signature_top:")
    for row in event_summary["dma_signature_top"]:
        print("    - {} -> {}".format(row["key"], row["count"]))


def main():
    args = parse_args()
    profile_dir = Path(args.profile_dir).resolve()
    prefix = detect_prefix(profile_dir, args.snapshot_prefix)

    status = load_json(profile_dir / (prefix + "_status.json"))
    events = load_json(profile_dir / (prefix + "_events.json"))
    meta = load_json(profile_dir / (prefix + "_meta.json"))

    status_summary = summarize_status(status)
    event_summary = summarize_events(events, args.top_k)
    payload = {
        "snapshot": prefix,
        "scheme": meta.get("scheme") if isinstance(meta, dict) else None,
        "phase": meta.get("phase") if isinstance(meta, dict) else None,
        "status_summary": status_summary,
        "event_summary": event_summary,
    }

    print_summary(prefix, meta, status_summary, event_summary)

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")


if __name__ == "__main__":
    main()
