#!/usr/bin/env python3
"""Shared helpers for capturing VTA runtime profiler snapshots."""

from __future__ import absolute_import, print_function

import json
import os
import time


def ensure_dir(path):
    if not path:
        return ""
    os.makedirs(path, exist_ok=True)
    return path


def fetch_runtime_profiler_hooks(remote):
    hooks = {
        "clear": None,
        "status": None,
        "events": None,
    }
    try:
        hooks["clear"] = remote.get_function("vta.runtime.profiler_clear")
    except Exception:  # pylint: disable=broad-except
        pass
    try:
        hooks["status"] = remote.get_function("vta.runtime.profiler_status")
    except Exception:  # pylint: disable=broad-except
        pass
    try:
        hooks["events"] = remote.get_function("vta.runtime.profiler_events")
    except Exception:  # pylint: disable=broad-except
        pass
    return hooks


def hooks_available(hooks):
    return bool(hooks and hooks.get("status") is not None)


def clear_runtime_profiler(hooks):
    if hooks and hooks.get("clear") is not None:
        hooks["clear"]()


def _load_json_blob(blob, default):
    if blob in [None, ""]:
        return default
    if isinstance(blob, (dict, list)):
        return blob
    return json.loads(blob)


def read_runtime_status(hooks):
    if not hooks_available(hooks):
        return {}
    return _load_json_blob(hooks["status"](), {})


def read_runtime_events(hooks, max_events=200):
    if not hooks or hooks.get("events") is None:
        return []
    return _load_json_blob(hooks["events"](int(max_events)), [])


def _write_json(path, payload):
    with open(path, "w", encoding="utf-8") as out:
        json.dump(payload, out, indent=2, sort_keys=True)
        out.write("\n")


def dump_runtime_snapshot(profile_dir, snapshot_name, hooks, events_limit=200, extra=None):
    if not profile_dir or not hooks_available(hooks):
        return {}
    ensure_dir(profile_dir)
    timestamp_ms = int(time.time() * 1000)
    meta = {
        "snapshot": snapshot_name,
        "timestamp_ms": timestamp_ms,
        "events_limit": int(events_limit),
    }
    if extra:
        meta.update(extra)
    status = read_runtime_status(hooks)
    events = read_runtime_events(hooks, max_events=events_limit)
    status_path = os.path.join(profile_dir, "{}_status.json".format(snapshot_name))
    events_path = os.path.join(profile_dir, "{}_events.json".format(snapshot_name))
    meta_path = os.path.join(profile_dir, "{}_meta.json".format(snapshot_name))
    _write_json(status_path, status)
    _write_json(events_path, events)
    _write_json(meta_path, meta)
    return {
        "status_path": status_path,
        "events_path": events_path,
        "meta_path": meta_path,
    }


def dump_status_only(profile_dir, snapshot_name, hooks, extra=None):
    if not profile_dir or not hooks_available(hooks):
        return {}
    ensure_dir(profile_dir)
    payload = {
        "snapshot": snapshot_name,
        "timestamp_ms": int(time.time() * 1000),
        "status": read_runtime_status(hooks),
    }
    if extra:
        payload.update(extra)
    out_path = os.path.join(profile_dir, "{}_status.json".format(snapshot_name))
    _write_json(out_path, payload)
    return {"status_path": out_path}


def write_manifest(path, payload):
    ensure_dir(os.path.dirname(path) or ".")
    _write_json(path, payload)
