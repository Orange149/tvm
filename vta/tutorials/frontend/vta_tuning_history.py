"""Validate tuning evidence and record the configurations used during stage builds.

An audit context is itself an AutoTVM ``DispatchContext``.  If it is installed
directly above the root fallback context, Relay no longer performs its normal
automatic TopHub setup.  This module therefore resolves TopHub explicitly and
places the audit *inside* it.
"""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

import numpy as np
from tvm import autotvm
from tvm.ir.container import Array
from tvm.target import Target


_TOPHUB_ALIASES = {
    "vtacpu": "vta",
    "webgpu": "opencl",
    "vulkan": "opencl",
    "nvptx": "cuda",
}


def _target_list(targets):
    if targets is None:
        return []
    if isinstance(targets, dict):
        targets = list(targets.values())
    elif not isinstance(targets, (Array, list, tuple)):
        targets = [targets]
    return [value if isinstance(value, Target) else Target(value) for value in targets]


def _tophub_packages(targets):
    """Return the local TopHub packages selected by TVM for ``targets``."""
    packages = []
    seen = set()
    for target in _target_list(targets):
        possible_names = []
        device = target.attrs.get("device", "")
        if device:
            possible_names.append(str(device))
        possible_names.extend(str(key) for key in target.keys)
        possible_names.append(target.kind.name)
        for raw_name in possible_names:
            name = _TOPHUB_ALIASES.get(raw_name, raw_name)
            if name not in autotvm.tophub.PACKAGE_VERSION:
                continue
            version = autotvm.tophub.PACKAGE_VERSION[name]
            path = Path(autotvm.tophub.AUTOTVM_TOPHUB_ROOT_PATH) / "{}_{}.log".format(
                name, version
            )
            key = str(path.resolve())
            if key not in seen:
                if not path.is_file():
                    raise RuntimeError("TopHub context did not provide expected package: " + str(path))
                data = path.read_bytes()
                packages.append(
                    {
                        "backend": name,
                        "version": version,
                        "path": key,
                        "size_bytes": len(data),
                        "sha256": hashlib.sha256(data).hexdigest(),
                    }
                )
                seen.add(key)
            break
    if not packages:
        raise RuntimeError("No TopHub package resolved for build targets")
    return packages


def tophub_identity(targets, ensure_available=True):
    """Describe the exact cached TopHub files used for a build."""
    normalized = _target_list(targets)
    if not normalized:
        raise ValueError("TopHub provenance requires explicit build targets")
    configured_location = autotvm.tophub._get_tophub_location()
    if configured_location == autotvm.tophub.AUTOTVM_TOPHUB_NONE_LOC:
        raise RuntimeError("TopHub is disabled by TOPHUB_LOCATION=NONE")
    if ensure_available:
        # This follows TVM's own download/cache policy.  The returned context is
        # discarded here; audited_history installs a fresh one for dispatch.
        autotvm.tophub.context(normalized)
    packages = _tophub_packages(normalized)
    digest_payload = json.dumps(packages, sort_keys=True, separators=(",", ":"))
    return {
        "mode": "tophub",
        "configured_location": configured_location,
        "targets": [str(target) for target in normalized],
        "files": packages,
        "sha256": hashlib.sha256(digest_payload.encode("utf-8")).hexdigest(),
    }


def history_identity(path, tophub_targets=None):
    if not path:
        return tophub_identity(tophub_targets)
    data = Path(path).read_bytes()
    if not data.strip():
        raise ValueError("Tuning log is empty: " + str(path))
    records = list(autotvm.record.load_from_file(str(path)))
    good = [(inp, result) for inp, result in records if result.error_no == 0
            and len(result.costs) and all(np.isfinite(x) and x > 0 for x in result.costs)]
    if not good:
        raise ValueError("Tuning log has no successful measurements: " + str(path))
    return {"mode": "history_best", "path": str(Path(path).resolve()),
            "sha256": hashlib.sha256(data).hexdigest(),
            "successful_records": len(good)}


class DispatchAudit(autotvm.task.DispatchContext):
    def __init__(self):
        super().__init__()
        self.observed = {}

    def _query_inside(self, target, workload):
        cfg = self._old_ctx.query(target, workload)
        if target is not None and target.device_name == "vta" and workload and workload[0] == "conv2d_packed.vta":
            key = json.dumps([str(target), workload])
            self.observed[key] = (target, workload, cfg)
        return cfg

    def summary(self):
        rows = [{"target": str(target), "workload": workload,
                 "is_fallback": bool(cfg.is_fallback),
                 "config": (cfg.to_json_dict() if hasattr(cfg, "to_json_dict") else
                            autotvm.task.space.ConfigEntity(-1, None, cfg._entity_map, []).to_json_dict())}
                for target, workload, cfg in self.observed.values()]
        return {"coverage_scope": "conv2d_packed.vta", "queried_workloads": len(rows),
                "tuned_workloads": sum(not r["is_fallback"] for r in rows),
                "fallback_workloads": sum(r["is_fallback"] for r in rows), "rows": rows}


@contextmanager
def audited_history(path="", result=None, require_tuned=False, tophub_targets=None):
    if path:
        history = history_identity(path)
        normalized = _target_list(tophub_targets)
        if normalized:
            base = tophub_identity(normalized)
            identity = {
                "mode": "history_overlay_on_tophub",
                "overlay": history,
                "base": base,
                "sha256": hashlib.sha256(
                    json.dumps([base["sha256"], history["sha256"]]).encode("utf-8")
                ).hexdigest(),
            }
            base_context = autotvm.tophub.context(normalized)
        else:
            identity = history
            base_context = None
        dispatch_context = autotvm.apply_history_best(path)
    else:
        normalized = _target_list(tophub_targets)
        dispatch_context = autotvm.tophub.context(normalized)
        identity = tophub_identity(normalized, ensure_available=False)
    if path and base_context is not None:
        # ApplyHistoryBest delegates a missing workload to its enclosing context.
        # This makes a partial experimental log a safe overlay rather than an
        # accidental replacement of the known-good TopHub baseline.
        with base_context:
            with dispatch_context:
                with DispatchAudit() as audit:
                    yield
    else:
        with dispatch_context:
            with DispatchAudit() as audit:
                yield
    summary = dict(identity, **audit.summary())
    if result is not None:
        result.update(summary)
    if require_tuned and (not summary["queried_workloads"] or summary["fallback_workloads"]):
        raise RuntimeError("Incomplete VTA tuning coverage: {} tuned / {} queried".format(
            summary["tuned_workloads"], summary["queried_workloads"]))
