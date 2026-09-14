"""Explicit, fail-closed residency routes for Relay VTA compilation.

The normal ``conv2d_packed.vta`` implementation remains TopHub compatible.  A
caller may install this context around ``relay.build`` to bind an exact packed
conv workload to both a complete AutoTVM ConfigEntity and an experimental
residency schedule.  Missing workloads delegate to the enclosing dispatch
context; a matched workload with a different ConfigEntity is rejected.
"""

from __future__ import absolute_import

import json
import threading

from tvm import autotvm
from tvm.autotvm.task.space import ConfigEntity


_STATE = threading.local()
_SUPPORTED_MODES = {
    "original": 0,
    "input_stationary": 1,
    "weight_resident_barrier": 4,
}


def _json_value(value):
    """Convert TVM containers and tuple-heavy payloads to comparable JSON."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if hasattr(value, "items"):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)) or hasattr(value, "__iter__"):
        return [_json_value(item) for item in value]
    if hasattr(value, "__len__") and hasattr(value, "__getitem__"):
        return [_json_value(value[index]) for index in range(len(value))]
    if hasattr(value, "value"):
        return _json_value(value.value)
    raise TypeError("unsupported residency-dispatch value: {}".format(type(value).__name__))


def _workload_key(workload):
    return json.dumps(_json_value(workload), sort_keys=True, separators=(",", ":"))


def _semantic_config(config):
    raw = _json_value(config.to_json_dict())
    raw.pop("index", None)
    return raw


def _route_payload(raw):
    identity = raw.get("identity", raw)
    workload = identity.get("workload")
    config = identity.get("complete_config_entity")
    public_mode = identity.get("public_mode", raw.get("public_mode"))
    implementation_mode = identity.get("implementation_mode", raw.get("implementation_mode"))
    if not workload or workload[0] != "conv2d_packed.vta":
        raise ValueError("residency route requires an exact conv2d_packed.vta workload")
    if public_mode not in _SUPPORTED_MODES:
        raise ValueError("unsupported production residency mode: {}".format(public_mode))
    expected_mode = _SUPPORTED_MODES[public_mode]
    if implementation_mode is None:
        implementation_mode = expected_mode
    if int(implementation_mode) != expected_mode:
        raise ValueError("public/implementation residency mode mismatch")
    if not isinstance(config, dict) or not config.get("entity"):
        raise ValueError("residency route lacks a complete ConfigEntity")
    normalized_config = _json_value(config)
    normalized_config.pop("index", None)
    config_with_index = dict(normalized_config, index=-1)
    return {
        "candidate_id": raw.get("candidate_id"),
        "workload": _json_value(workload),
        "workload_key": _workload_key(workload),
        "public_mode": public_mode,
        "implementation_mode": expected_mode,
        "complete_config_entity": normalized_config,
        "config": ConfigEntity.from_json_dict(config_with_index),
    }


def _find_conv_workload(outs):
    """Find the registered packed-conv workload under fused elementwise ops."""
    pending = [outs] if hasattr(outs, "op") else list(outs)
    seen = set()
    while pending:
        tensor = pending.pop()
        op = tensor.op
        if id(op) in seen:
            continue
        seen.add(id(op))
        workload = op.attrs.get("workload") if hasattr(op, "attrs") else None
        if workload and str(workload[0]) == "conv2d_packed.vta":
            return workload
        pending.extend(getattr(op, "input_tensors", []))
    return None


def _stack():
    return getattr(_STATE, "stack", [])


def active_context():
    # Relay may invoke TOPI scheduling from a worker thread, where Python
    # ``threading.local`` state does not propagate.  AutoTVM's own dispatch
    # context is process-global, so walk that authoritative chain first.
    context = autotvm.task.DispatchContext.current
    while context is not None:
        if isinstance(context, ExplicitResidencyDispatch):
            return context
        context = getattr(context, "_old_ctx", None)
    stack = _stack()
    return stack[-1] if stack else None


def resolve_schedule_route(cfg, outs):
    """Return an explicitly routed mode, or ``original`` outside the context."""
    # AutoTVM first invokes the schedule with a ConfigSpace while constructing a
    # Task.  Define the historical space through the original branch; the exact
    # mode is selected later when Task.instantiate supplies a ConfigEntity.
    if not hasattr(cfg, "to_json_dict"):
        return "original"
    context = active_context()
    if context is None:
        return "original"
    workload = _find_conv_workload(outs)
    route = context.routes.get(_workload_key(workload)) if workload is not None else None
    route_source = "workload_attr" if route is not None else None
    if route is None:
        # The registered AutoTVM schedule wrapper consumes the workload attribute
        # before calling this function.  It passes through the exact ConfigEntity
        # returned by our workload-keyed DispatchContext, so object identity is a
        # precise fallback that cannot match a delegated TopHub configuration.
        matches = [item for item in context.routes.values() if cfg is item["config"]]
        if len(matches) == 1:
            route = matches[0]
            route_source = "config_object_from_exact_workload_query"
    context.schedule_queries.append(
        {
            "workload_found": workload is not None,
            "workload": _json_value(workload) if workload is not None else None,
            "config_type": type(cfg).__name__,
            "route_source": route_source,
        }
    )
    if route is None:
        return "original"
    actual = _semantic_config(cfg)
    if actual != route["complete_config_entity"]:
        raise RuntimeError(
            "explicit residency route matched workload but ConfigEntity differed"
        )
    context.schedule_hits.append(
        {
            "candidate_id": route["candidate_id"],
            "workload": route["workload"],
            "public_mode": route["public_mode"],
            "implementation_mode": route["implementation_mode"],
            "complete_config_entity": actual,
        }
    )
    return route["implementation_mode"]


class ExplicitResidencyDispatch(autotvm.task.DispatchContext):
    """Bind exact packed-conv workloads to certified configs and schedules."""

    def __init__(self, routes):
        super(ExplicitResidencyDispatch, self).__init__()
        parsed = [_route_payload(route) for route in routes]
        self.routes = {}
        for route in parsed:
            key = route["workload_key"]
            if key in self.routes:
                raise ValueError("duplicate explicit residency workload")
            self.routes[key] = route
        if not self.routes:
            raise ValueError("at least one explicit residency route is required")
        self.config_hits = []
        self.schedule_queries = []
        self.schedule_hits = []

    def _query_inside(self, target, workload):
        route = self.routes.get(_workload_key(workload))
        if route is None:
            return None
        self.config_hits.append(
            {
                "candidate_id": route["candidate_id"],
                "workload": route["workload"],
                "public_mode": route["public_mode"],
            }
        )
        return route["config"]

    def update(self, target, workload, cfg):
        route = self.routes.get(_workload_key(workload))
        if route is not None and _semantic_config(cfg) != route["complete_config_entity"]:
            raise RuntimeError("alter-layout attempted to replace an explicit residency config")

    def __enter__(self):
        super(ExplicitResidencyDispatch, self).__enter__()
        stack = list(_stack())
        stack.append(self)
        _STATE.stack = stack
        return self

    def __exit__(self, ptype, value, trace):
        stack = list(_stack())
        if not stack or stack[-1] is not self:
            raise RuntimeError("explicit residency dispatch stack corruption")
        stack.pop()
        _STATE.stack = stack
        return super(ExplicitResidencyDispatch, self).__exit__(ptype, value, trace)

    def summary(self):
        return {
            "registered_routes": len(self.routes),
            "config_hits": list(self.config_hits),
            "schedule_queries": list(self.schedule_queries),
            "schedule_hits": list(self.schedule_hits),
            "all_routes_scheduled": len(
                {row["candidate_id"] for row in self.schedule_hits}
            ) == len(self.routes),
        }
