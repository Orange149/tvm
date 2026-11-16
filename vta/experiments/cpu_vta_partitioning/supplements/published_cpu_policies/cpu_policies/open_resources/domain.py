"""Execution identities, not just thread-count labels, for nine contexts."""
from copy import deepcopy
from hashlib import sha256
from itertools import combinations
import json


CORES = (0, 1, 2, 3)
CONFIGURATION_FIELDS = (
    "context_id", "compute_pool", "reserved_cores", "stages", "affinity_mode",
    "execution_mode", "host_mask", "control_mask", "core_partition",
)


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def contexts():
    result = [{"id": "shared4", "family": "shared", "label": "四核共享",
               "compute_pool": list(CORES), "host_mask": list(CORES),
               "control_mask": list(CORES), "reserved_cores": []}]
    for core in CORES:
        result.append({"id": f"host_shared_{core}", "family": "pinned_shared",
                       "label": f"主机固定{core}，仍共享", "compute_pool": list(CORES),
                       "host_mask": [core], "control_mask": list(CORES), "reserved_cores": []})
    for core in CORES:
        result.append({"id": f"host_reserved_{core}", "family": "reserved",
                       "label": f"为主机保留{core}",
                       "compute_pool": [c for c in CORES if c != core],
                       "host_mask": [core], "control_mask": [core], "reserved_cores": [core]})
    return tuple(result)


def context(value):
    """Resolve an immutable, explicitly selected context; no host-mask inference."""
    if not isinstance(value, (str, dict)):
        raise ValueError("resource context must be a name or explicit object")
    identifier = value if isinstance(value, str) else value.get("context_id", value.get("id"))
    match = next((c for c in contexts() if c["id"] == identifier), None)
    if match is None:
        raise ValueError("unknown resource context: " + str(identifier))
    if isinstance(value, dict):
        for key in ("compute_pool", "host_mask", "control_mask", "reserved_cores"):
            if key in value and value[key] != match[key]:
                raise ValueError("resource context field differs: " + key)
    return deepcopy(match)


def nonempty_masks(cores=CORES):
    return tuple(sorted(tuple(c) for n in range(1, len(cores) + 1)
                        for c in combinations(tuple(cores), n)))


def stage(device, begin, end, threads, mask):
    mask = list(mask)
    program = f"{device}_{begin:02}_{end - 1:02}"
    return {"device": device, "begin": begin, "end": end, "threads": threads,
            "mask": mask, "program": program,
            "signature": f"{program}_t{threads}_a{'-'.join(map(str, mask))}",
            "worker_core_map": list(mask) if device == "cpu" else [],
            "caller_core": mask[0] if device == "cpu" and mask else None}


def compatible_groups(masks):
    masks = tuple(tuple(m) for m in masks)
    return all(a == b or not set(a).intersection(b) for a in masks for b in masks)


def configuration(value):
    return {key: deepcopy(value[key]) for key in CONFIGURATION_FIELDS}


def candidate(stages, context_value="shared4", gate="off", **kwargs):
    # A keyword spelling is convenient for callers; reject two different values.
    if "context" in kwargs:
        if context_value != "shared4":
            raise TypeError("resource context supplied twice")
        context_value = kwargs.pop("context")
    if kwargs:
        raise TypeError("unknown candidate arguments: " + ",".join(kwargs))
    ctx = context(context_value)
    rows = [stage(r["device"], r["begin"], r["end"], r["threads"], r["mask"])
            for r in stages]
    value = {"context_id": ctx["id"], "compute_pool": ctx["compute_pool"],
             "reserved_cores": ctx["reserved_cores"], "stages": rows,
             "affinity_mode": "pinned", "execution_mode": gate,
             "host_mask": ctx["host_mask"], "control_mask": ctx["control_mask"],
             "core_partition": sorted({tuple(s["mask"]) for s in rows
                                       if s["device"] == "cpu"}) if gate == "resource_group" else []}
    value["core_partition"] = [list(m) for m in value["core_partition"]]
    value["id"] = "cfg_" + digest(value)[:24]
    return value


def canonical_tuple(value):
    """Physical execution order used for ties; a cryptographic hash is not an order."""
    return (tuple((s["device"], s["begin"], s["end"], s["threads"], tuple(s["mask"]))
                  for s in value["stages"]), tuple(value["compute_pool"]),
            tuple(value["host_mask"]), tuple(value["control_mask"]),
            tuple(value["reserved_cores"]), value["execution_mode"], value["affinity_mode"])


def validate_candidate(value, programs=None, component=False, unit_count=21,
                       max_vta_islands=3):
    if type(unit_count) is not int or unit_count < 1 or type(max_vta_islands) is not int or max_vta_islands < 1:
        raise ValueError("invalid graph bounds")
    ctx = context(value)
    if any(value.get(key) != ctx[key] for key in
           ("compute_pool", "host_mask", "control_mask", "reserved_cores")):
        raise ValueError("incomplete resource context")
    rows = value.get("stages")
    if not isinstance(rows, list) or not rows or (component and len(rows) != 1):
        raise ValueError("empty or invalid stage list")
    if value.get("execution_mode") not in ("off", "resource_group"):
        raise ValueError("unsupported stage execution mode")
    if component and value["execution_mode"] != "off":
        raise ValueError("isolated components do not execute pipeline group gates")
    if value.get("affinity_mode") != "pinned" and not (
            component and rows[0].get("device") == "vta" and value.get("affinity_mode") == "group"):
        raise ValueError("open CPU profiles require fixed per-thread binding")
    masks = []
    for index, row in enumerate(rows):
        device = row.get("device")
        begin, end, threads, mask = (row.get(k) for k in ("begin", "end", "threads", "mask"))
        if (device not in ("cpu", "vta") or type(begin) is not int or type(end) is not int
                or not 0 <= begin < end <= unit_count or type(threads) is not int or threads < 1
                or not isinstance(mask, list) or not mask
                or any(type(c) is not int for c in mask) or mask != sorted(set(mask))
                or not set(mask).issubset(CORES)):
            raise ValueError("invalid stage or physical core set")
        if row != stage(device, begin, end, threads, mask):
            raise ValueError("stage identity, caller, or worker map differs")
        if programs is not None:
            p = programs.get(row["program"])
            if p is None or any(p.get(k) != row[k] for k in ("device", "begin", "end")):
                raise ValueError("unqualified stage program")
        if device == "cpu":
            if threads != len(mask) or not set(mask).issubset(ctx["compute_pool"]):
                raise ValueError("CPU mapping uses a reserved core or differs from thread budget")
            masks.append(tuple(mask))
        elif threads != 1 or mask != ctx["host_mask"]:
            raise ValueError("VTA host affinity differs from context")
        if index and (rows[index - 1]["end"] != begin or rows[index - 1]["device"] == device):
            raise ValueError("noncontiguous or nonalternating graph")
    expected_groups = [list(m) for m in sorted(set(masks))] if value["execution_mode"] == "resource_group" else []
    if value.get("core_partition") != expected_groups:
        raise ValueError("CPU group identity differs")
    if value["execution_mode"] == "resource_group" and not compatible_groups(masks):
        raise ValueError("serial CPU groups partially overlap")
    if not component and (rows[0]["begin"] != 0 or rows[-1]["end"] != unit_count
            or rows[0]["device"] != "cpu" or rows[-1]["device"] != "cpu"
            or not 1 <= sum(r["device"] == "vta" for r in rows) <= max_vta_islands):
        raise ValueError("candidate does not cover the legal graph")
    if value.get("id") != "cfg_" + digest(configuration(value))[:24]:
        raise ValueError("candidate identity differs")
    return value


def component(row, context_value="shared4"):
    # CPU isolation has no VTA actor or producer/collector. Normalize its context.
    ctx = context("shared4" if row["device"] == "cpu" else context_value)
    cfg = candidate([row], ctx)
    if row["device"] == "vta":
        cfg["affinity_mode"] = "group"
        cfg["id"] = "cfg_" + digest(configuration(cfg))[:24]
    return cfg


def component_key(value):
    """Key effective isolated work; reservation/control is a documented alias.

    Asset identities are separately verified by the profile importer. This key
    must never be used alone as proof that two executable builds are identical.
    """
    if len(value["stages"]) != 1:
        raise ValueError("component key needs one stage")
    row = value["stages"][0]
    effective = {k: row[k] for k in ("device", "begin", "end", "threads", "mask", "worker_core_map")}
    return "cmp_" + digest({"stage": effective, "affinity_mode": value["affinity_mode"]})[:24]


def requirements(programs):
    result = {}
    for _, p in sorted(programs.items()):
        if p["device"] == "cpu":
            variants = [("shared4", n, CORES[:n]) for n in range(1, 5)]
        else:
            variants = [("shared4", 1, CORES)] + [(f"host_shared_{c}", 1, (c,)) for c in CORES]
        for ctx, n, mask in variants:
            cfg = component(stage(p["device"], p["begin"], p["end"], n, mask), ctx)
            result[component_key(cfg)] = cfg
    return dict(sorted(result.items()))


def traffic_requirements(programs):
    return {name: component(stage("vta", p["begin"], p["end"], 1, CORES))
            for name, p in sorted(programs.items()) if p["device"] == "vta"}
