"""Exact, cross-topology resource-label DP for frozen CPU--VTA models.

The algorithm merges only identical integer resource labels at identical future
interfaces.  Each such label retains the first K distinct canonical prefixes.
It never truncates different labels by their current scalar score.  Monotone
resource bounds may prune only when strictly worse than a complete Kth path.

This module imports no TVM, transport, performance records, or learning model.
Checkpoints are streaming gzip JSON, not executable Python pickles.
"""
from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
import gzip
import hashlib
from itertools import product
import json
from pathlib import Path
import resource
import time

from cpu_policies.open_resources import domain
from cpu_policies.open_resources.costs import (
    Costs, allocate_core_ns, ddr_stage_service, quantize_ns,
)


ALGORITHM = "cross_topology_exact_resource_labels_v1"
METHODS = ("tarnawski", "complete")
# complete: D[4], H, VTA, DDR, max_CPU, team[4], serial_group[15].
ZERO_COMPLETE = (0,) * 27
# baseline: physical_CPU_wall[4], VTA, DDR.
ZERO_TARNAWSKI = (0,) * 6


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _implementation():
    from cpu_policies.open_resources import costs as costs_module
    from resource_envelope import model as envelope_module
    root = Path(__file__).resolve().parents[1]
    files = (Path(__file__), Path(domain.__file__), Path(costs_module.__file__),
             Path(envelope_module.__file__), root / "model_ablation/model.py",
             root / "model_ablation/common.py", root / "staged_ablation/model.py",
             root / "staged_ablation/common.py")
    return {p.name + ":" + p.parent.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files}


def _normalize(programs, unit_count):
    if isinstance(programs, list):
        programs = {p.get("name", p.get("program")): p for p in programs}
    if not isinstance(programs, dict) or not programs:
        raise ValueError("qualified programs must be a nonempty mapping")
    result, intervals = {}, set()
    for name, raw in sorted(programs.items()):
        row = dict(raw)
        if row.get("name", name) != name:
            raise ValueError("program name differs from its mapping key")
        device, begin, end = row.get("device"), row.get("begin"), row.get("end")
        if (device not in ("cpu", "vta") or type(begin) is not int or type(end) is not int
                or not 0 <= begin < end <= unit_count):
            raise ValueError("invalid qualified interval: " + str(name))
        key = device, begin, end
        if key in intervals:
            raise ValueError("ambiguous modules for the same device interval")
        intervals.add(key)
        result[name] = {**row, "name": name}
    return result


def _frontiers(programs, unit_count, supplied=None):
    """Check that a boundary really has one future interface across intervals."""
    observed = {}
    for p in programs.values():
        for side, position in (("input", p["begin"]), ("output", p["end"])):
            ids, types = p.get(side + "_node_ids"), p.get(side + "_types")
            if ids is None and types is None:
                continue
            if not isinstance(ids, list) or not isinstance(types, list) or len(ids) != len(types):
                raise ValueError("incomplete frontier metadata")
            signature = _digest(list(zip(ids, types)))
            if position in observed and observed[position] != signature:
                raise ValueError("inconsistent frontier/layout at boundary " + str(position))
            observed[position] = signature
    if supplied is not None:
        if set(map(int, supplied)) != set(range(unit_count + 1)):
            raise ValueError("explicit frontier signatures must cover all boundaries")
        # The public mapping can describe more metadata than the module projection;
        # retain it in the state identity, while still checking module consistency.
        return {i: _digest(supplied.get(i, supplied.get(str(i)))) for i in range(unit_count + 1)}
    required = {p[k] for p in programs.values() for k in ("begin", "end")}
    if any("canonical_sha256" in p for p in programs.values()) and not required <= set(observed):
        raise ValueError("real compiled programs require complete frontier metadata")
    return {i: observed.get(i, _digest({"declared_synthetic_boundary": i}))
            for i in range(unit_count + 1)}


def _compatible(used, bits):
    return all(not (used & (1 << old)) or old == bits or not old & bits
               for old in range(1, 16))


def domain_counts(programs, *, unit_count, max_vta_islands=3, contexts=None):
    """Independent integer path/cardinality DP; it never expands complete paths."""
    programs = _normalize(programs, unit_count)
    contexts = _contexts(contexts)
    edges = defaultdict(list)
    for row in programs.values():
        edges[row["begin"], row["device"]].append(row["end"])

    @lru_cache(None)
    def paths(position, device, islands):
        if position == unit_count:
            return tuple(int(device == "vta" and islands == n) for n in range(1, max_vta_islands + 1))
        counts = [0] * max_vta_islands
        for end in edges.get((position, device), ()):
            next_islands = islands + (device == "vta")
            if next_islands > max_vta_islands:
                continue
            suffix = paths(end, "vta" if device == "cpu" else "cpu", next_islands)
            counts = [a + b for a, b in zip(counts, suffix)]
        return tuple(counts)

    per_island = paths(0, "cpu", 0)
    totals = {"complete": 0, "tarnawski": 0}
    context_counts = {}
    for ctx in contexts:
        masks = [sum(1 << c for c in mask) for mask in domain.nonempty_masks(ctx["compute_pool"])]
        ncores = len(ctx["compute_pool"])
        complete = baseline = 0
        for islands, topologies in enumerate(per_island, 1):
            cpus = islands + 1
            serial = _serial_count(ncores, cpus)
            complete += topologies * (len(masks) ** cpus + serial)
            baseline += topologies * ncores ** cpus
        context_counts[ctx["id"]] = {"complete": complete, "tarnawski": baseline}
        totals["complete"] += complete
        totals["tarnawski"] += baseline
    return {"topologies": sum(per_island), "topologies_per_island": list(per_island),
            "contexts": len(contexts), **totals, "per_context": context_counts,
            "configuration_count_is_not_DP_state_count": True}


@lru_cache(None)
def _serial_count(ncores, cpus):
    masks = range(1, 1 << ncores)
    return sum(all(a == b or not a & b for i, a in enumerate(groups)
                   for b in groups[i + 1:]) for groups in product(masks, repeat=cpus))


def _contexts(values):
    return tuple(domain.context(value) for value in (values if values is not None else domain.contexts()))


@dataclass(frozen=True)
class Option:
    row: dict
    canonical: tuple
    service: int
    run: int
    boundary: int
    ddr: int
    demand: tuple
    bits: int


def _options(programs, costs, ctx, method):
    masks = (tuple((c,) for c in ctx["compute_pool"]) if method == "tarnawski"
             else domain.nonempty_masks(ctx["compute_pool"]))
    options = []
    for name, p in programs.items():
        for mask in masks if p["device"] == "cpu" else (tuple(ctx["host_mask"]),):
            row = domain.stage(p["device"], p["begin"], p["end"],
                               len(mask) if p["device"] == "cpu" else 1, mask)
            if row["program"] != name:
                raise ValueError("qualified program uses a noncanonical interval name: " + name)
            cost = costs.get(row, ctx)
            service, run, boundary = (quantize_ns(cost[k]) for k in
                                     ("service_ms", "run_core_ms", "boundary_core_ms"))
            ddr = quantize_ns(ddr_stage_service(costs.ddr, row))
            if max(service, run, boundary, ddr) > ((1 << 63) - 1) // 64:
                raise ValueError("component exceeds the frozen int64 search safety bound")
            work = [0] * 4
            if p["device"] == "cpu":
                shares = allocate_core_ns(run, cost["cpu_core_weights"], threads=len(mask))
                for core, value in zip(mask, shares):
                    work[core] += value
                work[mask[0]] += boundary
            canonical = p["device"], p["begin"], p["end"], row["threads"], tuple(mask)
            options.append(Option(row, canonical, service, run, boundary,
                                  ddr,
                                  tuple(work), sum(1 << c for c in mask)))
    return tuple(sorted(options, key=lambda x: x.canonical))


def _advance(label, opt, method, gate):
    value = list(label)
    if method == "tarnawski":
        if opt.row["device"] == "cpu":
            value[opt.row["mask"][0]] += opt.service
        else:
            value[4] += opt.service
        value[5] += opt.ddr
        return tuple(value)
    if opt.row["device"] == "cpu":
        for core in range(4):
            value[core] += opt.demand[core]
        value[7] = max(value[7], opt.service)
        for core in opt.row["mask"]:
            value[8 + core] += opt.service
        if gate == "resource_group":
            value[11 + opt.bits] += opt.service
    else:
        value[4] += opt.run + opt.boundary
        value[5] += opt.service
    value[6] += opt.ddr
    return tuple(value)


def _value(label, ctx, method, gate, extra=(0, 0, 0, 0, 0)):
    """An exact final score, or a safe lower bound with relaxed suffix minima.

    extra is min future (VTA wall, host CPU, DDR, max CPU service, CPU work).
    Its entries need not occur in one common suffix, which only weakens bounds.
    """
    future_vta, future_host, future_ddr, future_cpu_max, future_cpu_work = extra
    if method == "tarnawski":
        return max(*label[:4], label[4] + future_vta, label[5] + future_ddr,
                   future_cpu_max,
                   (sum(label[:4]) + future_cpu_work + len(ctx["compute_pool"]) - 1)
                   // len(ctx["compute_pool"]))
    demand = label[:4]
    host = label[4] + future_host
    host_bits = sum(1 << c for c in ctx["host_mask"])
    result = max(label[5] + future_vta, label[6] + future_ddr,
                 label[7], future_cpu_max, *label[8:12],
                 *(demand[c] + (host if c in ctx["host_mask"] else 0) for c in range(4)))
    # This module implements ONLY complete/four-on (and the separate baseline).
    # For nonnegative integer D,H, R_host dominates each subset capacity:
    # if M <= S, sum(D[S])+H <= sum(D[c]+H*1[c in M] for c in S);
    # otherwise average(D[S]) <= max(D).  Integer ceilings preserve the bound.
    # R_team likewise dominates CPU_max and every exact-group service sum.
    # The complete resource label retains these fields; independent scorers use
    # the original full expression.  This is algebra, not a changed score.
    # Every remaining CPU task must stay in the compute pool.  Host work is
    # included here only when the host set is entirely confined to that pool.
    pool_bits = sum(1 << c for c in ctx["compute_pool"])
    pool_work = sum(demand[c] for c in ctx["compute_pool"]) + future_cpu_work
    if pool_bits & host_bits == host_bits:
        pool_work += host
    width = len(ctx["compute_pool"])
    return max(result, (pool_work + width - 1) // width)


def _suffixes(options, unit_count, max_islands, method):
    edges = defaultdict(list)
    for opt in options:
        edges[opt.row["begin"], opt.row["device"]].append(opt)

    @lru_cache(None)
    def suffix(position, device, islands):
        if position == unit_count:
            return (0,) * 5 if device == "vta" and 1 <= islands <= max_islands else None
        values = []
        for opt in edges.get((position, device), ()):
            n = islands + (device == "vta")
            if n > max_islands:
                continue
            tail = suffix(opt.row["end"], "vta" if device == "cpu" else "cpu", n)
            if tail is None:
                continue
            cpu = device == "cpu"
            values.append((tail[0] + (0 if cpu else opt.service),
                           tail[1] + (0 if cpu else opt.run + opt.boundary),
                           tail[2] + opt.ddr,
                           max(tail[3], opt.service if cpu else 0),
                           tail[4] + ((opt.service if method == "tarnawski"
                                       else opt.run + opt.boundary) if cpu else 0)))
        return tuple(min(v[i] for v in values) for i in range(5)) if values else None

    return edges, suffix


def _insert_prefix(paths, prefix, k):
    at = bisect_left(paths, prefix)
    if at < len(paths) and paths[at] == prefix:
        return "duplicate"
    paths.insert(at, prefix)
    if len(paths) > k:
        paths.pop()
        return "equivalent_discard"
    return "inserted"


def _add_complete(incumbents, cfg, score, k):
    value = {**cfg, "score_ns": int(score), "score_ms": score / 1_000_000}
    if any(row["id"] == value["id"] for row in incumbents):
        return
    incumbents.append(value)
    incumbents.sort(key=lambda row: (row["score_ns"], domain.canonical_tuple(row)))
    del incumbents[k:]


def _threshold(rows, k):
    return rows[-1]["score_ns"] if len(rows) == k else None


def _seed_one_island(options, contexts, methods, incumbents, unit_count, k):
    """Get deterministic feasible upper bounds solely from this cost table.

    All qualified three-stage paths and their resources are checked.  Their
    paths are not guaranteed winners; the full DP subsequently certifies K.
    No historical ranking, FPS, stochastic optimizer, or filtered domain is used.
    """
    started = time.monotonic(); evaluated = 0
    per_method = {}
    for method in methods:
        before = evaluated
        for ctx in contexts:
            opts = options[method, ctx["id"]]
            first = [o for o in opts if o.row["device"] == "cpu" and o.row["begin"] == 0]
            last = defaultdict(list)
            vtas = defaultdict(list)
            for o in opts:
                if o.row["device"] == "cpu" and o.row["end"] == unit_count:
                    last[o.row["begin"]].append(o)
                elif o.row["device"] == "vta":vtas[o.row["begin"]].append(o)
            for gate in (("resource_group",) if method == "tarnawski" else ("off", "resource_group")):
                zero = ZERO_COMPLETE if method == "complete" else ZERO_TARNAWSKI
                for a in first:
                    one = _advance(zero, a, method, gate)
                    for v in vtas.get(a.row["end"], ()):
                        two = _advance(one, v, method, gate)
                        for b in last.get(v.row["end"], ()):
                            if gate == "resource_group" and a.bits != b.bits and a.bits & b.bits:continue
                            label = _advance(two, b, method, gate)
                            score = _value(label, ctx, method, gate)
                            evaluated += 1
                            threshold = _threshold(incumbents[method], k)
                            if threshold is not None and score > threshold:continue
                            cfg = domain.candidate([a.row, v.row, b.row], ctx, gate)
                            _add_complete(incumbents[method], cfg, score, k)
        per_method[method] = evaluated - before
    return dict(evaluated_configurations=evaluated, per_method=per_method,
                seconds=time.monotonic() - started,
                source="current_isolated_costs_qualified_one_island_paths",
                historical_rankings_used=False, pipeline_FPS_used=False,
                complete_domain_still_certified_by_DP=True)


def _check_incumbents(incumbents, programs, costs, methods, unit_count, max_islands, contexts, k):
    """Optional static-only warm starts are feasible upper bounds, never answers."""
    from model_ablation.model import score as full_score
    from staged_ablation.model import score as baseline_score
    result = {method: [] for method in methods}
    baseline_costs = Costs(costs.records, programs=costs.programs, ddr=costs.ddr,
                           homogeneous=costs.homogeneous, distribution="balanced")
    allowed = {ctx["id"] for ctx in contexts}
    for method in methods:
        for raw in (incumbents or {}).get(method, ()):
            ctx = domain.context(raw["context_id"])
            if ctx["id"] not in allowed:
                continue
            cfg = domain.candidate(raw["stages"], ctx, raw["execution_mode"])
            domain.validate_candidate(cfg, programs, unit_count=unit_count, max_vta_islands=max_islands)
            score = (full_score(cfg, costs, "complete") if method == "complete"
                     else baseline_score(cfg, baseline_costs, "tarnawski"))["score_ns"]
            _add_complete(result[method], cfg, score, k)
    return result


def _checkpoint(path, data, buckets):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    h = hashlib.sha256()
    records = 0
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=1) as f:
        f.write(json.dumps({"header": data}, separators=(",", ":")) + "\n")
        for key, labels in buckets.items():
            for label, prefixes in labels.items():
                text = json.dumps([key, label, prefixes], separators=(",", ":")) + "\n"
                h.update(text.encode()); records += 1; f.write(text)
        f.write(json.dumps({"footer": {"states": records, "sha256": h.hexdigest()}}) + "\n")
    temporary.replace(path)


def _restore(path, fingerprint):
    buckets = {}
    h = hashlib.sha256(); count = 0
    with gzip.open(path, "rt", encoding="utf-8") as f:
        header = json.loads(next(f))["header"]
        if header.get("input_sha256") != fingerprint or header.get("algorithm") != ALGORITHM:
            raise ValueError("checkpoint input or implementation differs")
        footer = None
        for text in f:
            value = json.loads(text)
            if isinstance(value, dict):
                footer = value.get("footer")
                if f.read():raise ValueError("checkpoint footer is not last")
                break
            key, label, paths = value
            key = tuple(key); label = tuple(label); paths = [tuple(p) for p in paths]
            if label in buckets.setdefault(key, {}):
                raise ValueError("duplicate checkpoint state")
            buckets[key][label] = paths
            h.update(text.encode()); count += 1
        if footer != {"states": count, "sha256": h.hexdigest()}:
            raise ValueError("checkpoint content hash differs")
    return header, buckets


def _rss_bytes():
    # Linux ru_maxrss is KiB; the experiment host is Linux.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def rank(programs, costs, *, unit_count, max_vta_islands=3, k=20,
         methods=METHODS, contexts=None, frontier_signatures=None, output_dir=None,
         budget_seconds=7200, memory_bytes=8 * 1024**3, resume=False,
         incumbents=None, prune=True, max_expansions=None, initialize_one_island=True):
    """Return certified exact rankings, or an explicitly incomplete checkpoint.

    ``incumbents`` may contain independently rescored *static* configurations.
    They only provide upper thresholds and do not eliminate equal-score paths.
    ``max_expansions`` is an explicit interruption hook for recovery tests.
    Time budgets include prior invocations when resuming the same checkpoint.
    """
    invocation_start = time.monotonic()
    if type(unit_count) is not int or unit_count < 3 or not 1 <= max_vta_islands <= 3:
        raise ValueError("invalid graph or island bound")
    if type(k) is not int or k < 1 or any(m not in METHODS for m in methods):
        raise ValueError("invalid Top-K or method")
    methods = tuple(methods)
    if len(set(methods)) != len(methods):raise ValueError("duplicate methods")
    if budget_seconds <= 0 or memory_bytes <= 0:raise ValueError("invalid engineering budget")
    programs = _normalize(programs, unit_count)
    ctxs = _contexts(contexts)
    frontier = _frontiers(programs, unit_count, frontier_signatures)
    if not isinstance(costs, Costs):
        costs = Costs(costs, programs=programs, distribution="session_weights")
    if "complete" in methods and costs.distribution != "session_weights":
        raise ValueError("complete-model inputs require frozen session thread weights")
    options = {(method, ctx["id"]): _options(programs, costs, ctx, method)
               for method in methods for ctx in ctxs}
    identity = dict(algorithm=ALGORITHM, implementation=_implementation(), unit_count=unit_count,
                    max_vta_islands=max_vta_islands, k=k, methods=methods, contexts=ctxs,
                    frontier_signatures=frontier,
                    options={m + ":" + c: [[o.canonical, o.service, o.run, o.boundary,
                                             o.ddr, o.demand] for o in values]
                             for (m, c), values in options.items()}, prune=prune,
                    initialize_one_island=initialize_one_island)
    fingerprint = _digest(identity)
    jobs = [(method, ctx, gate) for method in methods for ctx in ctxs
            for gate in (("resource_group",) if method == "tarnawski"
                         else ("off", "resource_group"))]
    root = Path(output_dir) if output_dir is not None else None
    checkpoint = root / "search_checkpoint.jsonl.gz" if root is not None else None
    initial = _check_incumbents(incumbents, programs, costs, methods, unit_count,
                                max_vta_islands, ctxs, k)
    stats = dict(labels_expanded=0, transitions=0, exact_labels_merged=0,
                 equivalent_prefixes_discarded=0, strict_bound_prunes=0,
                 terminal_paths=0, peak_pending_labels=0, peak_RSS_bytes=_rss_bytes(),
                 warm_start_paths={m: len(rows) for m, rows in initial.items()})
    saved = dict(algorithm=ALGORITHM, input_sha256=fingerprint, job_index=0,
                 active=None, initialized=False, elapsed_seconds=0., incumbents=initial,
                 statistics=stats)
    buckets = {}
    if resume:
        if checkpoint is None or not checkpoint.is_file():raise ValueError("no checkpoint to resume")
        saved, buckets = _restore(checkpoint, fingerprint)
        stats = saved["statistics"]
    elif checkpoint is not None and checkpoint.exists():
        raise ValueError("checkpoint already exists; use resume or a new output directory")
    elif (initialize_one_island and time.monotonic() - invocation_start < budget_seconds
          and _rss_bytes() < memory_bytes):
        stats["one_island_initializer"] = _seed_one_island(
            options, ctxs, methods, saved["incumbents"], unit_count, k)
    else:
        stats["one_island_initializer"] = dict(evaluated_configurations=0, seconds=0,
            source="disabled_or_engineering_budget_reached", historical_rankings_used=False,
            pipeline_FPS_used=False)
    start = invocation_start
    last_progress = time.monotonic()
    last_checkpoint = last_progress
    pending = sum(len(values) for values in buckets.values())
    invocation_expansions = 0
    reason = None
    while saved["job_index"] < len(jobs):
        method, ctx, gate = jobs[saved["job_index"]]
        opts = options[method, ctx["id"]]
        edge_indices = defaultdict(list)
        for index, opt in enumerate(opts):edge_indices[opt.row["begin"], opt.row["device"]].append(index)
        _, suffix = _suffixes(opts, unit_count, max_vta_islands, method)
        if not saved["initialized"]:
            buckets = {(0, 0, 0, 0): {(ZERO_COMPLETE if method == "complete" else ZERO_TARNAWSKI): [()]}}
            pending = 1; saved["initialized"] = True
        while buckets or saved["active"] is not None:
            elapsed = saved["elapsed_seconds"] + time.monotonic() - start
            rss = _rss_bytes(); stats["peak_RSS_bytes"] = max(stats["peak_RSS_bytes"], rss)
            if root is not None and time.monotonic() - last_progress >= 5:
                root.mkdir(parents=True, exist_ok=True)
                progress = dict(status="RUNNING", algorithm=ALGORITHM,
                    input_sha256=fingerprint, method=method, context=ctx["id"], gate=gate,
                    completed_jobs=saved["job_index"], total_jobs=len(jobs),
                    elapsed_seconds=elapsed, pending_labels=pending, statistics=stats)
                temporary = root / "progress.json.tmp"
                temporary.write_text(json.dumps(progress, sort_keys=True) + "\n")
                temporary.replace(root / "progress.json")
                last_progress = time.monotonic()
            if checkpoint is not None and time.monotonic() - last_checkpoint >= 120:
                _checkpoint(checkpoint, {**saved, "elapsed_seconds": elapsed}, buckets)
                last_checkpoint = time.monotonic()
            if elapsed >= budget_seconds:reason = "time_budget"; break
            if rss >= memory_bytes:reason = "memory_budget"; break
            if max_expansions is not None and invocation_expansions >= max_expansions:
                reason = "explicit_expansion_limit"; break
            if saved["active"] is None:
                # Positions are topologically increasing.  This permits exact
                # merging of all incoming labels before a boundary is expanded.
                key = min(buckets)
                labels = buckets[key]
                label, paths = labels.popitem(); pending -= 1
                if not labels:del buckets[key]
                position, device_int, islands, used = key
                device = "cpu" if device_int == 0 else "vta"
                tail = suffix(position, device, islands)
                threshold = _threshold(saved["incumbents"][method], k)
                if tail is None:
                    continue
                if prune and threshold is not None and _value(label, ctx, method, gate, tail) > threshold:
                    stats["strict_bound_prunes"] += 1; continue
                saved["active"] = [list(key), list(label), [list(p) for p in paths], 0]
                stats["labels_expanded"] += 1; invocation_expansions += 1
            active = saved["active"]
            key, label, paths, at = active
            position, device_int, islands, used = key
            device = "cpu" if device_int == 0 else "vta"
            choices = edge_indices.get((position, device), ())
            if at >= len(choices):saved["active"] = None; continue
            index = choices[at]; active[3] += 1
            opt = opts[index]
            next_islands = islands + (device == "vta")
            if next_islands > max_vta_islands:continue
            next_used = used
            if device == "cpu" and gate == "resource_group":
                if not _compatible(used, opt.bits):continue
                next_used |= 1 << opt.bits
            end = opt.row["end"]; next_device = 1 - device_int
            tail = suffix(end, "cpu" if next_device == 0 else "vta", next_islands)
            if tail is None:continue
            new_label = _advance(label, opt, method, gate)
            threshold = _threshold(saved["incumbents"][method], k)
            stats["transitions"] += 1
            if prune and threshold is not None and _value(new_label, ctx, method, gate, tail) > threshold:
                stats["strict_bound_prunes"] += 1; continue
            if end == unit_count:
                value = _value(new_label, ctx, method, gate)
                for path in paths:
                    cfg = domain.candidate([opts[i].row for i in (*path, index)], ctx, gate)
                    _add_complete(saved["incumbents"][method], cfg, value, k)
                    stats["terminal_paths"] += 1
                continue
            new_key = (end, next_device, next_islands, next_used)
            labels = buckets.setdefault(new_key, {})
            if new_label not in labels:
                labels[new_label] = []; pending += 1
            else:stats["exact_labels_merged"] += 1
            target = labels[new_label]
            for path in paths:
                action = _insert_prefix(target, (*path, index), k)
                stats["equivalent_prefixes_discarded"] += action == "equivalent_discard"
            stats["peak_pending_labels"] = max(stats["peak_pending_labels"], pending)
        if reason is not None:break
        saved["job_index"] += 1; saved["initialized"] = False
    saved["elapsed_seconds"] += time.monotonic() - start
    complete = saved["job_index"] == len(jobs)
    result = dict(status="COMPLETE" if complete else "SEARCH_BUDGET_EXCEEDED",
                  algorithm=ALGORITHM, input_sha256=fingerprint, implementation=_implementation(),
                  methods={}, counts=domain_counts(programs, unit_count=unit_count,
                      max_vta_islands=max_vta_islands, contexts=ctxs), statistics=stats,
                  elapsed_seconds=saved["elapsed_seconds"], completed_jobs=saved["job_index"],
                  total_jobs=len(jobs), stop_reason=reason, pipeline_FPS_used=False,
                  exact_complete_domain=complete, independent_global_enumeration=False,
                  partial_rankings_are_not_certified=not complete)
    for method, rows in saved["incumbents"].items():
        result["methods"][method] = {"rankings": [{**r, "static_rank": i}
                                                    for i, r in enumerate(rows, 1)]}
    if checkpoint is not None:
        _checkpoint(checkpoint, saved, buckets)
        result["checkpoint"] = str(checkpoint)
        root.mkdir(parents=True, exist_ok=True)
        (root / "search_result.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    return result
