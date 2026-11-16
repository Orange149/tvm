"""Complete interval/resource DP, bounded terminal collectors and frozen selection.

The DP retains every half-path resource label.  Only COMPLETE configurations
are reduced to Top-K/Top-2.  A separate C++ scalar Cartesian enumerator verifies
the global and every stratified list.  Pipeline timings are never an input.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from itertools import combinations, product
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from . import domain
from .costs import Costs, score_configuration, quantize_ns, integer_weights

METHODS = ("enhanced", "single_core", "prefix")
GATES = ("off", "resource_group")
SCORE_SCALE = 1_000_000  # integer nanoseconds per millisecond
MAX_INT = np.iinfo(np.int64).max
MAX_COMPONENT_NS = MAX_INT // 64


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                   separators=(",", ":"), allow_nan=False) + "\n")
    temporary.replace(path)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _topologies(programs, unit_count, max_vta_islands):
    """Interval-DAG recurrence; retain legal paths rather than a scalar prefix."""
    edges = {}
    for program in programs.values():
        if not 0 <= program["begin"] < program["end"] <= unit_count:
            raise ValueError("invalid interval in qualified manifest")
        edges.setdefault((program["begin"], program["device"]), []).append(program)
    for rows in edges.values():
        rows.sort(key=lambda row: (row["end"], row["name"]))

    @lru_cache(None)
    def paths(position, device, islands):
        if position == unit_count:
            return ((),) if device == "vta" and 1 <= islands <= max_vta_islands else ()
        result = []
        for row in edges.get((position, device), ()):
            count = islands + (device == "vta")
            if count > max_vta_islands:
                continue
            for suffix in paths(row["end"], "vta" if device == "cpu" else "cpu", count):
                result.append((row["name"], *suffix))
        return tuple(result)

    return paths(0, "cpu", 0)


def _masks(pool):
    return tuple(mask for n in range(1, len(pool) + 1) for mask in combinations(pool, n))


def _shape(widths, bits):
    relations = []
    for a, b in combinations(bits, 2):
        relations.append(0 if a == b else 1 if not a & b else
                         2 if (a & b) in (a, b) else 3)
    return (len(widths), tuple(widths), tuple(relations))


def _shape_string(shape):
    n, widths, relations = shape
    return str(n) + ":" + ",".join(map(str, widths)) + ":" + ",".join(map(str, relations))


@lru_cache(maxsize=24)
def _allocation_domain(pool, n):
    options = _masks(pool)
    m = len(options)
    allocations = np.indices((m,) * n, dtype=np.int16).reshape(n, -1).T
    bit_options = np.asarray([sum(1 << c for c in mask) for mask in options], dtype=np.int16)
    widths = np.asarray([len(mask) for mask in options], dtype=np.int16)
    bits = bit_options[allocations]
    chosen_widths = widths[allocations]
    compatible = np.ones(len(allocations), dtype=bool)
    relation_columns = []
    for i, j in combinations(range(n), 2):
        a, b = bits[:, i], bits[:, j]
        intersection = a & b
        compatible &= (a == b) | (intersection == 0)
        relation_columns.append(np.where(a == b, 0, np.where(intersection == 0, 1,
                                np.where((intersection == a) | (intersection == b), 2, 3))))
    feature_columns = [chosen_widths]
    if relation_columns:
        feature_columns.append(np.stack(relation_columns, axis=1))
    features = np.concatenate(feature_columns, axis=1)
    unique, labels = np.unique(features, axis=0, return_inverse=True)
    shape_keys = [_shape_string((n, tuple(row[:n]), tuple(row[n:]))) for row in unique]
    prefix_options = {tuple(pool[:i]) for i in range(1, len(pool) + 1)}
    prefix = np.isin(allocations, [i for i, mask in enumerate(options) if mask in prefix_options]).all(axis=1)
    return {"options": options, "allocations": allocations, "bits": bits,
            "widths": chosen_widths, "compatible": compatible,
            "single": (chosen_widths == 1).all(axis=1), "prefix": prefix,
            "shape_labels": labels, "shape_keys": shape_keys}


def _ddr_stage(stage, ddr):
    if ddr is None:
        return 0.0                 # synthetic/offline fixtures only
    if hasattr(ddr, "stage"):
        value = ddr.stage(stage)
    else:
        table = ddr.get("stage_service_ms", ddr)
        if stage["program"] not in table:
            raise ValueError("missing DDR stage: " + stage["program"])
        value = table[stage["program"]]
        if isinstance(value, dict):
            if str(stage["threads"]) not in value:
                raise ValueError("missing DDR thread cost")
            value = value[str(stage["threads"])]
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("invalid DDR cost")
    return value


def _prepare(programs, costs, context, ddr):
    options = _masks(tuple(context["compute_pool"]))
    result = {}
    for name, program in sorted(programs.items()):
        masks = options if program["device"] == "cpu" else (tuple(context["host_mask"]),)
        values = []
        for mask in masks:
            row = domain.stage(program["device"], program["begin"], program["end"],
                               len(mask) if program["device"] == "cpu" else 1, mask)
            cost = costs.get(row, context)
            value = [quantize_ns(cost[key]) for key in ("service_ms", "run_core_ms", "boundary_core_ms")]
            value.append(quantize_ns(_ddr_stage(row, ddr)))
            if any(x > MAX_COMPONENT_NS for x in value):
                raise ValueError("isolated cost exceeds safe integer accumulation range")
            if program["device"] == "cpu":
                weights = integer_weights(None if getattr(costs, "distribution", "balanced") == "balanced"
                                          else cost.get("cpu_core_weights"), threads=len(mask))
            else:
                weights = ()
            value.extend([*weights, *([0] * (4 - len(weights)))])
            values.append(value)
        result[name] = {"cpu": program["device"] == "cpu", "begin": program["begin"],
                        "end": program["end"], "costs": values}
    return result


@dataclass(frozen=True)
class Terminal:
    score: int
    topology: int
    allocation: tuple
    context: str
    gate: str

    def json(self):
        return [self.score, self.topology, list(self.allocation), self.context, self.gate]

    @classmethod
    def from_json(cls, row):
        return cls(int(row[0]), int(row[1]), tuple(row[2]), row[3], row[4])


class Collectors:
    """Only compact terminal descriptors are retained for the stratified pool."""
    def __init__(self, topologies, programs, context, k):
        self.topologies, self.programs, self.context, self.k = topologies, programs, context, k
        self.options = _masks(tuple(context["compute_pool"]))
        self.global_best = {method: [] for method in METHODS}
        self.topology_best, self.shape_best = {}, {}
        self.counts = {method: 0 for method in METHODS}
        self.gate_counts = {gate: 0 for gate in GATES}

    def canonical(self, terminal):
        allocation = iter(terminal.allocation)
        rows = []
        for name in self.topologies[terminal.topology]:
            program = self.programs[name]
            mask = self.options[next(allocation)] if program["device"] == "cpu" else tuple(self.context["host_mask"])
            rows.append((program["device"], program["begin"], program["end"],
                         len(mask) if program["device"] == "cpu" else 1, mask))
        return (tuple(rows), tuple(self.context["compute_pool"]), tuple(self.context["host_mask"]),
                tuple(self.context["control_mask"]), tuple(self.context["reserved_cores"]), terminal.gate)

    def key(self, terminal):
        return terminal.score, self.canonical(terminal)

    def insert(self, pool, terminal, k):
        if len(pool) == k and terminal.score > pool[-1].score:
            return
        key = self.key(terminal)
        if len(pool) == k and key >= self.key(pool[-1]):
            return
        if terminal in pool:
            return
        pool.append(terminal)
        pool.sort(key=self.key)
        del pool[k:]

    def dump(self):
        return {"counts": self.counts, "gate_counts": self.gate_counts,
                "rankings": {key: [row.json() for row in rows] for key, rows in self.global_best.items()},
                "topology_strata": {key: [row.json() for row in rows] for key, rows in sorted(self.topology_best.items())},
                "resource_strata": {key: [row.json() for row in rows] for key, rows in sorted(self.shape_best.items())}}


def _first_k(scores, eligible, k):
    indices = np.flatnonzero(eligible)
    if len(indices) <= k:
        return indices[np.lexsort((indices, scores[indices]))]
    cutoff = np.partition(scores[indices], k - 1)[k - 1]
    strict = indices[scores[indices] < cutoff]
    tied = indices[scores[indices] == cutoff][:k - len(strict)]
    chosen = np.concatenate((strict, tied))
    return chosen[np.lexsort((chosen, scores[chosen]))]


def _group_first_two(scores, eligible, labels, count):
    """Linear-size group minima; no sorting of every complete allocation."""
    values = np.where(eligible, scores, MAX_INT)
    first_value = np.full(count, MAX_INT, dtype=np.int64)
    np.minimum.at(first_value, labels, values)
    indices = np.arange(len(scores), dtype=np.int64)
    winners = eligible & (values == first_value[labels])
    first_index = np.full(count, MAX_INT, dtype=np.int64)
    np.minimum.at(first_index, labels[winners], indices[winners])
    remaining = eligible & (indices != first_index[labels])
    second_value = np.full(count, MAX_INT, dtype=np.int64)
    np.minimum.at(second_value, labels[remaining], values[remaining])
    winners = remaining & (values == second_value[labels])
    second_index = np.full(count, MAX_INT, dtype=np.int64)
    np.minimum.at(second_index, labels[winners], indices[winners])
    return first_index, second_index


def _allocate_integer(run_ns, units):
    """Exact conservative apportionment, independently repeated by the oracle."""
    units = [int(unit) for unit in units]
    denominator = sum(units)
    if denominator <= 0:
        raise ValueError("empty integer CPU work weights")
    products = [int(run_ns) * int(unit) for unit in units]
    result = [value // denominator for value in products]
    missing = int(run_ns) - sum(result)
    order = sorted(range(len(units)), key=lambda i: (-(products[i] % denominator), i))
    for i in order[:missing]:
        result[i] += 1
    return result


def _half_vectors(cpu_names, table, options):
    """Resource-vector DP, extending every label by one stage option.

    Each half has at most two CPU stages.  No half label is removed based on
    its standalone score: its load can complement the other half.
    """
    states = {"core": np.zeros((1, 4), dtype=np.int64),
              "groups": np.zeros((1, len(options)), dtype=np.int64),
              "prefix": np.zeros((1, 4), dtype=np.int64),
              "cpu_max": np.zeros(1, dtype=np.int64), "ddr": np.zeros(1, dtype=np.int64)}
    m = len(options)
    for name in cpu_names:
        values = np.asarray(table[name]["costs"], dtype=np.int64)
        core = np.zeros((m, 4), dtype=np.int64)
        groups = np.zeros((m, m), dtype=np.int64)
        prefix = np.zeros((m, 4), dtype=np.int64)
        for j, mask in enumerate(options):
            core[j, list(mask)] = _allocate_integer(values[j, 1], values[j, 4:4 + len(mask)])
            core[j, mask[0]] += values[j, 2]
            groups[j, j] = values[j, 0]
            # Keep conserved confined work. Divide once, after both halves have
            # joined, so addition order cannot affect ties.
            prefix[j, len(mask) - 1:] = values[j, 1]
        count = len(states["cpu_max"])
        for key, addition in (("core", core), ("groups", groups), ("prefix", prefix)):
            states[key] = (states[key][:, None, :] + addition[None, :, :]).reshape(count * m, -1)
        states["cpu_max"] = np.maximum(states["cpu_max"][:, None], values[None, :, 0]).reshape(-1)
        states["ddr"] = (states["ddr"][:, None] + values[None, :, 3]).reshape(-1)
    return states


def _join_values(cpu_names, topology, table, context, alloc):
    """Join matching interval halves; at most 225 x 225 terminal vectors."""
    options = alloc["options"]
    split = 1 if len(cpu_names) == 2 else 2
    left = _half_vectors(cpu_names[:split], table, options)
    right = _half_vectors(cpu_names[split:], table, options)
    nr = len(right["cpu_max"])
    nl = len(left["cpu_max"])
    def add(key):
        return (left[key][:, None, ...] + right[key][None, :, ...]).reshape(nl * nr, *left[key].shape[1:])
    core, groups, prefix = add("core"), add("groups"), add("prefix")
    cpu_max = np.maximum(left["cpu_max"][:, None], right["cpu_max"][None, :]).reshape(-1)
    ddr = add("ddr")
    vta = host = 0
    for name in topology:
        if not table[name]["cpu"]:
            wall, run, boundary, memory = table[name]["costs"][0][:4]
            vta += wall
            host += run + boundary
            ddr += memory
    service = np.maximum(np.maximum(cpu_max, vta), ddr)
    demand = core.max(axis=1)
    host_mask = context["host_mask"]
    if len(host_mask) == 1:
        demand = np.maximum(demand, core[:, host_mask[0]] + host)
    elif len(host_mask) == 4:
        demand = np.maximum(demand, (core.sum(axis=1) + host + 3) // 4)
    else:
        # General allowed-set form, retained for small adversarial fixtures.
        host_bits = sum(1 << c for c in host_mask)
        for subset in range(1, 16):
            mask = [c for c in range(4) if subset & (1 << c)]
            work = core[:, mask].sum(axis=1)
            if host_bits & subset == host_bits:
                work += host
            demand = np.maximum(demand, (work + len(mask) - 1) // len(mask))
    enhanced = np.maximum(service, demand)
    serial = np.maximum(enhanced, groups.max(axis=1))
    old_prefix = np.maximum(service, (core.sum(axis=1) + host + 3) // 4)
    widths = np.arange(1, 5, dtype=np.int64)
    old_prefix = np.maximum(old_prefix, ((prefix + widths - 1) // widths).max(axis=1))
    old_prefix = np.maximum(old_prefix, (host + len(host_mask) - 1) // len(host_mask))
    return {"enhanced": enhanced, "serial": serial, "prefix": old_prefix}


def _rank_context(programs, graphs, table, context, k):
    collectors = Collectors(graphs, programs, context, k)
    pool = tuple(context["compute_pool"])
    started = time.monotonic()
    for graph_index, topology in enumerate(graphs):
        cpus = tuple(name for name in topology if table[name]["cpu"])
        alloc = _allocation_domain(pool, len(cpus))
        scores = _join_values(cpus, topology, table, context, alloc)
        all_ = np.ones(len(alloc["allocations"]), dtype=bool)
        def terminal(i, score, gate):
            return Terminal(int(score), graph_index, tuple(map(int, alloc["allocations"][i])), context["id"], gate)
        for gate in GATES:
            eligible = all_ if gate == "off" else alloc["compatible"]
            values = scores["enhanced" if gate == "off" else "serial"]
            number = int(eligible.sum())
            collectors.counts["enhanced"] += number
            collectors.gate_counts[gate] += number
            topology_key = str(graph_index) + "/" + gate
            local = []
            for i in _first_k(values, eligible, max(k, 2)):
                row = terminal(i, values[i], gate)
                collectors.insert(collectors.global_best["enhanced"], row, k)
                collectors.insert(local, row, 2)
            collectors.topology_best[topology_key] = local
            first, second = _group_first_two(values, eligible, alloc["shape_labels"], len(alloc["shape_keys"]))
            for shape_index in np.flatnonzero(first != MAX_INT):
                key = alloc["shape_keys"][shape_index] + "/" + gate
                target = collectors.shape_best.setdefault(key, [])
                for index in (first[shape_index], second[shape_index]):
                    if index != MAX_INT:
                        collectors.insert(target, terminal(index, values[index], gate), 2)
        for method, eligible, source, gate in (
            ("single_core", alloc["single"], "serial", "resource_group"),
            ("prefix", alloc["prefix"], "prefix", "off")):
            collectors.counts[method] += int(eligible.sum())
            for i in _first_k(scores[source], eligible, k):
                collectors.insert(collectors.global_best[method], terminal(i, scores[source][i], gate), k)
        if (graph_index + 1) % 500 == 0:
            print(f"[完整资源DP] {context['id']} {graph_index + 1}/{len(graphs)} 拓扑", flush=True)
    result = collectors.dump()
    result.update(context=context, topologies=len(graphs), algorithm="bidirectional_interval_resource_vector_dp",
                  seconds=time.monotonic() - started, maximum_terminal_block=15**4,
                  half_label_pruning=False)
    return result


def _oracle_executable(output_dir):
    source = Path(__file__).with_name("oracle.cc")
    if not source.is_file():
        raise ValueError("independent scalar oracle source missing")
    # The verified source closure travels with a relocated bundle.  Never pick
    # an unrelated repository header merely because one is available locally.
    include = source.parent
    if not (include / "picojson.h").is_file():
        raise ValueError("picojson header missing from oracle source closure")
    compiler = shutil.which("c++") or shutil.which("g++")
    if compiler is None:
        raise RuntimeError("C++ compiler required for independent full-domain verification")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256(source.read_bytes() + (include / "picojson.h").read_bytes()).hexdigest()
    executable = directory / ("scalar_oracle_" + fingerprint[:20])
    if not executable.is_file():
        result = subprocess.run([compiler, "-O3", "-std=c++17", "-I", str(include), str(source), "-o", str(executable)],
                                capture_output=True, text=True, timeout=120)
        if result.returncode:
            raise RuntimeError("scalar oracle compilation failed: " + result.stderr)
    return executable


def _verify_context(payload, result, output_dir):
    executable = _oracle_executable(output_dir)
    started = time.monotonic()
    checked = subprocess.run([str(executable)], input=json.dumps(payload, separators=(",", ":")),
                             text=True, capture_output=True, timeout=7200)
    if checked.returncode:
        raise RuntimeError("independent scalar enumeration failed: " + checked.stderr[-3000:])
    reference = json.loads(checked.stdout)
    fields = ("counts", "gate_counts", "rankings", "topology_strata", "resource_strata")
    for field in fields:
        if result[field] != reference[field]:
            _write(Path(output_dir) / (payload["context"]["id"] + "_oracle_mismatch.json"), reference)
            raise ValueError("DP / independent scalar enumeration mismatch: " + field)
    return {"passed": True, "algorithm": "independent_cpp_scalar_cartesian_enumeration",
            "compared": list(fields), "seconds": time.monotonic() - started,
            "result_sha256": _digest({field: result[field] for field in fields}),
            "source_sha256": hashlib.sha256(Path(__file__).with_name("oracle.cc").read_bytes()).hexdigest()}


def _configuration(terminal, graphs, programs, contexts):
    context = contexts[terminal.context]
    options = _masks(tuple(context["compute_pool"]))
    allocation = iter(terminal.allocation)
    stages = []
    for name in graphs[terminal.topology]:
        program = programs[name]
        mask = options[next(allocation)] if program["device"] == "cpu" else tuple(context["host_mask"])
        stages.append(domain.stage(program["device"], program["begin"], program["end"],
                                   len(mask) if program["device"] == "cpu" else 1, mask))
    return domain.candidate(stages, context, terminal.gate)


def topology_key(configuration):
    return tuple((s["device"], s["begin"], s["end"]) for s in configuration["stages"])


def resource_key(configuration):
    masks = [tuple(s["mask"]) for s in configuration["stages"] if s["device"] == "cpu"]
    return _shape([len(m) for m in masks], [sum(1 << c for c in m) for m in masks])


def _row_key(row):
    score = row.get("score_ns")
    if score is None:
        score = quantize_ns(row["score_ms"])
    if type(score) is not int or score < 0:
        raise ValueError("invalid integer static score")
    return score, domain.canonical_tuple(row)


def select_diverse(pure_static20, stratified_pool, *, count=20, pure_count=10, near_optimal_ratio=1.15):
    if count != 20 or pure_count != 10 or not math.isfinite(near_optimal_ratio) or near_optimal_ratio < 1:
        raise ValueError("unsupported frozen selection protocol")
    if not pure_static20:
        raise ValueError("cannot select from an empty legal domain")
    pure = sorted(pure_static20, key=_row_key)
    chosen, identities = [], set()
    def add(row, reason):
        if row["id"] not in identities:
            chosen.append({**row, "selection_reason": reason})
            identities.add(row["id"])
    for row in pure[:pure_count]:
        add(row, "pure_static_top10")
    ratio = Fraction(str(near_optimal_ratio))
    best_score = _row_key(pure[0])[0]
    pool = {row["id"]: row for row in [*pure, *stratified_pool]
            if _row_key(row)[0] * ratio.denominator <= best_score * ratio.numerator}
    ordered = sorted(pool.values(), key=_row_key)
    known_topologies = {topology_key(row) for row in chosen}
    cut_added = 0
    for row in ordered:
        key = topology_key(row)
        if row["id"] not in identities and key not in known_topologies:
            add(row, "new_topology_within_15_percent")
            known_topologies.add(key)
            cut_added += 1
            if cut_added == 5:
                break
    resource_added = 0
    known_resources = {resource_key(row) for row in chosen}
    for row in ordered:
        key = resource_key(row)
        if row["id"] not in identities and key not in known_resources:
            add(row, "new_cpu_resource_shape_within_15_percent")
            known_resources.add(key)
            resource_added += 1
            if resource_added == 5:
                break
    # If CPU shape diversity is exhausted, host role/gate is secondary diversity.
    known_roles = {(domain.context(row["context_id"])["family"], row["execution_mode"]) for row in chosen}
    for row in ordered:
        if resource_added == 5:
            break
        role = (domain.context(row["context_id"])["family"], row["execution_mode"])
        if row["id"] not in identities and role not in known_roles:
            add(row, "new_host_or_execution_role_within_15_percent")
            known_roles.add(role)
            resource_added += 1
    for row in pure:
        if len(chosen) == count:
            break
        add(row, "pure_static_fallback")
    if len(chosen) != min(count, len(pure)):
        raise ValueError("selection failed to fill the available Top-20 budget")
    for rank, row in enumerate(chosen, 1):
        row["selection_order"] = rank
    return chosen


def rank_all(programs, records, *, ddr=None, output_dir=None, unit_count=21,
             max_vta_islands=3, verify=True, contexts=None, k=20):
    """Joint full-domain ranking; output_dir stores resumable context checkpoints.

    `records` contains isolated component costs only.  No path to measured
    pipeline FPS is accepted.  None DDR is reserved for synthetic fixtures.
    """
    if type(k) is not int or k < 1:
        raise ValueError("invalid static ranking budget")
    selected_contexts = tuple(contexts or domain.contexts())
    context_map = {c["id"]: c for c in selected_contexts}
    if len(context_map) != len(selected_contexts):
        raise ValueError("duplicate context identity")
    costs = records if hasattr(records, "get") and not isinstance(records, dict) else Costs(records, programs=programs, ddr=ddr)
    graphs = _topologies(programs, unit_count, max_vta_islands)
    if not graphs:
        raise ValueError("no complete legal CPU-VTA topology")
    if max(sum(programs[p]["device"] == "cpu" for p in g) for g in graphs) > 4:
        raise ValueError("this bounded DP implementation supports at most four CPU stages")
    outputs = []
    started = time.monotonic()
    implementation = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                      for name in ("search.py", "oracle.cc", "picojson.h")}
    for context in selected_contexts:
        table = _prepare(programs, costs, context, ddr)
        payload = {"programs": table, "topologies": graphs, "context": context,
                   "k": k, "score_scale": SCORE_SCALE, "implementation": implementation}
        fingerprint = _digest(payload)
        saved = Path(output_dir) / (context["id"] + ".json") if output_dir else None
        if saved is not None and saved.is_file():
            result = json.loads(saved.read_text())
            if result.get("checkpoint_sha256") != _digest({key: value for key, value in result.items()
                                                           if key != "checkpoint_sha256"}):
                raise ValueError("search checkpoint content hash differs")
            if result.get("input_sha256") != fingerprint:
                raise ValueError("search checkpoint inputs changed; use a new search directory")
            proof = result.get("verification", {})
            if proof.get("passed"):
                fields = ("counts", "gate_counts", "rankings", "topology_strata", "resource_strata")
                if (proof.get("source_sha256") != implementation["oracle.cc"]
                        or proof.get("result_sha256") != _digest({field: result[field] for field in fields})):
                    raise ValueError("search checkpoint oracle proof differs")
        else:
            result = _rank_context(programs, graphs, table, context, k)
            result["input_sha256"] = fingerprint
        if verify and not result.get("verification", {}).get("passed"):
            oracle_dir = Path(output_dir) if output_dir else Path("/tmp/cpu-vta-open-resource-oracle")
            result["verification"] = _verify_context(payload, result, oracle_dir)
        if saved is not None:
            result["checkpoint_sha256"] = _digest({key: value for key, value in result.items()
                                                  if key != "checkpoint_sha256"})
            _write(saved, result)
        outputs.append(result)
    def make(terminal):
        config = _configuration(terminal, graphs, programs, context_map)
        return {**config, "score_ns": terminal.score, "score_ms": terminal.score / SCORE_SCALE}
    per_method = {method: [] for method in METHODS}
    pool_refs = set()
    for output in outputs:
        for method in METHODS:
            per_method[method].extend(Terminal.from_json(row) for row in output["rankings"][method])
        for name in ("topology_strata", "resource_strata"):
            for rows in output[name].values():
                pool_refs.update(Terminal.from_json(row) for row in rows)
    methods = {}
    for method, terminals in per_method.items():
        rows = sorted((make(row) for row in terminals), key=_row_key)[:k]
        for rank, row in enumerate(rows, 1):
            row["static_rank"] = rank
        methods[method] = {"rankings": rows, "method": method,
                           "candidate_count": sum(o["counts"][method] for o in outputs)}
    pure = methods["enhanced"]["rankings"]
    # Build JSON configurations only for the bounded terminal pool, never the
    # hundreds of millions of discarded allocations.
    pool = [make(row) for row in pool_refs]
    diverse = select_diverse(pure, pool) if k == 20 else []
    result = {"experiment": "resnet18_open_cpu_resource", "algorithm": "bidirectional_interval_resource_vector_dp",
              "ranking_uses_pipeline_feedback": False, "methods": methods,
              "pure_static20": pure, "diverse20": diverse,
              "counts": {"topologies": len(graphs), "contexts": len(outputs),
                         **{m: sum(o["counts"][m] for o in outputs) for m in METHODS}},
              "per_context": {o["context"]["id"]: {"context": o["context"], "counts": o["counts"],
                  "gate_counts": o["gate_counts"], "seconds": o["seconds"],
                  "verification": o.get("verification", {"passed": False}),
                  "input_sha256": o["input_sha256"]} for o in outputs},
              "selection_protocol": {"pure": 10, "new_topologies": 5, "new_resources": 5,
                  "near_optimal_ratio": 1.15, "stratum_retention": 2,
                  "fallback": "pure_static_order", "optimal_diversity_claimed": False,
                  "resource_relations": ["same", "disjoint", "containment", "crossing_overlap"],
                  "tie_order": "quantized_score_then_canonical_execution_tuple"},
              "score_quantization_ms": 1 / SCORE_SCALE,
              "integer_accounting": {"primitive_rounding": "Decimal(str(ms))*1e6 ROUND_HALF_UP",
                  "core_allocation": "integer weighted largest remainder; ties by logical slot; sum conserved",
                  "capacity_division": "ceil once after conserved work accumulation",
                  "weight_units": "balanced integer ones; session weights rounded to 1e12 units",
                  "oracle_transport": "exact JSON int64, not floating score rounding"},
              "implementation": implementation,
              "model_formulas": {
                  "enhanced": "max(CPU stage service, all VTA service, physical-core demand + movable-host capacity, enabled serial-group service, DDR)",
                  "single_core": "enhanced model restricted to singleton physical CPU devices with reused-core serial service",
                  "prefix": "max(CPU stage service, all VTA service, all core work / 4, nested CPU run-work / prefix width, host work / host-mask width, DDR); adapted original prefix model, no new CPU hotspot term"},
              "stratified_pool_size": len(pool_refs), "seconds": time.monotonic() - started,
              "verification": {"passed": verify and all(o.get("verification", {}).get("passed") for o in outputs),
                  "scope": "all complete allocations, global Top-K and every Top-2 stratum"},
              "assumptions": {"cpu_work_split": "balanced run_core_ms / configured_threads; boundary to caller",
                  "host_migration": "allowed-set ideal capacity bound", "control_work": "not separately modeled",
                  "same_size_cpu_masks": "homogeneous core permutation, not individually measured",
                  "ddr": "synthetic zero" if ddr is None else "explicit stage table; summed once"}}
    if output_dir:
        _write(Path(output_dir) / "stratified_pool.json", pool)
        _write(Path(output_dir) / "rankings.json", result)
    return result


def select_pool(rankings, references=()):
    """Pure20 + diverse20 + baseline5 each + three historical references <=43."""
    selected = [*rankings["pure_static20"], *rankings["diverse20"],
                *rankings["methods"]["single_core"]["rankings"][:5],
                *rankings["methods"]["prefix"]["rankings"][:5], *list(references)]
    result = {}
    for row in selected:
        config = domain.candidate(row["stages"], row["context_id"], row["execution_mode"])
        if row.get("id", config["id"]) != config["id"]:
            raise ValueError("pool contains a configuration with altered identity")
        if config["id"] in result and result[config["id"]] != config:
            raise ValueError("duplicate configuration identity differs")
        result[config["id"]] = config
    if len(result) > 43:
        raise ValueError("frozen pipeline measurement budget exceeds 43")
    return dict(sorted(result.items()))
