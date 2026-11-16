#!/usr/bin/env python3
"""Independently audit the frozen C1 fair-support identity and order invariants."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    contract = json.loads(Path(args.contract).read_text(encoding="utf-8"))
    failures = []
    support = contract.get("support", [])
    ids = [row.get("candidate_id") for row in support]
    support_set = set(ids)
    topology_counts = {}
    for row in support:
        topology_counts[row.get("topology_id")] = topology_counts.get(row.get("topology_id"), 0) + 1
        if row.get("realized_policy") != "ordinary_copy":
            failures.append("non-copy realized policy")
    if len(support) != 16 or len(support_set) != 16:
        failures.append("support identity cardinality")
    if len(topology_counts) != 8 or set(topology_counts.values()) != {2}:
        failures.append("topology multiplicity")
    if contract.get("support_identity_sha256") != canonical_hash(sorted(support_set)):
        failures.append("support identity hash")
    orders = contract.get("orders", {})
    deterministic = [orders.get(name, []) for name in ("O0_m1", "O1_topology_first", "O2_semantic_band")]
    random_orders = orders.get("O3_random", {})
    all_orders = deterministic + list(random_orders.values())
    if any(len(order) != 16 or set(order) != support_set for order in all_orders):
        failures.append("order support mismatch")
    first_candidates = {order[0] for order in all_orders if order}
    if len(first_candidates) != 1:
        failures.append("common first candidate")
    by_id = {row["candidate_id"]: row for row in support}
    expected_o0 = sorted(ids, key=lambda value: (float(by_id[value]["m1_score_ms"]), value))
    if deterministic[0] != expected_o0:
        failures.append("O0 M1 order")
    if len(deterministic[1]) == 16:
        first_topologies = [by_id[value]["topology_id"] for value in deterministic[1][:8]]
        second_topologies = [by_id[value]["topology_id"] for value in deterministic[1][8:]]
        if len(set(first_topologies)) != 8 or sorted(first_topologies) != sorted(second_topologies):
            failures.append("O1 topology-first partition")
    seeds = list(range(58001, 58021))
    if sorted(map(int, random_orders)) != seeds:
        failures.append("random seed set")
    elif len(first_candidates) == 1:
        fixed = next(iter(first_candidates))
        rest = sorted(support_set - {fixed})
        for seed in seeds:
            permutation = np.random.Generator(np.random.PCG64(seed)).permutation(len(rest))
            expected = [fixed] + [rest[int(index)] for index in permutation]
            if random_orders[str(seed)] != expected:
                failures.append("random order {}".format(seed))
                break
    base = sorted(support_set, key=lambda value: (hashlib.sha256(value.encode()).hexdigest(), value))
    for boot, offset in enumerate((0, 5, 10)):
        expected = base[offset:] + base[:offset]
        if contract.get("measurement_orders", {}).get(str(boot)) != expected:
            failures.append("measurement boot rotation {}".format(boot))
    if contract.get("label_free") is not True or contract.get("status") != "frozen_before_new_performance_labels":
        failures.append("label-free status")
    result = {
        "schema": "thesis_c1_fair_support_audit_v1",
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "support_size": len(support),
        "unique_candidate_ids": len(support_set),
        "unique_topologies": len(topology_counts),
        "orders_checked": len(all_orders),
        "random_orders_checked": len(random_orders),
        "hardware_calls": 0,
        "formal_stage04_support_frozen": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
