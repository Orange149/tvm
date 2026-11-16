"""Exhaustive finite-domain scoring, vectorized per topology with stable ties."""
import bisect
import random

from cpu_vta_partitioning.domain import configurations, candidate_id, topologies, signatures
from cpu_vta_partitioning.models import score


def rank(programs, costs, cores=4, k=20, seed=20261023):
    import numpy as np
    groups = "RCHAOGD"
    best = {g: [] for g in groups}
    selected = {}
    counts = {"topologies": 0, "shared_configurations": 0, "dedicated_configurations": 0}
    reservoir = {i: [] for i in (1, 2, 3)}
    seen = dict.fromkeys(reservoir, 0)
    rng = random.Random(seed)
    sigs = signatures(programs, cores)
    missing = set(sigs) - set(costs)
    if missing:
        raise ValueError("missing calibration signatures: " + str(sorted(missing)[:5]))
    # Additive vectors: CPU wall max is separate; VTA wall sums; all other
    # columns sum. This is independent of the scalar reference implementation.
    vectors = {}
    for key, s in sigs.items():
        c = costs[key]
        mask = sum(1 << x for x in s.mask)
        vectors[key] = np.array([
            c.service_ms if s.device == "cpu" else 0,
            c.service_ms if s.device == "vta" else 0,
            c.run_core_ms if s.device == "cpu" else 0,
            c.run_core_ms if s.device == "vta" else 0,
            c.boundary_core_ms,
            *[c.run_core_ms / t if s.device == "cpu" and s.threads <= t else 0 for t in range(1, cores + 1)],
            *[(c.run_core_ms + c.boundary_core_ms) / a.bit_count() if mask & a == mask else 0
              for a in range(1, 1 << cores)]])
    for topology in topologies(programs):
        counts["topologies"] += 1
        islands = sum(p["device"] == "vta" for p in topology)
        for dedicated in (False, True):
            candidates = list(configurations(topology, cores, dedicated))
            data = np.array([[vectors[s.signature] for s in c] for c in candidates])
            total = data.sum(axis=1)
            service = np.maximum(data[:, :, 0].max(axis=1), total[:, 1])
            c = np.maximum(service, (total[:, 2] + total[:, 4]) / cores)
            h = np.maximum(service, total[:, 2:5].sum(axis=1) / cores)
            old = total[:, 5:5 + cores].max(axis=1)
            generic = np.maximum(service, total[:, 5 + cores:].max(axis=1))
            values = {"R": service, "C": c, "H": h, "A": np.maximum(c, old),
                      "O": np.maximum(h, old), "G": generic, "D": generic}
            active = "D" if dedicated else "RCHAOG"
            counts["dedicated_configurations" if dedicated else "shared_configurations"] += len(candidates)
            for i, candidate in enumerate(candidates):
                key = None
                if not dedicated:
                    seen[islands] += 1
                    slot = rng.randrange(seen[islands])
                    if len(reservoir[islands]) < 39:
                        reservoir[islands].append(candidate)
                    elif slot < 39:
                        reservoir[islands][slot] = candidate
                for g in active:
                    value = round(float(values[g][i]), 9)
                    if len(best[g]) == k and value > best[g][-1][0]:
                        continue
                    if key is None:
                        key = candidate_id(candidate)
                    item = (value, key)
                    if len(best[g]) < k or item < best[g][-1]:
                        bisect.insort(best[g], item)
                        if len(best[g]) > k:
                            best[g].pop()
                        selected[key] = candidate
    output = {}
    for g, items in best.items():
        output[g] = [{"id": key, "score_ms": value, "stages": [s.to_dict() for s in selected[key]],
                      "components": score(selected[key], costs, cores)["components"]} for value, key in items]
    top_five = {r["id"] for rows in output.values() for r in rows[:5]}
    # Reservoir sampling makes the set uniform, not its slot order. Randomize
    # before taking a prefix, including strata with fewer than 39 candidates.
    for pool in reservoir.values():
        rng.shuffle(pool)
    random_rows = [{"id": candidate_id(c), "stages": [s.to_dict() for s in c]}
                   for pool in reservoir.values() for c in [c for c in pool if candidate_id(c) not in top_five][:4]]
    return {"counts": counts, "rankings": output, "stratified_random": random_rows,
            "identical_top1": {g: [h for h in groups if output[h] and output[g] and output[h][0]["id"] == output[g][0]["id"]]
                               for g in groups}}


def enumerate_reference(programs, costs, cores=4, k=20):
    """Slow independent scalar enumeration for small fixtures / full-domain audit."""
    best = {g: [] for g in "RCHAOGD"}
    for topology in topologies(programs):
        for dedicated in (False, True):
            for candidate in configurations(topology, cores, dedicated):
                scores, key = score(candidate, costs, cores), candidate_id(candidate)
                for g in ("D" if dedicated else "RCHAOG"):
                    item = (round(scores[g], 9), key)
                    if len(best[g]) < k or item < best[g][-1]:
                        bisect.insort(best[g], item)
                        del best[g][k:]
    return best


def screening_pool(ranking):
    rows = [r for group in ranking["rankings"].values() for r in group[:5]] + ranking["stratified_random"]
    return {r["id"]: r for r in rows}


def freeze_selection(ranking, measurements):
    # Measurements maps candidate -> boot -> II; never accepts confirmation rows.
    import statistics
    result = {}
    for group, rows in ranking["rankings"].items():
        five = rows[:5]
        for r in five:
            if len(measurements[r["id"]]) != 2:
                raise ValueError("selection requires two screening boots per candidate")
        winner = min(five, key=lambda r: (statistics.mean(measurements[r["id"]].values()), r["id"]))
        result[group] = {"1": rows[0]["id"], "5": winner["id"]}
    return result
