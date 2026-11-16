"""Independent implementation of Packrat's published replica/batch optimizer.

Source: Bhardwaj et al., ICML 2025, Sections 3.2--3.4, equations (1)--(2)
and the unnumbered ``opt[t,b]`` recurrence on PDF page 4.
https://proceedings.mlr.press/v267/bhardwaj25a.html

The input is a *single model's* isolated replica latency L[threads][batch].
Replicas process disjoint parts of one batch concurrently on disjoint core
groups. This module does not optimize dependent CPU subgraphs in a pipeline,
place an FPGA host, partition a graph, or provide a TVM execution adapter.

Only the published mathematical method is implemented here; no code from the
author's repository is incorporated. Its public reference implementation has
different zero-batch boundaries and a fragile equality-based reconstruction.
The paper's ``sum(threads) <= T`` constraint is implemented explicitly, so a
nonmonotonic thread curve is retained and unused cores are allowed.
"""

from collections.abc import Mapping
from math import inf, isfinite
from numbers import Real


PAPER_URL = "https://proceedings.mlr.press/v267/bhardwaj25a.html"
METHOD_SCOPE = "same_model_independent_replicas_and_batch_partitioning"


class InfeasiblePackratError(ValueError):
    """The supplied measured pairs cannot cover the batch within the budget."""


def _positive_integer(value, label):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _profile_items(latencies):
    if not isinstance(latencies, Mapping) or not latencies:
        raise ValueError("latencies must be a nonempty thread-to-batch mapping")
    items = []
    for threads, batches in latencies.items():
        _positive_integer(threads, "profile thread count")
        if not isinstance(batches, Mapping) or not batches:
            raise ValueError("each profile thread count needs measured batch latencies")
        for batch, value in batches.items():
            _positive_integer(batch, "profile batch size")
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError("profile latency must be a finite positive number")
            value = float(value)
            if not isfinite(value) or value <= 0:
                raise ValueError("profile latency must be a finite positive number")
            items.append((threads, batch, value))
    return sorted(items)


def solve_packrat(latencies, total_threads, total_batch):
    """Minimize the slowest replica's latency under Packrat's two constraints.

    Args:
        latencies: ``{threads: {batch: isolated_latency_ms}}`` for one model.
            Sparse measurements are supported, as in the paper's power-of-two
            batch profiling. Missing pairs are excluded, never interpolated.
        total_threads: Positive available physical-core/thread budget T.
        total_batch: Positive number B of inputs to divide among replicas.

    Returns:
        A JSON-serializable plan, with the explicit paper-method scope, predicted
        latency, budget use, and measured (threads, batch) pairs for each replica.

    Raises:
        ValueError: Invalid arguments or a missing/nonpositive/nonfinite cost.
        InfeasiblePackratError: No measured-pair combination satisfies both
            ``sum(batch) == B`` and ``sum(threads) <= T``.

    The first pass is the published min/max DP. A second, threshold-constrained
    batch DP reconstructs a valid plan without floating-point equality searches
    through partially unrelated states. Among minimum-latency plans it chooses
    fewer total threads, then fewer instances, then lexicographic (t,b) order.
    This deterministic tie rule is an implementation convention, not an extra
    optimization objective claimed by the paper.
    """
    budget = _positive_integer(total_threads, "total_threads")
    batch_size = _positive_integer(total_batch, "total_batch")
    measured = _profile_items(latencies)
    items = [x for x in measured if x[0] <= budget and x[1] <= batch_size]

    # Empty work costs zero for *every* available budget. This is what permits
    # leaving cores unused when adding threads does not improve the latency.
    opt = [[inf] * (batch_size + 1) for _ in range(budget + 1)]
    for threads in range(budget + 1):
        opt[threads][0] = 0.0
    for threads in range(1, budget + 1):
        for batch in range(1, batch_size + 1):
            best = inf
            for item_threads, item_batch, latency in items:
                if item_threads <= threads and item_batch <= batch:
                    previous = opt[threads - item_threads][batch - item_batch]
                    best = min(best, max(previous, latency))
            opt[threads][batch] = best
    optimum = opt[budget][batch_size]
    if not isfinite(optimum):
        raise InfeasiblePackratError(
            "no combination of measured positive-batch replicas covers "
            f"{batch_size} inputs with at most {budget} threads; "
            "missing profile pairs are not inferred"
        )

    # Every replica in an optimum has L(t,b) <= optimum. At this threshold,
    # minimum used threads is an ordinary additive exact-batch subproblem.
    feasible = [None] * (batch_size + 1)
    feasible[0] = (0, 0, ())
    allowed = [x for x in items if x[2] <= optimum]
    for batch in range(1, batch_size + 1):
        best = None
        for threads, item_batch, _ in allowed:
            if item_batch > batch or feasible[batch - item_batch] is None:
                continue
            old_threads, old_instances, old_plan = feasible[batch - item_batch]
            used = old_threads + threads
            if used > budget:
                continue
            plan = tuple(sorted((*old_plan, (threads, item_batch))))
            value = (used, old_instances + 1, plan)
            if best is None or value < best:
                best = value
        feasible[batch] = best
    reconstructed = feasible[batch_size]
    if reconstructed is None:
        raise RuntimeError("Packrat reconstruction disagrees with its latency DP")
    used, instances, plan = reconstructed
    lookup = {(t, b): latency for t, b, latency in items}
    allocations = [
        {"threads": t, "batch": b, "latency_ms": lookup[t, b]} for t, b in plan
    ]
    if (sum(a["batch"] for a in allocations) != batch_size
            or sum(a["threads"] for a in allocations) != used
            or max(a["latency_ms"] for a in allocations) != optimum):
        raise RuntimeError("invalid Packrat reconstruction")
    return {
        "method": "packrat_paper_minmax_dp",
        "scope": METHOD_SCOPE,
        "paper_url": PAPER_URL,
        "total_threads": budget,
        "total_batch": batch_size,
        "predicted_latency_ms": optimum,
        "used_threads": used,
        "unused_threads": budget - used,
        "num_instances": instances,
        "allocations": allocations,
        "unprofiled_pairs": "excluded_without_interpolation",
        "tie_break": "fewest_threads_then_fewest_instances_then_sorted_threads_batch",
    }


optimize_replicas = solve_packrat
