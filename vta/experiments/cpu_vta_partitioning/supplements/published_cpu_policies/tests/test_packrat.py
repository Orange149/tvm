"""Paper objective, source-boundary counterexamples, and independent enumeration."""
import math
import random

import pytest

from cpu_policies.packrat import InfeasiblePackratError, METHOD_SCOPE, solve_packrat


CLEAN = {
    1: {1: 9.0, 2: 16.0, 4: 30.0},
    2: {1: 5.0, 2: 8.0, 4: 16.0},
    3: {1: 3.4, 2: 6.0, 4: 12.0},
    4: {1: 3.1, 2: 5.9, 4: 10.9},
}


def _pairs(result):
    return [(r["threads"], r["batch"]) for r in result["allocations"]]


def _enumerate(profile, threads, batch):
    """Enumerate replica multisets; no recurrence shared with the implementation."""
    items = sorted((t, b, cost) for t, row in profile.items() for b, cost in row.items())
    answers = []

    def visit(first, remaining_threads, remaining_batch, chosen):
        if remaining_batch == 0:
            pairs = tuple((t, b) for t, b, _ in chosen)
            answers.append((max(cost for _, _, cost in chosen),
                            sum(t for t, _, _ in chosen), len(chosen), pairs))
            return
        for i in range(first, len(items)):
            t, b, _ = items[i]
            if t <= remaining_threads and b <= remaining_batch:
                visit(i, remaining_threads - t, remaining_batch - b, chosen + [items[i]])

    visit(0, threads, batch, [])
    return min(answers) if answers else None


def test_original_normal_batch_split():
    result = solve_packrat(CLEAN, 4, 4)
    assert result["predicted_latency_ms"] == 8.0
    assert _pairs(result) == [(2, 2), (2, 2)]
    assert result["scope"] == METHOD_SCOPE
    assert result["used_threads"] == 4


def test_batch_one_has_no_replica_parallelism():
    result = solve_packrat(CLEAN, 4, 1)
    assert _pairs(result) == [(4, 1)]
    assert result["num_instances"] == 1


def test_nonmonotonic_curve_and_unused_core_boundary():
    # The public author's prototype returns L(4,1)=3 here. Equations (1)--(2)
    # permit using only three threads, whose actual measured latency is lower.
    profile = {1: {1: 8.0}, 2: {1: 4.0}, 3: {1: 2.0}, 4: {1: 3.0}}
    result = solve_packrat(profile, 4, 1)
    assert result["predicted_latency_ms"] == 2.0
    assert _pairs(result) == [(3, 1)]
    assert result["unused_threads"] == 1


def test_tied_latency_reconstruction_does_not_overallocate():
    # The unmodified prototype's equality-based solution() appends several
    # incompatible items for this tie and eventually raises Too many threads.
    profile = {
        1: {1: 6.0, 2: 10.0, 4: 16.0},
        2: {1: 4.0, 2: 6.0, 4: 10.0},
        3: {1: 3.8, 2: 5.0, 4: 8.0},
        4: {1: 3.7, 2: 4.7, 4: 7.5},
    }
    result = solve_packrat(profile, 4, 4)
    assert result["predicted_latency_ms"] == 6.0
    assert _pairs(result) == [(2, 2), (2, 2)]
    assert sum(t for t, b in _pairs(result)) == 4
    assert sum(b for t, b in _pairs(result)) == 4


def test_sparse_measured_pairs_only_and_explicit_infeasibility():
    assert _pairs(solve_packrat({2: {2: 3.0}}, 4, 4)) == [(2, 2), (2, 2)]
    with pytest.raises(InfeasiblePackratError, match="missing profile pairs"):
        solve_packrat({2: {2: 3.0}}, 4, 3)
    with pytest.raises(InfeasiblePackratError):
        solve_packrat({5: {1: 3.0}}, 4, 1)


def test_tie_uses_fewer_cores_then_fewer_instances():
    result = solve_packrat({1: {1: 1.0, 2: 1.0}, 2: {2: 1.0}}, 4, 2)
    assert _pairs(result) == [(1, 2)]
    assert result["used_threads"] == 1


@pytest.mark.parametrize("value", [None, True, "2", 0, -1, math.inf, math.nan])
def test_invalid_or_missing_cost_is_rejected(value):
    with pytest.raises(ValueError):
        solve_packrat({1: {1: value}}, 4, 1)


@pytest.mark.parametrize("profile", [{}, {0: {1: 1.0}}, {True: {1: 1.0}},
                                     {1: {}}, {1: {0: 1.0}}, {1: {True: 1.0}},
                                     {"cpu_subgraph": {1: 1.0}}])
def test_profile_is_not_a_heterogeneous_stage_table(profile):
    with pytest.raises(ValueError):
        solve_packrat(profile, 4, 1)


@pytest.mark.parametrize("threads,batch", [(0, 1), (-1, 1), (True, 1),
                                         (4, 0), (4, -1), (4, True), (4.0, 1)])
def test_budgets_are_positive_integers(threads, batch):
    with pytest.raises(ValueError):
        solve_packrat(CLEAN, threads, batch)


def test_independent_exhaustive_oracle_on_sparse_nonmonotonic_profiles():
    randomizer = random.Random(20261105)
    for _ in range(24):
        profile = {}
        for t in range(1, 5):
            row = {b: randomizer.choice([0.5, 1.0, 1.5, 2.0, 3.0, 5.0])
                   for b in (1, 2, 4) if randomizer.random() < 0.7}
            if row:
                profile[t] = row
        if not profile:
            profile = {1: {1: 1.0}}
        for threads in range(1, 5):
            for batch in range(1, 7):
                expected = _enumerate(profile, threads, batch)
                if expected is None:
                    with pytest.raises(InfeasiblePackratError):
                        solve_packrat(profile, threads, batch)
                else:
                    result = solve_packrat(profile, threads, batch)
                    actual = (result["predicted_latency_ms"], result["used_threads"],
                              result["num_instances"], tuple(_pairs(result)))
                    assert actual == expected
