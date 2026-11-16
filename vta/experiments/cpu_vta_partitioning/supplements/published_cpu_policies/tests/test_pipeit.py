"""Hand-computable synthetic fixtures, not numerical data from Pipe-it."""
from copy import deepcopy
from itertools import combinations

import pytest

from cpu_policies.pipeit import CoreGroup, find_split, merge_stage, work_flow


def test_find_split_transfers_suffix_and_keeps_layer_order():
    layers = [0, 1, 2, 3]
    assert find_split(layers, [1, 1, 1, 1], [2, 2, 2, 2]) == ([0, 1, 2], [3])
    assert layers == [0, 1, 2, 3]


def test_paper_strict_inequality_rejects_equal_balance():
    assert find_split([0, 1], [1, 1], [1, 1]) == ([0, 1], [])


def test_decimal_equal_balance_is_not_changed_by_subtraction_roundoff():
    assert find_split([0, 1], [.1, .2], [.1, .1]) == ([0, 1], [])


def test_paper_heuristic_is_not_replaced_with_exact_minimax():
    fast, slow = [2, 2], [3, 3]
    left, right = find_split([0, 1], fast, slow)
    actual = max(sum(fast[i] for i in left), sum(slow[i] for i in right))
    oracle = min(max(sum(fast[:cut]), sum(slow[cut:])) for cut in range(3))
    assert actual == 4
    assert oracle == 3


def test_workflow_two_stage_hand_calculation_and_input_preservation():
    columns = {("fast", 1): [1, 1, 1, 1], ("slow", 1): [2, 2, 2, 2]}
    saved = deepcopy(columns)
    result = work_flow(columns, [CoreGroup("fast", (0,)), CoreGroup("slow", (1,))])
    assert result["layers"] == [[0, 1, 2], [3]]
    assert result["stage_times"] == [3, 2]
    assert result["interval"] == 3
    assert result["throughput"] == pytest.approx(1 / 3)
    assert result["passes"] == 2
    assert columns == saved


def test_workflow_three_stage_strict_pseudocode_fixture():
    result = work_flow({("cpu", 1): [1] * 7}, [CoreGroup("cpu", (i,)) for i in range(3)])
    assert result["layers"] == [[0, 1, 2, 3], [4, 5], [6]]
    assert result["stage_times"] == [4, 2, 1]
    assert [i for group in result["layers"] for i in group] == list(range(7))


def test_merge_retries_merged_stage_and_conserves_every_physical_core():
    columns = {("cpu", 1): [8, 8, 1, 1], ("cpu", 2): [5, 5, .6, .6],
               ("cpu", 3): [3.6, 3.6, .4, .4], ("cpu", 4): [3, 3, .3, .3]}
    result = merge_stage(columns, {"cpu": [0, 1, 2, 3]}, variant="paper_break_retry_merged")
    assert result["groups"] == [{"core_type": "cpu", "core_count": 4, "cores": [0, 1, 2, 3]}]
    assert result["layers"] == [[0, 1, 2, 3]]
    assert result["interval"] == pytest.approx(6.6)
    assert sum(step["accepted"] for step in result["merge_history"]) == 3


def test_merge_never_combines_different_core_types():
    result = merge_stage({("big", 1): [8, 2, 1], ("big", 2): [5, 1.2, .6],
                          ("small", 1): [16, 4, 2], ("small", 2): [10, 2.4, 1.2]},
                         {"big": [4, 5], "small": [0, 1]}, variant="paper_break_retry_merged")
    inventory = {g["core_type"]: set() for g in result["groups"]}
    for group in result["groups"]:
        inventory[group["core_type"]].update(group["cores"])
    assert inventory == {"big": {4, 5}, "small": {0, 1}}
    assert [i for group in result["layers"] for i in group] == [0, 1, 2]


def test_merge_eq12_is_strict_and_empty_stage_is_retained():
    result = merge_stage({("cpu", 1): [1], ("cpu", 2): [1]}, {"cpu": [0, 1]},
                         variant="paper_break_retry_merged")
    assert len(result["groups"]) == 2
    assert result["layers"] == [[0], []]
    assert result["merge_history"][0]["accepted"] is False


def test_merge_traversal_interpretation_is_explicit_and_tested():
    # Deliberately non-concave layer speedups expose traversal behavior. This
    # fixture is not evidence for the paper's concavity/order assumptions.
    columns = {("cpu", 1): [4, 4, 4, 4], ("cpu", 2): [8, 8, 8, 1], ("cpu", 3): [9, 9, 9, 1]}
    stopped = merge_stage(columns, {"cpu": [0, 1, 2]}, variant="paper_break_retry_merged")
    continued = merge_stage(columns, {"cpu": [0, 1, 2]}, variant="continue_adjacent_retry_merged")
    assert len(stopped["groups"]) == 3
    assert len(continued["groups"]) == 2
    assert continued["groups"][1]["cores"] == [1, 2]


def test_nonoverlap_is_a_pipeit_resource_contract():
    with pytest.raises(ValueError, match="share physical cores"):
        work_flow({("cpu", 1): [1, 2]}, [CoreGroup("cpu", (0,)), CoreGroup("cpu", (0,))])
    with pytest.raises(ValueError, match="share physical cores"):
        merge_stage({("big", 1): [1], ("small", 1): [2]}, {"big": [0], "small": [0]},
                    variant="paper_break_retry_merged")


@pytest.mark.parametrize("times", [[float("nan")], [float("inf")], [-1], [True]])
def test_invalid_measured_costs_are_rejected(times):
    with pytest.raises(ValueError):
        work_flow({("cpu", 1): times}, [CoreGroup("cpu", (0,))])


def test_mismatched_cost_coverage_and_unordered_workload_are_rejected():
    with pytest.raises(ValueError, match="layer counts"):
        work_flow({("cpu", 1): [1], ("cpu", 2): [1, 2]}, [CoreGroup("cpu", (0,))])
    with pytest.raises(ValueError, match="missing matched"):
        work_flow({("cpu", 1): [1]}, [CoreGroup("cpu", (0, 1))])
    with pytest.raises(ValueError, match="original order"):
        find_split([1, 0], [1, 1], [2, 2])
    with pytest.raises(ValueError, match="traversal"):
        merge_stage({("cpu", 1): [1]}, {"cpu": [0]}, variant="unreported_variant")


def test_small_workflow_results_cover_every_layer_without_reordering():
    # Property fixture checks ownership/order, not equality to an exact solver.
    for boundaries in combinations(range(1, 5), 2):
        values = [1 + int(i in boundaries) for i in range(5)]
        result = work_flow({("cpu", 1): values}, [CoreGroup("cpu", (i,)) for i in range(3)])
        assert [i for group in result["layers"] for i in group] == list(range(5))
