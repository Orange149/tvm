"""Research gates must preserve the meaning of the frozen ablations."""
from cpu_vta_partitioning.analysis import report


def example(values, mechanism=True):
    selection = {g: {b: g + b for b in ("1", "5")} for g in "RCHAOGD"}
    measured = {g + b: {f"boot{i}": values[b][g] * (1 + i / 100) for i in range(6)}
                for g in selection for b in ("1", "5")}
    return report(selection, measured, mechanism_supported=mechanism)


def test_local_capacity_effect_is_not_host_work_effect():
    selection = {g: {b: ("base" if g in "RCH" else "ours") for b in ("1", "5")} for g in "RCHAOGD"}
    result = report(selection, {k: {f"b{i}": v for i in range(6)} for k, v in (("base", 12.), ("ours", 10.))})
    assert all(v["gain"] == 0 and v["same_configuration"] for v in result["host_work_ablation_secondary"].values())
    assert not result["host_work_gate"] and result["local_cpu_constraint_gate"]
    assert result["direction"] == "effect_but_generic_or_isolation_explanation_unresolved"
    assert not result["continue_sensitivity"]


def test_better_dedicated_baseline_blocks_method_recommendation():
    values = {g: (10. if g == "O" else 8. if g == "D" else 12.) for g in "RCHAOGD"}
    result = example({"1": values, "5": values})
    assert result["comparisons"]["O_vs_D_budget1"]["gain"] < 0
    assert not result["continue_sensitivity"] and not result["performance_gate"]
    assert result["direction"] != "continue_method_research"


def test_comparisons_cannot_be_combined_across_budgets():
    first = dict.fromkeys("RCHAOGD", 12.); first.update(O=10., D=8.)
    fifth = dict.fromkeys("RCHAOGD", 12.); fifth.update(O=10., H=8.)
    result = example({"1": first, "5": fifth})
    assert result["budget_gates"]["1"]["local_cpu_constraint"]
    assert result["budget_gates"]["5"]["core_isolation"]
    assert result["eligible_budgets"] == []
    assert not result["continue_sensitivity"] and result["direction"] != "continue_method_research"


def test_direction_and_sensitivity_share_primary_and_mechanism_gate():
    values = dict.fromkeys("RCHAOGD", 12.); values.update(O=10., A=10.)
    for mechanism in (False, True):
        result = example({"1": values, "5": values}, mechanism)
        assert result["performance_gate"] and result["eligible_budgets"] == ["1", "5"]
        assert not result["host_work_gate"]  # O/A and H/C have zero selection effect
        assert result["continue_sensitivity"] == mechanism
        assert (result["direction"] == "continue_method_research") == mechanism
        if not mechanism:
            assert result["direction"] == "performance_gain_but_mechanism_unconfirmed"


def test_secondary_host_ablation_does_not_replace_primary_gate():
    values = dict.fromkeys("RCHAOGD", 10.); values.update(C=12., A=12.)
    result = example({"1": values, "5": values})
    assert result["host_work_gate"] and not result["local_cpu_constraint_gate"]
    assert not result["performance_gate"] and not result["continue_sensitivity"]
