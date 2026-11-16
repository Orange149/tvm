"""Unit tests for the real-task board boot qualification and selection rules."""

import inspect

from freeze_vta_c3_real_task_confirmation import confirmation_programs
import run_vta_c3_real_task_board_boot as board


def _selector():
    return {
        "workloads": {
            "W": {
                "incumbent_program_id": "inc",
                "orders": {
                    "bytes_only": ["bad", "fast", "slow"],
                    "compiler_cost_aware": ["fast", "bad", "slow"],
                    "fixed_random": ["slow", "bad", "fast"],
                },
            }
        }
    }


def _observations(program, latency):
    return [
        {"workload_id": "W", "program_id": program, "round": round_index, "latency_ms": latency}
        for round_index in range(board.SELECTION_ROUNDS)
        for _ in board.SEEDS
    ]


def test_fpga_rejected_program_is_removed_from_timing_union():
    plan = board.selection_timing_plan(_selector(), {"inc", "fast", "slow"})
    assert plan == {"W": ["fast", "inc", "slow"]}


def test_rejected_attempt_consumes_frozen_budget_without_latency():
    observations = (
        _observations("inc", 10.0)
        + _observations("fast", 8.0)
        + _observations("slow", 12.0)
    )
    decisions = board.choose_programs(
        _selector(), observations, {"inc", "fast", "slow"}
    )["W"]
    budget1 = decisions["budget_decisions"]["bytes_only"]["1"]
    assert budget1["attempted_new_program_ids"] == ["bad"]
    assert budget1["fpga_rejected_attempted_program_ids"] == ["bad"]
    assert budget1["eligible_program_ids"] == ["inc"]
    assert budget1["selected_program_id"] == "inc"
    budget2 = decisions["budget_decisions"]["bytes_only"]["2"]
    assert budget2["eligible_program_ids"] == ["inc", "fast"]
    assert budget2["selected_program_id"] == "fast"


def test_incumbent_rejection_disables_selector_evaluation():
    observations = _observations("fast", 8.0) + _observations("slow", 12.0)
    decision = board.choose_programs(_selector(), observations, {"fast", "slow"})["W"]
    assert decision["status"] == "incumbent_fpga_rejected_no_selector_evaluation"
    assert decision["budget_decisions"] == {}


def test_source_orders_qualification_before_any_selection_timing():
    source = inspect.getsource(board.run)
    assert source.index("\"fpga_correctness\"") < source.index(
        "\"selection_timing\", round_index, position"
    )
    assert source.index("write_json(output / \"selection_decisions_frozen.json\"") < source.index(
        "\"reference_warmup\""
    )


def test_confirmation_identity_union_always_keeps_incumbent():
    selector = _selector()
    decisions = {
        "W": {
            "budget_decisions": {
                "bytes_only": {
                    "1": {"selected_program_id": "inc"},
                    "2": {"selected_program_id": "fast"},
                },
                "fixed_random": {"1": {"selected_program_id": "slow"}},
            }
        }
    }
    assert confirmation_programs(selector, decisions) == {
        "W": ["fast", "inc", "slow"]
    }
