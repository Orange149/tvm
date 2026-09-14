import pytest

from validate_vta_full_pool_command_capacity import make_plan


def _row(index, workload, mode, insn, uop, outcome="passed_local_signature"):
    return {
        "candidate_id": "candidate-{}".format(index),
        "workload_id": workload,
        "public_mode": mode,
        "outcome": outcome,
        "command_signature": {
            "structural": {"peaks": {"insn_bytes": insn, "uop_bytes": uop}}
        },
    }


def test_make_plan_uses_independent_queue_maxima_and_page_alignment():
    rows = [
        _row(i, "W{:02d}".format(i % 3), "mode{}".format(i % 2), 608, 20)
        for i in range(197)
    ]
    rows[7]["command_signature"]["structural"]["peaks"]["insn_bytes"] = 23008
    rows[19]["command_signature"]["structural"]["peaks"]["uop_bytes"] = 10820
    plan = make_plan(rows)
    assert plan["insn_capacity_bytes"] == 24576
    assert plan["uop_capacity_bytes"] == 12288
    assert plan["total_capacity_bytes"] == 36864


def test_make_plan_rejects_a_nonpassing_baseline():
    rows = [_row(i, "W00", "original", 608, 20) for i in range(197)]
    rows[0]["outcome"] = "negative_wrong_answer"
    with pytest.raises(ValueError, match="must pass"):
        make_plan(rows)
