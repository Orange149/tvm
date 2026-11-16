import json
from pathlib import Path

import run_vta_c3_holdout_board_boot_v2 as subject


HERE = Path(__file__).resolve().parent
GROUPED = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout"


def test_selection_uses_three_rounds_and_one_sample_per_round():
    rows = [
        {"program_id": "p", "phase": "selection_timing", "round": index,
         "latency_ms": value}
        for index, value in enumerate((3.0, 1.0, 2.0))
    ]
    assert subject.selection_statistics(rows, "p") == {
        "values_ms": [3.0, 1.0, 2.0], "median_ms": 2.0,
    }


def test_formal_statistics_require_six_rounds_times_three_seeds():
    rows = []
    for round_index in range(subject.FORMAL_ROUNDS):
        for seed_index in range(len(subject.SEEDS)):
            rows.append({
                "program_id": "p", "phase": "confirmation_timing",
                "round": round_index, "latency_ms": 10 * round_index + seed_index,
            })
    stats = subject.three_seed_round_statistics(rows, "p", "confirmation_timing")
    assert stats["round_medians_ms"] == [1.0, 11.0, 21.0, 31.0, 41.0, 51.0]
    assert stats["median_of_round_medians_ms"] == 26.0


def test_frozen_shortlists_are_budget_prefixes_and_board_pool_is_role_scoped():
    policies = json.loads((GROUPED / "20260924_c3_holdout_shortlists_v2_run02/task_policies.json").read_text())
    tasks = json.loads((GROUPED / "20260924_c3_holdout_task_freeze_v2_run01/contract.json").read_text())
    board = [json.loads(line) for line in
             (GROUPED / "20260924_c3_holdout_board_pool_freeze_v2_run01/candidates.jsonl").read_text().splitlines()
             if line.strip()]
    assert len(board) == 24
    full = set(tasks["full_pool_task_ids"])
    limited = set(tasks["limited_budget_task_ids"])
    for task_id, policy in policies.items():
        shortlist = policy["shortlist"]["shortlist_candidate_ids"]
        assert len(shortlist) == 4
        assert policy["B1_B2_B3"]["B1"] == shortlist[1:2]
        assert policy["B1_B2_B3"]["B2"] == shortlist[1:3]
        assert policy["B1_B2_B3"]["B3"] == shortlist[1:4]
        observed = {row["program_id"] for row in board if row["execution_task_id"] == task_id}
        expected = set(policy["complete_local_program_pool"] if task_id in full else shortlist)
        if task_id in limited:
            expected.update(shortlist)
        assert observed == expected
