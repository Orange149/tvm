import copy
import math

import pytest

from freeze_thesis_e2e_ablation_contract import freeze
from thesis_e2e_ablation_protocol import analyze_performance, frozen_schedule, window_metrics


def freeze_input():
    return {
        "schema": "thesis_e2e_ablation_freeze_input_v1",
        "result_labels_read": False,
        "common_artifacts": {
            "model_hash": "1" * 64,
            "qir_hash": "2" * 64,
            "input_manifest_hash": "3" * 64,
            "label_manifest_hash": "4" * 64,
        },
        "conditions": [
            {
                "condition": condition,
                "package_hash": "a" * 64,
                "actual_policy_hash": "b" * 64,
                "plan_hash": "c" * 64,
                "capacity_certificate_hash": "d" * 64,
            }
            for condition in ("A", "B", "C", "Ca", "D", "E")
        ],
    }


def performance_payload():
    scale = {"A": 1.0, "B": 0.9, "C": 0.8, "Ca": 0.75, "D": 0.75, "E": 0.7}
    boots = []
    for schedule in frozen_schedule():
        blocks = []
        for block_index in (1, 2):
            rows = []
            for condition in schedule["block_{}".format(block_index)]:
                factor = scale[condition] * (1 + schedule["boot_index"] * 0.01)
                rows.append({
                    "condition": condition,
                    "serial_latency_ms": [10.0 * factor] * 15,
                    "completion_seconds": [1.0 + index * 0.01 * factor for index in range(302)],
                })
            blocks.append({"conditions": rows})
        boots.append({
            "boot_index": schedule["boot_index"],
            "boot_id": "boot-{}".format(schedule["boot_index"]),
            "blocks": blocks,
        })
    return {"schema": "thesis_e2e_ablation_performance_v1", "boots": boots}


def test_freeze_preserves_exact_six_condition_schedule_and_counts():
    result = freeze(freeze_input())
    assert [row["condition"] for row in result["conditions"]] == ["A", "B", "C", "Ca", "D", "E"]
    assert result["planned_counts"] == {
        "windows": 60,
        "completion_rows": 18120,
        "scored_ii_gaps": 17940,
        "serial_scored_calls": 900,
        "warmup_calls": 300,
        "accuracy_inferences": 6000,
        "numeric_qualification_calls": 402,
    }
    assert result["schedule"][0]["block_2"] == ["E", "D", "Ca", "C", "B", "A"]


def test_freeze_rejects_result_leak_and_condition_reordering():
    value = freeze_input()
    value["conditions"][0]["measured_latency_ms"] = 1.0
    with pytest.raises(ValueError, match="forbidden"):
        freeze(value)
    value = freeze_input()
    value["conditions"].reverse()
    with pytest.raises(ValueError, match="A/B/C/Ca/D/E"):
        freeze(value)


def test_window_scoring_discards_two_rows_and_scores_299_gaps():
    result = window_metrics([1.0 + index * 0.02 for index in range(302)])
    assert result["rows_retained"] == 300
    assert result["gaps_scored"] == 299
    assert math.isclose(result["ii_seconds"], 0.02)


def test_window_scoring_keeps_last_gap_and_drops_both_initial_rows():
    completed = [1.0, 2.0] + [10.0 + index * 0.02 for index in range(299)]
    completed.append(completed[-1] + 0.5)
    result = window_metrics(completed)
    assert result["rows_discarded_front"] == 2
    assert math.isclose(result["ii_seconds"], (298 * 0.02 + 0.5) / 299)


def test_complete_payload_uses_boot_as_independent_layer():
    result = analyze_performance(performance_payload())
    assert result["counts"]["windows"] == 60
    assert result["counts"]["serial_scored_calls"] == 900
    effect = result["paired_effects"]["serial_latency_ms"]["A->E"]
    assert effect["n_boots"] == 5 and effect["df"] == 4
    assert all(math.isclose(value, 0.7) for value in effect["raw_ratios_after_over_before"])
    assert math.isclose(effect["geometric_mean_ratio"], 0.7)


def test_incomplete_or_nonindependent_payload_fails_closed():
    value = performance_payload()
    value["boots"][1]["boot_id"] = value["boots"][0]["boot_id"]
    with pytest.raises(ValueError, match="distinct"):
        analyze_performance(value)
    value = performance_payload()
    value["boots"][0]["blocks"][0]["conditions"][0]["serial_latency_ms"].pop()
    with pytest.raises(ValueError, match="15"):
        analyze_performance(value)
    value = copy.deepcopy(performance_payload())
    value["boots"][0]["blocks"][0]["conditions"].reverse()
    with pytest.raises(ValueError, match="order mismatch"):
        analyze_performance(value)
