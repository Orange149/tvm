import sys
import time

import pytest

from thesis_c3_online_protocol import (
    PaidInformationBroker,
    freeze_attempt_plan,
    freeze_requested_space,
    random_requested_order,
    run_phase_subprocess,
)
from freeze_thesis_c3_online_campaign import freeze as freeze_campaign


def space():
    return freeze_requested_space("N1", ["tile{:02d}".format(index) for index in range(12)])


def campaign_input():
    return {
        "schema": "thesis_c3_online_source_freeze_input_v1",
        "target_labels_read": False,
        "tasks": [
            {
                "task_id": task,
                "source_model": "synthetic_{}".format(task),
                "workload_signature": "signature_{}".format(task),
                "history_exclusion_sha256": (str(index + 1) * 64)[:64],
                "tile_ids": ["{}_tile{:02d}".format(task, tile) for tile in range(12)],
            }
            for index, task in enumerate(("N1", "N2", "N3"))
        ],
    }


def passing_lower():
    return {
        "lower_pass": True,
        "logical_dma_bytes": 1000,
        "logical_dma_calls": 12,
        "barrier_indicator": 0,
        "effective_program_hash": "a" * 64,
    }


def passing_fsim():
    return {
        "fsim_pass": True,
        "actual_submissions": 2,
        "serialized_insn_peak_bytes": 4096,
        "serialized_uop_peak_bytes": 1024,
    }


def test_freezes_48_identities_and_27_rotated_attempts():
    identities = space()
    assert len(identities) == 48
    assert {row["mode"] for row in identities} == {"O", "I", "W", "IW"}
    plan = freeze_attempt_plan()
    assert len(plan) == 27
    assert len({row["attempt_id"] for row in plan}) == 27
    assert [row["strategy"] for row in plan[:3]] == [
        "random",
        "mode_aware_xgb",
        "staged_pareto",
    ]
    assert [row["strategy"] for row in plan[3:6]] == [
        "mode_aware_xgb",
        "staged_pareto",
        "random",
    ]
    assert [row["strategy"] for row in plan[6:9]] == [
        "staged_pareto",
        "random",
        "mode_aware_xgb",
    ]


def test_campaign_freeze_contains_three_isolated_spaces_and_27_attempts():
    result = freeze_campaign(campaign_input())
    assert result["attempt_count"] == 27
    assert list(result["tasks"]) == ["N1", "N2", "N3"]
    assert all(task["requested_count"] == 48 for task in result["tasks"].values())
    assert result["target_labels_read"] is False


def test_campaign_freeze_rejects_target_labels_and_task_reselection():
    value = campaign_input()
    value["tasks"][0]["measured_latency_ms"] = 1.0
    with pytest.raises(ValueError, match="target label"):
        freeze_campaign(value)
    value = campaign_input()
    value["tasks"].reverse()
    with pytest.raises(ValueError, match="N1/N2/N3"):
        freeze_campaign(value)


def test_random_order_is_seeded_and_complete():
    identities = space()
    first = random_requested_order(identities, 57001)
    assert first == random_requested_order(identities, 57001)
    assert first != random_requested_order(identities, 57002)
    assert set(first) == {row["requested_id"] for row in identities}


def test_unknown_paid_information_cannot_be_read():
    broker = PaidInformationBroker(space(), "attempt")
    requested_id = space()[0]["requested_id"]
    assert broker.read(requested_id, "mode") == "O"
    with pytest.raises(KeyError, match="unknown"):
        broker.read(requested_id, "logical_dma_bytes")


def test_request_budget_duplicates_and_out_of_space_fail_closed():
    identities = space()
    broker = PaidInformationBroker(identities, "attempt")
    assert broker.request(identities[0]["requested_id"], 1, "first")
    assert not broker.request(identities[0]["requested_id"], 2, "duplicate")
    for index in range(1, 24):
        assert broker.request(identities[index]["requested_id"], index + 2, "fill")
    assert not broker.request(identities[24]["requested_id"], 30, "over budget")
    assert len(broker.requested_ids) == 24
    with pytest.raises(KeyError, match="outside"):
        broker.request("missing", 1, "bad")


def test_paid_phase_chain_and_known_mask():
    identities = space()
    requested_id = identities[0]["requested_id"]
    broker = PaidInformationBroker(identities, "attempt")
    broker.request(requested_id, 1, "test")
    with pytest.raises(ValueError, match="passing lower"):
        broker.complete_phase(requested_id, "fsim", passing_fsim(), 2)
    assert broker.complete_phase(requested_id, "lower", passing_lower(), 3)
    assert broker.complete_phase(requested_id, "fsim", passing_fsim(), 4)
    assert broker.complete_phase(
        requested_id,
        "correctness",
        {"numeric_correct": True, "output_hash": "b" * 64},
        5,
    )
    samples = [10.0, 9.0, 11.0, 10.5, 9.5, 10.25, 9.75]
    assert broker.complete_phase(
        requested_id,
        "timing",
        {"latency_samples_ms": samples, "latency_median_ms": 10.0},
        6,
    )
    mask = broker.known_mask(requested_id)
    assert all(mask.values())
    assert broker.read(requested_id, "latency_median_ms") == 10.0


def test_failed_phase_has_no_fabricated_features():
    identities = space()
    requested_id = identities[0]["requested_id"]
    broker = PaidInformationBroker(identities, "attempt")
    broker.request(requested_id, 1, "test")
    failed = {
        "lower_pass": False,
        "logical_dma_bytes": None,
        "logical_dma_calls": None,
        "barrier_indicator": None,
        "effective_program_hash": None,
    }
    assert broker.complete_phase(requested_id, "lower", failed, 2)
    with pytest.raises(ValueError, match="passing lower"):
        broker.complete_phase(requested_id, "fsim", passing_fsim(), 3)


def test_late_result_is_preserved_but_not_admitted():
    identities = space()
    requested_id = identities[0]["requested_id"]
    broker = PaidInformationBroker(identities, "attempt")
    broker.request(requested_id, 1, "test")
    assert not broker.complete_phase(requested_id, "lower", passing_lower(), 600.001)
    assert len(broker.late_results) == 1
    with pytest.raises(KeyError, match="unknown"):
        broker.read(requested_id, "lower_pass")


def test_attempts_do_not_share_paid_information():
    identities = space()
    requested_id = identities[0]["requested_id"]
    first = PaidInformationBroker(identities, "first")
    second = PaidInformationBroker(identities, "second")
    first.request(requested_id, 1, "test")
    first.complete_phase(requested_id, "lower", passing_lower(), 2)
    with pytest.raises(KeyError, match="unknown"):
        second.read(requested_id, "lower_pass")


def test_timing_requires_seven_samples_and_exact_median():
    identities = space()
    requested_id = identities[0]["requested_id"]
    broker = PaidInformationBroker(identities, "attempt")
    broker.request(requested_id, 1, "test")
    broker.complete_phase(requested_id, "lower", passing_lower(), 2)
    broker.complete_phase(requested_id, "fsim", passing_fsim(), 3)
    broker.complete_phase(
        requested_id, "correctness", {"numeric_correct": True, "output_hash": "c" * 64}, 4
    )
    with pytest.raises(ValueError, match="seven"):
        broker.complete_phase(
            requested_id,
            "timing",
            {"latency_samples_ms": [1.0] * 6, "latency_median_ms": 1.0},
            5,
        )


def test_live_subprocess_success_failure_timeout_and_prestart_deadline():
    start = time.monotonic()
    success = run_phase_subprocess(
        [sys.executable, "-c", "print('ok')"], "lower", start
    )
    assert success["status"] == "completed" and success["admit_result"]
    failure = run_phase_subprocess(
        [sys.executable, "-c", "raise SystemExit(7)"], "lower", time.monotonic()
    )
    assert failure["status"] == "failed" and failure["returncode"] == 7
    timeout = run_phase_subprocess(
        [sys.executable, "-c", "import time; time.sleep(1)"],
        "rpc",
        time.monotonic() - 599.95,
    )
    assert timeout["status"] == "timeout" and not timeout["admit_result"]
    before = run_phase_subprocess(
        [sys.executable, "-c", "raise SystemExit(99)"],
        "rpc",
        0,
        now=lambda: 601.0,
    )
    assert before["status"] == "deadline_before_start"
    assert before["returncode"] is None
