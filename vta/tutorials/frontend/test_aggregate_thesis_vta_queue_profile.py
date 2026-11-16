import copy

import pytest

from aggregate_thesis_vta_queue_profile import aggregate


MAX_XFER = 1 << 25
ENVIRONMENT = {
    "VTA_REPLAY_POLICY": "disabled",
    "VTA_INSN_BUFFER_BYTES": str(MAX_XFER),
    "VTA_UOP_BUFFER_BYTES": str(MAX_XFER),
}


def row(queue, submit, insn, uop, insn_peak, uop_peak):
    return {
        "queue_id": queue,
        "submit": submit,
        "reason": "synchronize",
        "insn_capacity": MAX_XFER,
        "uop_capacity": MAX_XFER,
        "submit_threshold": MAX_XFER,
        "submit_threshold_configured": False,
        "insn_bytes": insn,
        "uop_bytes": uop,
        "insn_peak": insn_peak,
        "uop_peak": uop_peak,
        "timeout": 0,
    }


def records():
    return [
        row("0xa", 1, 4096, 1024, 4096, 1024),
        row("0xb", 1, 6144, 512, 6144, 512),
        row("0xa", 2, 2048, 2048, 4096, 2048),
    ]


def test_aggregates_all_worker_queues_and_running_peaks():
    result = aggregate(records(), ENVIRONMENT)
    assert result["insn_peak_bytes"] == 6144
    assert result["uop_peak_bytes"] == 2048
    assert result["submissions"] == 3
    assert result["queue_instances"] == 2
    assert result["finish_included"] is True


@pytest.mark.parametrize(
    "mutator,match",
    [
        (lambda rows, env: env.update(VTA_REPLAY_POLICY="enabled"), "disable replay"),
        (lambda rows, env: env.update(VTA_INSN_SUBMIT_THRESHOLD_BYTES="4096"), "early submit"),
        (lambda rows, env: env.update(VTA_UOP_BUFFER_BYTES="4096"), "VTA_MAX_XFER"),
        (lambda rows, env: rows[1].update(timeout=1), "device timeout"),
        (lambda rows, env: rows[2].update(submit=3), "non-contiguous"),
        (lambda rows, env: rows[2].update(insn_peak=2048), "running peak"),
        (lambda rows, env: rows[0].update(submit_threshold_configured=True), "early submit"),
        (lambda rows, env: rows[0].update(insn_capacity=4096), "capacity differs"),
    ],
)
def test_invalid_worker_profile_fails_closed(mutator, match):
    rows = copy.deepcopy(records())
    environment = dict(ENVIRONMENT)
    mutator(rows, environment)
    with pytest.raises(ValueError, match=match):
        aggregate(rows, environment)
