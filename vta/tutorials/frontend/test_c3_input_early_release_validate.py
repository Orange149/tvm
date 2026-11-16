"""Replay fixed native call/lease ledgers without a board or timing labels."""

import copy

import pytest

from c3_input_early_release_validate import INPUT_BYTES, MODE_FRAMES, validate_native


def native_fixture(variant="E1", mode="qualify", proxy=True):
    warmup, count = MODE_FRAMES[mode]
    reference = mode == "reference"
    frames, events = [], []

    def emit(name, frame, stage=-1, edge=-1, slot=-1, generation=0, time=None):
        sequence = len(events)
        row = {"event": name, "frame": frame, "stage": stage, "edge": edge,
               "slot": slot, "generation": generation, "time_ms": sequence if time is None else time,
               "sequence": sequence}
        events.append(row)
        return row

    # Use intervals with enough room for previous E1 execution to overlap the
    # next producer. Exact frame durations are irrelevant to protocol replay.
    for position, frame_id in enumerate(range(-warmup, count)):
        base = 1000 * position
        frames.append({"frame_id": frame_id, "input_index": position if position < warmup else frame_id,
                       "warmup": position < warmup, "complete": True, "enqueue_ms": base,
                       "completion_ms": base + 900,
                       "stage_run_begin_ms": [base + 10, base + 300, base + 800],
                       "stage_run_end_ms": [base + 100, base + 700, base + 850]})
    emit("score_begin", -1, time=0)
    for position, frame in enumerate(frames):
        fid, base = frame["frame_id"], frame["enqueue_ms"]
        slots = [position % (2 if variant == "L2" else 1), position % 2]
        generations = [position // (2 if variant == "L2" else 1) + 1, position // 2 + 1]
        emit("stage_begin", fid, 0, time=base)
        if not reference:
            emit("producer_acquired", fid, 0, 0, slots[0], generations[0], base + 1)
            emit("producer_published", fid, 0, 0, slots[0], generations[0], base + 110)
        emit("stage_end", fid, 0, time=base + 120)
        emit("stage_begin", fid, 1, time=base + 200)
        if not reference:
            emit("consumer_acquired", fid, 1, 0, slots[0], generations[0], base + 210)
            if variant == "C1":
                emit("copy_begin", fid, 1, 0, slots[0], generations[0], base + 220)
                emit("copy_end", fid, 1, 0, slots[0], generations[0], base + 230)
                emit("input_lease_released", fid, 1, 0, slots[0], generations[0], base + 240)
            emit("producer_acquired", fid, 1, 1, slots[1], generations[1], base + 250)
        if proxy:
            emit("last_reader_start", fid, 1, 0, slots[0], generations[0], base + 310)
            emit("last_reader_return", fid, 1, 0, slots[0], generations[0], base + 320)
            emit("dynamic_ranges_verified", fid, 1, 0, slots[0], generations[0], base + 325)
        if proxy and variant == "E1" and not reference:
            emit("validated_release", fid, 1, 0, slots[0], generations[0], base + 326)
        if not reference:
            if variant == "E1":
                emit("input_lease_released", fid, 1, 0, slots[0], generations[0], base + 330)
        if proxy:
            emit("node2_execute", fid, 1, 0, slots[0], generations[0], base + 335)
        if not reference:
            emit("producer_published", fid, 1, 1, slots[1], generations[1], base + 710)
            if variant in ("L1", "L2"):
                emit("input_lease_released", fid, 1, 0, slots[0], generations[0], base + 720)
        emit("stage_end", fid, 1, time=base + 730)
        emit("stage_begin", fid, 2, time=base + 740)
        if not reference:
            emit("consumer_acquired", fid, 2, 1, slots[1], generations[1], base + 750)
            emit("input_lease_released", fid, 2, 1, slots[1], generations[1], base + 860)
        emit("stage_end", fid, 2, time=base + 870)
    emit("score_end", -1, time=frames[-1]["completion_ms"])
    range_evidence = []
    for frame in frames:
        def data_range(offset, size=INPUT_BYTES):
            return {"handle": hex(0x40000000 + offset), "cpu_address": hex(0x10000000 + offset),
                    "physical_address": hex(0x20000000 + offset), "bytes": size}
        copying = variant == "C1" and not reference
        range_evidence.append({"frame_id": frame["frame_id"], "verified": True,
                               "input": data_range(0x100000), "reader_output": data_range(0x200000),
                               "next_input": data_range(0x200000), "packed_output": data_range(0x300000, INPUT_BYTES // 4),
                               "copy_source": data_range(0x400000) if copying else None,
                               "copy_destination": data_range(0x100000) if copying else None,
                               "copy_source_private_separate": True if copying else None})
    return {"schema": "c3-input-last-read-result-v1", "status": "passed", "error": None,
            "variant": variant, "mode": mode, "warmup_count": warmup, "frame_count": count,
            "frames": frames, "events": events, "reference_serial": reference,
            "proxy_enabled": proxy, "stage_call_counts": [len(frames)] * 3,
            "target_call_count": len(frames) if proxy else None,
            "packed_call_count": len(frames) if proxy else None,
            "dynamic_ranges_verified": True if proxy else None,
            "dynamic_range_evidence": range_evidence if proxy else [],
            "allocation_snapshots": [{"phase": "slots_allocated", "initialized": 1,
                                      "high_water_bytes": 0x1000000, "virtual_base": 0x10000000,
                                      "physical_base": 0x20000000}],
            "affinity": [{"stage": stage, "caller_tid": 100 + stage,
                          "workers": [{"tid": 100 + stage, "cpus": cpus}],
                          "actual_cpu_initial": cpus[0], "actual_cpu_final": cpus[0]}
                         for stage, cpus in enumerate(([0, 1], [3], [2]))]}


def renumber(result):
    for sequence, row in enumerate(result["events"]):
        row["sequence"] = sequence


def event(result, name, frame=0, edge=None):
    return next(row for row in result["events"] if row["event"] == name and row["frame"] == frame
                and (edge is None or row["edge"] == edge))


@pytest.mark.parametrize("variant", ["L1", "L2", "C1", "E1"])
def test_complete_shared_lease_and_call_ledger(variant):
    result = native_fixture(variant)
    original = copy.deepcopy(result)
    evidence = validate_native(result)
    assert result == original
    assert evidence["call_ledger_verified"] is True
    assert evidence["lease_trace_verified"] is True
    assert evidence["observed_stage_calls"] == 96
    assert evidence["formal_copy_count"] == (32 if variant == "C1" else 0)
    assert evidence["formal_copy_bytes"] == (32 * INPUT_BYTES if variant == "C1" else 0)
    assert evidence["early_release_overlap_verified"] is (False if variant == "E1" else None)


def test_ordinary_reference_has_no_shared_leases_but_same_read_only_range_proxy():
    evidence = validate_native(native_fixture("L1", "reference"))
    assert evidence["observed_complete_inferences"] == 100
    assert evidence["lease_trace_verified"] is False


def test_proxy_off_calibration_is_the_only_allowed_nonreference_proxy_off():
    assert validate_native(native_fixture("L2", "proxycheck", proxy=False))["observed_stage_calls"] == 63
    with pytest.raises(ValueError, match="proxy-off"):
        validate_native(native_fixture("L1", "qualify", proxy=False))


@pytest.mark.parametrize("field,value", [("status", "INVALID"), ("status", "partial"),
                                        ("global_stop", True), ("error", "timeout"),
                                        ("target_call_count", 31), ("stage_call_counts", [32, 33, 32])])
def test_failed_or_unexpected_call_count_rejected(field, value):
    result = native_fixture()
    result[field] = value
    with pytest.raises(ValueError):
        validate_native(result)


@pytest.mark.parametrize("name", ["producer_acquired", "producer_published", "consumer_acquired",
                                  "input_lease_released", "last_reader_return", "stage_end"])
def test_missing_transition_or_function_boundary_rejected(name):
    result = native_fixture()
    result["events"].remove(event(result, name))
    renumber(result)
    with pytest.raises(ValueError):
        validate_native(result)


def test_duplicate_release_and_wrong_generation_rejected():
    result = native_fixture()
    row = event(result, "input_lease_released", edge=0)
    result["events"].insert(row["sequence"] + 1, dict(row))
    renumber(result)
    with pytest.raises(ValueError):
        validate_native(result)
    result = native_fixture()
    event(result, "producer_acquired", frame=1, edge=0)["generation"] = 1
    with pytest.raises(ValueError):
        validate_native(result)


def test_output_error_incomplete_frame_and_early_return_fail():
    result = native_fixture()
    result["frames"][0]["complete"] = False
    with pytest.raises(ValueError):
        validate_native(result)
    result = native_fixture()
    event(result, "input_lease_released", edge=0)["time_ms"] = 319
    with pytest.raises(ValueError):
        validate_native(result)


def test_c1_cannot_release_before_copy_completes():
    result = native_fixture("C1")
    event(result, "copy_end")["time_ms"] = 250
    with pytest.raises(ValueError):
        validate_native(result)


def test_late_release_arm_cannot_silently_use_early_release():
    result = native_fixture("L1")
    event(result, "input_lease_released", edge=0)["time_ms"] = 330
    with pytest.raises(ValueError):
        validate_native(result)


def overlap_first_two(result):
    """Keep actual FIFO frame completion while producing next input earlier."""
    events = result["events"]
    next_stage0 = [row for row in events if row["frame"] == 1 and row["stage"] == 0]
    for row in next_stage0:
        events.remove(row)
        row["time_ms"] -= 1600
    location = event(result, "input_lease_released", edge=0)["sequence"] + 1
    events[location:location] = next_stage0
    result["frames"][1]["enqueue_ms"] = 400
    result["frames"][1]["stage_run_begin_ms"][0] = 410
    result["frames"][1]["stage_run_end_ms"][0] = 500
    for row in next_stage0:
        row["time_ms"] += 1000  # stage0 original base1000 -> newbase400
    renumber(result)
    return result


def test_e1_observed_next_generation_overlap_is_separate_from_safe_early_return():
    result = overlap_first_two(native_fixture())
    evidence = validate_native(result)
    assert evidence["early_release_overlap_verified"] is True
    assert evidence["early_release_overlap_examples"][0]["next_frame"] == 1


def test_event_before_proves_waiting_not_just_a_marker():
    result = native_fixture(mode="event-before")
    marker = dict(event(result, "last_reader_start"), event="before_reader_producer_blocked", time_ms=315)
    wait = dict(marker, event="producer_wait", stage=0, edge=0, frame=1, slot=0, generation=0, time_ms=314)
    at = marker["sequence"] + 1
    result["events"][at:at] = [wait, marker]
    renumber(result)
    assert validate_native(result)["deterministic_event_proof_verified"] is True
    result["events"].remove(wait)
    renumber(result)
    with pytest.raises(ValueError, match="blocked producer"):
        validate_native(result)


def test_event_after_requires_completed_next_production_before_old_stage_end():
    result = overlap_first_two(native_fixture(mode="event-after"))
    published = event(result, "producer_published", frame=1, edge=0)
    marker = dict(event(result, "last_reader_return"), event="after_reader_next_frame_published", time_ms=600)
    event(result, "node2_execute")["time_ms"] = 650
    result["events"].insert(published["sequence"] + 1, marker)
    renumber(result)
    assert validate_native(result)["deterministic_event_proof_verified"] is True
    marker["time_ms"] = 900
    # The marker must also be inside the target invocation, not after it.
    with pytest.raises(ValueError):
        validate_native(result)


@pytest.mark.parametrize("change", [
    lambda row: row["reader_output"].__setitem__("cpu_address", row["input"]["cpu_address"]),
    lambda row: row["packed_output"].__setitem__("physical_address", row["input"]["physical_address"]),
    lambda row: row["next_input"].__setitem__("cpu_address", "0x10200100"),
    lambda row: row["input"].__setitem__("bytes", INPUT_BYTES - 1),
    lambda row: row["input"].__setitem__("physical_address", "0x1000"),
    lambda row: row.__setitem__("verified", False),
])
def test_actual_abi_ranges_must_be_complete_disjoint_and_within_startup_allocation(change):
    result = native_fixture()
    change(result["dynamic_range_evidence"][0])
    with pytest.raises(ValueError):
        validate_native(result)


def test_missing_frame_range_proof_or_extra_packed_call_fails():
    result = native_fixture()
    result["dynamic_range_evidence"].pop()
    with pytest.raises(ValueError):
        validate_native(result)
    result = native_fixture()
    result["packed_call_count"] += 1
    with pytest.raises(ValueError):
        validate_native(result)


def test_c1_cpu_source_must_not_alias_private_destination():
    result = native_fixture("C1")
    proof = result["dynamic_range_evidence"][0]
    proof["copy_source"]["physical_address"] = proof["copy_destination"]["physical_address"]
    with pytest.raises(ValueError):
        validate_native(result)


def test_affinity_caller_worker_and_actual_cpu_all_verified():
    result = native_fixture()
    result["affinity"][1]["workers"][0]["cpus"] = [2, 3]
    with pytest.raises(ValueError):
        validate_native(result)
    result = native_fixture()
    result["affinity"][1]["actual_cpu_final"] = 0
    with pytest.raises(ValueError):
        validate_native(result)


def test_e1_cannot_release_before_range_certificate_or_skip_packed_entry_event():
    result = native_fixture()
    result["events"].remove(event(result, "dynamic_ranges_verified"))
    renumber(result)
    with pytest.raises(ValueError):
        validate_native(result)
    result = native_fixture()
    event(result, "validated_release")["time_ms"] = 319
    with pytest.raises(ValueError):
        validate_native(result)
