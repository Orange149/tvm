"""Read-only, fixed-campaign verification of native calls and input lease events.

No native success boolean replaces this replay. This checker proves software
event/ownership relationships, not physical DDR transaction timing or ELF safety.
"""

import math


INPUT_BYTES = 802816
MODE_FRAMES = {"reference": (0, 100), "qualify": (0, 32), "event-before": (0, 3),
               "event-after": (0, 3), "proxycheck": (5, 16), "score": (5, 100)}
TRANSITIONS = ("producer_acquired", "producer_published", "consumer_acquired", "input_lease_released")
OTHER_EVENTS = {"producer_wait", "stage_begin", "stage_end", "last_reader_start", "last_reader_return",
                "copy_begin", "copy_end", "score_begin", "score_end",
                "before_reader_producer_blocked", "after_reader_next_frame_published",
                "reader_ranges_verified", "dynamic_ranges_verified", "validated_release", "node2_execute"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value):
    return type(value) is int


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _address(value):
    _require(isinstance(value, str) and value.startswith("0x"), "range addresses must be explicit hexadecimal strings")
    try:
        address = int(value, 16)
    except ValueError as error:
        raise ValueError("malformed actual address") from error
    _require(address > 0, "actual address must be nonzero")
    return address


def _verify_ranges(result, frames, proxy):
    evidence = result.get("dynamic_range_evidence")
    if not proxy:
        _require(evidence == [] and result.get("dynamic_ranges_verified") is None,
                 "proxy-off cannot claim actual ABI range verification")
        return None
    _require(isinstance(evidence, list) and [row.get("frame_id") for row in evidence] == [row["frame_id"] for row in frames],
             "one actual ABI range certificate per frame required")
    startup = [row for row in result.get("allocation_snapshots", []) if row.get("phase") == "slots_allocated"]
    _require(len(startup) == 1 and startup[0].get("initialized") == 1, "startup DMA allocation extent missing")
    extent = startup[0]
    _require(_integer(extent.get("high_water_bytes")) and extent["high_water_bytes"] > 0,
             "actual allocated DMA extent unknown")
    domains = (("cpu_address", "virtual_base"), ("physical_address", "physical_base"))
    for row in evidence:
        _require(row.get("verified") is True, "actual tensor range proof failed")
        expected_sizes = {"input": INPUT_BYTES, "reader_output": INPUT_BYTES,
                          "next_input": INPUT_BYTES, "packed_output": INPUT_BYTES // 4}
        for label, expected_size in expected_sizes.items():
            value = row.get(label)
            _require(isinstance(value, dict) and value.get("bytes") == expected_size,
                     "actual tensor range shape/byte contract changed")
            _address(value.get("handle"))
            for address_key, base_key in domains:
                begin = _address(value.get(address_key))
                base = extent.get(base_key)
                _require(_integer(base) and base > 0 and base <= begin and begin + expected_size <= base + extent["high_water_bytes"],
                         "actual tensor range escapes startup allocated DMA extent")
        for address_key, _ in domains:
            input_range, reader, next_input, packed = [row[name] for name in expected_sizes]
            _require(reader[address_key] == next_input[address_key], "packed function does not consume the certified reader output")
            for left, right in ((input_range, reader), (input_range, packed), (reader, packed)):
                a, b = _address(left[address_key]), _address(right[address_key])
                _require(a + left["bytes"] <= b or b + right["bytes"] <= a, "input/intermediate/packed data ranges alias")
        copying = result["variant"] == "C1" and result["mode"] != "reference"
        if copying:
            source, destination = row.get("copy_source"), row.get("copy_destination")
            _require(isinstance(source, dict) and isinstance(destination, dict)
                     and source.get("bytes") == destination.get("bytes") == INPUT_BYTES
                     and row.get("copy_source_private_separate") is True, "C1 missing real CPU source/private destination proof")
            for address_key, base_key in domains:
                src, dst = _address(source.get(address_key)), _address(destination.get(address_key))
                _require(destination[address_key] == row["input"][address_key], "C1 target does not read the private copy")
                _require(src + INPUT_BYTES <= dst or dst + INPUT_BYTES <= src, "C1 source overlaps private input")
                base = extent[base_key]
                _require(base <= src and src + INPUT_BYTES <= base + extent["high_water_bytes"], "C1 source escapes allocated extent")
        else:
            _require(row.get("copy_source") is None and row.get("copy_destination") is None
                     and row.get("copy_source_private_separate") is None, "noncopying arm carries unexpected copy evidence")
    return True


def _verify_affinity(result):
    records = result.get("affinity")
    _require(isinstance(records, list) and [row.get("stage") for row in records] == [0, 1, 2],
             "actual three-actor affinity records missing")
    tids = set()
    for record, allowed in zip(records, ({0, 1}, {3}, {2})):
        workers = record.get("workers")
        _require(isinstance(workers, list) and bool(workers), "actual actor/helper thread masks missing")
        local = []
        for worker in workers:
            tid, cpus = worker.get("tid"), worker.get("cpus")
            _require(_integer(tid) and tid > 0 and tid not in tids and tid not in local, "thread assigned to more than one actor")
            _require(isinstance(cpus, list) and bool(cpus) and all(_integer(cpu) for cpu in cpus)
                     and set(cpus).issubset(allowed), "actual actor/helper escaped frozen CPU mask")
            local.append(tid)
        _require(record.get("caller_tid") in local and record.get("actual_cpu_initial") in allowed
                 and record.get("actual_cpu_final") in allowed, "caller actual CPU/identity outside stage assignment")
        tids.update(local)
    return True


def validate_native(result):
    """Return verified evidence fields or raise on an incomplete/invalid result.

    Inputs follow c3-input-last-read-result-v1 from the isolated native runner.
    References have no shared leases but may use the identical read-only proxy
    to certify actual tensor ranges. Proxy-off is confined to calibration.
    Formal overlap=False is a valid negative finding.
    Full output bytes, actual address separation and DSO identities remain the
    main controller's independent gates.
    """
    _require(result.get("schema") == "c3-input-last-read-result-v1", "native result schema differs")
    _require(result.get("status") in ("passed", "COMPLETE"), "native result is incomplete or failed")
    _require(not result.get("error") and not result.get("global_stop", False), "native attempt failed")
    mode, variant = result.get("mode"), result.get("variant")
    _require(mode in MODE_FRAMES and variant in ("L1", "L2", "C1", "E1"), "unsupported fixed trial mode/variant")
    warmups, scored_count = MODE_FRAMES[mode]
    if mode.startswith("event-"):
        _require(variant == "E1", "deterministic event tests require E1")
    reference = mode == "reference"
    _require(result.get("reference_serial") is reference, "reference execution mode mismatch")
    _require(result.get("warmup_count") == warmups and result.get("frame_count") == scored_count,
             "native planned frame counts differ")
    frames = result.get("frames")
    _require(isinstance(frames, list) and len(frames) == warmups + scored_count, "native frame ledger incomplete")
    expected_ids = list(range(-warmups, scored_count))
    _require([row.get("frame_id") for row in frames] == expected_ids and
             all(_integer(row.get("frame_id")) for row in frames), "native frame identities missing/non-FIFO")
    for position, frame in enumerate(frames):
        warmup = position < warmups
        _require(frame.get("complete") is True and frame.get("warmup") is warmup, "frame incomplete or warmup changed")
        _require(frame.get("input_index") == (position if warmup else frame["frame_id"]), "input index sequence differs")
        _require(_integer(frame.get("input_index")), "input index must be integer")
        begin, end = frame.get("enqueue_ms"), frame.get("completion_ms")
        _require(_finite(begin) and _finite(end) and 0 <= begin <= end, "invalid native frame interval")
        starts, ends = frame.get("stage_run_begin_ms"), frame.get("stage_run_end_ms")
        _require(isinstance(starts, list) and isinstance(ends, list) and len(starts) == len(ends) == 3,
                 "all three stage timing boundaries required")
        _require(all(_finite(value) for value in starts + ends), "nonfinite stage boundary")
        _require(begin <= starts[0] <= ends[0] <= starts[1] <= ends[1] <= starts[2] <= ends[2] <= end,
                 "stage ordering/complete-output boundary invalid")
        if position:
            _require(frames[position - 1]["completion_ms"] < end and frames[position - 1]["enqueue_ms"] <= begin,
                     "native completion/enqueue order not FIFO")
    if warmups:
        _require(frames[warmups]["enqueue_ms"] >= frames[warmups - 1]["completion_ms"], "warmup pipeline not drained")
    _require(result.get("stage_call_counts") == [len(frames)] * 3, "unexpected complete graph invocation count")
    proxy = result.get("proxy_enabled")
    _require(type(proxy) is bool, "missing actual proxy identity")
    if not proxy:
        _require(mode == "proxycheck" and variant == "L2", "proxy-off only allowed for L2 calibration")
    _require(result.get("target_call_count") == len(frames) if proxy else result.get("target_call_count") is None,
             "unexpected target wrapper invocation count")
    _require(result.get("packed_call_count") == len(frames) if proxy else result.get("packed_call_count") is None,
             "unexpected packed function wrapper invocation count")
    affinity_verified = _verify_affinity(result)
    ranges_verified = _verify_ranges(result, frames, proxy)
    events = result.get("events")
    _require(isinstance(events, list) and bool(events), "native event ledger unavailable")
    _require([row.get("sequence") for row in events] == list(range(len(events))), "event trace gap or duplicate sequence")
    by_frame = {frame["frame_id"]: {} for frame in frames}
    lease_by_frame = {}
    slot_state = {}
    leases = {}
    waits = []
    for event in events:
        name, frame_id, stage = event.get("event"), event.get("frame"), event.get("stage")
        _require(name in TRANSITIONS or name in OTHER_EVENTS, "unknown event cannot silently pass")
        _require(_finite(event.get("time_ms")), "invalid event timestamp")
        if name in ("score_begin", "score_end"):
            _require(stage == -1 and frame_id == -1, "score marker identity differs")
            continue
        _require(frame_id in by_frame and _integer(frame_id), "event references an unknown frame")
        if name == "producer_wait":
            _require(not reference and event.get("edge") in (0, 1), "invalid producer wait event")
            waits.append(event)
            continue
        if name in TRANSITIONS:
            _require(not reference, "reference unexpectedly used shared leases")
            edge, slot, generation = event.get("edge"), event.get("slot"), event.get("generation")
            count = 2 if edge == 1 or variant == "L2" else 1
            _require(edge in (0, 1) and _integer(edge) and _integer(slot) and 0 <= slot < count
                     and _integer(generation) and generation > 0, "invalid lease token")
            _require(stage == edge + (name in ("consumer_acquired", "input_lease_released")), "lease event stage differs")
            key, slot_key = (edge, slot, generation, frame_id), (edge, slot)
            if name == "producer_acquired":
                prior = slot_state.get(slot_key)
                _require(prior is None or prior["state"] == "free", "overwriting a live shared slot")
                _require(generation == (1 if prior is None else prior["generation"] + 1), "lease generation did not advance exactly once")
                _require((frame_id, edge) not in lease_by_frame, "frame acquired multiple edge leases")
                if prior is not None:
                    _require(event["time_ms"] >= prior["release"]["time_ms"], "new write precedes previous release")
                lease_by_frame[(frame_id, edge)] = key
                leases[key] = {name: event}
                slot_state[slot_key] = {"state": "writing", "generation": generation, "key": key}
            else:
                _require(key in leases and name not in leases[key], "stale/duplicate/unknown lease transition")
                current = slot_state.get(slot_key)
                states = {"producer_published": ("writing", "ready"), "consumer_acquired": ("ready", "using"),
                          "input_lease_released": ("using", "free")}
                before, after = states[name]
                _require(current is not None and current["key"] == key and current["state"] == before,
                         "lease transition violates ownership state")
                previous_name = TRANSITIONS[TRANSITIONS.index(name) - 1]
                _require(event["time_ms"] >= leases[key][previous_name]["time_ms"], "lease event timestamp runs backward")
                leases[key][name] = event
                current["state"] = after
                if after == "free":
                    current["release"] = event
            continue
        label = (name, stage)
        _require(label not in by_frame[frame_id], "duplicate frame function/event boundary")
        by_frame[frame_id][label] = event
    if not reference:
        _require(len(leases) == 2 * len(frames) and all(len(value) == 4 for value in leases.values()),
                 "incomplete shared lease lifecycle")
        _require(all(value["state"] == "free" for value in slot_state.values()), "shared lease leaked at end")
    copy_count = 0
    overlap = []
    for frame in frames:
        frame_id = frame["frame_id"]
        records = by_frame[frame_id]
        for stage in range(3):
            _require(("stage_begin", stage) in records and ("stage_end", stage) in records,
                     "missing complete graph call boundaries")
            before, after = records[("stage_begin", stage)], records[("stage_end", stage)]
            _require(before["sequence"] < after["sequence"] and before["time_ms"] <= frame["stage_run_begin_ms"][stage]
                     <= frame["stage_run_end_ms"][stage] <= after["time_ms"], "graph boundary ordering differs")
        reader_start, reader_end = records.get(("last_reader_start", 1)), records.get(("last_reader_return", 1))
        _require((reader_start is not None and reader_end is not None) == proxy, "reader callbacks missing or unexpected")
        if proxy:
            _require(frame["stage_run_begin_ms"][1] <= reader_start["time_ms"] <= reader_end["time_ms"]
                     <= frame["stage_run_end_ms"][1] and reader_start["sequence"] < reader_end["sequence"],
                     "last reader callback not within the one graph call")
            dynamic, node2 = records.get(("dynamic_ranges_verified", 1)), records.get(("node2_execute", 1))
            _require(dynamic is not None and node2 is not None and reader_end["sequence"] < dynamic["sequence"] < node2["sequence"]
                     and reader_end["time_ms"] <= dynamic["time_ms"] <= node2["time_ms"] <= frame["stage_run_end_ms"][1],
                     "actual range certificate not established before packed execution")
        copies = records.get(("copy_begin", 1)), records.get(("copy_end", 1))
        _require((copies[0] is not None and copies[1] is not None) == (variant == "C1" and not reference),
                 "copy callback count differs from variant")
        if reference:
            continue
        key = lease_by_frame[(frame_id, 0)]
        lease = leases[key]
        release = lease["input_lease_released"]
        for event in [reader_start, reader_end, *copies]:
            if event is not None:
                _require((event["edge"], event["slot"], event["generation"], event["frame"]) == key,
                         "reader/copy callback references wrong input generation")
        if variant == "E1":
            _require(reader_end["sequence"] < release["sequence"] and reader_end["time_ms"] <= release["time_ms"]
                     <= frame["stage_run_end_ms"][1], "E1 release precedes last reader or follows stage end")
            dynamic, validated, node2 = (records.get((name, 1)) for name in
                                         ("dynamic_ranges_verified", "validated_release", "node2_execute"))
            _require(dynamic is not None and validated is not None and node2 is not None,
                     "E1 requires actual range verification before validated release")
            _require(reader_end["sequence"] < dynamic["sequence"] < validated["sequence"] < release["sequence"]
                     < node2["sequence"] and reader_end["time_ms"] <= dynamic["time_ms"] <= validated["time_ms"]
                     <= release["time_ms"] <= node2["time_ms"] <= frame["stage_run_end_ms"][1],
                     "E1 release must follow range proof and precede the packed function")
        elif variant == "C1":
            copy_begin, copy_end = copies
            _require(lease["consumer_acquired"]["sequence"] < copy_begin["sequence"] < copy_end["sequence"]
                     < release["sequence"] < reader_start["sequence"], "C1 must release after copy and before graph use")
            _require(copy_begin["time_ms"] <= copy_end["time_ms"] <= release["time_ms"]
                     <= frame["stage_run_begin_ms"][1], "C1 copy not in frame service path")
            copy_count += not frame["warmup"]
        else:
            _require(frame["stage_run_end_ms"][1] <= release["time_ms"], "late-release arm unexpectedly returned early")
        outgoing = leases[lease_by_frame[(frame_id, 1)]]
        _require(frame["stage_run_end_ms"][2] <= outgoing["input_lease_released"]["time_ms"],
                 "output source returned before final consumer completed")
        if variant == "E1" and not frame["warmup"]:
            next_key = next((other for other in leases if other[0] == 0 and other[1] == key[1]
                             and other[2] == key[2] + 1), None)
            if next_key is not None and next_key[3] >= 0:
                published = leases[next_key]["producer_published"]
                end = records[("stage_end", 1)]
                if release["sequence"] < published["sequence"] < end["sequence"] and release["time_ms"] <= published["time_ms"] < end["time_ms"]:
                    overlap.append({"frame": frame_id, "next_frame": next_key[3], "release_ms": release["time_ms"],
                                    "next_published_ms": published["time_ms"], "consumer_stage_end_ms": end["time_ms"]})
    event_proof = None
    if mode == "event-before":
        marker = by_frame[0].get(("before_reader_producer_blocked", 1))
        zero = leases[lease_by_frame[(0, 0)]]
        one = leases[lease_by_frame[(1, 0)]]
        _require(marker is not None and any(event["edge"] == 0 and event["frame"] == 1 and event["sequence"] < marker["sequence"] for event in waits),
                 "before-reader test did not demonstrate a blocked producer")
        _require(marker["sequence"] < by_frame[0][("last_reader_return", 1)]["sequence"] < zero["input_lease_released"]["sequence"]
                 < one["producer_acquired"]["sequence"], "before-reader test illegally overwrote the held input")
        _require(by_frame[0][("last_reader_start", 1)]["time_ms"] <= marker["time_ms"]
                 <= by_frame[0][("last_reader_return", 1)]["time_ms"], "before-reader barrier marker outside reader")
        event_proof = True
    elif mode == "event-after":
        marker = by_frame[0].get(("after_reader_next_frame_published", 1))
        one = leases[lease_by_frame[(1, 0)]]
        _require(marker is not None and one["producer_published"]["sequence"] < marker["sequence"]
                 < by_frame[0][("stage_end", 1)]["sequence"] and bool(overlap),
                 "after-reader test did not demonstrate next-generation production before stage end")
        _require(one["producer_published"]["time_ms"] <= marker["time_ms"] <= frames[0]["stage_run_end_ms"][1],
                 "after-reader barrier marker outside consumer execution")
        _require(marker["sequence"] < by_frame[0][("node2_execute", 1)]["sequence"]
                 and marker["time_ms"] <= by_frame[0][("node2_execute", 1)]["time_ms"],
                 "after-reader control did not hold packed computation until next production")
        event_proof = True
    return {
        "call_ledger_verified": True,
        "lease_trace_verified": not reference,
        "affinity_verified": affinity_verified,
        "dynamic_ranges_verified": ranges_verified,
        "formal_copy_count": copy_count,
        "formal_copy_bytes": copy_count * INPUT_BYTES,
        "early_release_overlap_verified": bool(overlap) if variant == "E1" and not reference else None,
        "early_release_overlap_examples": overlap,
        "deterministic_event_proof_verified": event_proof,
        "observed_stage_calls": sum(result["stage_call_counts"]),
        "observed_complete_inferences": len(frames),
    }
