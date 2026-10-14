"""Bounded two-engine interleaving proof, not a cycle/performance simulator."""

from collections import deque


def certify_stream_window(release_groups=(12, 13), mapping=None):
    """Enumerate starts AND completions; reject overwrite during active MAC.

    Preconditions: existing spatial bridge completes preceding tile; resident
    slots are disjoint; one FIFO Load and one FIFO Compute engine. UOP/SRAM
    and complete compiled queue qualification are separate mandatory checks.
    """
    mapping = mapping or {12: 12, 13: 13, 14: 12, 15: 13}
    load = tuple(range(12, 16))
    compute = tuple((kind, group) for group in load for kind in ("RESET", "MAC"))
    # completed indices, busy instruction indices (-1 means idle), slot
    # contents, available Load->Compute and Compute->Load completion tokens.
    initial = (0, 0, -1, -1, (-1, -1), 0, 0)
    pending, visited = deque([initial]), {initial}
    overlaps, terminal = set(), 0
    while pending:
        state = pending.popleft()
        lp, cp, lb, cb, values, ready, released = state
        if lp == len(load) and cp == len(compute) and lb == cb == -1:
            if ready or released:
                raise ValueError("stream leaves unmatched completion tokens")
            terminal += 1
            continue
        successors = []
        if lb == -1 and lp < len(load) and (load[lp] < 14 or released):
            slot = mapping[load[lp]]
            if cb != -1 and compute[cb][0] == "MAC" and mapping[compute[cb][1]] == slot:
                raise ValueError("WGT overwrite can begin before previous read completes")
            successors.append((lp, cp, lp, cb, values, ready,
                               released - int(load[lp] >= 14)))
        if lb != -1:
            contents = list(values)
            contents[mapping[load[lb]] - 12] = load[lb]
            successors.append((lp + 1, cp, -1, cb, tuple(contents), ready + 1, released))
        if cb == -1 and cp < len(compute):
            kind, group = compute[cp]
            if kind == "RESET" or ready:
                if kind == "MAC" and values[mapping[group] - 12] != group:
                    raise ValueError("MAC can consume overwritten/not-yet-loaded WGT")
                if kind == "MAC" and lb != -1 and mapping[load[lb]] == mapping[group]:
                    raise ValueError("MAC can begin while same slot LOAD is active")
                successors.append((lp, cp, lb, cp, values, ready - int(kind == "MAC"), released))
        if cb != -1:
            kind, group = compute[cb]
            successors.append((lp, cp + 1, lb, -1, values, ready,
                               released + int(kind == "MAC" and group in release_groups)))
        if not successors:
            raise ValueError("FIFO circular wait: " + str(state))
        if lb != -1 and cb != -1 and compute[cb][0] == "MAC":
            overlaps.add((load[lb], compute[cb][1]))
        for nxt in successors:
            if nxt not in visited:
                visited.add(nxt)
                pending.append(nxt)
    if not terminal or (13, 12) not in overlaps or (14, 13) not in overlaps:
        raise ValueError("two slots did not expose the required LOAD/MAC windows")
    return {
        "status": "BOUNDED_STREAM_INTERLEAVINGS_PASSED", "states": len(visited),
        "terminal_states": terminal,
        "possible_independent_load_mac_pairs": [list(x) for x in sorted(overlaps)],
        "slot_mapping": mapping,
        "all_interleavings_no_RAW_WAR_or_circular_wait": True,
        "not_hardware_trace_or_latency_prediction": True,
        "preconditions": ["prior tile complete through unchanged spatial bridge",
                          "disjoint resident slots", "native FIFO/token completion semantics"],
        "remaining_evidence": ["actual compiled full queue/token mapping",
                               "UOP allocation/references", "FPGA correctness"],
    }
