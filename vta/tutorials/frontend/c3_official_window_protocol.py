#!/usr/bin/env python3
"""Pure local contracts for the bounded official-program aggregation study.

No compiler, runtime, RPC, board, or filesystem operation is performed here.
Selection uses structural evidence only. Missing/failed attempts yield no
performance result; diagnostic calls and qualification costs are never samples.
"""

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import math
from statistics import median


SCHEMA = "c3_official_window_protocol_v1"
SEEDS = (0, 20250901, 20260910)
CLASSES = (1, 2, 3, 4)
MAX_OBJECTS = 4
MAX_PROGRAMS = 8
MAX_SUBMITS = 16


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _classes(row):
    result = row.get("classes", [row["class_id"]] if "class_id" in row else [])
    _require((result or row.get("eligible") is False) and set(result) <= set(CLASSES), "unknown/empty structural classes")
    return sorted(set(result))


def freeze_pool(audit_rows, max_objects=MAX_OBJECTS):
    """Choose each class's first structural object, then deduplicate; no refill.

    Required row fields: object_id, network, earliest_graph_node, semantic_hash,
    classes, delta_mac_insns, official_baseline_bound, local_gates_passed,
    programs={B/A: {program_id, execution_sha256, abi_sha256, ...}}.
    Unsupported rows may have eligible=False and rejection_reason. Frozen
    structural selection is retained even while local execution gates are pending.
    """
    _require(max_objects == MAX_OBJECTS, "the four-class budget is fixed")
    audits = deepcopy(list(audit_rows))
    _require(len({row["object_id"] for row in audits}) == len(audits), "duplicate audit object IDs")
    for row in audits:
        _require(not row.get("performance_labels_used", False), "selection cannot use performance labels")
        _classes(row)
        _require(isinstance(row["delta_mac_insns"], int) and row["delta_mac_insns"] >= 0,
                 "dynamic MAC reduction must be a nonnegative integer")
        _require(isinstance(row["earliest_graph_node"], int) and row["earliest_graph_node"] >= 0,
                 "earliest graph node must be an integer")
        _require(row["semantic_hash"] and row["network"], "missing object identity")
    eligible = [row for row in audits if row.get("eligible", True) and row.get("official_baseline_bound", False)]
    key = lambda row: (-row["delta_mac_insns"], row["network"],
                       row["earliest_graph_node"], row["semantic_hash"])
    winners = {}
    for category in CLASSES:
        members = sorted((row for row in eligible if category in _classes(row)), key=key)
        winners[str(category)] = members[0]["object_id"] if members else None
    by_id = {row["object_id"]: row for row in audits}
    selected, seen = [], {}
    for category in CLASSES:
        oid = winners[str(category)]
        if oid is None:
            continue
        row = by_id[oid]
        programs = row["programs"]
        _require(set(programs) == {"B", "A"}, "each object requires B and A representations")
        for program in programs.values():
            _require(all(program.get(field) for field in ("program_id", "execution_sha256", "abi_sha256")),
                     "incomplete execution representation identity")
        _require(programs["B"]["abi_sha256"] == programs["A"]["abi_sha256"], "B/A complete ABI differs")
        # Both complete execution representations AND ABI must match. Names,
        # network labels, or mathematical equivalence alone are insufficient.
        identity = (programs["B"]["execution_sha256"], programs["B"]["abi_sha256"],
                    programs["A"]["execution_sha256"], programs["A"]["abi_sha256"])
        if identity in seen:
            selected[seen[identity]]["selection_classes"].append(category)
            continue
        entry = deepcopy(row)
        entry["selection_classes"] = [category]
        entry["comparison_applicable"] = programs["B"]["execution_sha256"] != programs["A"]["execution_sha256"]
        entry["comparison_status"] = "PENDING_MEASUREMENT" if entry["comparison_applicable"] else "DEDUP_NOT_APPLICABLE"
        entry["baseline_program"] = programs["B"]["program_id"]
        entry["aggregate_program"] = programs["A"]["program_id"] if entry["comparison_applicable"] else entry["baseline_program"]
        seen[identity] = len(selected)
        selected.append(entry)
    unique = []
    program_ids = set()
    execution_owners = {}
    for entry in selected:
        for role in ("B", "A") if entry["comparison_applicable"] else ("B",):
            program = deepcopy(entry["programs"][role])
            pid = program["program_id"]
            _require(pid not in program_ids, "program ID reused between distinct selected objects")
            representation = (program["abi_sha256"], program["execution_sha256"])
            _require(representation not in execution_owners,
                     "identical cross-object representation requires an explicit common object identity")
            execution_owners[representation] = entry["object_id"]
            program_ids.add(pid)
            program.update(object_id=entry["object_id"], role=role)
            unique.append(program)
    _require(len(selected) <= MAX_OBJECTS and len(unique) <= MAX_PROGRAMS, "fixed pool exceeded")
    different = sum(row["comparison_applicable"] for row in selected)
    gates = bool(selected) and all(row.get("local_gates_passed", False) for row in selected)
    allowed = gates and different > 0
    status = ("FROZEN_LOCAL_GATED_POOL" if allowed else
              "STOP_ALL_EXECUTION_REPRESENTATIONS_IDENTICAL" if selected and different == 0 else
              "STOP_NO_SELECTED_OBJECT" if not selected else "FROZEN_POOL_LOCAL_GATES_PENDING")
    funnel = {
        "audited_objects": len(audits), "official_baseline_bound": sum(bool(row.get("official_baseline_bound")) for row in audits),
        "structurally_eligible_objects": len(eligible),
        "eligible_per_class": {str(c): sum(c in _classes(row) for row in eligible) for c in CLASSES},
        "nonempty_class_selections": sum(value is not None for value in winners.values()),
        "unique_selected_objects": len(selected), "unique_programs": len(unique),
        "different_B_A_objects": different, "dedup_NA_controls": len(selected) - different,
        "local_gated_selected_objects": sum(bool(row.get("local_gates_passed")) for row in selected),
    }
    pool = {"schema": SCHEMA, "status": status, "board_allowed": allowed,
            "selection_rule": "each class: descending dynamic MAC instruction reduction, then network/earliest node/semantic hash ascending; deduplicate after first choices; never refill",
            "class_first_choices": winners, "objects": selected, "programs": unique,
            "coverage_funnel": funnel, "audit_rows_sha256": digest(audits),
            "seeds": list(SEEDS), "no_retry": True, "no_replacement_after_selection": True,
            "formal_runtime_mode": "minimal", "diagnostic_runtime_mode": "detailed"}
    pool["pool_sha256"] = digest(pool)
    return pool


def _verify_pool(pool):
    content = {key: value for key, value in pool.items() if key != "pool_sha256"}
    _require(pool.get("schema") == SCHEMA and digest(content) == pool.get("pool_sha256"), "frozen pool identity changed")


def build_call_manifest(pool, attempt):
    """Predeclare 60K calls, fresh sessions, rotations, and balanced B/A order.

    Every target invocation uses either once timer or once diagnostic helper.
    Timer results for warmups/ABA/prefix calls are excluded. Both helpers count
    every invocation; reset once before each six-call session, then counts 1..6.
    """
    _verify_pool(pool)
    _require(pool["board_allowed"], "pool is not admitted for board calls")
    _require(isinstance(attempt, str) and attempt and all(c.isalnum() or c in "_-" for c in attempt), "invalid attempt ID")
    calls, sessions = [], []
    by_id = {row["program_id"]: row for row in pool["programs"]}

    def add_session(phase, pid, rotation, round_id, specifications):
        sid = "%s:%s:%03d" % (attempt, phase, len(sessions))
        session = {"session_id": sid, "phase": phase, "program_id": pid,
                   "object_id": by_id[pid]["object_id"], "rotation": rotation, "round": round_id,
                   "mode": "detailed" if phase == "diagnostic" else "minimal",
                   "fresh_rpc": True, "helper_reset_once_before_first_call": True}
        sessions.append(session)
        for position, spec in enumerate(specifications, 1):
            row = {**session, **spec, "position": position, "helper_count_after": position,
                   "invocation_id": "%s:%06d" % (attempt, len(calls)),
                   "complete_operator_calls": 1, "hidden_warmup_calls": 0,
                   "timer_result_is_formal_sample": spec["call_kind"] == "timing"}
            row["helper"] = "diagnostic_once" if row.get("capture_dump", False) else "timed_once"
            calls.append(row)

    for program in pool["programs"]:
        specs = [{"call_kind": "correctness", "seed": seed, "weight_version": "W0", "capture_dump": False} for seed in SEEDS]
        specs += [{"call_kind": "weight_ABA", "seed": SEEDS[0], "weight_version": weight,
                   "aba_label": label, "capture_dump": False} for weight, label in (("W0", "W0"), ("W1", "W1"), ("W0", "W0_return"))]
        add_session("qualification", program["program_id"], 0, 0, specs)

    objects = pool["objects"]
    nobjects = len(objects)
    for phase, rounds in (("diagnostic", 3), ("formal", 6)):
        for r in range(rounds):
            seeds = list(SEEDS[r % 3:] + SEEDS[:r % 3])
            order = list(range(nobjects))[r % nobjects:] + list(range(nobjects))[:r % nobjects]
            if phase == "formal" and r == 5:
                order = list(reversed(range(nobjects)))
            for object_index in order:
                obj = objects[object_index]
                pids = [obj["baseline_program"]]
                if obj["comparison_applicable"]:
                    pids.append(obj["aggregate_program"])
                    if r % 2:
                        pids.reverse()
                for pid in pids:
                    specs = []
                    for position, seed in enumerate(seeds):
                        specs.append({"call_kind": "diagnostic_prefix" if phase == "diagnostic" else "warmup",
                                      "seed": seed, "weight_version": "W0",
                                      "capture_dump": phase == "diagnostic" and position == 0})
                    for seed in seeds:
                        specs.append({"call_kind": "warmed_diagnostic" if phase == "diagnostic" else "timing",
                                      "seed": seed, "weight_version": "W0", "capture_dump": phase == "diagnostic"})
                    add_session(phase, pid, r % 3, r + 1, specs)
    k = len(pool["programs"])
    expected = {"correctness": 3*k, "weight_ABA": 3*k, "diagnostic_prefix": 9*k,
                "warmed_diagnostic": 9*k, "warmup": 18*k, "timing": 18*k}
    _require(Counter(row["call_kind"] for row in calls) == expected and len(calls) == 60*k,
             "internal manifest budget error")
    result = {"schema": SCHEMA, "attempt": attempt, "pool_sha256": pool["pool_sha256"],
              "K": k, "target_calls": 60*k, "expected_call_counts": expected,
              "sessions": sessions, "calls": calls, "helper_count_protocol": "one reset per fresh session; both helpers count all six calls 1..6",
              "no_implicit_target_calls": True, "no_retry": True, "stop_on_first_error": True}
    result["manifest_sha256"] = digest(result)
    return result


def canonical_submit_sequence(metadata, expected_mode=None):
    """Validate bounded raw submit evidence; never hash/copy command contents."""
    _require(metadata.get("schema") == "c3_official_window_submit_ledger_v1", "unknown submit ledger schema")
    if expected_mode is not None:
        _require(metadata.get("mode") == expected_mode, "collection mode changed")
    submits = metadata.get("submits", [])
    _require(not metadata.get("overflow", True), "submit ledger overflow invalidates attempt")
    _require(0 < len(submits) <= MAX_SUBMITS, "missing/out-of-bound submit evidence")
    _require(metadata.get("attempted_submits") == metadata.get("recorded_submits") == len(submits), "submit ledger incomplete")
    canonical = []
    for position, row in enumerate(submits, 1):
        _require(row.get("ordinal") == position and row.get("completion") is True and row.get("timeout") == 0,
                 "submit ordinal/completion/timeout failed")
        _require(isinstance(row.get("insns"), int) and row["insns"] > 0 and row.get("insn_bytes") == 16*row["insns"], "instruction size unit mismatch")
        _require(isinstance(row.get("serialized_uop_bytes"), int) and row["serialized_uop_bytes"] >= 0 and row["serialized_uop_bytes"] % 4 == 0,
                 "serialized UOP byte unit mismatch")
        _require(isinstance(row.get("reason"), str) and row["reason"], "missing submit reason")
        canonical.append({key: row[key] for key in ("ordinal", "reason", "insns", "insn_bytes", "serialized_uop_bytes", "completion", "timeout")})
    return canonical


def validate_call(expected, actual):
    """One manifest invocation must complete exactly once; no retry acceptance."""
    for key in ("invocation_id", "session_id", "program_id", "object_id", "phase", "call_kind", "round", "rotation", "seed", "position", "weight_version"):
        _require(actual.get(key) == expected[key], "call manifest mismatch: " + key)
    _require(actual.get("status") == "PASS" and actual.get("correct") is True and actual.get("arguments_unchanged") is True,
             "failed output/arguments/status invalidates attempt")
    _require(actual.get("complete_operator_calls") == 1 and actual.get("hidden_warmup_calls") == 0, "unexpected target call")
    _require(actual.get("expected_sha256") and actual.get("output_sha256") == actual["expected_sha256"],
             "complete observable output hash mismatch")
    _require(actual.get("helper_count_after") == expected["helper_count_after"], "once-helper count changed")
    for field in ("helper_started_after", "helper_completed_after"):
        if field in actual:
            _require(actual[field] == expected["helper_count_after"], "helper did not complete exactly once")
    if expected["call_kind"] == "timing":
        value = actual.get("latency_ms")
        _require(isinstance(value, (int, float)) and math.isfinite(value) and value > 0, "invalid complete-call latency")
    return canonical_submit_sequence(actual["runtime_metadata"], expected["mode"])


def round_statistics(round_medians):
    """Describe six round medians; raw MAD is unscaled, not uncertainty."""
    _require(len(round_medians) == 6 and all(math.isfinite(v) and v > 0 for v in round_medians), "six finite positive round medians required")
    value = median(round_medians)
    return {"m_ms": list(round_medians), "T_ms": value,
            "MAD_ms": median(abs(v-value) for v in round_medians),
            "min_ms": min(round_medians), "max_ms": max(round_medians)}


def paired_statistics(candidate, baseline):
    _require(len(candidate) == len(baseline) == 6, "six paired round medians required")
    round_statistics(candidate)
    round_statistics(baseline)
    gains = [100*(1-a/b) for a,b in zip(candidate,baseline)]
    saved = [b-a for a,b in zip(candidate,baseline)]
    return {"g_pct": gains, "G_pct": median(gains), "ms_saved": saved,
            "median_ms_saved": median(saved), "wins": sum(a<b for a,b in zip(candidate,baseline)),
            "ties": sum(a==b for a,b in zip(candidate,baseline)),
            "losses": sum(a>b for a,b in zip(candidate,baseline))}


def analyse(pool, manifest, calls):
    """Check a complete attempt before producing any six-round conclusions.

    calls are raw once-invocation rows, including all qualification/diagnostic/
    warmup/timing calls in manifest order. Detailed diagnostics additionally
    supply native_certificate_passed=True for every requested dump. Collection
    modes share one DSO; the caller must bind binary/runtime/boot attestations.
    """
    _verify_pool(pool)
    _require(manifest == build_call_manifest(pool, manifest["attempt"]), "manifest differs from frozen schedule")
    raw = list(calls)
    _require(len(raw) == len(manifest["calls"]), "incomplete/additional calls invalidate attempt")
    _require(len({row.get("invocation_id") for row in raw}) == len(raw), "duplicate invocation identities")
    _require(all(isinstance(row.get("boot_id"), str) and row["boot_id"] for row in raw)
             and len({row["boot_id"] for row in raw}) == 1, "single boot identity missing/changed")
    sequences = {}
    for expected, actual in zip(manifest["calls"], raw):
        sequence = validate_call(expected, actual)
        sequences[expected["invocation_id"]] = sequence
        if expected["capture_dump"]:
            _require(actual.get("native_certificate_passed") is True, "uncertified actual diagnostic invocation")
    diagnostics = {(row["program_id"], row["rotation"], row["position"]): sequences[row["invocation_id"]]
                   for row in raw if row["phase"] == "diagnostic"}
    session_buffers = {}
    for row in raw:
        _require(row.get("buffer_handles"), "missing long-lived parameter identities")
        identity = tuple(row["buffer_handles"])
        session_buffers.setdefault(row["session_id"], identity)
        _require(session_buffers[row["session_id"]] == identity, "parameter handle changed within fresh session")
        if row["phase"] != "diagnostic":
            _require(sequences[row["invocation_id"]] == diagnostics[row["program_id"], row["rotation"], row["position"]],
                     "complete submit sequence differs from state-corresponding diagnostic")
    for program in pool["programs"]:
        aba = [row for row in raw if row["program_id"] == program["program_id"] and row["call_kind"] == "weight_ABA"]
        _require(len(aba) == 3, "W0/W1/W0 qualification missing")
        hashes = [row.get("expected_sha256") for row in aba]
        _require(all(hashes) and hashes[0] == hashes[2] != hashes[1], "W0/W1 references are not observably distinct")
        weights = [row.get("weight_payload_sha256") for row in aba]
        _require(all(weights) and weights[0] == weights[2] != weights[1], "W0/W1 weight switch not established")
    groups = defaultdict(list)
    for row in raw:
        if row["call_kind"] == "timing":
            groups[row["program_id"], row["round"]].append(row["latency_ms"])
    statistics = {}
    for program in pool["programs"]:
        pid = program["program_id"]
        _require(all(len(groups[pid,r]) == 3 for r in range(1,7)), "missing three-seed formal cell")
        statistics[pid] = round_statistics([median(groups[pid,r]) for r in range(1,7)])
    comparisons = {}
    for obj in pool["objects"]:
        oid = obj["object_id"]
        if not obj["comparison_applicable"]:
            comparisons[oid] = {"status": "DEDUP_NOT_APPLICABLE", "baseline_program": obj["baseline_program"],
                                "G_pct": None, "wins": None, "ties": None}
        else:
            pair = paired_statistics(statistics[obj["aggregate_program"]]["m_ms"], statistics[obj["baseline_program"]]["m_ms"])
            comparisons[oid] = {"status": "COMPLETE_PAIRED_RESULT", "baseline_program": obj["baseline_program"],
                                "aggregate_program": obj["aggregate_program"], **pair}
    applicable = [row for row in comparisons.values() if row["status"] == "COMPLETE_PAIRED_RESULT"]
    return {"schema": SCHEMA, "status": "COMPLETE_SINGLE_BOOT_OFFICIAL_FIXED_POOL_RESULT",
            "pool_sha256": pool["pool_sha256"], "manifest_sha256": manifest["manifest_sha256"],
            "call_ledger_sha256": digest(raw), "statistics": statistics, "comparisons": comparisons,
            "all_selected_negative": bool(applicable) and all(row["G_pct"] < 0 for row in applicable),
            "coverage_funnel": deepcopy(pool["coverage_funnel"]),
            "formal_timings": 18*manifest["K"], "formal_warmups": 18*manifest["K"],
            "qualification_calls": 6*manifest["K"], "diagnostic_calls": 18*manifest["K"],
            "statistical_significance_claimed": False, "graph_performance_measured": False,
            "pause_after_attempt": True, "no_retry_or_failed_sample_splicing": True,
            "MAD_definition": "unscaled median absolute deviation of six round medians; not a confidence interval",
            "formal_evidence_scope": "bounded per-submit metadata matched to independent detailed diagnostics; no timed instruction dump or direct SRAM readback"}


analyze = analyse
