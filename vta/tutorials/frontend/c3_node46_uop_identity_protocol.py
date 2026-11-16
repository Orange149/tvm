"""Pure, frozen TPS B/A protocol; this module never contacts a device.

``timed`` identifies a formal performance sample, not the helper to call.
Qualification, weight checks and warmups also record elapsed_ms, but their
times never enter the paired performance analysis.  The runner must certify
one target invocation and its helper/driver completion separately.
"""

from collections import Counter
import json
import math
from statistics import median, pstdev


SEEDS = (0, 20250901, 20260910)
PROGRAMS = ("B", "A")
ROUND_ORDERS = (("B", "A"), ("A", "B")) * 3
SCHEMA = "c3_node46_uop_identity_120_calls_v1"
_DIAGNOSTIC_PHASES = frozenset(("diagnostic_prefix", "diagnostic_warm"))


def _canonical(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("NON_JSON_OR_NONFINITE_IDENTITY") from exc


def _copy_json(value):
    return json.loads(_canonical(value))


def manifest(attempt, programs):
    """Create the complete 120-call ledger and its 20 fresh-RPC sessions.

    programs must map exactly B and A to caller-frozen JSON identity dicts.
    Each result must retain every field of its corresponding calls row.
    """
    if not isinstance(attempt, str) or not attempt.strip():
        raise ValueError("ATTEMPT_ID_REQUIRED")
    if not isinstance(programs, dict) or set(programs) != set(PROGRAMS):
        raise ValueError("EXACT_TPS_B_A_POOL_REQUIRED")
    if any(not isinstance(programs[p], dict) for p in PROGRAMS):
        raise ValueError("PROGRAM_IDENTITY_DICT_REQUIRED")
    identities = {p: _copy_json(programs[p]) for p in PROGRAMS}
    calls, sessions = [], []

    def add_session(program, phase, rotation, round_id, specs, collection):
        sid = "%s:s%03d" % (attempt, len(sessions) + 1)
        first = len(calls) + 1
        for position, (seed, role, weight) in enumerate(specs, 1):
            calls.append({
                "call_index": len(calls) + 1,
                "session_id": sid,
                "program": program,
                "program_identity": _copy_json(identities[program]),
                "phase": role,
                "round": round_id,
                "rotation": rotation,
                "seed": seed,
                "position": position,
                "timed": role == "formal",
                "collection": collection,
                "weight": weight,
            })
        sessions.append({
            "session_id": sid,
            "program": program,
            "program_identity": _copy_json(identities[program]),
            "phase": phase,
            "round": round_id,
            "rotation": rotation,
            "collection": collection,
            "first_call_index": first,
            "last_call_index": len(calls),
            "target_calls": len(specs),
        })

    for program in PROGRAMS:
        specs = [(seed, "qualification", "W0") for seed in SEEDS]
        specs += [(SEEDS[0], "weight_check", weight) for weight in ("W0", "W1", "W0")]
        add_session(program, "qualification", None, None, specs, "detailed")

    for rotation in range(3):
        seeds = SEEDS[rotation:] + SEEDS[:rotation]
        for program in PROGRAMS:
            specs = [(seed, "diagnostic_prefix", "W0") for seed in seeds]
            specs += [(seed, "diagnostic_warm", "W0") for seed in seeds]
            add_session(program, "diagnostic", rotation, None, specs, "detailed")

    for round_id, order in enumerate(ROUND_ORDERS, 1):
        rotation = (round_id - 1) % 3
        seeds = SEEDS[rotation:] + SEEDS[:rotation]
        for program in order:
            specs = [(seed, "warmup", "W0") for seed in seeds]
            specs += [(seed, "formal", "W0") for seed in seeds]
            add_session(program, "formal", rotation, round_id, specs, "minimal")

    return {
        "schema": SCHEMA,
        "attempt": attempt,
        "programs": identities,
        "seeds": list(SEEDS),
        "round_orders": [list(order) for order in ROUND_ORDERS],
        "calls": calls,
        "sessions": sessions,
        "phase_counts": dict(Counter(row["phase"] for row in calls)),
        "total_target_calls": len(calls),
        "formal_samples": sum(row["timed"] for row in calls),
        "implicit_calls_allowed": 0,
    }


def _check_manifest(ledger):
    if not isinstance(ledger, dict):
        raise ValueError("INVALID_FROZEN_MANIFEST")
    try:
        expected = manifest(ledger["attempt"], ledger["programs"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("INVALID_FROZEN_MANIFEST") from exc
    for key, value in expected.items():
        if key not in ledger or _canonical(ledger[key]) != _canonical(value):
            raise ValueError("FROZEN_MANIFEST_CHANGED:" + key)


def _positive_finite_ms(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value) and value > 0
    except OverflowError:
        return False


def validate_results(ledger, results):
    """Reject partial, reordered, changed-identity or invalid-time attempts.

    The runner owns physical buffer, helper-start/completion, output and
    submission-path checks.  PASS is its attestation that those checks passed.
    An optional per-row target_calls count, when supplied, must equal one.
    """
    _check_manifest(ledger)
    if not isinstance(results, (list, tuple)) or len(results) != len(ledger["calls"]):
        raise ValueError("INCOMPLETE_ATTEMPT_NO_PERFORMANCE_RESULT")
    for expected, actual in zip(ledger["calls"], results):
        if not isinstance(actual, dict):
            raise ValueError("RESULT_ROW_NOT_OBJECT")
        for key, value in expected.items():
            if key not in actual or _canonical(actual[key]) != _canonical(value):
                raise ValueError("CALL_MANIFEST_MISMATCH:%d:%s" % (expected["call_index"], key))
        if actual.get("status") != "PASS":
            raise ValueError("FAILED_TARGET_CALL:%d" % expected["call_index"])
        if "target_calls" in actual and (
                type(actual["target_calls"]) is not int or actual["target_calls"] != 1):
            raise ValueError("UNEXPECTED_TARGET_CALL_COUNT:%d" % expected["call_index"])
        if "elapsed_ms" not in actual:
            raise ValueError("MISSING_ELAPSED_MS:%d" % expected["call_index"])
        elapsed = actual["elapsed_ms"]
        if elapsed is None and expected["phase"] in _DIAGNOSTIC_PHASES:
            continue
        if not _positive_finite_ms(elapsed):
            raise ValueError("INVALID_ELAPSED_MS:%d" % expected["call_index"])
    return {
        "status": "COMPLETE_120_CALL_ATTEMPT_VALIDATED",
        "target_calls": len(results),
        "sessions": len(ledger["sessions"]),
        "formal_samples": ledger["formal_samples"],
    }


def _dispersion(values):
    center = median(values)
    return {
        "range_ms": [min(values), max(values)],
        "population_stdev_ms": pstdev(values),
        "MAD_ms": median(abs(value - center) for value in values),
    }


def analyse(ledger, results):
    """Analyse only the 36 formal samples after validating all 120 calls."""
    validation = validate_results(ledger, results)
    samples = {
        (row["program"], row["round"], row["seed"]): row["elapsed_ms"]
        for row in results if row["phase"] == "formal"
    }
    round_medians = {
        program: [median(samples[(program, round_id, seed)] for seed in SEEDS)
                  for round_id in range(1, 7)]
        for program in PROGRAMS
    }
    totals = {program: median(values) for program, values in round_medians.items()}
    pairwise = {}
    for candidate, baseline in (("A", "B"),):
        candidate_ms, baseline_ms = round_medians[candidate], round_medians[baseline]
        gains = [100 * (1 - c / b) for c, b in zip(candidate_ms, baseline_ms)]
        absolute = [b - c for c, b in zip(candidate_ms, baseline_ms)]
        gain = median(gains)
        wins = sum(c < b for c, b in zip(candidate_ms, baseline_ms))
        pairwise[candidate + "/" + baseline] = {
            "round_gains_percent": gains,
            "Gain_percent": gain,
            "wins": wins,
            "round_absolute_ms": absolute,
            "absolute_ms": median(absolute),
            "gain_range_percent": [min(gains), max(gains)],
            "gain_MAD_percent": median(abs(value - gain) for value in gains),
            "engineering_goal_met": gain >= 2 and wins >= 5,
        }
    return {
        "schema": "c3_node46_uop_identity_paired_analysis_v1",
        "status": "COMPLETE_SINGLE_BOOT_PAIRED_RESULTS",
        "validation": validation,
        "round_medians": round_medians,
        "T": totals,
        "time_unit": "ms",
        "dispersion": {p: _dispersion(values) for p, values in round_medians.items()},
        "pairwise": pairwise,
        "primary_comparison": "A/B",
        "engineering_goal_met": pairwise["A/B"]["engineering_goal_met"],
        "statistical_significance_claimed": False,
        "independent_boot_confirmation": False,
    }


def transport_contract(ledger, base_port=9600):
    _check_manifest(ledger)
    if type(base_port) is not int or not 1024 <= base_port <= 32000:
        raise ValueError('PORT_RANGE_OUTSIDE_FROZEN_SUPPORT')
    sessions = ledger['sessions']
    if len(sessions) != 20 or len({s['session_id'] for s in sessions}) != 20:
        raise ValueError('TRANSPORT_REQUIRES_20_UNIQUE_SESSIONS')
    return {'schema': 'c3_node46_uop_identity_transport_v1', 'base_port': base_port,
            'sessions': [{'ordinal': i, 'session_id': s['session_id'], 'port': base_port+i-1}
                         for i, s in enumerate(sessions, 1)],
            'rpc_connect_attempts_per_session': 1, 'target_retries': 0,
            'port_reuse_within_attempt': False,
            'shutdown': 'release remote references, explicit CloseRPCConnection, stop owned server'}
