"""Frozen node24 tradeoff protocol and local interpolation, without board I/O."""

from __future__ import annotations

from collections import Counter
from itertools import combinations
import hashlib
import json
import math
from pathlib import Path
from statistics import median

import numpy as np


SCHEMA = "c3_node24_residency_instruction_tradeoff_v1_1"
SEEDS = (0, 20250901, 20260910)
PROGRAMS = ("F13-B", "R13-G", "R0-G", "R8-G", "R12-G", "Input-7x7")
CALIBRATION = ("R0-G", "R13-G", "F13-B")
VALIDATION = ("R8-G", "R12-G")
RESIDENT = {"R0-G": 0, "R8-G": 8, "R12-G": 12, "R13-G": 13, "F13-B": 13}
ORDERS = tuple(tuple(PROGRAMS[ord(label) - ord("A")] for label in row.split())
               for row in ("A B F C E D", "B C A D F E", "C D B E A F",
                           "D E C F B A", "E F D A C B", "F A E B D C"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    """Never overwrite a sealed protocol, model, or previous attempt."""
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def expectations(name):
    if name not in PROGRAMS:
        raise ValueError("undeclared program")
    reset = 4 if name == "Input-7x7" else 112
    mac = 8 if name == "Input-7x7" else 896 if name == "F13-B" else 112
    loads = 8 if name == "Input-7x7" else 112 - 6 * RESIDENT[name]
    return {"reset": reset, "mac": mac, "total_gemm_push": reset + mac,
            "wgt_loads": loads,
            "wgt_payload_bytes": 1152 * 1024 if name == "Input-7x7" else loads * 18432}


def balance():
    positions = Counter((name, pos) for row in ORDERS for pos, name in enumerate(row))
    adjacent = Counter((a, b) for row in ORDERS for a, b in zip(row, row[1:]))
    valid = (len(positions) == 36 and set(positions.values()) == {1}
             and len(adjacent) == 30 and set(adjacent.values()) == {1})
    if not valid:
        raise ValueError("frozen six-program Williams order is not balanced")
    return {"position_balanced": True, "within_row_predecessor_balanced": True,
            "positions": {"%s@%d" % key: count for key, count in positions.items()},
            "adjacent": {"%s->%s" % key: count for key, count in adjacent.items()}}


def seed_order(round_id):
    if round_id not in range(6):
        raise ValueError("round outside frozen six-round protocol")
    offset = round_id % 3
    return SEEDS[offset:] + SEEDS[:offset]


def verify_call_profile(name, profile, signature):
    """The existing profiler includes RESET; use a qualified command signature."""
    exp = expectations(name)
    if signature.get("reset") != exp["reset"] or signature.get("mac") != exp["mac"]:
        raise ValueError("RESET/MAC command signature changed")
    if (profile.get("push_gemm_op_calls") != signature["reset"] + signature["mac"]
            or profile.get("driver_timeout_calls", 0)
            or profile.get("driver_run_calls", 0) <= 0):
        raise ValueError("target call profile does not match RESET+MAC signature")
    # Return the qualified non-RESET feature, not total profiler pushes.
    return {"wgt_loads": signature["wgt_loads"], "mac": signature["mac"]}


def round_medians(rows, programs=PROGRAMS):
    selected = [row for row in rows if row.get("call_kind") == "timing"]
    if len(selected) != len(programs) * 18:
        raise ValueError("incomplete or extra formal timing calls")
    if len({row.get("boot_id") for row in selected}) != 1 or \
            any(not row.get("boot_id") for row in selected):
        raise ValueError("mixed or absent boot identity")
    result = {r: {} for r in range(6)}
    seen, invocations = set(), set()
    for row in selected:
        r, name, seed = row["round"], row["program"], row["seed"]
        key = (r, name, seed)
        invocation = row["invocation_id"]
        if (r not in range(6) or name not in programs or seed not in SEEDS
                or key in seen or invocation in invocations):
            raise ValueError("duplicate or out-of-domain timing identity")
        if row.get("status") != "PASS" or not row.get("correct") or \
                not row.get("arguments_unchanged"):
            raise ValueError("invalid timing sample cannot enter statistics")
        t = row.get("latency_ms")
        if not isinstance(t, (float, int)) or not math.isfinite(t) or t <= 0:
            raise ValueError("missing or nonpositive latency is not zero")
        seen.add(key)
        invocations.add(invocation)
    for r in range(6):
        for name in programs:
            samples = [row["latency_ms"] for row in selected
                       if row["round"] == r and row["program"] == name]
            if len(samples) != 3:
                raise ValueError("missing three-seed unit")
            result[r][name] = median(samples)
    return result


def compare(medians, candidate, baseline):
    gains = [100 * (1 - medians[r][candidate] / medians[r][baseline]) for r in range(6)]
    return {"candidate": candidate, "baseline": baseline,
            "round_improvement_pct": gains, "median_improvement_pct": median(gains),
            "wins": sum(medians[r][candidate] < medians[r][baseline] for r in range(6))}


def summarize(rows):
    rm = round_medians(rows)
    overall = {p: median(rm[r][p] for r in range(6)) for p in PROGRAMS}
    pairs = [("R13-G", "F13-B"), ("R8-G", "R0-G"),
             ("R12-G", "R8-G"), ("R13-G", "R12-G")]
    pairs += [(p, b) for p in PROGRAMS if p not in ("R13-G", "Input-7x7")
              for b in ("R13-G", "Input-7x7")]
    pairs += [("R13-G", "Input-7x7")]
    comparisons = {"%s_vs_%s" % (a, b): compare(rm, a, b) for a, b in pairs}
    eligible = []
    for p in ("R0-G", "R8-G", "R12-G"):
        primary, practical = comparisons[p + "_vs_R13-G"], comparisons[p + "_vs_Input-7x7"]
        if (primary["median_improvement_pct"] >= 2 and primary["wins"] >= 5
                and practical["median_improvement_pct"] > 0):
            eligible.append(p)
    return {"status": "SINGLE_BOOT_COMPLETE", "round_medians_ms": rm,
            "program_medians_ms": overall, "comparisons": comparisons,
            "future_confirmation_candidates": eligible, "not_graph_fps": True}


def fit_model(targets, features):
    """Three fixed calibration rows. Enumerating 8 faces solves this small NNLS."""
    x = np.asarray([[1, features[p]["wgt_loads"] / 34, features[p]["mac"] / 112]
                    for p in CALIBRATION], dtype=float)
    y = np.asarray([targets[p] for p in CALIBRATION], dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(y).all() or (y <= 0).any():
        raise ValueError("invalid calibration value")
    singular = np.linalg.svd(x, compute_uv=False)
    tol = np.finfo(float).eps * max(x.shape) * singular[0]
    rank = int(sum(singular > tol))
    base = {"schema": SCHEMA, "calibration_programs": list(CALIBRATION),
            "validation_programs": list(VALIDATION), "design_matrix": x.tolist(),
            "singular_values": singular.tolist(), "rank": rank,
            "condition_number": float(singular[0] / singular[-1]) if rank == 3 else None,
            "calibration_targets_ms": y.tolist(), "threshold_absolute_relative_pct": 10,
            "interpretation": "local residency-direction interpolation; not organization generalization"}
    if rank != 3:
        return {**base, "status": "MODEL_UNIDENTIFIABLE"}
    solutions = []
    for count in range(4):
        for active in combinations(range(3), count):
            coefficient = np.zeros(3)
            if active:
                fitted = np.linalg.lstsq(x[:, active], y, rcond=None)[0]
                if (fitted < -1e-12).any():
                    continue
                coefficient[list(active)] = np.maximum(fitted, 0)
            residual = x @ coefficient - y
            solutions.append((float(residual @ residual), tuple(coefficient), residual))
    _, coef, residual = min(solutions, key=lambda item: (item[0], item[1]))
    return {**base, "status": "MODEL_SEALED", "coefficients_ms": list(coef),
            "active_zero_constraints": ["abc"[i] for i, v in enumerate(coef) if v == 0],
            "calibration_residual_ms": residual.tolist(),
            "calibration_fit_is_not_validation": True}


def validate_model(model, targets, features):
    if model["status"] != "MODEL_SEALED":
        return {"status": model["status"], "passed": None}
    rows = []
    a, b, c = model["coefficients_ms"]
    for name in VALIDATION:
        observed = targets[name]
        predicted = a + b * features[name]["wgt_loads"] / 34 + c * features[name]["mac"] / 112
        if not math.isfinite(observed) or observed <= 0:
            raise ValueError("missing validation target")
        error = predicted - observed
        rows.append({"program": name, "predicted_ms": predicted, "observed_ms": observed,
                     "signed_relative_error_pct": 100 * error / observed,
                     "absolute_error_ms": abs(error),
                     "absolute_relative_error_pct": 100 * abs(error) / observed})
    return {"status": "LOCAL_INTERPOLATION_EVALUATED", "rows": rows,
            "passed": all(row["absolute_relative_error_pct"] <= 10 for row in rows),
            "does_not_resolve_two_percent_differences": True}
