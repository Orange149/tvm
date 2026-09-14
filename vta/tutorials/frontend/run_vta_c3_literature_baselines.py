#!/usr/bin/env python3
"""Unified, auditable runner for the C3 literature-aligned tuning baselines.

This file owns policy decisions and information visibility.  Board/compiler work
is supplied by an immutable outcome ledger during replay and will be replaced by
the live executor in Node C/D.  A workload contract must contain identities and
label-free features only; target validity and latency are revealed after a policy
has paid for the corresponding phase.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path


SCHEMA = "c3_literature_baseline_runner_v1"
POLICIES = (
    "random",
    "stock_mode_aware_xgb",
    "rieber_hw_init_xgb",
    "ml2tuner_pva",
    "cheng_minimum_access",
    "ours_dma_multifidelity",
)
RIEBER_LEVELS = (
    "random_initialization",
    "neighbor_presampling",
    "balanced_e0",
    "balanced_e0_validity_bias",
)
CHENG_MODES = (
    "original",
    "input_stationary",
    "weight_resident_barrier",
    "input_weight_resident_barrier",
)
VISIBLE_KNOBS = (
    "tile_b",
    "tile_h",
    "tile_w",
    "tile_ci",
    "tile_co",
    "oc_nthread",
    "h_nthread",
)
HIDDEN_FEATURES = (
    "loop_count",
    "loop_extent_sum",
    "loop_extent_product_log2",
    "branch_count",
    "partial_tile_count",
    "allocation_bytes",
    "tensorize_count",
)
WORKLOAD_FEATURES = ("ci", "co", "height", "width", "kernel", "stride")
DMA_FEATURES = (
    "total_dma_bytes",
    "total_dma_calls",
    "padding_dma_calls",
    "submissions",
)
PERFORMANCE_KEYS = frozenset(
    (
        "latency",
        "latency_ms",
        "paired_ratio",
        "performance",
        "runtime_ms",
        "is_valid",
        "valid",
        "correct",
        "oracle",
    )
)
TERMINAL_PHASES = ("lower", "fsim", "compile", "fpga_correctness", "timing")


def canonical_bytes(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def digest_value(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")


def stable_rank(seed, workload_id, candidate_id, salt):
    return hashlib.sha256(
        "{}:{}:{}:{}".format(seed, workload_id, candidate_id, salt).encode("utf-8")
    ).hexdigest()


def _scan_forbidden_labels(value, path="$"):
    found = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key.lower() in PERFORMANCE_KEYS:
                found.append("{}.{}".format(path, key))
            found.extend(_scan_forbidden_labels(item, "{}.{}".format(path, key)))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_scan_forbidden_labels(item, "{}[{}]".format(path, index)))
    return found


def validate_workload_contract(contract):
    if contract.get("schema") != "c3_literature_workload_contract_v1":
        raise ValueError("unexpected workload contract schema")
    candidates = contract.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("workload contract requires candidates")
    forbidden = _scan_forbidden_labels(candidates)
    if forbidden:
        raise ValueError("future target label leaked into workload contract: {}".format(forbidden[0]))
    identifiers = [row.get("candidate_id") for row in candidates]
    if any(not isinstance(item, str) or len(item) != 64 for item in identifiers):
        raise ValueError("every candidate requires a SHA-256 identity")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("candidate identities must be unique")
    for row in candidates:
        if row.get("residence_mode") not in CHENG_MODES:
            raise ValueError("unsupported residence mode")
        if not isinstance(row.get("complete_config_entity"), dict):
            raise ValueError("complete ConfigEntity is required")
        if not isinstance(row.get("visible_features"), dict):
            raise ValueError("visible label-free features are required")
        if not isinstance(row.get("static_features"), dict):
            raise ValueError("static label-free features are required")
    return candidates


def _feature(row, key, default=0.0):
    if key in row.get("visible_features", {}):
        return float(row["visible_features"][key])
    if key in row.get("static_features", {}):
        return float(row["static_features"][key])
    if key in row.get("hidden_features", {}):
        return float(row["hidden_features"][key])
    return float(default)


def visible_vector(row):
    values = [_feature(row, name) for name in VISIBLE_KNOBS]
    values.extend(float(row["residence_mode"] == mode) for mode in CHENG_MODES)
    values.extend(float(row.get("workload_features", {}).get(name, 0.0)) for name in WORKLOAD_FEATURES)
    return values


def advanced_vector(row, include_dma=False):
    values = visible_vector(row)
    values.extend(_feature(row, name) for name in HIDDEN_FEATURES)
    if include_dma:
        values.extend(_feature(row, name) for name in DMA_FEATURES)
    return values


def knob_coordinate(row):
    return tuple(int(_feature(row, name)) for name in VISIBLE_KNOBS)


def manhattan(left, right):
    return sum(abs(a - b) for a, b in zip(knob_coordinate(left), knob_coordinate(right)))


def neighbor_map(candidates):
    """Return one-knob, one-discrete-level neighbours.

    Factor knobs are not uniformly spaced numerically (for example 1, 2, 4,
    7, 14).  Hardware-aware neighbourhood therefore means moving one position
    in that knob's observed discrete domain while keeping every other knob
    fixed; raw Manhattan distance one would silently miss 2->4 and 7->14.
    """

    result = {row["candidate_id"]: set() for row in candidates}
    groups = {}
    for row in candidates:
        groups.setdefault((row.get("workload_id"), row["residence_mode"]), []).append(row)
    for rows in groups.values():
        domains = {
            name: sorted({int(_feature(row, name)) for row in rows})
            for name in VISIBLE_KNOBS
        }
        by_coordinate = {knob_coordinate(row): row["candidate_id"] for row in rows}
        if len(by_coordinate) != len(rows):
            raise ValueError("neighbourhood requires unique visible knob coordinates")
        for row in rows:
            coordinate = list(knob_coordinate(row))
            for knob_index, name in enumerate(VISIBLE_KNOBS):
                values = domains[name]
                value_index = values.index(coordinate[knob_index])
                for adjacent_index in (value_index - 1, value_index + 1):
                    if not 0 <= adjacent_index < len(values):
                        continue
                    neighbour = list(coordinate)
                    neighbour[knob_index] = values[adjacent_index]
                    candidate_id = by_coordinate.get(tuple(neighbour))
                    if candidate_id is not None:
                        result[row["candidate_id"]].add(candidate_id)
    return result


def rieber_presampling_order(candidates, seed, validity_callback, limit=1000, parallel=8):
    """Algorithm-1-style locality presampling with sequential feedback.

    ``validity_callback`` is invoked only after a candidate has been placed in the
    current compiler batch.  Consequently the function cannot inspect future
    validity labels while constructing that batch.
    """

    by_id = {row["candidate_id"]: row for row in candidates}
    neighbours = neighbor_map(candidates)
    unseen = set(by_id)
    rng = random.Random(seed)
    pending = rng.sample(sorted(unseen), min(parallel, len(unseen)))
    order = []
    while pending and len(order) < min(limit, len(candidates)):
        current = [item for item in pending if item in unseen]
        current = current[: min(parallel, limit - len(order))]
        if not current:
            break
        frontier = set()
        for candidate_id in current:
            order.append(candidate_id)
            unseen.remove(candidate_id)
            is_valid = bool(validity_callback(candidate_id))
            if is_valid:
                frontier.update(neighbours[candidate_id] & unseen)
            elif unseen:
                frontier.add(rng.choice(sorted(unseen)))
        # A neighbour inserted while processing an early item in this parallel
        # batch may itself be processed by a later item in the same batch.
        # Re-filter after the entire batch so a now-seen ID cannot consume one
        # slot in the next pending wave.
        frontier.intersection_update(unseen)
        count = min(parallel, limit - len(order), len(unseen))
        selected = rng.sample(sorted(frontier), min(count, len(frontier)))
        if len(selected) < count:
            selected.extend(rng.sample(sorted(unseen - set(selected)), count - len(selected)))
        pending = selected
    return order


def max_min_subset(rows, count):
    """Deterministic farthest-first E0 diversity selection."""

    rows = sorted(rows, key=lambda item: item["candidate_id"])
    if count >= len(rows):
        return rows
    selected = [rows[0]]
    remaining = rows[1:]
    while len(selected) < count:
        best = max(
            remaining,
            key=lambda row: (
                min(manhattan(row, other) for other in selected),
                sum(manhattan(row, other) for other in selected),
                row["candidate_id"],
            ),
        )
        selected.append(best)
        remaining.remove(best)
    return selected


def balanced_e0(compiler_rows, per_class=25):
    valid = [row for row in compiler_rows if row["is_valid"]]
    invalid = [row for row in compiler_rows if not row["is_valid"]]
    selected = max_min_subset(valid, min(per_class, len(valid)))
    selected += max_min_subset(invalid, min(per_class, len(invalid)))
    return sorted(selected, key=lambda row: row["candidate_id"])


def _xgb_regression(train_x, train_y, predict_x, seed):
    if len(train_y) < 4 or not predict_x:
        return [statistics.mean(train_y) if train_y else 0.0] * len(predict_x)
    import numpy as np
    import xgboost as xgb

    model = xgb.XGBRegressor(
        n_estimators=64,
        max_depth=3,
        learning_rate=0.1,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=seed,
        n_jobs=1,
        verbosity=0,
    )
    model.fit(np.asarray(train_x), np.asarray(train_y))
    return [float(item) for item in model.predict(np.asarray(predict_x))]


def _xgb_validity(train_x, train_y, predict_x, seed):
    if len(set(train_y)) < 2 or len(train_y) < 4 or not predict_x:
        prior = statistics.mean(train_y) if train_y else 0.5
        return [float(prior)] * len(predict_x)
    import numpy as np
    import xgboost as xgb

    model = xgb.XGBClassifier(
        n_estimators=64,
        max_depth=3,
        learning_rate=0.1,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=seed,
        n_jobs=1,
        verbosity=0,
        use_label_encoder=False,
    )
    model.fit(np.asarray(train_x), np.asarray(train_y))
    return [float(item) for item in model.predict_proba(np.asarray(predict_x))[:, 1]]


def performance_training_rows(development, observed):
    rows = []
    for item in list(development) + list(observed):
        if item.get("is_valid") and item.get("latency_ms") is not None:
            rows.append(item)
    return rows


def validity_training_rows(development, observed):
    return [item for item in list(development) + list(observed) if "is_valid" in item]


def select_model_rank(candidates, development, observed, seed, vector_fn, validity=False):
    train = validity_training_rows(development, observed) if validity else performance_training_rows(
        development, observed
    )
    train_x = [vector_fn(row) for row in train]
    predict_x = [vector_fn(row) for row in candidates]
    if validity:
        predictions = _xgb_validity(train_x, [int(row["is_valid"]) for row in train], predict_x, seed)
        return sorted(zip(candidates, predictions), key=lambda item: (-item[1], item[0]["candidate_id"]))
    predictions = _xgb_regression(
        train_x, [float(row["latency_ms"]) for row in train], predict_x, seed
    )
    return sorted(zip(candidates, predictions), key=lambda item: (item[1], item[0]["candidate_id"]))


def cheng_applicability(row):
    mode = row["residence_mode"]
    # Cheng's resource fallback constrains the residency mechanism, not the
    # existence of the original schedule.  A crude accumulator bound must not
    # pre-reject original: real lowering remains the authoritative legality
    # decision for that tile.
    if mode in ("input_stationary", "input_weight_resident_barrier"):
        if not bool(row.get("applicability", {}).get("accumulator_fits_sram", True)):
            return False, "accumulator_exceeds_sram"
    if mode in ("weight_resident_barrier", "input_weight_resident_barrier"):
        if not bool(row.get("applicability", {}).get("full_weight_fits_sram", False)):
            return False, "full_weight_exceeds_sram"
    return True, None


def cheng_minimum_access_order(candidates):
    """Choose one applicable minimum-access mode per same-tile family."""

    groups = {}
    for row in candidates:
        groups.setdefault(row["family_id"], []).append(row)
    selected = []
    fallbacks = []
    for family_id, group in sorted(groups.items()):
        originals = [row for row in group if row["residence_mode"] == "original"]
        if len(originals) != 1:
            raise ValueError("each Cheng family requires exactly one original")
        applicable = []
        for row in group:
            ok, reason = cheng_applicability(row)
            if ok:
                applicable.append(row)
            else:
                fallbacks.append(
                    {
                        "family_id": family_id,
                        "candidate_id": row["candidate_id"],
                        "status": "not_applicable",
                        "reason": reason,
                        "fallback_candidate_id": originals[0]["candidate_id"],
                    }
                )
        chosen = min(
            applicable,
            key=lambda row: (
                _feature(row, "total_dma_bytes", math.inf),
                _feature(row, "total_dma_calls", math.inf),
                row["candidate_id"],
            ),
        )
        selected.append(chosen)
    return selected, fallbacks


class OutcomeLedger:
    """One-way target outcome reveal API; policies never receive the raw mapping."""

    def __init__(self, payload):
        if payload.get("schema") != "c3_literature_outcome_ledger_v1":
            raise ValueError("unexpected outcome ledger schema")
        self._outcomes = payload.get("outcomes", {})
        self.pool_complete = bool(payload.get("pool_complete", False))
        self.session_status = payload.get("session_status", "unknown")
        self.revealed = []

    def reveal(self, candidate_id, phase):
        if phase not in TERMINAL_PHASES:
            raise ValueError("unknown execution phase")
        if candidate_id not in self._outcomes or phase not in self._outcomes[candidate_id]:
            raise KeyError("missing {} outcome for {}".format(phase, candidate_id))
        value = json.loads(json.dumps(self._outcomes[candidate_id][phase]))
        self.revealed.append((candidate_id, phase))
        return value

    def oracle_after_completion(self, candidate_ids):
        if not self.pool_complete:
            raise RuntimeError("oracle cannot be connected before complete-pool execution")
        values = []
        for candidate_id in candidate_ids:
            timing = self._outcomes.get(candidate_id, {}).get("timing", {})
            correct = self._outcomes.get(candidate_id, {}).get("fpga_correctness", {})
            if timing.get("status") == "ok" and correct.get("status") == "ok":
                values.append((float(timing["latency_ms"]), candidate_id))
        if not values:
            raise RuntimeError("complete pool has no correct timed candidate")
        latency, candidate_id = min(values)
        return {"candidate_id": candidate_id, "latency_ms": latency}


def outcome_to_observation(candidate, phase_rows):
    lower = phase_rows.get("lower", {})
    fsim = phase_rows.get("fsim", {})
    compile_row = phase_rows.get("compile", {})
    correctness = phase_rows.get("fpga_correctness", {})
    timing = phase_rows.get("timing", {})
    is_valid = all(
        row.get("status") == "ok" for row in (lower, fsim, compile_row, correctness)
    )
    merged = json.loads(json.dumps(candidate))
    merged["is_valid"] = is_valid
    if lower.get("hidden_features"):
        merged["hidden_features"] = lower["hidden_features"]
    if is_valid and timing.get("status") == "ok":
        merged["latency_ms"] = float(timing["latency_ms"])
    return merged


def compile_stage_observation(candidate, phase_rows):
    """Return a censored observation after lowering/cross-compilation only.

    A compile-success candidate is *not* labelled globally valid here: FSim,
    device execution and numerical correctness remain unknown.  A compile
    failure is terminal and may safely contribute a negative validity label.
    This distinction prevents Model V from learning optimistic labels for the
    2N candidates that Model A does not promote to FPGA profiling.
    """

    lower = phase_rows.get("lower", {})
    compile_row = phase_rows.get("compile", {})
    compile_valid = (
        lower.get("status") == "ok" and compile_row.get("status") == "ok"
    )
    merged = json.loads(json.dumps(candidate))
    if lower.get("hidden_features"):
        merged["hidden_features"] = lower["hidden_features"]
    merged["compile_valid"] = compile_valid
    merged["final_validity_observed"] = not compile_valid
    if not compile_valid:
        merged["is_valid"] = False
    return merged


def execute_candidate(candidate, ledger, timeline, cumulative_wall, phases=TERMINAL_PHASES):
    phase_rows = {}
    terminal = False
    for phase in phases:
        if terminal:
            break
        result = ledger.reveal(candidate["candidate_id"], phase)
        wall = float(result.get("wall_seconds", 0.0))
        if wall < 0 or not math.isfinite(wall):
            raise ValueError("invalid wall time")
        cumulative_wall += wall
        event = {
            "event_index": len(timeline) + 1,
            "candidate_id": candidate["candidate_id"],
            "phase": phase,
            "status": result.get("status"),
            "phase_wall_seconds": wall,
            "cumulative_wall_seconds": cumulative_wall,
        }
        for key in (
            "logical_dma_bytes",
            "logical_dma_calls",
            "fpga_kernel_invocations",
            "compiler_attempts",
        ):
            if key in result:
                event[key] = result[key]
        timeline.append(event)
        phase_rows[phase] = result
        if result.get("status") not in ("ok",):
            terminal = True
    return phase_rows, cumulative_wall


def choose_sequential(policy, remaining, development, observed, seed, dispatch_index):
    remaining = list(remaining)
    workload_id = remaining[0]["workload_id"]
    if policy == "random":
        return min(
            remaining,
            key=lambda row: stable_rank(seed, workload_id, row["candidate_id"], "random"),
        )
    if policy == "ours_dma_multifidelity":
        return min(
            remaining,
            key=lambda row: (
                _feature(row, "total_dma_bytes", math.inf),
                _feature(row, "total_dma_calls", math.inf),
                row["candidate_id"],
            ),
        )
    if policy == "stock_mode_aware_xgb":
        if dispatch_index < 4:
            return min(
                remaining,
                key=lambda row: stable_rank(seed, workload_id, row["candidate_id"], "xgb-warmup"),
            )
        return select_model_rank(
            remaining, development, observed, seed, visible_vector, validity=False
        )[0][0]
    raise ValueError("policy is not sequential")


def replay_standard(
    policy,
    candidates,
    budget,
    seed,
    development,
    ledger,
    rieber_level="balanced_e0_validity_bias",
):
    timeline = []
    results = []
    observed = []
    cumulative_wall = 0.0
    remaining = list(candidates)
    fallback_rows = []
    diagnostic = {}
    rieber_initial_performance_count = 0
    if policy == "cheng_minimum_access":
        remaining, fallback_rows = cheng_minimum_access_order(candidates)
    if policy == "rieber_hw_init_xgb":
        originals = [row for row in candidates if row["residence_mode"] == "original"]
        by_id = {row["candidate_id"]: row for row in originals}
        compiler_rows = []

        def reveal_compiler_validity(candidate_id):
            nonlocal cumulative_wall
            row = by_id[candidate_id]
            phase_rows, cumulative_wall = execute_candidate(
                row, ledger, timeline, cumulative_wall, phases=("lower",)
            )
            compiler_rows.append(
                {**row, "is_valid": phase_rows["lower"].get("status") == "ok"}
            )
            return compiler_rows[-1]["is_valid"]

        if rieber_level not in RIEBER_LEVELS:
            raise ValueError("unknown Rieber ablation level")
        if rieber_level == "random_initialization":
            order = sorted(
                (row["candidate_id"] for row in originals),
                key=lambda candidate_id: stable_rank(
                    seed, originals[0]["workload_id"], candidate_id, "rieber-random-e0"
                ),
            )[: min(50, len(originals))]
            for candidate_id in order:
                reveal_compiler_validity(candidate_id)
            initial = list(compiler_rows)
        else:
            order = rieber_presampling_order(
                originals,
                seed,
                reveal_compiler_validity,
                limit=min(1000, len(originals)),
            )
            if len(order) != min(1000, len(originals)):
                raise AssertionError("Rieber presampling did not consume the frozen compiler budget")
            if rieber_level == "neighbor_presampling":
                initial = compiler_rows[: min(50, len(compiler_rows))]
            else:
                initial = balanced_e0(compiler_rows, 25)
        initial_ids = {row["candidate_id"] for row in initial}
        valid_initial = [row for row in initial if row["is_valid"]]
        rieber_initial_performance_count = len(valid_initial)
        outside = [row for row in originals if row["candidate_id"] not in initial_ids]
        if rieber_level == "balanced_e0_validity_bias":
            outside = [
                row
                for row, _ in select_model_rank(
                    outside,
                    development,
                    compiler_rows,
                    seed,
                    visible_vector,
                    validity=True,
                )
            ]
        else:
            outside = sorted(
                outside,
                key=lambda row: stable_rank(
                    seed, row["workload_id"], row["candidate_id"], "rieber-post-e0"
                ),
            )
        remaining = valid_initial + outside
        first_25_valid_calls = None
        count = 0
        for index, row in enumerate(compiler_rows, 1):
            count += int(row["is_valid"])
            if count == 25:
                first_25_valid_calls = index
                break
        diagnostic = {
            "rieber_level": rieber_level,
            "presampling_compiler_calls": len(compiler_rows),
            "presampling_valid": sum(row["is_valid"] for row in compiler_rows),
            "presampling_invalid": sum(not row["is_valid"] for row in compiler_rows),
            "presampling_valid_yield": (
                sum(row["is_valid"] for row in compiler_rows) / len(compiler_rows)
                if compiler_rows
                else None
            ),
            "compiler_calls_to_first_25_valid": first_25_valid_calls,
            "e0_size": len(initial),
            "e0_valid": sum(row["is_valid"] for row in initial),
            "e0_invalid": sum(not row["is_valid"] for row in initial),
        }
    seen = set()
    while remaining and len(results) < budget:
        if policy in ("random", "stock_mode_aware_xgb", "ours_dma_multifidelity"):
            candidate = choose_sequential(
                policy, remaining, development, observed, seed, len(results)
            )
        elif policy == "rieber_hw_init_xgb" and len(results) >= rieber_initial_performance_count:
            candidate = select_model_rank(
                remaining, development, observed, seed, visible_vector, validity=False
            )[0][0]
        else:
            candidate = remaining[0]
        remaining = [row for row in remaining if row["candidate_id"] != candidate["candidate_id"]]
        if candidate["candidate_id"] in seen:
            continue
        seen.add(candidate["candidate_id"])
        phases = TERMINAL_PHASES
        if policy == "rieber_hw_init_xgb" and any(
            event["candidate_id"] == candidate["candidate_id"] and event["phase"] == "lower"
            for event in timeline
        ):
            phases = TERMINAL_PHASES[1:]
        phase_rows, cumulative_wall = execute_candidate(
            candidate, ledger, timeline, cumulative_wall, phases=phases
        )
        # Restore the already revealed lowering validity for Rieber E0 entries.
        if phases != TERMINAL_PHASES:
            prior = next(row for row in compiler_rows if row["candidate_id"] == candidate["candidate_id"])
            phase_rows["lower"] = {"status": "ok" if prior["is_valid"] else "invalid"}
        observation = outcome_to_observation(candidate, phase_rows)
        observed.append(observation)
        results.append(
            {
                "dispatch": len(results) + 1,
                "candidate_id": candidate["candidate_id"],
                "is_valid": observation["is_valid"],
                "latency_ms": observation.get("latency_ms"),
                "cumulative_wall_seconds": cumulative_wall,
            }
        )
    return timeline, results, fallback_rows, diagnostic


def replay_ml2tuner(candidates, budget, seed, development, ledger, include_dma=False):
    """Repeated P/V -> compile 2N -> A -> profile N, with N=10 and alpha=1."""

    timeline = []
    cumulative_wall = 0.0
    results = []
    observed = []
    remaining = list(candidates)
    waves = []
    total_compiled = 0
    total_compile_invalid = 0
    total_profiled_invalid = 0
    while remaining and len(results) < budget:
        wave_index = len(waves)
        n_measure = min(10, budget - len(results))
        n_compile = min(2 * n_measure, len(remaining))
        p_rank = select_model_rank(
            remaining, development, observed, seed + wave_index, visible_vector, validity=False
        )
        v_rank = select_model_rank(
            remaining, development, observed, seed + wave_index, visible_vector, validity=True
        )
        p_position = {row["candidate_id"]: index for index, (row, _) in enumerate(p_rank)}
        v_position = {row["candidate_id"]: index for index, (row, _) in enumerate(v_rank)}
        compile_wave = sorted(
            remaining,
            key=lambda row: (
                p_position[row["candidate_id"]] + v_position[row["candidate_id"]],
                row["candidate_id"],
            ),
        )[:n_compile]
        compile_ids = {row["candidate_id"] for row in compile_wave}
        remaining = [row for row in remaining if row["candidate_id"] not in compile_ids]
        compiled = []
        invalid = []
        for candidate in compile_wave:
            phase_rows, cumulative_wall = execute_candidate(
                candidate, ledger, timeline, cumulative_wall, phases=("lower", "compile")
            )
            is_valid = all(row.get("status") == "ok" for row in phase_rows.values())
            merged = compile_stage_observation(candidate, phase_rows)
            if is_valid:
                compiled.append(merged)
            else:
                invalid.append(merged)
                observed.append(merged)
        vector_fn = lambda row: advanced_vector(row, include_dma=include_dma)
        advanced = select_model_rank(
            compiled, development, observed, seed + wave_index, vector_fn, validity=False
        )
        promoted = [row for row, _ in advanced[:n_measure]]
        promoted_ids = {row["candidate_id"] for row in promoted}
        # Model V may learn that a lowered-but-not-promoted row is valid, but
        # Model P/A must not receive a latency that was never measured.
        observed.extend(row for row in compiled if row["candidate_id"] not in promoted_ids)
        for candidate in promoted:
            phase_rows, cumulative_wall = execute_candidate(
                candidate,
                ledger,
                timeline,
                cumulative_wall,
                phases=("fsim", "fpga_correctness", "timing"),
            )
            phase_rows["lower"] = {
                "status": "ok",
                "hidden_features": candidate.get("hidden_features", {}),
            }
            phase_rows["compile"] = {"status": "ok"}
            observation = outcome_to_observation(candidate, phase_rows)
            observation["final_validity_observed"] = True
            total_profiled_invalid += int(not observation["is_valid"])
            observed.append(observation)
            results.append(
                {
                    "dispatch": len(results) + 1,
                    "candidate_id": candidate["candidate_id"],
                    "is_valid": observation["is_valid"],
                    "latency_ms": observation.get("latency_ms"),
                    "cumulative_wall_seconds": cumulative_wall,
                    "ml2_wave": wave_index,
                }
            )
        total_compiled += len(compile_wave)
        total_compile_invalid += len(invalid)
        waves.append(
            {
                "wave": wave_index,
                "compile_candidates": len(compile_wave),
                "compile_valid": len(compiled),
                "compile_invalid": len(invalid),
                "model_a_promoted": len(promoted),
            }
        )
    return timeline, results, {
        "model_p_candidates": total_compiled,
        "model_v_candidates": total_compiled,
        "compiled_candidates": total_compiled,
        "model_a_promoted": len(results),
        "invalid_before_profiling": total_compile_invalid,
        "invalid_during_profiling": total_profiled_invalid,
        "invalid_profiling_ratio": (
            total_profiled_invalid / len(results) if results else None
        ),
        "censored_compile_valid_not_profiled": sum(
            bool(row.get("compile_valid"))
            and not bool(row.get("final_validity_observed"))
            for row in observed
        ),
        "model_a_includes_dma": bool(include_dma),
        "waves": waves,
    }


def summarize_cost(timeline, results):
    phase_wall_seconds = {
        phase: sum(
            float(row.get("phase_wall_seconds", 0.0))
            for row in timeline
            if row["phase"] == phase
        )
        for phase in TERMINAL_PHASES
    }
    return {
        "gross_candidates": len({row["candidate_id"] for row in timeline}),
        "compiler_attempts": sum(
            int(row.get("compiler_attempts", row["phase"] in ("lower", "compile")))
            for row in timeline
        ),
        "fpga_dispatches": sum(row["phase"] == "fpga_correctness" for row in timeline),
        "fpga_kernel_invocations": sum(int(row.get("fpga_kernel_invocations", 0)) for row in timeline),
        "logical_dma_bytes": sum(int(row.get("logical_dma_bytes", 0)) for row in timeline),
        "logical_dma_calls": sum(int(row.get("logical_dma_calls", 0)) for row in timeline),
        "wall_seconds": timeline[-1]["cumulative_wall_seconds"] if timeline else 0.0,
        "phase_wall_seconds": phase_wall_seconds,
        "qualification_wall_seconds": phase_wall_seconds["lower"]
        + phase_wall_seconds["fsim"],
        "cross_compile_wall_seconds": phase_wall_seconds["compile"],
        "correctness_wall_seconds": phase_wall_seconds["fpga_correctness"],
        "timing_wall_seconds": phase_wall_seconds["timing"],
        "invalid_candidates": sum(not row["is_valid"] for row in results),
        "timed_valid_candidates": sum(row.get("latency_ms") is not None for row in results),
    }


def freeze_contract(args, workload, candidates, output):
    if args.policy == "rieber_hw_init_xgb":
        originals = [row for row in candidates if row["residence_mode"] == "original"]
        if len(originals) != len(candidates) or workload.get("role") != "complete_original_space_for_hw_aware":
            raise ValueError("HW-Aware baseline requires the declared complete original ConfigSpace")
        declared = workload.get("complete_config_space_count")
        if declared != len(originals):
            raise ValueError("HW-Aware complete ConfigSpace cardinality contract mismatch")
    if args.policy == "ml2tuner_pva" and (args.ml2_n != 10 or args.ml2_alpha != 1):
        raise ValueError("frozen ML2Tuner baseline requires N=10 and alpha=1")
    if args.policy == "cheng_minimum_access":
        modes = {}
        for row in candidates:
            modes.setdefault(row["family_id"], set()).add(row["residence_mode"])
        if any(value != set(CHENG_MODES) for value in modes.values()):
            raise ValueError("Cheng baseline requires same-tile four-mode families")
    contract = {
        "schema": SCHEMA,
        "status": "frozen_local_only_no_target_labels_no_board",
        "policy": args.policy,
        "phase": "freeze",
        "workload_id": workload["workload_id"],
        "workload_contract_sha256": sha256_file(args.workload_contract),
        "candidate_universe_sha256": digest_value(candidates),
        "candidate_count": len(candidates),
        "candidate_ids": [row["candidate_id"] for row in candidates],
        "candidate_budget": args.candidate_budget,
        "seed": args.seed,
        "policy_parameters": {
            "rieber": {
                "ablation_level": args.rieber_level,
                "e0": 50,
                "per_class": 25,
                "presampling": min(1000, len(candidates)),
            },
            "ml2tuner": {"N": args.ml2_n, "alpha": args.ml2_alpha},
            "cheng_modes": list(CHENG_MODES),
        },
        "identity_fields": [
            "model_hash",
            "workload_id",
            "complete_config_entity",
            "residence_mode",
            "lowered_tir_sha256",
            "hardware_fingerprint",
        ],
        "target_labels_present_in_contract": False,
        "board_contacted": False,
        "runner_source_sha256": sha256_file(Path(__file__).resolve()),
    }
    write_json(output / "contract.json", contract)
    (output / "timeline.jsonl").write_text("", encoding="utf-8")
    (output / "results.jsonl").write_text("", encoding="utf-8")
    write_json(
        output / "summary.json",
        {
            "status": "frozen",
            "candidate_count": len(candidates),
            "performance_labels_read": False,
            "board_contacted": False,
        },
    )


def replay_contract(args, workload, candidates, output):
    if not args.outcome_ledger:
        raise ValueError("--outcome-ledger is required for replay")
    ledger_path = Path(args.outcome_ledger).resolve()
    ledger = OutcomeLedger(read_json(ledger_path))
    if ledger.session_status != "complete":
        raise RuntimeError("interrupted/RPC-failed session cannot be replayed or spliced")
    development = read_json(args.development_data).get("rows", []) if args.development_data else []
    if args.policy == "ml2tuner_pva":
        timeline, results, diagnostic = replay_ml2tuner(
            candidates,
            args.candidate_budget,
            args.seed,
            development,
            ledger,
            include_dma=args.ml2_include_dma,
        )
        fallback = []
    else:
        timeline, results, fallback, diagnostic = replay_standard(
            args.policy,
            candidates,
            args.candidate_budget,
            args.seed,
            development,
            ledger,
            rieber_level=args.rieber_level,
        )
    # Pool oracle is connected only after selection and only for a declared complete ledger.
    oracle = ledger.oracle_after_completion([row["candidate_id"] for row in candidates])
    for row in timeline:
        append_jsonl(output / "timeline.jsonl", row)
    for row in results:
        append_jsonl(output / "results.jsonl", row)
    contract = {
        "schema": SCHEMA,
        "status": "complete_replay",
        "policy": args.policy,
        "phase": "replay",
        "workload_id": workload["workload_id"],
        "workload_contract_sha256": sha256_file(args.workload_contract),
        "outcome_ledger_sha256": sha256_file(ledger_path),
        "development_data_sha256": sha256_file(args.development_data)
        if args.development_data
        else None,
        "candidate_budget": args.candidate_budget,
        "seed": args.seed,
        "target_label_access": list(ledger.revealed),
        "oracle_connected_after_selection": True,
        "board_contacted": False,
        "runner_source_sha256": sha256_file(Path(__file__).resolve()),
    }
    write_json(output / "contract.json", contract)
    best = min(
        (row for row in results if row.get("latency_ms") is not None),
        key=lambda row: row["latency_ms"],
        default=None,
    )
    write_json(
        output / "summary.json",
        {
            "status": "complete_replay",
            "policy": args.policy,
            "cost": summarize_cost(timeline, results),
            "best_observed": best,
            "pool_oracle": oracle,
            "fallbacks": fallback,
            "diagnostic": diagnostic,
            "claim_boundary": "replay validates policy semantics and accounting; it is not a new board result",
        },
    )


def finish_output(output):
    artifacts = {
        str(path.relative_to(output)): sha256_file(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    write_json(output / "artifact_hashes.json", {"artifacts": artifacts})


def run(args):
    if args.policy not in POLICIES:
        raise ValueError("unknown policy")
    if args.candidate_budget <= 0:
        raise ValueError("candidate budget must be positive")
    workload_path = Path(args.workload_contract).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite {}".format(output))
    # Never allow an output directory to be nested in an evidence input.
    for source in (workload_path, Path(args.outcome_ledger).resolve() if args.outcome_ledger else None):
        if source is not None and (source == output or source in output.parents):
            raise ValueError("output cannot overwrite or nest inside an evidence input")
    workload = read_json(workload_path)
    candidates = validate_workload_contract(workload)
    output.mkdir(parents=True)
    started = time.perf_counter()
    try:
        if args.phase == "freeze":
            freeze_contract(args, workload, candidates, output)
        elif args.phase == "replay":
            replay_contract(args, workload, candidates, output)
        else:
            raise ValueError("unsupported phase")
        (output / "command.txt").write_text(
            " ".join([sys.executable, str(Path(__file__).resolve())] + sys.argv[1:]) + "\n",
            encoding="utf-8",
        )
        finish_output(output)
    except BaseException as err:
        write_json(
            output / "invalid_session.json",
            {
                "status": "invalid_complete_session",
                "error_type": type(err).__name__,
                "message": str(err),
                "elapsed_seconds": time.perf_counter() - started,
                "cannot_splice_with_later_session": True,
            },
        )
        raise
    print(json.dumps(read_json(output / "summary.json"), indent=2, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True, choices=POLICIES)
    parser.add_argument("--workload-contract", required=True)
    parser.add_argument("--candidate-budget", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--phase", required=True, choices=("freeze", "replay"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--outcome-ledger")
    parser.add_argument("--development-data")
    parser.add_argument("--ml2-n", type=int, default=10)
    parser.add_argument("--ml2-alpha", type=int, default=1)
    parser.add_argument("--ml2-include-dma", action="store_true")
    parser.add_argument("--rieber-level", choices=RIEBER_LEVELS, default="balanced_e0_validity_bias")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
