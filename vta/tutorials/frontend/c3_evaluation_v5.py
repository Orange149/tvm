#!/usr/bin/env python3
"""Pure protocol helpers for the frozen C3 selector-v4/v5 evaluation.

This module deliberately contains no TVM or board imports.  The freeze,
operator runner, analyzer, and graph runner share these routines so that
candidate membership, budgets, call schedules, tie-breaking, and campaign
state cannot drift between phases.
"""

from __future__ import annotations

import hashlib
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


SCHEMA = "c3_selector_v4_evaluation_v5"
SEEDS = (0, 20250901, 20260910)
GRAPH_SEED_SEQUENCE = (0, 20250901, 20260910, 0, 20250901,
                       20260910, 0, 20250901, 20260910, 0)
METHODS = ("selector_v4", "bytes_only", "calls_only", "command_only", "fixed_random")
BUDGETS = (1, 2, 3)
REFERENCE_ROUNDS = 6
CONFIRMATION_ROUNDS = 6
RANDOM_REPLAYS = 1000
RANDOM_SEED = 20260926
STATES = (
    "PRE_LATENCY_FROZEN",
    "SELECTION_RUNNING",
    "WINNERS_SEALED",
    "CONFIRMATION",
    "REFERENCE",
    "COMPLETE",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def write_json(path: Path, value: Any) -> None:
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, value: Any) -> None:
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")
        stream.flush()


def verify_flat(directory: Path) -> str:
    directory = Path(directory).resolve()
    ledger = read_json(directory / "artifact_hashes.json")
    for name, expected in ledger.get("artifacts", {}).items():
        path = directory / name
        if isinstance(expected, str) and (not path.is_file() or sha256_file(path) != expected):
            raise ValueError("artifact hash mismatch: {}".format(path))
    return sha256_file(directory / "artifact_hashes.json")


def seal_flat(output: Path, source_paths: Iterable[Path] = ()) -> None:
    output = Path(output)
    write_json(output / "artifact_hashes.json", {
        "artifacts": {
            path.name: sha256_file(path)
            for path in sorted(output.iterdir())
            if path.is_file() and path.name != "artifact_hashes.json"
        },
        "sources": {str(Path(path).resolve()): sha256_file(path) for path in source_paths},
    })


def _rotate(values: Sequence[str], offset: int) -> List[str]:
    values = list(values)
    if not values:
        return []
    offset %= len(values)
    return values[offset:] + values[:offset]


def _program_rows(candidates: Sequence[Mapping[str, Any]]) -> Dict[str, Mapping[str, Any]]:
    rows: Dict[str, Mapping[str, Any]] = {}
    for row in candidates:
        program_id = str(row["program_id"])
        if program_id in rows:
            raise ValueError("duplicate program_id: {}".format(program_id))
        rows[program_id] = row
    return rows


def freeze_task_policies(
    candidates: Sequence[Mapping[str, Any]],
    source_policies: Mapping[str, Mapping[str, Any]],
    correctness_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Apply the completed FPGA gate and freeze v5 pools and static orders."""
    by_program = _program_rows(candidates)
    passed = {str(row["program_id"]) for row in correctness_rows if row["status"] == "passed"}
    rejected = {str(row["program_id"]) for row in correctness_rows if row["status"] != "passed"}
    if passed & rejected:
        raise ValueError("a correctness identity is both passed and rejected")
    if passed | rejected != set(by_program):
        raise ValueError("correctness identities do not cover the declared candidate pool")

    result: Dict[str, Dict[str, Any]] = {}
    for task_id, source in sorted(source_policies.items()):
        incumbent = str(source["incumbent_program_id"])
        declared = list(source["complete_isolated_reference_program_pool"])
        production_declared = list(source["production_program_pool"])
        production = [program for program in production_declared if program in passed]
        invalid = [program for program in declared if program in rejected]
        legacy = [program for program in declared if program in passed and program not in production]
        if incumbent not in production:
            raise ValueError("task lacks a FPGA-valid incumbent: {}".format(task_id))
        if len(declared) != 15 or len(production) != 8 or len(legacy) != 6 or len(invalid) != 1:
            raise ValueError("unexpected v5 pool cardinality for {}".format(task_id))
        if set(production) & set(legacy) or set(production) & set(invalid) or set(legacy) & set(invalid):
            raise ValueError("v5 pools overlap for {}".format(task_id))

        new_programs = [program for program in production if program != incumbent]
        selector_order = [program for program in source["selector_v4_order"]
                          if program in new_programs]
        if set(selector_order) != set(new_programs):
            raise ValueError("selector-v4 does not order every valid production candidate")

        def metric_order(fields: Sequence[str]) -> List[str]:
            def key(program_id: str) -> Tuple[Any, ...]:
                metrics = by_program[program_id]["timing_freeze_metrics"]
                return tuple(int(metrics[field]) for field in fields) + (program_id,)
            return sorted(new_programs, key=key)

        orders = {
            "selector_v4": selector_order,
            "bytes_only": metric_order(("total_request_bytes",)),
            "calls_only": metric_order(("total_request_calls",)),
            "command_only": metric_order((
                "submissions", "uop_load_bytes", "instruction_bytes",
                "total_request_bytes", "total_request_calls",
            )),
        }
        task_seed = RANDOM_SEED ^ int(hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:16], 16)
        rng = random.Random(task_seed)
        random_orders = []
        for _ in range(RANDOM_REPLAYS):
            order = list(new_programs)
            rng.shuffle(order)
            random_orders.append(order)
        orders["fixed_random"] = random_orders[0]

        dispositions = source["production_gate"]["candidate_dispositions"]
        legacy_contracts = []
        for program_id in legacy:
            candidate = by_program[program_id]
            disposition = dispositions[candidate["candidate_id"]]
            legacy_contracts.append({
                "program_id": program_id,
                "candidate_id": candidate["candidate_id"],
                "residence_mode": candidate["residence_mode"],
                "production_rejection_reason": disposition["reason"],
                "allowed_predecessor_state": "only_after_v5_empty_sync_boundary",
                "pre_call_boundary": "un-timed VTASynchronize on the experiment runtime queue",
                "timed_boundary": "operator main plus post-call VTASynchronize",
                "post_call_boundary": "included_in_timed_wrapper",
                "warmup_policy": "one wrapped untimed call for each frozen seed before reference",
                "certificate_status": "pending_v5_transition_audit",
            })

        result[task_id] = {
            "execution_task_id": task_id,
            "workload_id": source["workload_id"],
            "incumbent_program_id": incumbent,
            "declared_program_pool": declared,
            "production_program_pool": production,
            "legacy_controlled_program_pool": legacy,
            "fpga_invalid_programs": invalid,
            "orders": orders,
            "budget_prefixes": {
                method: {"B{}".format(budget): order[:budget] for budget in BUDGETS}
                for method, order in orders.items()
            },
            "random_replay_orders": random_orders,
            "legacy_boundary_contracts": legacy_contracts,
            "invalid_budget_policy": (
                "known FPGA-invalid identities consume qualification cost only and are absent "
                "from post-qualification B1/B2/B3"
            ),
        }
    return result


def qualification_schedule(policies: Mapping[str, Mapping[str, Any]], pool: str) -> List[Dict[str, Any]]:
    if pool not in ("production", "legacy"):
        raise ValueError("unknown qualification pool")
    field = "production_program_pool" if pool == "production" else "legacy_controlled_program_pool"
    schedule = []
    ordinal = 0
    for task_index, task_id in enumerate(sorted(policies)):
        programs = list(policies[task_id][field])
        for position, program_id in enumerate(programs):
            for seed in SEEDS:
                schedule.append({
                    "ordinal": ordinal, "phase": pool + "_correctness",
                    "execution_task_id": task_id, "program_id": program_id,
                    "position": position, "seed": seed,
                    "task_index": task_index,
                })
                ordinal += 1
    return schedule


def selection_schedule(policies: Mapping[str, Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Three one-seed rounds over all production candidates: 16 * 3 = 48 calls."""
    schedule = []
    ordinal = 0
    task_ids = sorted(policies)
    for round_index, seed in enumerate(SEEDS):
        for task_position, task_id in enumerate(_rotate(task_ids, round_index)):
            programs = list(policies[task_id]["production_program_pool"])
            order = _rotate(programs, round_index + task_position)
            if (round_index + task_position) % 2:
                order.reverse()
            for position, program_id in enumerate(order):
                schedule.append({
                    "ordinal": ordinal, "phase": "selection_timing", "round": round_index,
                    "execution_task_id": task_id, "program_id": program_id,
                    "position": position, "seed": seed,
                })
                ordinal += 1
    return schedule


def reference_schedule(
    policies: Mapping[str, Mapping[str, Any]], pool: str = "production",
) -> List[Dict[str, Any]]:
    if pool not in ("production", "legacy"):
        raise ValueError("unknown reference pool")
    field = "production_program_pool" if pool == "production" else "legacy_controlled_program_pool"
    schedule = []
    ordinal = 0
    task_ids = sorted(policies)
    for round_index in range(REFERENCE_ROUNDS):
        for task_position, task_id in enumerate(_rotate(task_ids, round_index)):
            programs = list(policies[task_id][field])
            order = _rotate(programs, round_index + task_position)
            if (round_index + task_position) % 2:
                order.reverse()
            for position, program_id in enumerate(order):
                seed_order = _rotate(SEEDS, round_index + position)
                for seed_position, seed in enumerate(seed_order):
                    schedule.append({
                        "ordinal": ordinal, "phase": pool + "_reference_timing",
                        "round": round_index, "execution_task_id": task_id,
                        "program_id": program_id, "position": position,
                        "seed_position": seed_position, "seed": seed,
                    })
                    ordinal += 1
    return schedule


def confirmation_schedule(
    policies: Mapping[str, Mapping[str, Any]], winners: Mapping[str, str],
) -> List[Dict[str, Any]]:
    schedule = []
    ordinal = 0
    for round_index in range(CONFIRMATION_ROUNDS):
        for task_id in _rotate(sorted(policies), round_index):
            incumbent = policies[task_id]["incumbent_program_id"]
            winner = winners[task_id]
            programs = [incumbent] if winner == incumbent else [incumbent, winner]
            if round_index % 2:
                programs.reverse()
            for position, program_id in enumerate(programs):
                for seed in _rotate(SEEDS, round_index + position):
                    schedule.append({
                        "ordinal": ordinal, "phase": "confirmation_timing",
                        "round": round_index, "execution_task_id": task_id,
                        "program_id": program_id, "position": position, "seed": seed,
                    })
                    ordinal += 1
    return schedule


def round_statistics(rows: Sequence[Mapping[str, Any]], rounds: int) -> Dict[str, Any]:
    medians = []
    for round_index in range(rounds):
        values = [float(row["latency_ms"]) for row in rows if row["round"] == round_index]
        if not values:
            raise ValueError("missing timing round {}".format(round_index))
        medians.append(float(statistics.median(values)))
    return {
        "round_medians_ms": medians,
        "median_of_round_medians_ms": float(statistics.median(medians)),
        "single_call_values_ms": [float(row["latency_ms"]) for row in rows],
    }


def select_winners(
    policies: Mapping[str, Mapping[str, Any]], rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Replay every frozen method/budget using only its allowed selection samples."""
    by_key: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_key[str(row["execution_task_id"]), str(row["program_id"])].append(row)
    result = {}
    for task_id, policy in sorted(policies.items()):
        production = policy["production_program_pool"]
        stats = {program: round_statistics(by_key[task_id, program], len(SEEDS))
                 for program in production}
        incumbent = policy["incumbent_program_id"]
        method_rows = {}
        for method in METHODS:
            method_rows[method] = {}
            order = policy["orders"][method]
            for budget in BUDGETS:
                exposed = [incumbent] + order[:budget]
                winner = min(exposed, key=lambda program: (
                    stats[program]["median_of_round_medians_ms"],
                    0 if program == incumbent else 1,
                    program,
                ))
                method_rows[method]["B{}".format(budget)] = {
                    "exposed_program_ids": exposed,
                    "new_candidate_calls": 3 * budget,
                    "incumbent_common_calls": 3,
                    "search_winner": winner,
                    "selection_latency_ms": stats[winner]["median_of_round_medians_ms"],
                }
        result[task_id] = {
            "program_statistics": stats,
            "methods": method_rows,
            "winner_tie_policy": "exact latency tie: incumbent first, then program_id",
        }
    return result


def transition_state(current: str, requested: str) -> str:
    if current not in STATES or requested not in STATES:
        raise ValueError("unknown campaign state")
    if STATES.index(requested) != STATES.index(current) + 1:
        raise ValueError("illegal campaign transition {} -> {}".format(current, requested))
    return requested


def v4_b3_winners(decisions: Mapping[str, Mapping[str, Any]]) -> Dict[str, str]:
    return {task_id: row["methods"]["selector_v4"]["B3"]["search_winner"]
            for task_id, row in decisions.items()}


def validate_call_counts(policies: Mapping[str, Mapping[str, Any]]) -> Dict[str, int]:
    counts = {
        "production_correctness": len(qualification_schedule(policies, "production")),
        "legacy_correctness": len(qualification_schedule(policies, "legacy")),
        "selection": len(selection_schedule(policies)),
        "production_reference": len(reference_schedule(policies, "production")),
        "legacy_reference": len(reference_schedule(policies, "legacy")),
    }
    expected = {
        "production_correctness": 48,
        "legacy_correctness": 36,
        "selection": 48,
        "production_reference": 288,
        "legacy_reference": 216,
    }
    if counts != expected:
        raise ValueError("v5 call-count drift: {} != {}".format(counts, expected))
    return counts
