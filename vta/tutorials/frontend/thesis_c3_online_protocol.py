#!/usr/bin/env python3
"""Protocol core for the thesis C3 paid-information online experiments."""

from __future__ import annotations

import hashlib
import json
import math
import os
import signal
import statistics
import subprocess
import time
from dataclasses import dataclass, field

import numpy as np


MODES = ("O", "I", "W", "IW")
STRATEGIES = ("random", "mode_aware_xgb", "staged_pareto")
SEEDS = (57001, 57002, 57003)
PHASE_LIMIT_SECONDS = {
    "lower": 60.0,
    "fsim": 120.0,
    "cross_compile": 120.0,
    "rpc": 60.0,
}
PAID_FIELDS = {
    "lower": (
        "lower_pass",
        "logical_dma_bytes",
        "logical_dma_calls",
        "barrier_indicator",
        "effective_program_hash",
    ),
    "fsim": (
        "fsim_pass",
        "actual_submissions",
        "serialized_insn_peak_bytes",
        "serialized_uop_peak_bytes",
    ),
    "correctness": ("numeric_correct", "output_hash"),
    "timing": ("latency_samples_ms", "latency_median_ms"),
}


def stable_hash(*parts):
    return hashlib.sha256("\0".join(map(str, parts)).encode("utf-8")).hexdigest()


def freeze_requested_space(task_id, tiles):
    if len(tiles) != 12 or len(set(tiles)) != 12:
        raise ValueError("each task requires exactly 12 unique Tile identities")
    identities = []
    for tile in tiles:
        if not isinstance(tile, str) or not tile:
            raise ValueError("Tile identity must be a non-empty string")
        for mode in MODES:
            requested_id = stable_hash(task_id, tile, mode)
            identities.append(
                {
                    "requested_id": requested_id,
                    "task_id": task_id,
                    "tile_id": tile,
                    "family_id": stable_hash(task_id, tile),
                    "mode": mode,
                }
            )
    if len({row["requested_id"] for row in identities}) != 48:
        raise AssertionError("requested identity collision")
    return identities


def freeze_attempt_plan(task_ids=("N1", "N2", "N3")):
    if tuple(task_ids) != ("N1", "N2", "N3"):
        raise ValueError("formal unseen plan requires N1/N2/N3 order")
    rotations = (
        ("random", "mode_aware_xgb", "staged_pareto"),
        ("mode_aware_xgb", "staged_pareto", "random"),
        ("staged_pareto", "random", "mode_aware_xgb"),
    )
    attempts = []
    for task_id in task_ids:
        for seed_index, seed in enumerate(SEEDS):
            for strategy in rotations[seed_index]:
                attempt_id = stable_hash("thesis_c3_online_attempt_v1", task_id, seed, strategy)[:20]
                attempts.append(
                    {
                        "attempt_index": len(attempts),
                        "attempt_id": attempt_id,
                        "task_id": task_id,
                        "seed": seed,
                        "strategy": strategy,
                        "request_budget": 24,
                        "online_deadline_seconds": 600,
                    }
                )
    if len(attempts) != 27 or len({row["attempt_id"] for row in attempts}) != 27:
        raise AssertionError("invalid 27-attempt plan")
    return attempts


def random_requested_order(identities, seed):
    ordered = sorted(row["requested_id"] for row in identities)
    permutation = np.random.Generator(np.random.PCG64(int(seed))).permutation(len(ordered))
    return [ordered[int(index)] for index in permutation]


@dataclass
class CandidateState:
    static: dict
    requested: bool = False
    known: dict = field(default_factory=dict)
    acquired_at_seconds: dict = field(default_factory=dict)
    phase_status: dict = field(default_factory=dict)


class PaidInformationBroker:
    """Per-attempt state. It has no import path for another attempt's ledger."""

    def __init__(self, identities, attempt_id, deadline_seconds=600.0, request_budget=24):
        if len(identities) != 48 or len({row["requested_id"] for row in identities}) != 48:
            raise ValueError("broker requires one exact 48-identity task space")
        self.attempt_id = str(attempt_id)
        self.deadline_seconds = float(deadline_seconds)
        self.request_budget = int(request_budget)
        self.states = {row["requested_id"]: CandidateState(dict(row)) for row in identities}
        self.requested_ids = []
        self.events = []
        self.late_results = []

    def request(self, requested_id, elapsed_seconds, reason):
        elapsed = self._elapsed(elapsed_seconds)
        state = self._state(requested_id)
        if elapsed > self.deadline_seconds:
            self.events.append({"event": "request_rejected_deadline", "requested_id": requested_id, "elapsed_seconds": elapsed})
            return False
        if state.requested:
            self.events.append({"event": "duplicate_request", "requested_id": requested_id, "elapsed_seconds": elapsed})
            return False
        if len(self.requested_ids) >= self.request_budget:
            self.events.append({"event": "request_rejected_budget", "requested_id": requested_id, "elapsed_seconds": elapsed})
            return False
        state.requested = True
        self.requested_ids.append(requested_id)
        self.events.append(
            {
                "event": "requested",
                "requested_id": requested_id,
                "elapsed_seconds": elapsed,
                "reason": str(reason),
                "proposal_count": len(self.requested_ids),
            }
        )
        return True

    def complete_phase(self, requested_id, phase, values, elapsed_seconds):
        elapsed = self._elapsed(elapsed_seconds)
        state = self._state(requested_id)
        if phase not in PAID_FIELDS:
            raise ValueError("unknown paid phase")
        if not state.requested:
            raise ValueError("cannot complete an unrequested identity")
        expected = set(PAID_FIELDS[phase])
        if set(values) != expected:
            raise ValueError("phase fields differ from the frozen paid-information schema")
        if phase in state.phase_status:
            raise ValueError("phase already completed")
        self._require_predecessor(state, phase)
        self._validate_values(phase, values)
        event = {
            "event": "phase_completed" if elapsed <= self.deadline_seconds else "late_phase_result",
            "requested_id": requested_id,
            "phase": phase,
            "elapsed_seconds": elapsed,
            "values": dict(values),
        }
        if elapsed > self.deadline_seconds:
            self.late_results.append(event)
            self.events.append(event)
            return False
        for name, value in values.items():
            state.known[name] = value
            state.acquired_at_seconds[name] = elapsed
        state.phase_status[phase] = "completed"
        self.events.append(event)
        return True

    def read(self, requested_id, field):
        state = self._state(requested_id)
        if field in state.static:
            return state.static[field]
        if field not in state.known:
            raise KeyError("paid field is unknown in this attempt: {}".format(field))
        return state.known[field]

    def known_mask(self, requested_id):
        state = self._state(requested_id)
        return {field: field in state.known for fields in PAID_FIELDS.values() for field in fields}

    def export(self):
        return {
            "schema": "thesis_c3_paid_information_broker_v1",
            "attempt_id": self.attempt_id,
            "request_budget": self.request_budget,
            "deadline_seconds": self.deadline_seconds,
            "requested_ids": list(self.requested_ids),
            "proposal_count": len(self.requested_ids),
            "events": list(self.events),
            "late_results": list(self.late_results),
            "candidates": {
                key: {
                    "static": state.static,
                    "requested": state.requested,
                    "known": state.known,
                    "acquired_at_seconds": state.acquired_at_seconds,
                    "phase_status": state.phase_status,
                }
                for key, state in sorted(self.states.items())
            },
        }

    def _state(self, requested_id):
        if requested_id not in self.states:
            raise KeyError("identity outside the frozen 48-point space")
        return self.states[requested_id]

    @staticmethod
    def _elapsed(value):
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError("invalid monotonic elapsed time")
        return value

    @staticmethod
    def _require_predecessor(state, phase):
        if phase == "fsim" and state.known.get("lower_pass") is not True:
            raise ValueError("FSim requires a paid passing lower result")
        if phase == "correctness" and state.known.get("fsim_pass") is not True:
            raise ValueError("correctness requires a paid passing FSim result")
        if phase == "timing" and state.known.get("numeric_correct") is not True:
            raise ValueError("timing requires paid numeric correctness")

    @staticmethod
    def _validate_values(phase, values):
        if phase == "lower":
            if values["lower_pass"] is not True and any(
                values[name] is not None for name in PAID_FIELDS[phase][1:]
            ):
                raise ValueError("failed lower cannot reveal successful lower features")
            if values["lower_pass"] is True and (
                min(int(values["logical_dma_bytes"]), int(values["logical_dma_calls"])) < 0
                or int(values["barrier_indicator"]) not in (0, 1)
                or not values["effective_program_hash"]
            ):
                raise ValueError("invalid lower feature values")
        elif phase == "fsim":
            if values["fsim_pass"] is not True and any(
                values[name] is not None for name in PAID_FIELDS[phase][1:]
            ):
                raise ValueError("failed FSim cannot reveal successful FSim features")
            if values["fsim_pass"] is True and min(
                int(values["actual_submissions"]),
                int(values["serialized_insn_peak_bytes"]),
                int(values["serialized_uop_peak_bytes"]),
            ) <= 0:
                raise ValueError("invalid FSim feature values")
        elif phase == "correctness":
            if values["numeric_correct"] is not True and values["output_hash"] is not None:
                raise ValueError("failed correctness cannot publish a qualified output hash")
            if values["numeric_correct"] is True and not values["output_hash"]:
                raise ValueError("passing correctness requires an output hash")
        elif phase == "timing":
            samples = values["latency_samples_ms"]
            if not isinstance(samples, list) or len(samples) != 7 or any(
                not math.isfinite(float(value)) or float(value) <= 0 for value in samples
            ):
                raise ValueError("timing requires seven positive finite samples")
            expected = statistics.median(map(float, samples))
            if abs(float(values["latency_median_ms"]) - expected) > 1.0e-12:
                raise ValueError("timing median mismatch")


def run_phase_subprocess(command, phase, online_start_monotonic, deadline_seconds=600.0, now=None):
    """Run one real phase under min(phase limit, remaining online time)."""

    if phase not in PHASE_LIMIT_SECONDS:
        raise ValueError("unknown executable phase")
    clock = now or time.monotonic
    start = clock()
    elapsed = start - float(online_start_monotonic)
    remaining = float(deadline_seconds) - elapsed
    if remaining <= 0:
        return {
            "status": "deadline_before_start",
            "returncode": None,
            "elapsed_at_start_seconds": elapsed,
            "wall_seconds": 0.0,
            "stdout": "",
            "stderr": "",
            "admit_result": False,
        }
    timeout = min(PHASE_LIMIT_SECONDS[phase], remaining)
    process = subprocess.Popen(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
    end = clock()
    completed_elapsed = end - float(online_start_monotonic)
    within_deadline = completed_elapsed <= float(deadline_seconds)
    status = "timeout" if timed_out else ("completed" if process.returncode == 0 else "failed")
    return {
        "status": status,
        "returncode": process.returncode,
        "elapsed_at_start_seconds": elapsed,
        "completed_elapsed_seconds": completed_elapsed,
        "timeout_seconds": timeout,
        "wall_seconds": end - start,
        "stdout": stdout,
        "stderr": stderr,
        "admit_result": bool(not timed_out and process.returncode == 0 and within_deadline),
    }
