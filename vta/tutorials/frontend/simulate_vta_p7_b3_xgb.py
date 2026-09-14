#!/usr/bin/env python3
"""Replay the frozen B3 sequential ConfigEntity-only XGB policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import xgboost as xgb


SEED = 20250901
FEATURE_NAMES = (
    "tile_b",
    "tile_h",
    "tile_w",
    "tile_ci",
    "tile_co",
    "oc_nthread",
    "h_nthread",
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_frozen_contract(path):
    path = Path(path).resolve()
    ledger = json.loads((path.parent / "artifact_hashes.json").read_text())
    if ledger["output_sha256"].get(path.name) != sha256(path):
        raise ValueError("qualified timing contract hash mismatch")
    value = json.loads(path.read_text())
    if value.get("schema") != "c3_p7_qualified_timing_contract_v1":
        raise ValueError("unexpected qualified timing contract")
    return value


def vector(entry):
    knobs = {}
    for name, kind, value in entry["complete_config_entity"]["entity"]:
        knobs[name] = float(value[-1] if kind == "sp" else value)
    return [knobs[name] for name in FEATURE_NAMES]


def model():
    return xgb.XGBRegressor(
        n_estimators=64,
        max_depth=3,
        learning_rate=0.1,
        min_child_weight=1,
        subsample=1.0,
        colsample_bytree=1.0,
        reg_alpha=0.0,
        reg_lambda=1.0,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=SEED,
        n_jobs=1,
        verbosity=0,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--timing-summary", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("refusing to overwrite B3 replay")

    contract = load_frozen_contract(args.contract)
    latency = {}
    for path in args.timing_summary:
        summary = json.loads(Path(path).read_text())
        workload_id = summary["workload_id"]
        latency[workload_id] = {
            candidate_id: float(row["median_ms"])
            for candidate_id, row in summary["candidate_summaries"].items()
        }

    results = {}
    for workload_id in contract["workload_order"]:
        workload = contract["workloads"][workload_id]
        entries = {row["candidate_id"]: row for row in workload["gross_candidates"]}
        gross_ids = list(entries)
        valid_ids = set(workload["timed_candidate_ids"])
        invalid_ids = set(gross_ids) - valid_ids
        if set(latency.get(workload_id, {})) != valid_ids:
            raise ValueError("timing coverage mismatch for " + workload_id)
        remaining = set(gross_ids)
        warmup = sorted(
            gross_ids,
            key=lambda cid: hashlib.sha256(
                "{}:{}:{}".format(SEED, workload_id, cid).encode()
            ).hexdigest(),
        )[:4]
        observed_x = []
        observed_y = []
        dispatches = []
        while remaining:
            if len(dispatches) < 4:
                candidate_id = warmup[len(dispatches)]
                selection = "deterministic_random_warmup"
                predicted = None
            else:
                estimator = model()
                estimator.fit(np.asarray(observed_x), np.asarray(observed_y))
                choices = sorted(remaining)
                predictions = estimator.predict(
                    np.asarray([vector(entries[candidate_id]) for candidate_id in choices])
                )
                ranked = sorted(zip(predictions.tolist(), choices), key=lambda item: (item[0], item[1]))
                predicted, candidate_id = ranked[0]
                selection = "refit_xgb_min_prediction"
            remaining.remove(candidate_id)
            valid = candidate_id in valid_ids
            row = {
                "gross_dispatch": len(dispatches) + 1,
                "candidate_id": candidate_id,
                "selection": selection,
                "predicted_ms_before_dispatch": predicted,
                "valid": valid,
                "observed_ms": latency[workload_id].get(candidate_id),
            }
            dispatches.append(row)
            if valid:
                observed_x.append(vector(entries[candidate_id]))
                observed_y.append(latency[workload_id][candidate_id])
        results[workload_id] = {
            "warmup_candidate_ids": warmup,
            "invalid_candidate_ids": sorted(invalid_ids),
            "dispatches": dispatches,
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "schema": "c3_p7_b3_xgb_replay_v1",
                "seed": SEED,
                "xgboost_version": xgb.__version__,
                "feature_names": FEATURE_NAMES,
                "invalid_dispatch_policy": "consume gross position, do not train regressor",
                "workloads": results,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
