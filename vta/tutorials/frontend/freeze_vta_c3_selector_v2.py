#!/usr/bin/env python3
"""Fit and freeze selector v2, or freeze its specified automatic fallback."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

from c3_selector_v2 import fixed_random_orders, train_selector


HERE = Path(__file__).resolve().parent
GROUPED = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout"
DEFAULT_DATASET = GROUPED / "20260924_c3_selector_v2_dataset_run01"
DEFAULT_OUTPUT = GROUPED / "20260924_c3_selector_v2_freeze_run01"
SCHEMA = "c3_selector_v2_freeze_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite immutable output: {args.output}")
    args.output.mkdir(parents=True)
    rows_path = args.dataset / "training_rows.jsonl"
    rows = load_jsonl(rows_path)
    contract = train_selector(rows)
    contract.update({
        "freeze_schema": SCHEMA,
        "status": "frozen",
        "dataset_path": str(args.dataset.resolve()),
        "training_rows_sha256": sha256(rows_path),
        "training_execution_task_ids": sorted({row["execution_task_id"] for row in rows}),
        "excluded_source_reasons": [
            source for source in json.loads((args.dataset / "source_admission.json").read_text())
            if not source.get("accepted")
        ],
        "prediction_target": "log(candidate_latency_ms/incumbent_latency_ms)",
        "shortlist_policy": ["incumbent", "selector_v2", "bytes_only", "command_only"],
        "budgets": {"B1": 1, "B2": 2, "B3": 3, "unit": "additional new candidates"},
        "fpga_invalid_policy": "consumes budget; cannot be timed; no replacement",
        "random_baseline": {"seed": 20260924, "orders": 1000,
                            "scope": "offline replay on complete pools only"},
    })
    write_json(args.output / "selector_v2_contract.json", contract)
    # Freeze the random policy without producing task-specific orders.  The
    # canary makes implementation drift detectable; real orders are generated
    # only from a frozen complete-pool program-ID list.
    canary = fixed_random_orders(["candidate-a", "candidate-b", "candidate-c"], count=1000)
    write_jsonl(args.output / "random_order_canary.jsonl",
                ({"replicate": i, "order": order} for i, order in enumerate(canary)))
    summary = {
        "schema": SCHEMA,
        "status": "frozen",
        "selector_kind": contract["selector_kind"],
        "eligibility": contract["eligibility"],
        "selected_alpha": contract.get("ridge", {}).get("alpha"),
        "training_rows_sha256": contract["training_rows_sha256"],
        "performance_labels_collected": False,
        "board_contacted": False,
    }
    write_json(args.output / "summary.json", summary)
    write_json(args.output / "artifact_hashes.json", {
        "selector_v2_contract.json": sha256(args.output / "selector_v2_contract.json"),
        "random_order_canary.jsonl": sha256(args.output / "random_order_canary.jsonl"),
        "summary.json": sha256(args.output / "summary.json"),
        "source.c3_selector_v2.py": sha256(HERE / "c3_selector_v2.py"),
        "source.freeze_vta_c3_selector_v2.py": sha256(Path(__file__).resolve()),
    })
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
