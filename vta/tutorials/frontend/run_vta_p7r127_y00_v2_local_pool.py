#!/usr/bin/env python3
"""Build and locally qualify a frozen YOLO three-mode v2 pool.

P7R127/Y00 remains the default.  Later prospective geometries reuse the exact
same qualification path by supplying an explicit workload ID and immutable
candidate contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

from c3_candidate_identity import canonical_json_bytes
from run_vta_p7r117_yolo_no_leak_local import sram_features, static_feature_vector
from run_vta_p7r119_yolo_barrier_pilot import (
    MODES,
    candidate_identity,
    config_knobs,
    load_jsonl,
    run_fsim,
    sha256_file,
    static_result,
    verify_frozen,
    write_json,
)
from run_vta_p7r124_y01_v2_local_pool import GUARDS, fsim_not_runnable


SCHEMA = "c3_p7r127_y00_v2_local_pool_v1"
SEEDS = (0, 20250901, 20260910)
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
C3 = HERE / "report_out" / "stage_tile_cotuning" / "c3_dma_residency_autotune"
P7 = C3 / "07_grouped_holdout"
P7R115B = P7 / "20260911_p7r115b_yolo_confirmation_96_contract_run01"
DEFAULT_CANDIDATES = P7R115B / "candidates.jsonl"
DEFAULT_OUTPUT = P7 / "20260912_p7r127_y00_v2_local_pool_run01"


def canonical_sha256(value):
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def build_candidates(rows, workload_id="Y00", schema=SCHEMA):
    selected = [row for row in rows if row.get("workload_id") == workload_id]
    originals = [row for row in selected if row.get("residence_mode") == "original"]
    if len(originals) != 8 or len(selected) not in (8, 32):
        raise ValueError(
            "input must provide 8 {} original ConfigEntities, optionally expanded over 4 legacy modes".format(
                workload_id
            )
        )
    if len({row["family_id"] for row in originals}) != 8:
        raise ValueError("{} original family IDs are not unique".format(workload_id))
    candidates = []
    for original in sorted(originals, key=lambda row: row["family_ordinal"]):
        base = {
            "identity": original["identity"],
            "complete_config_entity": original["identity"]["complete_config_entity"],
            "debug": original["debug"],
        }
        knobs = config_knobs(base["complete_config_entity"])
        for mode, mode_number in MODES.items():
            semantic = candidate_identity(base, mode, mode_number)
            candidates.append(
                {
                    **semantic,
                    "schema": schema,
                    "workload_id": workload_id,
                    "family_id": original["family_id"],
                    "family_ordinal": int(original["family_ordinal"]),
                    "public_mode": mode,
                    "implementation_mode": int(mode_number),
                    "knobs": knobs,
                    "hardware_predicate": original["hardware_predicate"],
                    "selection_rank_sha256": original["selection_rank_sha256"],
                    "selection_basis": (
                        "frozen deterministic label-free ConfigEntity; no {} board ".format(workload_id)
                        + "latency, TopHub entity/cost, FSim outcome, or replacement"
                    ),
                    "performance_label": None,
                    "board_status": "not_dispatched",
                }
            )
    if len(candidates) != 24 or len({row["candidate_id"] for row in candidates}) != 24:
        raise ValueError("expected 24 unique schema-v2 {} candidates".format(workload_id))
    return candidates


def structural_vector(candidate):
    knobs = candidate["knobs"]
    required = candidate["hardware_predicate"]["required"]
    opportunities = candidate["hardware_predicate"]["opportunities"]
    return {
        "tile_h": int(knobs["tile_h"]),
        "tile_w": int(knobs["tile_w"]),
        "tile_ci": int(knobs["tile_ci"]),
        "tile_co": int(knobs["tile_co"]),
        "oc_nthread": int(knobs["oc_nthread"]),
        "h_nthread": int(knobs["h_nthread"]),
        "input_vectors": int(required["input_vectors"]),
        "weight_vectors": int(required["weight_vectors"]),
        "accumulator_vectors": int(required["accumulator_vectors"]),
        "output_channel_outer_tiles": int(opportunities["output_channel_outer_tiles"]),
        "output_width_outer_tiles": int(opportunities["output_width_outer_tiles"]),
    }


def structural_coverage_order(candidates, workload_id="Y00"):
    """Frozen greedy max-min family order using only structural fields."""
    families = {}
    for row in candidates:
        families.setdefault(row["family_id"], structural_vector(row))
    if len(families) != 8:
        raise ValueError("expected eight {} families".format(workload_id))
    keys = tuple(next(iter(families.values())))
    ranges = {
        key: max(row[key] for row in families.values()) - min(row[key] for row in families.values())
        for key in keys
    }

    def distance(first, second):
        return sum(
            abs(families[first][key] - families[second][key]) / ranges[key]
            for key in keys
            if ranges[key]
        )

    ids = sorted(families)
    pairwise = {
        first: {second: distance(first, second) for second in ids if second != first}
        for first in ids
    }
    first = min(ids, key=lambda family: (-sum(pairwise[family].values()), family))
    selected = [first]
    trace = [
        {
            "ordinal": 0,
            "family_id": first,
            "criterion": "maximum total normalized L1 distance to all other families",
            "total_distance": sum(pairwise[first].values()),
            "vector": families[first],
        }
    ]
    while len(selected) < len(ids):
        remaining = [family for family in ids if family not in selected]
        scores = {
            family: (
                min(pairwise[family][chosen] for chosen in selected),
                sum(pairwise[family][other] for other in ids if other != family),
            )
            for family in remaining
        }
        chosen = min(
            remaining,
            key=lambda family: (-scores[family][0], -scores[family][1], family),
        )
        trace.append(
            {
                "ordinal": len(selected),
                "family_id": chosen,
                "criterion": "maximum minimum normalized L1 distance to selected set",
                "minimum_distance_to_selected": scores[chosen][0],
                "total_distance": scores[chosen][1],
                "vector": families[chosen],
            }
        )
        selected.append(chosen)
    return trace


def complete_families(candidates, static_rows, fsim_rows):
    static = {row["candidate_id"]: row for row in static_rows}
    fsim = {row["candidate_id"]: row for row in fsim_rows}
    by_family = defaultdict(list)
    for row in candidates:
        by_family[row["family_id"]].append(row["candidate_id"])
    complete = []
    for family, ids in sorted(by_family.items()):
        if len(ids) == 3 and all(
            static[candidate_id]["status"] == "ok"
            and fsim[candidate_id]["status"] == "passed"
            and len(fsim[candidate_id].get("seeds", [])) == 3
            and all(seed.get("correct") for seed in fsim[candidate_id]["seeds"])
            and fsim[candidate_id].get("command_evidence", {}).get("status")
            == "available_three_seed_consistent"
            for candidate_id in ids
        ):
            complete.append(family)
    return complete


def run(args):
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable P7R127 output {}".format(output))
    fsim_config = Path(args.fsim_hw_path) / "config" / "vta_config.json"
    fsim_pkg = Path(args.fsim_hw_path) / "config" / "pkg_config.py"
    if not fsim_config.is_file() or not fsim_pkg.is_file():
        raise FileNotFoundError(
            "incomplete FSim hardware path; require {} and {}".format(fsim_config, fsim_pkg)
        )
    input_certificate = verify_frozen(args.candidates)
    workload_id = args.workload_id
    schema = SCHEMA if workload_id == "Y00" else "c3_{}_v2_local_pool_v1".format(workload_id.lower())
    candidates = build_candidates(load_jsonl(args.candidates), workload_id, schema)
    coverage = structural_coverage_order(candidates, workload_id)
    guards = {str(path.relative_to(REPO)): sha256_file(path) for path in GUARDS}
    contract = {
        "schema": schema,
        "status": "frozen_before_static_or_fsim_observations",
        "study_tier": "{}_formal_local_pool".format(workload_id),
        "input": input_certificate,
        "source_guards_sha256": guards,
        "gross_pool": {
            "families": 8,
            "modes": MODES,
            "identities": 24,
            "candidate_order": [row["candidate_id"] for row in candidates],
            "commitment_sha256": canonical_sha256(candidates),
            "failure_policy": "all failures stay charged; no replacement",
        },
        "local_protocol": {
            "static": "lowered TIR, DMA/request, SRAM, dependency and synchronization signatures for all 24",
            "fsim": "three exact seeds and command signature for every static-buildable identity",
            "seeds": list(SEEDS),
            "fsim_timeout_seconds_per_identity": args.fsim_timeout,
            "performance_measurement": "forbidden",
            "board_contact": "forbidden",
        },
        "future_board_order": {
            "rule": (
                "structure-only greedy max-min normalized L1 coverage; first maximize total "
                "distance, then maximize minimum distance to selected, total distance, family ID"
            ),
            "full_family_order": [row["family_id"] for row in coverage],
            "trace": coverage,
            "future_eligibility": "retain order while filtering to complete three-mode local families",
            "performance_fields_used": False,
            "board_status": "not_dispatched",
        },
        "prohibitions": {
            "rpc_or_board_contact": True,
            "board_latency_or_tophub_entity_cost": True,
            "replacement_after_failure": True,
        },
    }
    output.mkdir(parents=True)
    write_json(output / "contract.json", contract)
    (output / "candidates_v2.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in candidates), encoding="utf-8"
    )
    write_json(
        output / "pre_observation_hashes.json",
        {"artifacts": {
            "contract.json": sha256_file(output / "contract.json"),
            "candidates_v2.jsonl": sha256_file(output / "candidates_v2.jsonl"),
        }},
    )

    static_rows = []
    with (output / "static_results.jsonl").open("x", encoding="utf-8") as stream:
        for candidate in candidates:
            row = static_result(candidate)
            row["schema"] = schema
            sram = sram_features(candidate)
            row["sram_features"] = sram
            if row["status"] == "ok":
                row["static_feature_vector"] = static_feature_vector(row["transfer_signature"], sram)
            static_rows.append(row)
            stream.write(json.dumps(row, sort_keys=True) + "\n")
            stream.flush()
            print("static {} {} {}".format(candidate["family_id"], candidate["public_mode"], row["status"]), flush=True)

    static = {row["candidate_id"]: row for row in static_rows}
    fsim_rows = []
    stderr_path = output / "stderr.log"
    stderr_path.touch(exist_ok=False)
    with (output / "fsim_results.jsonl").open("x", encoding="utf-8") as stream:
        for candidate in candidates:
            static_row = static[candidate["candidate_id"]]
            if static_row["status"] != "ok":
                row = fsim_not_runnable(candidate, static_row)
                row["schema"] = schema
                # The legacy helper was introduced for Y01 and carries that
                # workload label in its prerequisite-only row.  Candidate
                # identity and outcome are already correct; normalize the
                # descriptive label for later generic Yxx pools.
                row["workload_id"] = workload_id
                stderr = ""
            else:
                row, stderr = run_fsim(
                    candidate,
                    static_row["tir_sha256"],
                    static_row["sync"]["residency_drains"],
                    args.fsim_timeout,
                    args.fsim_hw_path,
                )
                row["schema"] = schema
            fsim_rows.append(row)
            stream.write(json.dumps(row, sort_keys=True) + "\n")
            stream.flush()
            safe = "\n".join(
                "[VTA_QUEUE] <parsed; timing discarded>" if "[VTA_QUEUE] " in line else line
                for line in stderr.splitlines()
            )
            if safe.strip():
                with stderr_path.open("a", encoding="utf-8") as errstream:
                    errstream.write("[{} {}]\n{}\n".format(candidate["family_id"], candidate["public_mode"], safe[-4000:]))
            print("fsim {} {} {}".format(candidate["family_id"], candidate["public_mode"], row["status"]), flush=True)

    complete = complete_families(candidates, static_rows, fsim_rows)
    frozen_order = [row["family_id"] for row in coverage]
    eligible_order = [family for family in frozen_order if family in complete]
    write_json(
        output / "future_board_order.json",
        {
            "schema": schema,
            "full_structural_order": frozen_order,
            "complete_three_mode_families": complete,
            "eligible_order_without_reordering": eligible_order,
            "selection_uses_performance": False,
            "board_status": "not_dispatched",
        },
    )
    by_mode = {}
    for mode in MODES:
        srows = [row for row in static_rows if row["public_mode"] == mode]
        frows = [row for row in fsim_rows if row["public_mode"] == mode]
        by_mode[mode] = {
            "gross": 8,
            "static_ok": sum(row["status"] == "ok" for row in srows),
            "fsim_passed": sum(row["status"] == "passed" for row in frows),
        }
    summary = {
        "schema": schema,
        "status": "completed_local_no_board",
        "gross_identities": 24,
        "static_ok": sum(row["status"] == "ok" for row in static_rows),
        "fsim_passed": sum(row["status"] == "passed" for row in fsim_rows),
        "by_mode": by_mode,
        "complete_three_mode_family_count": len(complete),
        "complete_three_mode_families": complete,
        "future_board_family_order": eligible_order,
        "board_contacted": False,
        "performance_labels_used": False,
        "claim_boundary": "local {} legality/correctness/resource evidence only; no latency or FPGA correctness".format(workload_id),
    }
    write_json(output / "summary.json", summary)
    (output / "command.txt").write_text(
        " ".join([sys.executable, str(Path(__file__).resolve())] + sys.argv[1:]) + "\n",
        encoding="utf-8",
    )
    hashes = {
        path.name: sha256_file(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    write_json(
        output / "artifact_hashes.json",
        {"artifacts": hashes, "source_sha256": {str(Path(__file__).resolve()): sha256_file(__file__)}},
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--workload-id", default="Y00")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fsim-hw-path", type=Path, default=Path("/tmp/vta-hw-fsim"))
    parser.add_argument("--fsim-timeout", type=int, default=300)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
