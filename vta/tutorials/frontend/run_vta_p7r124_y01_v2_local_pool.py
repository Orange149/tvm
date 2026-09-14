#!/usr/bin/env python3
"""Build and locally qualify the frozen 8-family Y01 schema-v2 pool."""

from __future__ import annotations

import argparse
import hashlib
import itertools
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
    load_json,
    load_jsonl,
    run_fsim,
    sha256_file,
    static_result,
    verify_frozen,
    write_json,
)


SCHEMA = "c3_p7r124_y01_v2_local_pool_v1"
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
C3 = HERE / "report_out" / "stage_tile_cotuning" / "c3_dma_residency_autotune"
P7 = C3 / "07_grouped_holdout"
P7R115B = P7 / "20260911_p7r115b_yolo_confirmation_96_contract_run01"
DEFAULT_CANDIDATES = P7R115B / "candidates.jsonl"
DEFAULT_OUTPUT = P7 / "20260912_p7r124_y01_v2_local_pool_run01"
SEEDS = (0, 20250901, 20260910)
GUARDS = (
    REPO / "vta/python/vta/top/vta_conv2d.py",
    REPO / "vta/python/vta/top/vta_conv2d_residency.py",
    REPO / "vta/python/vta/transform.py",
    REPO / "src/tir/transforms/coproc_sync.cc",
    REPO / "3rdparty/vta-hw/config/vta_config.json",
    REPO / "build/libtvm.so",
)


def canonical_sha256(value):
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def build_candidates(rows):
    y01 = [row for row in rows if row.get("workload_id") == "Y01"]
    originals = [row for row in y01 if row.get("residence_mode") == "original"]
    if len(y01) != 32 or len(originals) != 8:
        raise ValueError("P7R115b must provide 8 Y01 ConfigEntities x 4 legacy modes")
    if len({row["family_id"] for row in originals}) != 8:
        raise ValueError("Y01 original family IDs are not unique")
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
                    "schema": SCHEMA,
                    "workload_id": "Y01",
                    "family_id": original["family_id"],
                    "family_ordinal": int(original["family_ordinal"]),
                    "public_mode": mode,
                    "implementation_mode": int(mode_number),
                    "knobs": knobs,
                    "hardware_predicate": original["hardware_predicate"],
                    "selection_rank_sha256": original["selection_rank_sha256"],
                    "selection_basis": (
                        "P7R115b frozen deterministic label-free ConfigEntity; no Y01 "
                        "board latency, TopHub entity/cost, FSim outcome, or replacement"
                    ),
                    "performance_label": None,
                    "board_status": "not_dispatched",
                }
            )
    if len(candidates) != 24 or len({row["candidate_id"] for row in candidates}) != 24:
        raise ValueError("expected 24 unique schema-v2 candidates")
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


def pair_ranking(candidates):
    families = {}
    for row in candidates:
        families.setdefault(row["family_id"], structural_vector(row))
    keys = tuple(next(iter(families.values())).keys())
    ranges = {
        key: max(vector[key] for vector in families.values())
        - min(vector[key] for vector in families.values())
        for key in keys
    }
    ranking = []
    for first, second in itertools.combinations(sorted(families), 2):
        a, b = families[first], families[second]
        different = sum(a[key] != b[key] for key in keys)
        normalized_l1 = sum(
            abs(a[key] - b[key]) / ranges[key] for key in keys if ranges[key]
        )
        log_tile_l1 = sum(
            abs(math.log2(a[key]) - math.log2(b[key]))
            for key in ("tile_h", "tile_w", "tile_ci", "tile_co")
        )
        ranking.append(
            {
                "families": [first, second],
                "different_structural_fields": different,
                "normalized_l1": normalized_l1,
                "log2_tile_l1": log_tile_l1,
                "vectors": {first: a, second: b},
            }
        )
    ranking.sort(
        key=lambda row: (
            -row["different_structural_fields"],
            -row["normalized_l1"],
            -row["log2_tile_l1"],
            row["families"],
        )
    )
    if len(ranking) != 28:
        raise ValueError("expected all 8-choose-2 structural pairs")
    for ordinal, row in enumerate(ranking):
        row["pre_registered_rank"] = ordinal
    return ranking


def fsim_not_runnable(candidate, static):
    return {
        "schema": SCHEMA,
        "candidate_id": candidate["candidate_id"],
        "workload_id": "Y01",
        "family_id": candidate["family_id"],
        "public_mode": candidate["public_mode"],
        "implementation_mode": candidate["implementation_mode"],
        "status": "not_run_static_failed",
        "failure": {
            "category": "prerequisite",
            "phase": "fsim",
            "message": "static lowering failed; identity remains charged and is not replaced",
            "static_failure": static.get("failure"),
        },
        "seeds": [],
        "command_evidence": {"status": "unavailable", "values": None},
        "performance_measurement": "not_collected",
    }


def choose_board_families(ranking, static_rows, fsim_rows):
    static = {row["candidate_id"]: row for row in static_rows}
    fsim = {row["candidate_id"]: row for row in fsim_rows}
    by_family = defaultdict(list)
    for candidate_id, row in static.items():
        by_family[row["family_id"]].append(candidate_id)
    complete = []
    for family_id, ids in sorted(by_family.items()):
        if len(ids) != 3:
            continue
        if all(
            static[candidate_id]["status"] == "ok"
            and fsim[candidate_id]["status"] == "passed"
            and len(fsim[candidate_id].get("seeds", [])) == 3
            and all(seed["correct"] for seed in fsim[candidate_id]["seeds"])
            and fsim[candidate_id].get("command_evidence", {}).get("status")
            == "available_three_seed_consistent"
            for candidate_id in ids
        ):
            complete.append(family_id)
    selected = None
    for pair in ranking:
        if set(pair["families"]).issubset(complete):
            selected = pair
            break
    return complete, selected


def run(args):
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable P7R124 output {}".format(output))
    input_certificate = verify_frozen(args.candidates)
    raw = Path(args.candidates).read_bytes().lower()
    if b"board_latency" in raw or b"latency_ms" in raw or b"tophub" in raw:
        raise ValueError("P7R124 selection input contains forbidden performance/reference material")
    candidates = build_candidates(load_jsonl(args.candidates))
    ranking = pair_ranking(candidates)
    guards = {str(path.relative_to(REPO)): sha256_file(path) for path in GUARDS}
    contract = {
        "schema": SCHEMA,
        "status": "frozen_before_static_or_fsim_observations",
        "study_tier": "Y01_second_geometry_formal_local_pool",
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
            "static": "lowered TIR, DMA/access, SRAM, dependency and synchronization signatures for all 24",
            "fsim": "three exact seeds and command signature for every static-buildable identity",
            "seeds": list(SEEDS),
            "fsim_timeout_seconds_per_identity": args.fsim_timeout,
            "performance_measurement": "forbidden",
            "board_contact": "forbidden",
        },
        "board_family_preselection": {
            "eligibility": "all three modes static-ok, FSim 3/3 exact, command signature consistent",
            "rule": (
                "take the first eligible pair in the frozen ranking: maximize number of "
                "different structural fields, then normalized L1, then log2 tile L1, then family IDs"
            ),
            "pair_ranking": ranking,
            "performance_fields_used": False,
        },
    }
    output.mkdir(parents=True)
    write_json(output / "contract.json", contract)
    (output / "candidates_v2.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in candidates),
        encoding="utf-8",
    )
    write_json(
        output / "pre_observation_hashes.json",
        {
            "artifacts": {
                "contract.json": sha256_file(output / "contract.json"),
                "candidates_v2.jsonl": sha256_file(output / "candidates_v2.jsonl"),
            }
        },
    )

    static_rows = []
    with (output / "static_results.jsonl").open("x", encoding="utf-8") as stream:
        for candidate in candidates:
            row = static_result(candidate)
            row["schema"] = SCHEMA
            sram = sram_features(candidate)
            row["sram_features"] = sram
            if row["status"] == "ok":
                row["static_feature_vector"] = static_feature_vector(
                    row["transfer_signature"], sram
                )
            static_rows.append(row)
            stream.write(json.dumps(row, sort_keys=True) + "\n"); stream.flush()
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
                stderr = ""
            else:
                row, stderr = run_fsim(
                    candidate,
                    static_row["tir_sha256"],
                    static_row["sync"]["residency_drains"],
                    args.fsim_timeout,
                    args.fsim_hw_path,
                )
                row["schema"] = SCHEMA
            fsim_rows.append(row)
            stream.write(json.dumps(row, sort_keys=True) + "\n"); stream.flush()
            safe = "\n".join(
                "[VTA_QUEUE] <parsed; timing discarded>" if "[VTA_QUEUE] " in line else line
                for line in stderr.splitlines()
            )
            if safe.strip():
                with stderr_path.open("a", encoding="utf-8") as errstream:
                    errstream.write("[{} {}]\n{}\n".format(candidate["family_id"], candidate["public_mode"], safe[-4000:]))
            print("fsim {} {} {}".format(candidate["family_id"], candidate["public_mode"], row["status"]), flush=True)

    complete, selected = choose_board_families(ranking, static_rows, fsim_rows)
    selection = {
        "schema": SCHEMA,
        "complete_three_mode_family_count": len(complete),
        "complete_three_mode_families": complete,
        "selected_for_future_complete_board_pool": selected,
        "selection_uses_performance": False,
        "board_status": "not_dispatched",
    }
    if len(complete) >= 2 and selected is None:
        raise RuntimeError("pre-registered pair selection failed despite two complete families")
    write_json(output / "board_family_selection.json", selection)
    by_mode = {}
    for mode in MODES:
        srows = [row for row in static_rows if row["public_mode"] == mode]
        frows = [row for row in fsim_rows if row["public_mode"] == mode]
        by_mode[mode] = {
            "static_ok": sum(row["status"] == "ok" for row in srows),
            "fsim_passed": sum(row["status"] == "passed" for row in frows),
            "gross": 8,
        }
    summary = {
        "schema": SCHEMA,
        "status": "completed_local_no_board",
        "gross_identities": 24,
        "static_ok": sum(row["status"] == "ok" for row in static_rows),
        "fsim_passed": sum(row["status"] == "passed" for row in fsim_rows),
        "by_mode": by_mode,
        "complete_three_mode_family_count": len(complete),
        "selected_board_families": selected["families"] if selected else [],
        "selection_rank": selected["pre_registered_rank"] if selected else None,
        "board_contacted": False,
        "performance_labels_used": False,
        "claim_boundary": "local Y01 legality/correctness/resource evidence only; no latency or FPGA correctness",
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
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fsim-hw-path", type=Path, default=Path("/tmp/vta-hw-fsim"))
    parser.add_argument("--fsim-timeout", type=int, default=300)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
