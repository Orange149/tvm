#!/usr/bin/env python3
"""Close the Node-B protocol bundle with label-blind eight-round board orders.

This is a freeze operation, not a board runner. It uses the verified local
qualification unchanged and fills the final protocol's named records and order.
"""
import argparse
import random
from pathlib import Path
import time

from analyze_vta_c3_reuse_local import check_candidate
from run_vta_c3_reuse_supplement import freeze, read, write, append, seal, verify, reject_labels, check_bindings, file_hash, MODES

FAMILY_ORDER_SEED = 20260910


def latin_orders():
    modes = list(MODES)
    return [[modes[(offset + position) % 4] for position in range(4)] for offset in range(4)] * 2


def run(args):
    started = time.monotonic()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    append(output / "timeline.jsonl", {"event": "W0", "wall_unix": time.time(), "T0": None})
    try:
        freeze(args, output, started)
        contract = read(output / "contract.json")
        candidates, controls = read(output / "candidates.json"), read(output / "controls.json")
        for row in candidates + controls:
            reject_labels(row)
            check_candidate(row)
        families = sorted({row["family_id"] for row in candidates})
        random.Random(FAMILY_ORDER_SEED).shuffle(families)
        reordered = [row for family_id in families for mode in MODES for row in candidates
                     if row["family_id"] == family_id and row["residence_mode"] == mode]
        if len(reordered) != len(candidates):
            raise ValueError("invalid family/mode order")
        bindings = {**contract["source_bindings"], str(Path(__file__).resolve()): file_hash(Path(__file__)),
                    str(Path(check_candidate.__code__.co_filename).resolve()): file_hash(Path(check_candidate.__code__.co_filename))}
        contract.update(source_bindings=bindings, family_order_seed=FAMILY_ORDER_SEED,
                        frozen_family_order=families, candidate_order=[row["candidate_id"] for row in reordered],
                        frozen_mode_orders=latin_orders(), profile_complete_invocations=1,
                        profile_requires_extra_execution=True, controls_timed_in_independent_balanced_blocks=True,
                        session_timeout_seconds=7200, interruption_invalidates_whole_board_session=True,
                        positive_fpga_combination_results=False, board_protocol_implemented=False)
        write(output / "contract.json", contract)
        write(output / "candidates.json", reordered)
        for row in reordered + controls:
            append(output / "results.jsonl", {"phase": "freeze", "candidate_id": row["candidate_id"],
                                              "family_id": row["family_id"], "residence_mode": row["residence_mode"],
                                              "control": row.get("experimental_reload_control", False),
                                              "status": "local_and_cross_qualified", "fpga_status": "not_attempted",
                                              "tir_sha256": row["local_qualification"]["tir_sha256"],
                                              "binary_sha256": row["cross_qualification"]["binary_sha256"]})
        summary = read(output / "summary.json")
        summary.update(four_local_correct_families=len(families), rounds_per_family=8,
                       planned_main_timed_calls_per_boot=len(candidates) * 8,
                       wall_seconds=time.monotonic() - started,
                       protocol_record_files_complete=True)
        write(output / "summary.json", summary)
        check_bindings(bindings)
        verify(Path(args.contract).resolve().parent)
        append(output / "timeline.jsonl", {"event": "T1", "elapsed_seconds": time.monotonic() - started})
        seal(output)
    except BaseException as error:
        write(output / "invalid_session.json", {"type": type(error).__name__, "message": str(error), "board_contacted": False})
        seal(output)
        raise
    print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("freeze",), default="freeze")
    parser.add_argument("--contract", required=True)
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())
