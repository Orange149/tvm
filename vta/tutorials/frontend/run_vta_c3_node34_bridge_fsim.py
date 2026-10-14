#!/usr/bin/env python3
"""Local-only exact-fusion FSim gate for the node34 token bridge."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import traceback

import numpy as np
import tvm
from tvm import autotvm, relay, tir
from tvm.contrib import graph_executor
import vta
from vta.testing import simulator
from vta.top.residency_dispatch import ExplicitResidencyDispatch

from build_vta_resnet50_generic_residency_tir_audit import FullTIRArchive
from c3_node34_group_bridge import (
    assert_group_boundary, build_config_with_bridge,
    fused_relay_from_mode4_evidence,
)
from c3_reuse_semantics import extract_requests
from c3_thread_sync_audit import final_tir_audit


HERE = Path(__file__).resolve().parent
GROUPED = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout"
MODE4 = GROUPED / "20261008_c3_node34_mode4_local_gate_run02"
STANDALONE_SYMBOL = "tvmgen_default_fused_nn_conv2d_add_right_shift_clip_cast"
SEEDS = (0, 20250901, 20260910)


def sha(value):
    return hashlib.sha256(value).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def arrays(seed):
    rng = np.random.default_rng(seed)
    return (
        rng.integers(-4, 5, size=(1, 32, 7, 7, 1, 16), dtype=np.int8),
        rng.integers(-4, 5, size=(64, 32, 3, 3, 16, 16), dtype=np.int8),
        rng.integers(-20, 21, size=(64, 1, 1, 1, 16), dtype=np.int32),
    )


def run(output):
    baseline = json.loads((MODE4 / "local_evidence.json").read_text())
    previous = json.loads((MODE4 / "fsim_results.json").read_text())
    if previous["status"] != "passed":
        raise ValueError("mode4 prerequisite did not pass FSim")
    boundary = assert_group_boundary(baseline["address_audit"])
    expected = {row["seed"]: row["expected_sha256"] for row in previous["seeds"]}
    if set(expected) != set(SEEDS):
        raise ValueError("mode4 prerequisite seed set differs")
    route = json.loads((MODE4 / "candidate.json").read_text())["routes"][-1]
    program = fused_relay_from_mode4_evidence(MODE4 / "local_evidence.json")
    cfg = dict(vta.get_env().cfg_dict)
    cfg["TARGET"] = "sim"
    with vta.Environment(cfg):
        env = vta.get_env()
        target = tvm.target.Target(env.target, host=env.target_host)
        archive = FullTIRArchive(env)
        context, changed = build_config_with_bridge(STANDALONE_SYMBOL)
        relay.backend.te_compiler.get().clear()
        with autotvm.tophub.context(target), ExplicitResidencyDispatch([route]) as dispatch:
            with tvm.transform.PassContext(
                opt_level=3, config=context.config,
                disabled_pass={"AlterOpLayout", "tir.CommonSubexprElimTIR"},
                instruments=[archive],
            ):
                graph, lib, params = relay.build(program, target=target)
        if not dispatch.summary()["all_routes_scheduled"] or len(changed) != 1 or params:
            raise ValueError("standalone mode4 bridge was not applied exactly once")
        functions = []
        for snapshot in archive.snapshots:
            module = tvm.ir.load_json(snapshot["json"])
            for function in module.functions.values():
                if isinstance(function, tir.PrimFunc) and function.attrs is not None \
                        and "global_symbol" in function.attrs \
                        and str(function.attrs["global_symbol"]) == STANDALONE_SYMBOL:
                    functions.append(function)
        if len(functions) != 1:
            raise ValueError("missing exact standalone fused bridge TIR")
        function = functions[0]
        (output / "bridge.tir.json").write_text(tvm.ir.save_json(function))
        compact_tir = function.script()
        (output / "bridge.tir").write_text(compact_tir)
        single = tvm.IRModule({"main": function})
        requests = extract_requests(single, env)
        address = final_tir_audit(single, requests)
        if address["status"] != "passed":
            raise ValueError("bridge final TIR address audit failed")
        if requests["totals"] != baseline["request_semantics"]["totals"]:
            raise ValueError("bridge changed logical memory requests")
        if address["ordered_events"] != baseline["address_audit"]["ordered_events"]:
            raise ValueError("bridge changed data request or compute-kernel order")
        if len(address["sync_event_positions"]) != 1:
            raise ValueError("bridge retained a group drain or lost function completion")
        if address["max_kernel_uops"] != baseline["address_audit"]["max_kernel_uops"]:
            raise ValueError("bridge changed a UOP kernel")
        runner = graph_executor.create(graph, lib, tvm.device("ext_dev", 0))
        rows = []
        for seed in (*SEEDS, 0):  # WA -> WB -> WA reuse on the same runner.
            x, w, add = arrays(seed)
            identities = [sha(array.tobytes()) for array in (x, w, add)]
            runner.set_input("p09", x)
            runner.set_input("p15", w)
            runner.set_input("p24", add)
            simulator.clear_stats()
            runner.run()
            actual = runner.get_output(0).numpy()
            observed = sha(actual.tobytes())
            unchanged = identities == [sha(array.tobytes()) for array in (x, w, add)]
            row = {"seed": seed, "correct": observed == expected[seed] and unchanged,
                   "actual_sha256": observed, "expected_sha256": expected[seed],
                   "host_arguments_unchanged": unchanged, "fsim_counters": simulator.stats()}
            rows.append(row)
            if not row["correct"]:
                raise ValueError("bridge FSim output or independent operand changed")
        result = {
            "status": "BRIDGE_LOCAL_FSIM_PASSED_BOARD_NOT_QUALIFIED",
            "boundary_gate": boundary, "changed": changed,
            "exact_data_and_kernel_event_sequence": True,
            "request_totals": requests["totals"],
            "address_status": address["status"],
            "max_kernel_uops": address["max_kernel_uops"],
            "final_sync_count": len(address["sync_event_positions"]),
            "fsim_calls": rows,
            "runtime_submission_count": "requires queue diagnostic; not inferred from TIR",
            "on_chip_uop_peak": None,
            "board_contacted": False,
        }
        save(output / "result.json", result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        result = run(args.output_dir)
    except BaseException as error:
        result = {"status": "BRIDGE_LOCAL_GATE_FAILED", "error": str(error),
                  "traceback": traceback.format_exc(), "board_contacted": False}
        save(args.output_dir / "failure.json", result)
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("fsim_calls", "request_totals", "traceback")}, indent=2))
    if result["status"] == "BRIDGE_LOCAL_GATE_FAILED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
