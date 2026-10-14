#!/usr/bin/env python3
"""Traceable local FSim UOP and command statistics for the fixed mode4 fused control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback

import tvm
from tvm import autotvm, relay
from tvm.contrib import graph_executor
import vta
from vta.testing import simulator
from vta.top.residency_dispatch import ExplicitResidencyDispatch

from c3_node34_group_bridge import fused_relay_from_mode4_evidence
from run_vta_c3_node34_bridge_fsim import MODE4, SEEDS, arrays, sha, save


def run(output):
    expected_rows = json.loads((MODE4 / "fsim_results.json").read_text())
    if expected_rows["status"] != "passed":
        raise ValueError("mode4 local FSim prerequisite failed")
    expected = {row["seed"]: row["expected_sha256"] for row in expected_rows["seeds"]}
    route = json.loads((MODE4 / "candidate.json").read_text())["routes"][-1]
    program = fused_relay_from_mode4_evidence(MODE4 / "local_evidence.json")
    cfg = dict(vta.get_env().cfg_dict)
    cfg["TARGET"] = "sim"
    with vta.Environment(cfg):
        env = vta.get_env()
        target = tvm.target.Target(env.target, host=env.target_host)
        relay.backend.te_compiler.get().clear()
        with autotvm.tophub.context(target), ExplicitResidencyDispatch([route]) as dispatch:
            with vta.build_config(opt_level=3,
                                  disabled_pass={"AlterOpLayout", "tir.CommonSubexprElimTIR"}):
                graph, lib, params = relay.build(program, target=target)
        if not dispatch.summary()["all_routes_scheduled"] or params:
            raise ValueError("mode4 fused route not matched")
        runner = graph_executor.create(graph, lib, tvm.device("ext_dev", 0))
        rows = []
        for seed in SEEDS:
            x, w, add = arrays(seed)
            runner.set_input("p09", x)
            runner.set_input("p15", w)
            runner.set_input("p24", add)
            simulator.clear_stats()
            runner.run()
            output_hash = sha(runner.get_output(0).numpy().tobytes())
            row = {"seed": seed, "correct": output_hash == expected[seed],
                   "actual_sha256": output_hash, "expected_sha256": expected[seed],
                   "fsim_counters": simulator.stats()}
            rows.append(row)
            if not row["correct"]:
                raise ValueError("mode4 FSim output differs from frozen exact reference")
        result = {"status": "MODE4_FSIM_PROFILE_COMPLETE", "board_contacted": False,
                  "scope": "cumulative local simulator counters, not SRAM peak or board traffic",
                  "rows": rows}
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
        result = {"status": "MODE4_FSIM_PROFILE_FAILED", "error": str(error),
                  "traceback": traceback.format_exc(), "board_contacted": False}
        save(args.output_dir / "failure.json", result)
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))
    if result["status"] == "MODE4_FSIM_PROFILE_FAILED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
