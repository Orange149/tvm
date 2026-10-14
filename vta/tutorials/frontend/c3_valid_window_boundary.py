#!/usr/bin/env python3
"""Read-only VTA boundary audit: replay existing passes on one fresh parent IR."""

import argparse
import hashlib
import json
from pathlib import Path

import tvm
from tvm import autotvm, relay, tir
import vta

from run_vta_c3_node24_partial_local import frozen_fusion, route, SYMBOL
from vta.top.residency_dispatch import ExplicitResidencyDispatch
from vta.top.vta_node24_partial import (
    Node24WeightSlots, Node24GroupWeightDMA, Node24SpatialBoundaryBridge,
    SplitNode24InputDMA, Node24GroupGEMM,
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def summarize(function):
    kernels, attr_counts, calls = [], {}, {}

    def visit(node):
        if isinstance(node, tir.Call):
            name = node.op.name if isinstance(node.op, tvm.ir.Op) else str(node.op)
            if name == "tir.call_extern":
                name = str(node.args[0])
            calls[name] = calls.get(name, 0) + 1
        if isinstance(node, tir.AttrStmt):
            attr_counts[node.attr_key] = attr_counts.get(node.attr_key, 0) + 1
            if node.attr_key == "coproc_uop_scope":
                details = {"value": str(node.value), "base_loop_extents": [], "calls": {}}

                def kernel_visit(item):
                    if isinstance(item, tir.For):
                        details["base_loop_extents"].append(str(item.extent))
                    if isinstance(item, tir.Call):
                        name = item.op.name if isinstance(item.op, tvm.ir.Op) else str(item.op)
                        if name == "tir.call_extern":
                            name = str(item.args[0])
                        details["calls"][name] = details["calls"].get(name, 0) + 1
                tir.stmt_functor.post_order_visit(node.body, kernel_visit)
                kernels.append(details)
    tir.stmt_functor.post_order_visit(function.body, visit)
    return {"attributes": attr_counts, "calls": calls, "kernels": kernels}


@tvm.instrument.pass_instrument
class Archive:
    def __init__(self, output):
        self.output, self.events, self.counter = output, [], 0

    def capture(self, mod, direction, info):
        functions = [f for f in mod.functions.values() if isinstance(f, tir.PrimFunc)
                     and f.attrs is not None and "global_symbol" in f.attrs
                     and str(f.attrs["global_symbol"]) == SYMBOL]
        if len(functions) != 1:
            return
        name = str(info.name)
        interest = ("LiftAttrScope", "Node24WeightSlots", "Node24GroupWeightDMA",
                    "Node24SpatialBoundaryBridge", "LowerDeviceStorageAccessInfo",
                    "FoldUopLoop", "CPUAccessRewrite", "EarlyRewrite")
        if not any(s in name for s in interest):
            return
        f = functions[0]
        stem = "%02d_%s_%s" % (self.counter, direction, name.replace(".", "_"))
        self.counter += 1
        raw = tvm.ir.save_json(f)
        (self.output / (stem + ".tir.json")).write_text(raw)
        (self.output / (stem + ".tir")).write_text(f.script())
        self.events.append({"direction": direction, "pass": name, "stem": stem,
                            "sha256": digest(raw.encode()), "summary": summarize(f)})

    def run_before_pass(self, mod, info):
        self.capture(mod, "before", info)

    def run_after_pass(self, mod, info):
        self.capture(mod, "after", info)


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    archive = Archive(output)
    cfg = dict(vta.get_env().cfg_dict, TARGET="sim")
    with vta.Environment(cfg):
        env = vta.get_env()
        target = tvm.target.Target(env.target, host=env.target_host)
        candidate = route("node24_partial_input_experimental")
        relay.backend.te_compiler.get().clear()
        with autotvm.tophub.context(target), ExplicitResidencyDispatch([candidate]) as dispatch:
            passes = list(vta.build_config().config["tir.add_lower_pass"])
            passes.insert(1, (1, SplitNode24InputDMA()))
            passes.extend([(2, Node24WeightSlots(13, True)),
                           (2, Node24GroupWeightDMA()),
                           (2, Node24SpatialBoundaryBridge())])
            with tvm.transform.PassContext(opt_level=3,
                    disabled_pass={"AlterOpLayout", "tir.CommonSubexprElimTIR"},
                    instruments=[archive], config={"tir.add_lower_pass": passes}):
                relay.build(frozen_fusion(), target=target)
        if not dispatch.summary()["all_routes_scheduled"]:
            raise RuntimeError("fixed route did not schedule")
        save(output / "archive.json", archive.events)
        parent_row = next(row for row in archive.events
                          if row["direction"] == "before" and row["pass"].endswith("FoldUopLoop"))
        parent = tvm.IRModule({"main": tvm.ir.load_json((output / (parent_row["stem"] + ".tir.json")).read_text())})
        default = vta.transform.FoldUopLoop()(parent)
        folded = default["main"]
        (output / "parent_post_fold_pre_cpu.tir.json").write_text(tvm.ir.save_json(folded))
        (output / "parent_post_fold_pre_cpu.tir").write_text(folded.script())
        lift_uop = lambda: tir.transform.LiftAttrScope("coproc_uop_scope")
        lift_scope = lambda: tir.transform.LiftAttrScope("coproc_scope")
        cases = {
            "default_fold": [vta.transform.FoldUopLoop()],
            "lift_uop_then_fold": [lift_uop(), vta.transform.FoldUopLoop()],
            "lift_scope_uop_then_fold": [lift_scope(), lift_uop(), vta.transform.FoldUopLoop()],
            "lift_uop_scope_uop_then_fold": [lift_uop(), lift_scope(), lift_uop(), vta.transform.FoldUopLoop()],
            "fold_then_lift_uop": [vta.transform.FoldUopLoop(), lift_uop()],
            "fold_then_lift_scope_uop": [vta.transform.FoldUopLoop(), lift_scope(), lift_uop()],
            "fixed_group_gemm": [vta.transform.FoldUopLoop(), Node24GroupGEMM()],
        }
        rows = {}
        for label, pipeline in cases.items():
            try:
                mod = tvm.transform.Sequential(pipeline)(parent)
                raw = tvm.ir.save_json(mod["main"])
                (output / (label + ".tir.json")).write_text(raw)
                (output / (label + ".tir")).write_text(mod["main"].script())
                rows[label] = {"status": "PASS_APPLIED", "sha256": digest(raw.encode()),
                    "summary": summarize(mod["main"]),
                    "structural_equal_default": bool(tvm.ir.structural_equal(mod["main"], folded)),
                    "structural_equal_fixed_group": bool(tvm.ir.structural_equal(mod["main"],
                        Node24GroupGEMM()(default)["main"]))}
            except Exception as error:
                rows[label] = {"status": "REJECTED", "error": str(error)}
        save(output / "comparison.json", {"same_parent": parent_row,
            "post_fold_parent": str(output / "parent_post_fold_pre_cpu.tir.json"),
            "program_contract": {"symbol": SYMBOL, "tile": [1, 2, 14, 1, 1],
                "resident_groups": 13, "group_count": 16, "target": env.TARGET,
                "board_contacted": False}, "cases": rows})
    print(json.dumps({label: {"status": row["status"],
        "scopes": row.get("summary", {}).get("attributes", {}).get("coproc_uop_scope"),
        "structural_equal_fixed_group": row.get("structural_equal_fixed_group")}
        for label, row in rows.items()}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
