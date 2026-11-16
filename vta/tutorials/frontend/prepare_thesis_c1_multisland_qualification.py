#!/usr/bin/env python3
"""Build label-free same-qIR multi-island completion probes, not a ranking pool."""

import argparse
from collections import Counter
import os
from pathlib import Path
import shlex
import shutil
import sys

from prepare_thesis_c1_memory_inputs import CANONICAL, reference
from thesis_c1_memory_ablation import read_json, seal, verify_ref, write_new
from thesis_supplement_run import RunStore

FRONTEND = Path(__file__).resolve().parent
BASE = FRONTEND / "report_out/stage_tile_cotuning/c1_memory_ablation"
HISTORY = FRONTEND / "report_out/stage_tile_cotuning/thesis_supplement_experiments/results/06/20260916T122018Z_single_image_dcdeba41/raw/stock_compatible_history.log"
PROBES = {"two_islands": [(0, 3, "cpu"), (3, 8, "vta"), (8, 10, "cpu"),
                           (10, 18, "vta"), (18, 21, "cpu")],
          "three_islands": [(0, 3, "cpu"), (3, 5, "vta"), (5, 8, "cpu"),
                             (8, 10, "vta"), (10, 13, "cpu"), (13, 18, "vta"), (18, 21, "cpu")]}


def dma_calls(module):
    """Count all final fused TIR DMA call sites times static host-loop extents.

    Branch-dependent DMA is rejected rather than interpreted as unconditional.
    TIR uop loops do not multiply host API calls. This remains a prediction until
    checked against actual diagnostic counters.
    """
    import tvm
    from tvm import tir
    totals, analyzer = {}, tvm.arith.Analyzer()
    for global_var, function in module.functions.items():
        loops, counts = [], Counter()
        def before(node):
            if isinstance(node, tir.IfThenElse):
                calls = []
                tir.stmt_functor.post_order_visit(node, lambda n: calls.append(n) if
                    isinstance(n, tir.Call) and n.op.name == "tir.call_extern" and
                    n.args[0].value in ("VTALoadBuffer2D", "VTAStoreBuffer2D") else None)
                if calls:
                    raise ValueError("conditional final TIR DMA requires exact branch accounting")
            if isinstance(node, tir.For):
                extent = analyzer.simplify(node.extent)
                if not isinstance(extent, tir.IntImm):
                    raise ValueError("nonconstant final TIR host loop extent")
                loops.append(int(extent))
            if isinstance(node, tir.Call) and isinstance(node.op, tvm.ir.Op) and node.op.name == "tir.call_extern":
                name = node.args[0].value
                if name in ("VTALoadBuffer2D", "VTAStoreBuffer2D"):
                    import math
                    counts[name] += math.prod(loops)
        def after(node):
            if isinstance(node, tir.For):
                loops.pop()
        tir.stmt_functor.ir_transform(function.body, before, after, None)
        totals[global_var.name_hint] = dict(counts)
    return totals


def build(directory):
    import tvm
    from tvm import autotvm, relay
    import vta
    from thesis_canonical_graph_split import UnitGraph
    from thesis_single_image_c1_prepare import packed_stage
    from deploy_classification_stage_pipeline_native import export_stage_lib
    store = RunStore(directory)
    frozen = store.contract
    for key in ("canonical_qir", "history"):
        verify_ref(frozen[key])
    graph = UnitGraph(tvm.ir.load_json(Path(frozen["canonical_qir"]["path"]).read_text()))
    env = vta.get_env()
    @tvm.instrument.pass_instrument
    class Capture:
        def __init__(self, target):
            self.target, self.modules = target, []
        def run_after_pass(self, mod, info):
            if info.name == "tir.vta.CPUAccessRewrite":
                index = len(self.modules)
                (self.target / ("final_{:03}.json".format(index))).write_text(tvm.ir.save_json(mod))
                (self.target / ("final_{:03}.tir".format(index))).write_text(mod.script())
                self.modules.append(mod)
    rows = []
    specs = sorted({tuple(spec) for ranges in frozen["probes"].values() for spec in ranges})
    for begin, end, device in specs:
        name = "{}_{:02}_{:02}".format(device, begin, end - 1)
        target = store.directory / "raw/components" / name
        target.mkdir(parents=True)
        original, inputs, outputs = graph.stage(begin, end)
        mod = packed_stage(original, env) if device == "vta" else original
        (target / "lowering_qir.json").write_text(tvm.ir.save_json(mod))
        capture = Capture(target)
        build_target = tvm.target.Target(env.target if device == "vta" else env.target_vta_cpu, host=env.target_host)
        with autotvm.apply_history_best([frozen["history"]["path"]]), vta.build_config(
                opt_level=3, disabled_pass={"AlterOpLayout", "tir.CommonSubexprElimTIR"}, instruments=[capture]):
            factory = relay.build(mod, target=build_target)
        (target / "graph.json").write_text(factory.get_graph_json())
        (target / "params.bin").write_bytes(relay.save_param_dict(factory.get_params()))
        export_stage_lib(factory.get_lib(), target / "module.so")
        functions = {}
        for module in capture.modules:
            for function, counts in dma_calls(module).items():
                if function in functions and functions[function] != counts:
                    raise ValueError("ambiguous final lowered primitive")
                functions[function] = counts
        calls = 0
        for node in read_json(target / "graph.json")["nodes"]:
            if node["op"] != "tvm_op":
                continue
            function = node["attrs"]["func_name"]
            if function not in functions:
                raise ValueError("graph invocation is absent from final TIR: " + function)
            calls += sum(functions[function].values())
        if device == "vta" and calls <= 0:
            raise ValueError("VTA program has no final-TIR DMA provenance")
        summary = {"functions": functions, "vta_dma_calls": calls,
                   "scope": "final fused TIR static prediction; runtime counters still required"}
        write_new(target / "final_tir_manifest.json", summary)
        row = {"name": name, "begin": begin, "end": end, "device": device,
               "input_names": [p.name_hint for p in original["main"].params],
               "input_node_ids": [graph.ids[n] for n in inputs], "output_node_ids": [graph.ids[n] for n in outputs],
               "output_live_input_indices": [inputs.index(n) if n in inputs else -1 for n in outputs],
               "graph": reference(target / "graph.json"), "lib": reference(target / "module.so"),
               "params": reference(target / "params.bin"), "final_tir": reference(target / "final_tir_manifest.json"),
               "vta_dma_calls": calls, "lowered_modules": [reference(p) for p in sorted(target.glob("final_*.json")) if p.name != "final_tir_manifest.json"]}
        rows.append(row)
        write_new(target / "component.json", row)
        print(name + ": same-qIR compiled; final TIR DMA=" + str(calls), flush=True)
    write_new(store.directory / "components.json", seal({"schema": "c1_multisland_components_v1",
               "canonical_qir": frozen["canonical_qir"], "history": frozen["history"],
               "probes": frozen["probes"], "components": rows, "extra_quantize_calls": 0,
               "board_qualified": False, "ranking_pool": False}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--canonical-qir", type=Path, default=CANONICAL)
    parser.add_argument("--history", type=Path, default=HISTORY)
    parser.add_argument("--sdk", type=Path, default=Path("/home/orange/alinx_plsdk/install/environment-setup-aarch64-xilinx-linux"))
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return build(args.output_dir)
    store = RunStore.create(args.output_dir, {"run_id": args.output_dir.name, "stage": "C1_multisland_prepare",
            "campaign_id": "c1_memory_ablation", "canonical_qir": reference(args.canonical_qir),
            "history": reference(args.history), "probes": PROBES, "extra_quantize_calls": 0,
            "selection": "predeclared diagnostic probes; no M1 or candidate II selection",
            "source": reference(__file__), "sdk": reference(args.sdk),
            "environment": {k: os.environ.get(k) for k in ("TVM_NUM_THREADS", "OPENBLAS_NUM_THREADS", "PYTHONPATH", "VTA_TARGET", "VTA_HW_PATH")}})
    shutil.copy2(__file__, store.directory / "raw/executed_builder.py")
    script = store.directory / "raw/build.sh"
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--output-dir", str(store.directory)]
    script.write_text("set -e\nunset LD_LIBRARY_PATH\nsource " + shlex.quote(str(args.sdk.resolve())) +
                      "\nexec " + shlex.join(command) + "\n")
    result = store.execute(["bash", str(script)],
                           "compile_multisland_probes", "host_compilation", 1800)
    if result["status"] != "ok":
        write_new(store.directory / "FAILED.json", result)
        raise RuntimeError("compile failed; see " + str(store.directory / result["stderr"]))
    print(store.directory / "components.json")


if __name__ == "__main__":
    main()
