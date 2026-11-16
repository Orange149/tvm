"""Compile-only preparation of the frozen TPS synchronization experiment.

No target function, simulator invocation, RPC, or board operation occurs here.
All TVM imports are deliberately delayed until the frozen simulator DSO search
path is selected.  Historical artifacts are inputs and are never rewritten.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
HOLD = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout"
OFFICIAL = HOLD / "20261014_c3_official_valid_window_run01"
QUALIFIED = HOLD / "20261014_c3_tps_window_complete_run03"
TPS = HOLD / "20261013_c3_banerjee_baseline_completion_run02/local_final06/TPS-oc-reuse-ci4"
AUTHOR = HOLD / "20261012_c3_banerjee_double_buffer_local_run01/author_sources"
SELECTION = HOLD / "20261013_c3_banerjee_baseline_completion_run01/audit.json"
PROGRAM = "TPS-oc-reuse-ci4"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def canonical_config(value):
    """Config index is provenance, while every scheduling field is semantic."""
    result = json.loads(json.dumps(value))
    result.pop("index", None)
    return result


def complete_function(module, symbol):
    """Find the sole fused PrimFunc using the local DictAttrs interface."""
    from tvm import tir
    functions = [(gv, fn) for gv, fn in module.functions.items()
                 if isinstance(fn, tir.PrimFunc) and fn.attrs is not None
                 and "global_symbol" in fn.attrs and str(fn.attrs["global_symbol"]) == symbol]
    if len(functions) != 1 or len(module.functions) != 1:
        raise ValueError("FULL_FUSION_SYMBOL_OR_MODULE_SHAPE_CHANGED")
    return functions[0]


def compiled_module_contract(module, symbol, llvm):
    """Bind LLVM to the function owned by this module after JSON round trips.

    GlobalVar objects from another IRModule are not valid lookup keys, even
    when their names match.  Preparation and its read-only regressions share
    this entry point instead of transporting a parent module's GlobalVar.
    """
    from c3_official_window_baseline import buffer_abi
    from c3_official_window_workflow import llvm_alu_contract, llvm_cache_contract

    _, function = complete_function(module, symbol)
    return {"abi": buffer_abi(function),
            "alu_initializer_contract": llvm_alu_contract(function, llvm),
            "compiled_cache_contract": llvm_cache_contract(llvm)}


def ordered_reductions(events, expand, cells):
    """Exact per-observed-output operand sequence, independent of loop packing.

    The separate version/ACC certificate must already have accepted the trace.
    This comparison intentionally retains every product occurrence and order;
    hashing is only used after exact comparison for a compact stored identity.
    """
    versions, accumulators, outputs = {}, {}, {}
    for event in events:
        if event["kind"] == "load" and event["domain"] in (1, 2):
            for address, origin in cells(event):
                versions[event["domain"], address] = origin
        elif event["kind"] == "kernel":
            mode, reset = event["mode_reset"]
            for values in expand(event):
                dst, src, wgt = values[2:5]
                if reset:
                    accumulators[dst] = []
                elif mode == 0:
                    if dst not in accumulators:
                        raise ValueError("ORDERED_REDUCTION_WITHOUT_RESET")
                    accumulators[dst].append((versions[2, src], versions[1, wgt]))
        elif event["kind"] == "store":
            for address, output in cells(event):
                if output in outputs:
                    raise ValueError("ORDERED_REDUCTION_DUPLICATE_OBSERVATION")
                outputs[output] = tuple(accumulators[address])
    if not outputs:
        raise ValueError("ORDERED_REDUCTION_NO_OUTPUT")
    return outputs


def _semantic_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _bind(paths):
    return {str(Path(path).resolve()): sha(path) for path in paths}


def _hardware(root):
    path = root / "hardware_archive.json"
    hardware = json.loads(path.read_text())
    frozen = QUALIFIED / "hardware_archive.json"
    if sha(path) != sha(frozen):
        raise ValueError("HARDWARE_ARCHIVE_DIFFERS_FROM_QUALIFIED_SOURCE")
    for name, digest in hardware["source_bindings"].items():
        if sha(name) != digest:
            raise ValueError("HARDWARE_SOURCE_CHANGED:" + name)
    gap = hardware.get("verified_acc_interval")
    proof = hardware.get("gemm_feedback_proof", {})
    if not isinstance(gap, int) or gap <= 0 or gap != proof.get("conservative_minimum_issue_distance"):
        raise ValueError("FROZEN_HARDWARE_ACC_INTERVAL_UNRESOLVED")
    if proof.get("status") != "REVIEWED_BOUND_GEMM_ACC_FEEDBACK_RULE":
        raise ValueError("HARDWARE_GEMM_FEEDBACK_PROOF_NOT_BOUND")
    return hardware


def _compile(output, safe_order, hardware):
    import tvm
    from tvm import tir
    import vta
    from c3_banerjee_completion import CloseCallBoundary, ImprovedDoubleBuffer, SplitPaddedInputChannels
    from c3_banerjee_local import author_namespace, private_author_schedule
    from run_vta_c3_node24_partial_local import route, frozen_fusion, SYMBOL
    from c3_banerjee_baseline import Tile, paper_cost
    from c3_official_window_baseline import buffer_abi, normalize_llvm, compiler_sources, plain
    from c3_official_window_method import audit_function
    from c3_official_window_reference import from_workload, postprocess_from_output_terms, semantic_gates
    from c3_tps_window_method import acc_spacing
    from c3_valid_window import certify_trace, expanded_uops, dma_cells
    from vta.top.residency_dispatch import ExplicitResidencyDispatch

    output.mkdir(exist_ok=False)
    selection = json.loads(SELECTION.read_text())["selected_oc_thread"]
    tile = dict(selection["tile"], ci=4)
    expected_tile = {"b": 1, "h": 14, "w": 14, "ci": 4, "co": 4,
                     "oc_nthread": 2, "h_nthread": 1}
    if tile != expected_tile:
        raise ValueError("FROZEN_TILE_OR_THREADS_CHANGED")
    env = vta.get_env()
    selected = paper_cost(Tile(**tile), {"INP": env.INP_BUFF_SIZE,
                          "WGT": env.WGT_BUFF_SIZE, "ACC": env.ACC_BUFF_SIZE})
    if not selected["necessary_conditions_passed"] or selected["paper_score_bytes"] != selection["paper_score_bytes"]:
        raise ValueError("FROZEN_SELECTION_NO_LONGER_ANALYTICAL_TIE")
    candidate = route("input_stationary", tuple(tile[k] for k in ("b", "h", "w", "ci", "co")))
    candidate["candidate_id"] = "Banerjee-completion-" + PROGRAM + "-" + selected["id"]
    candidate["identity"].update(public_mode="original", implementation_mode=0)
    for entity in candidate["identity"]["complete_config_entity"]["entity"]:
        if entity[0] in ("oc_nthread", "h_nthread"):
            entity[2] = tile[entity[0]]
    sealed_contract = json.loads((TPS / "contract.json").read_text())
    if selected != sealed_contract["selection"] or candidate != sealed_contract["route"]:
        raise ValueError("FROZEN_ROUTE_RECONSTRUCTION_CHANGED")
    namespace = author_namespace(AUTHOR, safe_uop_order=safe_order)
    actual_configs = []
    original_schedule = namespace["schedule_conv2d_packed"]

    def schedule(cfg, outs):
        if bool(getattr(cfg, "is_fallback", False)):
            raise ValueError("FALLBACK_SCHEDULE_CONFIG_REJECTED")
        if hasattr(cfg, "to_json_dict"):
            actual = plain(cfg.to_json_dict())
            if canonical_config(actual) != canonical_config(candidate["identity"]["complete_config_entity"]):
                raise ValueError("ACTUAL_SCHEDULE_CONFIG_DIFFERS")
            actual_configs.append(actual)
        return original_schedule(cfg, outs)

    namespace["schedule_conv2d_packed"] = schedule

    @tvm.instrument.pass_instrument
    class Archive:
        def __init__(self):
            self.rows, self.pass_order = [], []

        def save(self, module, info, when):
            if when == "before":
                self.pass_order.append(info.name)
            if not any(part in info.name for part in (
                    "StorageFlatten", "LiftAttrScope", "InjectVirtualThread", "InjectDMAIntrin",
                    "InjectCopyIntrin", "InjectCoProcSync", "FoldUopLoop", "CPUAccessRewrite",
                    "ImprovedDoubleBuffer", "CloseCallBoundary", "SplitPaddedInputChannels")):
                return
            name = "%03d_%s_%s" % (len(self.rows), when, info.name.replace(".", "_"))
            (output / (name + ".json")).write_text(tvm.ir.save_json(module))
            (output / (name + ".tir")).write_text(module.script())
            self.rows.append({"file": name, "pass": info.name, "when": when})

        def run_before_pass(self, module, info):
            self.save(module, info, "before")

        def run_after_pass(self, module, info):
            self.save(module, info, "after")

    archive = Archive()
    config = dict(vta.build_config().config)
    passes = list(config["tir.add_lower_pass"])
    before_dma = next(i for i, (_, p) in enumerate(passes) if p.info.name == "tir.InjectCopyIntrin")
    passes.insert(before_dma, (1, SplitPaddedInputChannels()))
    position = next(i for i, (phase, _) in enumerate(passes) if phase >= 2)
    passes[position:position] = [(2, CloseCallBoundary()), (2, ImprovedDoubleBuffer())]
    config["tir.add_lower_pass"] = passes
    disabled = {"AlterOpLayout", "tir.CommonSubexprElimTIR"}
    target = tvm.target.Target(env.target, host=env.target_host)
    tvm.relay.backend.te_compiler.get().clear()
    with private_author_schedule(namespace), ExplicitResidencyDispatch([candidate]) as dispatch:
        with tvm.transform.PassContext(opt_level=3, config=config, instruments=[archive], disabled_pass=disabled):
            graph, relay_library, params = tvm.relay.build(frozen_fusion(), target=target)
    if params or not dispatch.summary()["all_routes_scheduled"] or not actual_configs:
        raise ValueError("FULL_RELAY_FUSION_DID_NOT_USE_EXACT_CONFIG")
    # The Relay library is never invoked or exported.  Only its captured final
    # function is compiled below under the already-qualified final-IR context.
    del relay_library
    def captured(suffix, when):
        matches = [row for row in archive.rows if row["pass"].endswith(suffix) and row["when"] == when]
        if len(matches) != 1:
            raise ValueError("NONUNIQUE_COMPLETE_FUSION_PASS_CAPTURE:" + suffix)
        return tvm.ir.load_json((output / (matches[0]["file"] + ".json")).read_text())
    parent = captured("FoldUopLoop", "after")
    before_cpu = captured("CPUAccessRewrite", "before")
    direct_final = captured("CPUAccessRewrite", "after")
    if not tvm.ir.structural_equal(parent, before_cpu, map_free_vars=False):
        raise ValueError("UNCAPTURED_PASS_BETWEEN_PARENT_AND_CPU_REWRITE")
    restored = tvm.ir.load_json(tvm.ir.save_json(parent))
    if not tvm.ir.structural_equal(parent, restored, map_free_vars=False):
        raise ValueError("PARENT_JSON_ROUNDTRIP_CHANGED")
    final = vta.transform.CPUAccessRewrite()(restored)
    if not tvm.ir.structural_equal(final, direct_final, map_free_vars=False):
        raise ValueError("REMAINING_CPU_REWRITE_ROUNDTRIP_CHANGED")
    gv, function = complete_function(parent, SYMBOL)
    abi = buffer_abi(function)
    if len(abi) != 4 or any(row["kind"] != "buffer" for row in abi):
        raise ValueError("FOUR_BUFFER_ABI_REQUIRED")
    audit = audit_function(function)
    semantic = certify_trace(audit["events"])
    sequence = ordered_reductions(audit["events"], expanded_uops, dma_cells)
    spacing = acc_spacing(audit["events"], hardware["verified_acc_interval"])
    workload = ["conv2d_packed.vta", ["TENSOR", abi[0]["shape"], "int8"],
                ["TENSOR", abi[1]["shape"], "int8"], [1, 1], [1, 1, 1, 1],
                [1, 1], "NCHW1n16c", "int32"]
    reference = from_workload(workload, abi, postprocess_from_output_terms(semantic["output_terms"], abi))
    independent = semantic_gates(reference, semantic["output_terms"])
    with tvm.transform.PassContext(opt_level=3, disabled_pass=disabled):
        lib = tvm.build(final, target=target)
        direct_lib = tvm.build(direct_final, target=target)
    llvm = lib.get_source("ll")
    if normalize_llvm(llvm) != normalize_llvm(direct_lib.get_source("ll")):
        raise ValueError("ROUNDTRIP_LLVM_OR_INITIALIZER_SHARING_CHANGED")
    (output / "parent.json").write_text(tvm.ir.save_json(parent))
    (output / "parent.tir").write_text(parent.script())
    (output / "final.json").write_text(tvm.ir.save_json(final))
    (output / "final.tir").write_text(final.script())
    (output / "module.ll").write_text(llvm)
    write(output / "graph.json", json.loads(graph))
    write(output / "events.json", audit["events"])
    write(output / "reference_contract.json", reference.contract())
    write(output / "independent_semantics.json", independent)
    compiled = compiled_module_contract(final, SYMBOL, llvm)
    if compiled["abi"] != abi:
        raise ValueError("FINAL_MODULE_ABI_CHANGED")
    write(output / "alu_initializer_contract.json", compiled["alu_initializer_contract"])
    write(output / "compiled_cache_contract.json", compiled["compiled_cache_contract"])
    write(output / "logical_certificate.json", {
        "status": "LOGICAL_AND_PIPELINE_GATES_PASSED_NATIVE_PHYSICAL_PENDING",
        "counts": audit["counts"], "statistics": audit["statistics"],
        "semantics": {k: v for k, v in semantic.items() if k not in ("consumer_versions", "output_terms")},
        "acc_spacing": spacing, "ordered_reduction_outputs": len(sequence),
        "ordered_reduction_sha256": _semantic_digest(sorted(sequence.items())),
        "roundtrip_parent_final_and_llvm_equal": True, "target_calls": 0})
    write(output / "compilation_context.json", {
        "program": PROGRAM, "variant": "O_rebuilt" if safe_order else "R",
        "safe_uop_order": safe_order, "selection": selected, "route": candidate,
        "actual_config_entities": actual_configs, "dispatch": dispatch.summary(),
        "environment": env.cfg_dict, "target": str(target), "host_target": str(target.host),
        "relay_opt_level": 3, "disabled_pass": sorted(disabled),
        "relay_config": {str(k): str(v) for k, v in config.items() if k != "tir.add_lower_pass"},
        "extra_lower_passes": [[int(phase), p.info.name] for phase, p in passes],
        "observed_pass_order": archive.pass_order, "captured_passes": archive.rows,
        "final_ir_build_context": {"opt_level": 3, "disabled_pass": sorted(disabled)},
        "compiler_sources": compiler_sources(), "target_calls": 0,
        "author_thread_extension_mechanically_ported": False,
        "adaptations_retained": ["cthread", "SplitPaddedInputChannels", "CloseCallBoundary", "ImprovedDoubleBuffer"],
        "uop_504_is_prediction_not_acceptance_condition": True})
    descriptor = {"variant": "O_rebuilt" if safe_order else "R", "program": PROGRAM,
        "parent_module": str(output / "parent.json"), "final_module": str(output / "final.json"),
        "parent_module_sha256": sha(output / "parent.json"), "final_module_sha256": sha(output / "final.json"),
        "global_var": str(gv.name_hint), "symbol": SYMBOL, "abi": abi, "workload": workload,
        "events": str(output / "events.json"), "reference_contract": str(output / "reference_contract.json"),
        "logical_certificate": str(output / "logical_certificate.json"),
        "independent_semantics": str(output / "independent_semantics.json"),
        "hardware_interval": hardware["verified_acc_interval"], "roundtrip_passed": True,
        "module_llvm": str(output / "module.ll"), "module_llvm_sha256": sha(output / "module.ll"),
        "target_calls": 0}
    if not safe_order:
        lib.export_library(str(output / "fsim_module.so"))
        descriptor.update(module_path=str(output / "fsim_module.so"),
                          module_sha256=sha(output / "fsim_module.so"))
    descriptor["artifact_bindings"] = _bind(sorted(path for path in output.iterdir() if path.is_file()))
    write(output / "descriptor.json", descriptor)
    return {"descriptor": descriptor, "parent": parent, "final": final, "audit": audit,
            "semantic": semantic, "sequence": sequence, "llvm": llvm}


def prepare_worker(out):
    """Create ``out/prepare`` and return descriptors; execute zero targets."""
    out = Path(out).resolve()
    runtime = OFFICIAL / "runtime_local_final"
    os.environ["TVM_LIBRARY_PATH"] = str(runtime)
    os.environ["C3_OFFICIAL_WINDOW_COLLECTION"] = "detailed"
    os.environ["VTA_REPLAY_POLICY"] = "disabled"
    os.environ.pop("VTA_COMMAND_MANIFEST_ID", None)
    import tvm
    import vta
    from vta.testing import simulator
    from c3_tps_aggregation_prepare import compare_initializer_bindings
    hardware = _hardware(out)
    loaded = Path(simulator.LIBS[0]._name).resolve()
    contract = json.loads((OFFICIAL / "runtime_contract.json").read_text())
    if loaded != (runtime / "libvta_fsim.so").resolve() or sha(loaded) != contract["runtime"][0]["binary_sha256"]:
        raise ValueError("WRONG_PREPARATION_RUNTIME_LOADED")
    preparation = out / "prepare"
    preparation.mkdir(exist_ok=False)
    inputs = [SELECTION, TPS / "contract.json", TPS / "013_after_tir_vta_FoldUopLoop.json",
              TPS / "015_after_tir_vta_CPUAccessRewrite.json", QUALIFIED / "TPS_B_detailed/parent.json",
              QUALIFIED / "TPS_B_detailed/final.json", QUALIFIED / "TPS_B_detailed/module.ll",
              QUALIFIED / "TPS_B_detailed/alu_initializer_contract.json", QUALIFIED / "hardware_archive.json",
              OFFICIAL / "runtime_contract.json", loaded]
    inputs += [AUTHOR / name for name in ("vta_conv2d.py", "inject_virtual_thread.cc", "coproc_sync.cc", "runtime.cc")]
    bindings = _bind(inputs)
    write(preparation / "input_bindings.json", bindings)
    with vta.Environment(dict(vta.get_env().cfg_dict, TARGET="sim")):
        original = _compile(preparation / "O_rebuilt", True, hardware)
        for label, suffix in (("parent", "parent.json"), ("final", "final.json")):
            sealed = tvm.ir.load_json((QUALIFIED / "TPS_B_detailed" / suffix).read_text())
            if not tvm.ir.structural_equal(original[label], sealed, map_free_vars=True):
                raise ValueError("O_REBUILT_DIFFERS_FROM_SEALED_" + label.upper())
        initializers = compare_initializer_bindings(
            (QUALIFIED / "TPS_B_detailed/module.ll").read_text(), original["llvm"],
            json.loads((QUALIFIED / "TPS_B_detailed/alu_initializer_contract.json").read_text()),
            json.loads((preparation / "O_rebuilt/alu_initializer_contract.json").read_text()))
        write(preparation / "O_rebuild_contract.json", {
            "status": "SEALED_O_AND_REBUILT_PARENT_FINAL_STRUCTURALLY_EQUAL",
            "normalization": "bijective bound/free variable mapping preserving sharing; all semantic fields retained",
            "bound_initializer_and_cache_sharing_comparison": initializers,
            "kernel_identity_shared_relations_preserved": True, "target_calls": 0,
            "O_shared_library_exported": False})
        restored = _compile(preparation / "R", False, hardware)
    o, r = original["audit"]["events"], restored["audit"]["events"]
    if original["descriptor"]["abi"] != restored["descriptor"]["abi"]:
        raise ValueError("R_CHANGED_FOUR_ARGUMENT_ABI")
    tensors = lambda events: [e for e in events if e["kind"] in ("load", "store")]
    controls = lambda events: [e for e in events if e["kind"] in ("push", "pop", "sync")]
    if tensors(o) != tensors(r):
        raise ValueError("R_CHANGED_ORDERED_TENSOR_REQUESTS")
    if controls(o) != controls(r):
        raise ValueError("R_CHANGED_NOTIFICATION_SEQUENCE")
    if original["semantic"]["output_terms"] != restored["semantic"]["output_terms"]:
        raise ValueError("R_CHANGED_OUTPUT_PRODUCTS_OR_POSTPROCESS")
    if original["sequence"] != restored["sequence"]:
        raise ValueError("R_CHANGED_PER_ACC_ORDERED_REDUCTION")
    for name, digest in bindings.items():
        if sha(name) != digest:
            raise ValueError("PREPARATION_INPUT_CHANGED:" + name)
    comparison = {"status": "O_R_EXACT_SEMANTIC_AND_REQUEST_COMPARISON_PASSED",
        "same_four_buffer_abi": True, "ordered_tensor_requests_equal": True,
        "notification_sequence_equal": True, "complete_output_products_and_tail_equal": True,
        "each_output_ordered_reduction_exactly_equal": True,
        "O_counts": original["audit"]["counts"], "R_counts": restored["audit"]["counts"],
        "O_maximum_base_uops": max(len(e["uops"]) for e in o if e["kind"] == "kernel"),
        "R_maximum_base_uops": max(len(e["uops"]) for e in r if e["kind"] == "kernel"),
        "prediction_504_required": False, "improvement_is_baseline_restoration_not_innovation": True,
        "native_physical_and_fsim_qualification_pending": True}
    write(preparation / "O_R_comparison.json", comparison)
    result = {"status": "COMPILE_AND_LOGICAL_PREPARATION_PASSED_QUALIFICATION_PENDING",
        "programs": {"O_rebuilt": original["descriptor"], "R": restored["descriptor"]},
        "input_bindings": bindings, "comparison": comparison,
        "hardware_archive": str(out / "hardware_archive.json"),
        "target_calls": 0, "new_fsim_calls": 0, "board_contacted": False,
        "S_built": False, "performance_conclusion": None}
    write(preparation / "result.json", result)
    return result
