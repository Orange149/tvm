#!/usr/bin/env python3
"""Cross-compile exact locally qualified Relay controls, without board access."""

import argparse
import json
from pathlib import Path
import subprocess
import time
import traceback

import tvm
import vta

from c3_banerjee_local import (author_namespace, complete_compute, private_author_schedule,
                             SpecializeNode24Contexts, BanerjeeNode24Reuse, Archive, SYMBOL)
from c3_residency_tradeoff import read, write_new, digest
from c3_residency_tradeoff_evidence import check_runtime
from run_vta_c3_node24_partial_local import frozen_fusion, route, TILE
from vta.top.residency_dispatch import ExplicitResidencyDispatch
from build_vta_yolov3_tiny_relay_residency_dispatch_pair import cross_compiler


def run(root, destination, reuse):
    destination.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        proof = read(root / "local_qualification.json")
        label = "reuse" if reuse else "ordinary"
        prior = Path(proof["programs"][label]["directory"])
        if proof["status"] != "LOCAL_FULL_FUSION_AND_NATIVE_LIFETIME_AUDIT_PASSED" or \
                digest(prior / "lowered.json") != proof["programs"][label]["TIR_sha256"]:
            raise ValueError("full-fusion local qualification identity changed")
        source = read(prior / "result.json")
        if source["adapter_source_sha256"] != digest(Path(__file__).with_name("c3_banerjee_local.py")):
            raise ValueError("qualified adapter changed")
        runtime = check_runtime()
        if vta.get_env().TARGET != "axu5evb":
            raise ValueError("compile only for the frozen physical hardware target")
        namespace = author_namespace(root / "author_sources", safe_uop_order=True)
        _, _, contract = complete_compute(namespace)
        candidate = route("input_stationary", TILE)
        candidate["candidate_id"] = "Banerjee-functional-adaptation" if reuse else "ordinary-double-buffer-control"
        candidate["identity"].update(public_mode="original", implementation_mode=0,
                                    complete_config_entity=contract["config_entity"])
        config = dict(vta.build_config().config)
        passes = list(config["tir.add_lower_pass"])
        pos = next(i for i, (phase, _) in enumerate(passes) if phase >= 2)
        additions = [(2, SpecializeNode24Contexts())]
        if reuse:
            additions.append((2, BanerjeeNode24Reuse()))
        passes[pos:pos] = additions
        config["tir.add_lower_pass"] = passes
        archive = Archive(destination)
        env = vta.get_env()
        target = tvm.target.Target(env.target, host=env.target_host)
        tvm.relay.backend.te_compiler.get().clear()
        with private_author_schedule(namespace), ExplicitResidencyDispatch([candidate]) as dispatch:
            with tvm.transform.PassContext(opt_level=3, config=config, instruments=[archive],
                         disabled_pass={"AlterOpLayout", "tir.CommonSubexprElimTIR"}):
                graph, lib, params = tvm.relay.build(frozen_fusion(), target=target)
        if params or not dispatch.summary()["all_routes_scheduled"]:
            raise ValueError("exact four-parameter compile route failed")
        final_snapshot = next(row for row in reversed(archive.rows)
                              if row["pass"].endswith("CPUAccessRewrite") and row["when"] == "after")
        actual = tvm.ir.load_json((destination / (final_snapshot["file"] + ".json")).read_text())
        functions = [f for f in actual.functions.values() if isinstance(f, tvm.tir.PrimFunc)
                     and f.attrs is not None and "global_symbol" in f.attrs and str(f.attrs["global_symbol"]) == SYMBOL]
        original = tvm.ir.load_json((prior / "lowered.json").read_text())[SYMBOL]
        if len(functions) != 1 or not tvm.ir.structural_equal(original, functions[0], map_free_vars=True):
            raise ValueError("cross-compiled final target differs from qualified native TIR")
        write_new(destination / "node24_final.tir.json", json.loads(tvm.ir.save_json(functions[0])))
        write_new(destination / "graph.json", json.loads(graph))
        lib.export_library(str(destination / "graphlib.so"), fcompile=cross_compiler())
        elf = subprocess.run(["readelf", "-h", str(destination / "graphlib.so")], capture_output=True, text=True, check=True).stdout
        if "AArch64" not in elf:
            raise ValueError("candidate ELF is not for the physical board")
        write_new(destination / "result.json", {"status": "ARM_BINARY_MATCHES_LOCAL_FSIM_TARGET_FPGA_PENDING",
            "program": label, "board_contacted": False, "latency_ms": None, "seconds": time.monotonic() - started,
            "binary_sha256": digest(destination / "graphlib.so"), "TIR_sha256": digest(destination / "node24_final.tir.json"),
            "source_sha256": digest(__file__), "adapter_source_sha256": source["adapter_source_sha256"],
            "local_qualification_sha256": digest(root / "local_qualification.json"), "dispatch": dispatch.summary(),
            "contract": contract, "runtime": runtime, "elf_header": elf})
        print("ARM_BINARY_MATCHES_LOCAL_FSIM_TARGET_FPGA_PENDING: " + label, flush=True)
    except BaseException as error:
        write_new(destination / "failure.json", {"status": "LOCAL_CROSS_COMPILE_FAILURE_NO_BOARD_WORK",
            "error": str(error), "traceback": traceback.format_exc(), "seconds": time.monotonic() - started,
            "board_contacted": False, "latency_ms": None})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()
    run(args.root.resolve(), args.output_dir.resolve(), args.reuse)
