#!/usr/bin/env python3
"""Cross-build the four sealed local Q functions; never contact a board."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import struct
import subprocess
import sys
import traceback

SDK = Path("/home/orange/alinx_plsdk/install/environment-setup-aarch64-xilinx-linux")
SYMBOL = "tvmgen_default_fused_nn_conv2d_add_right_shift_clip_cast"
CORE_SHA = "9e70184b7838ad9fc44917ada361a181dd16382c3d1d33fe1733e901563600b2"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def initializers(llvm):
    """Canonicalize names only; retain all integer operations and CFG edges."""
    result = []
    pattern = r"^define[^\n]*(@__tvm_static_init_lambda[^ (]*)\([^\n]*\) [^\n]*\{\n(.*?)^\}"
    for name, body in re.findall(pattern, llvm, re.M | re.S):
        names, rows = {}, []
        def local(match):
            value = match[0]
            return names.setdefault(value, "%%v%d" % len(names))
        for line in body.splitlines():
            line = line.split(";", 1)[0].strip()
            if not line:
                continue
            # Profiling/tbaa IDs refer to separately numbered metadata tables.
            line = re.sub(r", ![A-Za-z_]+ !\d+", "", line)
            label = re.fullmatch(r"([A-Za-z0-9_.$]+):", line)
            if label:
                line = local(re.match(r".*", "%" + label[1])) + ":"
            else:
                line = re.sub(r"%[A-Za-z0-9_.$]+", local, line)
            rows.append(line)
        canonical = "\n".join(rows)
        result.append({"initializer": name, "body": canonical,
            "sha256": hashlib.sha256(canonical.encode()).hexdigest()})
    if not result:
        raise ValueError("no LLVM cached initializer bodies found")
    return result


def llvm_contract(llvm, scope_count):
    groups, calls, control = {}, [], []
    current = None
    for line in llvm.splitlines():
        label = re.match(r"^([A-Za-z0-9_.$]+):", line)
        if label:
            current = label[1]
        if "call" not in line or "@VTAPushGEMMOp" not in line:
            continue
        match = re.search(r"@VTAPushGEMMOp\(([^)]*)\)", line)
        fields = match[1].split(", ") if match else []
        size = re.fullmatch(r"i32 (\d+)", fields[-1]) if len(fields) == 4 else None
        if size is None or int(size[1]) not in (0, 4) or not fields[2].startswith("ptr "):
            raise ValueError("unsupported ARM LLVM cache closure ABI")
        handle, initializer = fields[0].split()[-1], fields[1].split()[-1]
        closure = fields[2].split()[-1]
        key = (initializer, closure, int(size[1]))
        if handle in groups and groups[handle] != key:
            raise ValueError("ARM LLVM static handle changes initializer/closure/signature")
        groups[handle] = key
        calls.append({"cache_handle": handle, "initializer": initializer,
                      "signature_pointer": closure, "signature_bytes": int(size[1])})
        control.append({"basic_block": current, "call": line.strip()})
    if len(groups) != scope_count:
        raise ValueError("ARM TIR scopes differ from distinct LLVM cache handles")
    alu = []
    for line in llvm.splitlines():
        if "call" not in line or "@VTAUopPush(i32 1," not in line:
            continue
        match = re.search(r"@VTAUopPush\(([^)]*)\)", line)
        values = [re.fullmatch(r"i32 (-?\d+)", item) for item in match[1].split(", ")]
        if len(values) != 8 or any(value is None for value in values):
            raise ValueError("unsupported ARM LLVM ALU initializer field")
        alu.append([int(value[1]) for value in values])
    return {"distinct_gemm_cache_handles": len(groups), "gemm_calls": calls,
        "control_flow_locations": control, "alu_uop_fields": alu}


def worker(source, output):
    import tvm
    from tvm import tir
    from tvm.contrib import cc
    import vta
    from build_vta_yolov3_tiny_relay_residency_dispatch_pair import cross_compiler

    output.mkdir(parents=True, exist_ok=False)
    try:
        contract = json.loads((source / "contract.json").read_text())
        result = json.loads((source / "result.json").read_text())
        if contract["aggregation_source_sha256"] != CORE_SHA or digest(source / "aggregation_source.py") != CORE_SHA:
            raise ValueError("qualified aggregation source identity changed")
        if digest(source / "node24_final.tir.json") != result["final_tir_sha256"] or contract["ABI"] != ["X", "W", "ADD", "Y"]:
            raise ValueError("sealed simulation TIR/ABI mismatch")
        if len(result["calls"]) != 7 or not all(all(row[k] for k in ("correct", "same_handles", "arguments_unchanged")) for row in result["calls"]):
            raise ValueError("simulation qualification incomplete")
        function = tvm.ir.load_json((source / "node24_final.tir.json").read_text())
        if len(function.params) != 4 or str(function.attrs["global_symbol"]) != SYMBOL:
            raise ValueError("ARM function ABI changed")
        raw = tvm.ir.save_json(function)
        (output / "node24_final.tir.json").write_text(raw)
        (output / "node24_final.tir").write_text(function.script())
        loaded = tvm.ir.load_json((output / "node24_final.tir.json").read_text())
        if not tvm.ir.structural_equal(function, loaded, map_free_vars=False):
            raise ValueError("ARM loaded final TIR differs structurally from FSim input")
        abi = [{"parameter": i, "shape": [int(value) for value in function.buffer_map[param].shape],
                "dtype": str(function.buffer_map[param].dtype)} for i, param in enumerate(function.params)]
        cfg = dict(vta.get_env().cfg_dict, TARGET="axu5evb")
        with vta.Environment(cfg):
            target = tvm.target.Target(vta.get_env().target, host=vta.get_env().target_host)
            lib = tvm.build(tvm.IRModule({SYMBOL: function}), target=target)
            llvm = lib.get_source("ll")
            if 'target triple = "aarch64-linux-gnu"' not in llvm:
                raise ValueError("LLVM module is not frozen aarch64 target")
            (output / "module.ll").write_text(llvm)
            scopes = []
            tir.stmt_functor.post_order_visit(function.body, lambda node: scopes.append(node)
                if isinstance(node, tir.AttrStmt) and node.attr_key == "coproc_uop_scope"
                and isinstance(node.value, tir.StringImm) and node.value.value == "VTAPushGEMMOp" else None)
            arm = llvm_contract(llvm, len(scopes))
            sim = json.loads((source / "cache_identity_llvm.json").read_text())
            sim_alu = json.loads((source / "alu_fields_llvm.json").read_text())
            if arm["distinct_gemm_cache_handles"] != sim["distinct_gemm_cache_handles"] or sorted(arm["alu_uop_fields"]) != sorted(sim_alu["expected"]):
                raise ValueError("ARM cache census or ALU fields differ from qualified FSim")
            arm_init = initializers(llvm)
            sim_init = initializers((source / "module.ll").read_text())
            same_initializers = Counter(row["sha256"] for row in arm_init) == Counter(row["sha256"] for row in sim_init)
            write(output / "llvm_contract.json", {"ARM": arm, "ARM_initializers": arm_init,
                "FSim_initializers": sim_init, "canonical_initializer_bodies_equal": same_initializers,
                "canonicalization": "rename local SSA/labels and drop metadata reference IDs/comments only; retain integer operations, operands, calls and CFG edges",
                "final_tir_structural_equal": True, "ABI": abi,
                "ARM_triple": "aarch64-linux-gnu", "FSim_triple": "x86_64-pc-linux-gnu"})
            if not same_initializers:
                raise ValueError("target-specific LLVM initializer body differs; explicit semantic review required")
            records = []
            original_compile = cc._linux_compile
            def record_compile(destination, objects, options, compile_cmd, compile_shared=False):
                inputs = [objects] if isinstance(objects, str) else list(objects)
                argv = [compile_cmd]
                if compile_shared or str(destination).endswith((".so", ".dylib")):
                    argv += ["-shared", "-fPIC"]
                argv += ["-o", destination, *inputs, *(options or [])]
                retained = []
                folder = output / "link_inputs"
                folder.mkdir(exist_ok=True)
                for index, name in enumerate(inputs):
                    path = Path(name)
                    copied = folder / ("%02d_" % index + path.name)
                    shutil.copyfile(path, copied)
                    retained.append({"original": str(path), "retained": str(copied), "sha256": digest(copied)})
                records.append({"argv": argv, "inputs": retained})
                return original_compile(destination, objects, options, compile_cmd, compile_shared)
            cc._linux_compile = record_compile
            try:
                lib.export_library(str(output / "graphlib.so"), fcompile=cross_compiler())
            finally:
                cc._linux_compile = original_compile
            write(output / "compiler_invocations.json", records)
        blob = (output / "graphlib.so").read_bytes()
        if blob[:4] != b"\x7fELF" or blob[4:6] != bytes([2, 1]) or struct.unpack_from("<H", blob, 18)[0] != 183:
            raise ValueError("cross-compiled DSO is not little-endian ELF64 AArch64")
        cxx = shlex.split(os.environ["CXX"])
        compiler = Path(shutil.which(cxx[0])).resolve()
        version = subprocess.run([str(compiler), "--version"], check=True, stdout=subprocess.PIPE, text=True).stdout
        dumpmachine = subprocess.run([str(compiler), "-dumpmachine"], check=True, stdout=subprocess.PIPE, text=True).stdout.strip()
        readelf = shutil.which(os.environ.get("READELF", "aarch64-xilinx-linux-readelf"))
        exported = subprocess.run([readelf, "--dyn-syms", "--wide", str(output / "graphlib.so")], check=True, stdout=subprocess.PIPE, text=True).stdout
        if SYMBOL not in exported:
            raise ValueError("full four-parameter fusion symbol not exported")
        (output / "dynamic_symbols.txt").write_text(exported)
        subfiles = []
        for key in ("OECORE_TARGET_SYSROOT", "OECORE_NATIVE_SYSROOT"):
            folder = Path(os.environ[key]) / "environment-setup.d"
            subfiles.extend({"path": str(path), "sha256": digest(path)} for path in sorted(folder.glob("*.sh")))
        write(output / "build.json", {"program": contract["program"],
            "status": "ARM_MATCHES_FINAL_FSIM_IR_AND_INITIALIZER_CONTENT_FPGA_PENDING",
            "source": str(source), "source_contract_sha256": digest(source / "contract.json"),
            "source_final_tir_sha256": result["final_tir_sha256"], "loaded_arm_tir_sha256": digest(output / "node24_final.tir.json"),
            "aggregation_source_sha256": CORE_SHA, "ABI": abi, "target": "axu5evb", "hardware_cfg": cfg,
            "sdk": str(SDK), "sdk_sha256": digest(SDK), "sdk_subscripts": subfiles,
            "CXX_argv": cxx, "LDFLAGS_argv": shlex.split(os.environ.get("LDFLAGS", "")),
            "compiler_path": str(compiler), "compiler_sha256": digest(compiler),
            "compiler_version": version, "dumpmachine": dumpmachine,
            "cross_compiler_entry": "build_vta_yolov3_tiny_relay_residency_dispatch_pair.cross_compiler",
            "cross_compiler_entry_source_sha256": digest(Path(__file__).with_name("build_vta_yolov3_tiny_relay_residency_dispatch_pair.py")),
            "binary_sha256": digest(output / "graphlib.so"), "LLVM_sha256": digest(output / "module.ll"),
            "script_sha256": digest(__file__), "board_contacted": False, "latency_ms": None})
        print(json.dumps({"program": contract["program"], "status": "ARM_CROSS_BUILD_PASSED_FPGA_PENDING", "binary_sha256": digest(output / "graphlib.so")}))
    except Exception as error:
        write(output / "failure.json", {"status": "LOCAL_CROSS_BUILD_FAILED", "error": str(error),
            "traceback": traceback.format_exc(), "board_contacted": False})
        raise


def run(source, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    argv = [sys.executable, __file__, "worker", "--source", str(source), "--output-dir", str(output)]
    shell = "unset LD_LIBRARY_PATH; source " + shlex.quote(str(SDK)) + "; exec " + shlex.join(argv)
    log = output.with_suffix(".log")
    with log.open("x") as stream:
        process = subprocess.run(["bash", "-c", shell], stdout=stream, stderr=subprocess.STDOUT, timeout=300)
    write(output.with_name(output.name + "_exit.json"), {"returncode": process.returncode, "argv": ["bash", "-c", shell], "sdk_sha256": digest(SDK), "board_contacted": False})
    if process.returncode:
        raise RuntimeError("cross worker failed; retained attempt " + str(output))
    built = json.loads((output / "build.json").read_text())
    print(json.dumps({"program": built["program"], "status": built["status"],
        "binary_sha256": built["binary_sha256"], "directory": str(output)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("run", "worker"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    (run if args.phase == "run" else worker)(args.source.resolve(), args.output_dir.resolve())
