#!/usr/bin/env python3
"""Build one derived driver in a new directory; original source/DSO stay intact."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import time

from run_vta_c3_reuse_supplement import ROOT, HERE, read, write, verify, seal, file_hash, check_bindings
from recover_vta_c3_reuse_cross import sdk_environment


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError("driver patch anchor is ambiguous or absent")
    return source.replace(old, new, 1)


def derived_driver(source, header):
    source = replace_once(source, "#include <vta/driver.h>", '#include <vta/driver.h>\n#include "' + str(header) + '"')
    source = replace_once(source, "    double run_t0 = NowMicros();",
        '    const char* c3_trace = getenv("C3_COMPLETION_TRACE");\n'
        '    C3CompletionProbe c3_probe(c3_trace && strcmp(c3_trace, "1") == 0);\n'
        '    if (c3_probe.enabled) {\n'
        '      c3_probe.before_done = VTAReadMappedReg(vta_compute_handle_, VTA_COMPUTE_DONE_RD_OFFSET);\n'
        '    }\n    double run_t0 = NowMicros();')
    source = replace_once(source,
        '      flag = VTAReadMappedReg(vta_compute_handle_, VTA_COMPUTE_DONE_RD_OFFSET);',
        '      flag = VTAReadMappedReg(vta_compute_handle_, VTA_COMPUTE_DONE_RD_OFFSET);\n'
        '      c3_probe.Observe(t, flag);')
    source = replace_once(source, '      LogInfo("wait done cycles=%u flag=0x%x", t, flag);',
        '      c3_probe.Report(insn_phy_addr, insn_count, false, run_total_us);\n'
        '      LogInfo("wait done cycles=%u flag=0x%x", t, flag);')
    source = replace_once(source, '    LogWarn("wait timeout flag=0x%x fetch_ctrl=0x%x load_ctrl=0x%x compute_ctrl=0x%x store_ctrl=0x%x",',
        '    c3_probe.Report(insn_phy_addr, insn_count, true, run_total_us);\n'
        '    LogWarn("wait timeout flag=0x%x fetch_ctrl=0x%x load_ctrl=0x%x compute_ctrl=0x%x store_ctrl=0x%x",')
    return source


def run(args):
    started = time.monotonic()
    prepared = Path(args.contract).resolve().parent
    contract = verify(prepared)
    check_bindings(contract["source_bindings"])
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    original = ROOT / "3rdparty/vta-hw/src/axu5evb/axu5evb_driver.cc"
    header = HERE / "c3_completion_probe.h"
    build_dir = ROOT / "build_axu_aarch64"
    bindings = {**contract["source_bindings"], str(Path(__file__).resolve()): file_hash(__file__),
                str(header): file_hash(header)}
    write(output / "contract.json", {"phase": "driver_probe_build", "source_bindings": bindings,
          "prepared_directory": str(prepared), "prepared_manifest_sha256": file_hash(prepared / "artifact_hashes.json"),
          "original_driver_sha256": file_hash(original), "original_libvta_sha256": file_hash(build_dir / "libvta.so"),
          "permission": "user explicitly approved driver modification; isolated experimental library only",
          "mode": "trace unchanged level-done decision; no guessed valid register or register writes",
          "observer_effect": "one extra before-start done read; stderr record after measured driver interval",
          "formal_performance_labels": False})
    try:
        derived = output / "axu5evb_driver_probe.cc"
        derived.write_text(derived_driver(original.read_text(), header))
        commands = subprocess.run(["ninja", "-C", str(build_dir), "-t", "commands", "libvta.so"],
                                   check=True, capture_output=True, text=True).stdout.splitlines()
        compile_command = shlex.split(next(c for c in commands if " -c " + str(original) in c))
        for flag in ("-MT", "-MF", "-o", "-c"):
            index = compile_command.index(flag) + 1
            compile_command[index] = str(derived) if flag == "-c" else str(output / ("driver_probe.o.d" if flag == "-MF" else "driver_probe.o"))
        link = shlex.split(next(c for c in commands if "-Wl,-soname,libvta.so" in c))
        link = link[link.index("&&") + 1:link.index("&&", link.index("&&") + 1)]
        link[link.index("-o") + 1] = str(output / "libvta.so")
        objects = []
        for index, item in enumerate(link):
            if item.endswith(".o"):
                path = build_dir / item
                objects.append({"path": str(path), "sha256": file_hash(path)})
                link[index] = str(output / "driver_probe.o") if "axu5evb_driver" in item else str(path)
        write(output / "commands.json", {"compile": compile_command, "link": link, "original_object_bindings": objects})
        for name, command in (("compile", compile_command), ("link", link)):
            process = subprocess.run(command, cwd=output, env=sdk_environment(), capture_output=True,
                                     text=True, timeout=180, check=False)
            (output / (name + ".stderr.log")).write_text(process.stderr)
            if process.returncode:
                raise RuntimeError(name + " failed: " + process.stderr[-4000:])
        check_bindings(bindings)
        if file_hash(build_dir / "libvta.so") != read(output / "contract.json")["original_libvta_sha256"]:
            raise ValueError("original runtime unexpectedly changed")
        write(output / "summary.json", {"status": "ISOLATED_TRACE_DRIVER_BUILT", "libvta_sha256": file_hash(output / "libvta.so"),
              "derived_driver_sha256": file_hash(derived), "wall_seconds": time.monotonic() - started,
              "original_driver_and_library_unchanged": True, "completion_decision_changed": False})
    except BaseException as error:
        write(output / "invalid_session.json", {"message": str(error), "wall_seconds": time.monotonic() - started})
        raise
    finally:
        seal(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())
