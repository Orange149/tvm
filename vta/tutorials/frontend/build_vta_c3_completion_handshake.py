#!/usr/bin/env python3
"""Isolated output-valid experiment; no guessed MMIO writes or original edits."""
import argparse
from pathlib import Path
import subprocess
import time

from run_vta_c3_reuse_supplement import ROOT, HERE, verify, read, write, seal, check_bindings, file_hash
from build_vta_c3_completion_probe import replace_once
from recover_vta_c3_reuse_cross import sdk_environment


def source_variant(source, header, require_fresh):
    source = replace_once(source, '#include <vta/driver.h>', '#include <vta/driver.h>\n#include "' + str(header) + '"')
    source = replace_once(source, '    double run_t0 = NowMicros();',
        '    const char* c3_trace = getenv("C3_COMPLETION_TRACE");\n'
        '    C3FreshCompletion c3_probe(c3_trace && strcmp(c3_trace, "1") == 0);\n'
        '    c3_probe.before_done = VTAReadMappedReg(vta_compute_handle_, VTA_COMPUTE_DONE_RD_OFFSET);\n'
        '    c3_probe.before_valid = VTAReadMappedReg(vta_compute_handle_, VTA_COMPUTE_DONE_RD_OFFSET + 4);\n'
        '    c3_probe.after_clear_valid = VTAReadMappedReg(vta_compute_handle_, VTA_COMPUTE_DONE_RD_OFFSET + 4);\n'
        '    if (c3_probe.before_valid > 1 || c3_probe.after_clear_valid != 0) {\n'
        '      fprintf(stderr, "[C3_VALID_REJECT] unexpected output-valid register behavior\\n");\n'
        '      return 1;\n    }\n'
        '    double run_t0 = NowMicros();')
    source = replace_once(source,
        '      flag = VTAReadMappedReg(vta_compute_handle_, VTA_COMPUTE_DONE_RD_OFFSET);\n      if (flag == VTA_DONE) break;',
        '      uint32_t c3_valid = VTAReadMappedReg(vta_compute_handle_, VTA_COMPUTE_DONE_RD_OFFSET + 4);\n'
        '      flag = VTAReadMappedReg(vta_compute_handle_, VTA_COMPUTE_DONE_RD_OFFSET);\n'
        '      c3_probe.ObserveFresh(t, c3_valid, flag);\n'
        + ('      if (C3AcceptFreshCompletion(c3_valid, flag)) break;\n' if require_fresh else '      if (flag == VTA_DONE) break;\n')
        + '      if (NowMicros() - run_t0 > 1000000.0 || c3_valid > 1) { t = wait_cycles; break; }')
    source = replace_once(source, '      LogInfo("wait done cycles=%u flag=0x%x", t, flag);',
        '      c3_probe.ReportFresh(insn_phy_addr, insn_count, false, run_total_us);\n'
        '      LogInfo("wait done cycles=%u flag=0x%x", t, flag);')
    source = replace_once(source, '    LogWarn("wait timeout flag=0x%x fetch_ctrl=0x%x load_ctrl=0x%x compute_ctrl=0x%x store_ctrl=0x%x",',
        '    c3_probe.ReportFresh(insn_phy_addr, insn_count, true, run_total_us);\n'
        '    LogWarn("wait timeout flag=0x%x fetch_ctrl=0x%x load_ctrl=0x%x compute_ctrl=0x%x store_ctrl=0x%x",')
    return source


def run(args):
    started = time.monotonic()
    previous = Path(args.trace_build).resolve()
    original_build = verify(previous)
    check_bindings(original_build["source_bindings"])
    require_fresh = args.mode == "fresh_valid"
    header = HERE / "c3_fresh_completion.h"
    source = ROOT / "3rdparty/vta-hw/src/axu5evb/axu5evb_driver.cc"
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    bindings = {**original_build["source_bindings"], str(Path(__file__).resolve()): file_hash(__file__),
                str(header): file_hash(header)}
    write(output / "contract.json", {"phase": "completion_handshake_build", "source_bindings": bindings,
          "mode": args.mode, "require_fresh_event": require_fresh,
          "register_hypothesis": "HLS done_o_ap_vld at DONE_RD+4=0x1c; generated bitstream map unavailable",
          "unknown_MMIO_writes": 0, "before_start_action": "read assumed output-valid twice to consume old event and verify clear",
          "timeout_seconds_per_submit": 1, "fixed_completion_sleep_added": False,
          "trace_build_manifest_sha256": file_hash(previous / "artifact_hashes.json"),
          "original_libvta_sha256": file_hash(ROOT / "build_axu_aarch64/libvta.so"),
          "formal_performance_approved": False})
    try:
        derived = output / "axu5evb_driver_fresh.cc"
        derived.write_text(source_variant(source.read_text(), header, require_fresh))
        commands = read(previous / "commands.json")
        for bound in commands["original_object_bindings"]:
            if file_hash(bound["path"]) != bound["sha256"]:
                raise ValueError("original object changed")
        compile_command = list(commands["compile"])
        for flag in ("-MT", "-MF", "-o", "-c"):
            compile_command[compile_command.index(flag) + 1] = str(derived) if flag == "-c" else str(output / ("driver.o.d" if flag == "-MF" else "driver.o"))
        link = [str(output / "driver.o") if v == str(previous / "driver_probe.o") else v for v in commands["link"]]
        link[link.index("-o") + 1] = str(output / "libvta.so")
        write(output / "commands.json", {"compile": compile_command, "link": link})
        for name, command in (("compile", compile_command), ("link", link)):
            result = subprocess.run(command, env=sdk_environment(), cwd=output, capture_output=True,
                                    text=True, timeout=180, check=False)
            (output / (name + ".stderr.log")).write_text(result.stderr)
            if result.returncode:
                raise RuntimeError(name + " failed: " + result.stderr[-4000:])
        check_bindings(bindings)
        write(output / "summary.json", {"status": "EXPERIMENTAL_HANDSHAKE_BUILT", "mode": args.mode,
              "libvta_sha256": file_hash(output / "libvta.so"), "wall_seconds": time.monotonic() - started})
    except BaseException as error:
        write(output / "invalid_session.json", {"message": str(error)})
        raise
    finally:
        seal(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-build", required=True)
    parser.add_argument("--mode", choices=("valid_observer", "fresh_valid"), required=True)
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())
