#!/usr/bin/env python3
"""Build separate FSim/board runtime overlays with the author UOP extensions."""
import argparse
import json
from pathlib import Path
import subprocess
import shlex

from c3_banerjee_baseline import REPO, HERE, digest, write_new


def runtime_source():
    text = (REPO / "vta/runtime/runtime.cc").read_text()
    start = text.index("template <int kMaxBytes, bool kCoherent, bool kAlwaysCache>\nclass UopQueue")
    end = text.index("// Internal kernel structure\nclass UopKernelMap", start)
    text = text[:start] + (HERE / "c3_banerjee_uop_queue.h").read_text() + "\n\n" + text[end:]
    old = """      VTAMemInsn* insn = insn_queue_.CreateMemInsn(VTA_MEM_ID_UOP);
      insn->opcode = VTA_OPCODE_LOAD;
      uop_queue_.FlushUopLoad(insn);"""
    if text.count(old) != 2:
        raise ValueError("two GEMM/ALU UOP load seams required")
    return text.replace(old, "      uop_queue_.FlushUopLoad(insn_queue_);")


def build(output, target, enabled=True):
    output.mkdir(parents=True, exist_ok=False)
    source = output / "runtime.cc"
    source.write_text(runtime_source())  # Generated overlay, not a production edit.
    includes = [REPO / p for p in ("include", "3rdparty/dmlc-core/include", "3rdparty/dlpack/include",
                "3rdparty/vta-hw/include", "vta/runtime")]
    config = REPO / "3rdparty/vta-hw/config/vta_config.py"
    defs = shlex.split(subprocess.check_output(["/home/orange/miniconda3/envs/vta-resnet/bin/python", str(config), "--defs"], text=True))
    cxx = "g++"
    shared = [str(REPO / "vta/runtime/device_api.cc")]
    if target == "sim":
        shared += [str(REPO / "3rdparty/vta-hw" / s) for s in (
            "src/sim/sim_driver.cc", "src/sim/sim_tlpp.cc", "src/vmem/virtual_memory.cc")]
        defs += ["-DUSE_FSIM_TLPP"]
        name = "libvta_fsim.so"
    else:
        shared += [str(REPO / "3rdparty/vta-hw/src/axu5evb/axu5evb_driver.cc")]
        cxx = "aarch64-xilinx-linux-g++"
        name = "libvta.so"
        defs += ["--sysroot=/home/orange/alinx_plsdk/install/sysroots/aarch64-xilinx-linux",
                 "-march=armv8-a+crc", "-mtune=cortex-a72.cortex-a53"]
    command = [cxx, "-std=c++17", "-O2", "-shared", "-fPIC", "-pthread", str(source), *shared,
               *["-I" + str(p) for p in includes], *defs,
               "-DDMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>",
               "-DC3_BANERJEE_VALUE_UOPS=" + str(int(enabled)),
               "-DC3_BANERJEE_GREEDY_UOPS=" + str(int(enabled)), "-o", str(output / name)]
    if target == "sim":
        process = subprocess.run(command, capture_output=True, text=True)
    else:
        # SDK context affects this isolated compiler process only.
        shell = "unset LD_LIBRARY_PATH; source /home/orange/alinx_plsdk/install/environment-setup-aarch64-xilinx-linux; exec " + shlex.join(command)
        process = subprocess.run(["bash", "-c", shell], capture_output=True, text=True)
    (output / "build.log").write_text(process.stdout + process.stderr)
    write_new(output / "build.json", {"returncode": process.returncode, "target": target,
        "enabled": enabled, "command": command, "base_runtime_sha256": digest(REPO / "vta/runtime/runtime.cc"),
        "overlay_sha256": digest(source), "header_sha256": digest(HERE / "c3_banerjee_uop_queue.h"),
        "binary_sha256": digest(output / name) if process.returncode == 0 else None,
        "local_adaptations": ["local flat VTAUop ABI", "capacity checked before appending/auto-sync",
                              "single contiguous host copy", "existing profiler and serialization methods"],
        "board_contacted": False})
    if process.returncode:
        raise RuntimeError("experimental runtime build failed; see " + str(output / "build.log"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target", choices=("sim", "board"), required=True)
    parser.add_argument("--disable-uop-extensions", action="store_true")
    args = parser.parse_args()
    build(args.output_dir.resolve(), args.target, not args.disable_uop_extensions)
