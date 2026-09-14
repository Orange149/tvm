#!/usr/bin/env python3
"""Build isolated C3-S driver/runner; never overwrite frozen runtime artifacts."""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess

from audit_shared_buffer_storage import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-no-strict-aliasing", action="store_true",
                        help="Diagnostic runtime only: legacy instruction views alias common storage")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sysroot = os.environ["SDKTARGETSYSROOT"]
    records = json.loads((root / "build_axu_aarch64/compile_commands.json").read_text())
    selected = [r for r in records if "CMakeFiles/vta.dir/" in r["command"]]
    assert len(selected) == 3
    commands, objects = [], []
    for record in selected:
        command = shlex.split(record["command"])
        target = out / (Path(record["file"]).name + ".o")
        command[command.index("-o") + 1] = str(target)
        if args.audit_no_strict_aliasing and Path(record["file"]).name == "runtime.cc":
            command.append("-fno-strict-aliasing")
        commands.append(command)
        subprocess.run(command, cwd=record["directory"], check=True)
        objects.append(str(target))
    compiler = selected[0]["command"].split()[0]
    command = [compiler, "--sysroot=" + sysroot, "-shared", "-Wl,-soname,libvta.so",
               "-o", str(out / "libvta.so"), *objects, "-pthread"]
    commands.append(command)
    subprocess.run(command, check=True)
    source = root / "vta/apps/native_deploy/vta_stage_pipeline_runner.cc"
    command = [compiler, "--sysroot=" + sysroot, "-std=c++17", "-O2",
               *["-I" + str(root / p) for p in ("include", "3rdparty/dlpack/include",
                   "3rdparty/dmlc-core/include", "3rdparty/vta-hw/include", "vta/include")],
               str(source), "-o", str(out / "vta_stage_pipeline_runner"),
               "-L" + str(root / "build_axu_aarch64"), "-ltvm_runtime", "-lvta", "-ldl", "-pthread"]
    commands.append(command)
    subprocess.run(command, check=True)
    result = {"commands": commands, "source_sha256": {str(Path(r["file"])): sha(r["file"])
              for r in selected}, "artifacts": {name: sha(out / name) for name in
              ("libvta.so", "vta_stage_pipeline_runner")}}
    result["source_sha256"][str(source)] = sha(source)
    queue_header = root / "vta/runtime/queue_capacity.h"
    if queue_header.exists():
        result["source_sha256"][str(queue_header)] = sha(queue_header)
    header = source.with_name("vta_shared_buffer_plan.h")
    result["source_sha256"][str(header)] = sha(header)
    result["source_sha256"][str(root / "3rdparty/picojson/picojson.h")] = sha(root / "3rdparty/picojson/picojson.h")
    result["source_sha256"][str(Path(__file__).resolve())] = sha(__file__)
    # Preserve exact uncommitted experiment sources, not hashes alone.
    snapshot = out / "source_snapshot"
    sources = [Path(p) for p in result["source_sha256"]]
    for path in sources:
        relative = path.resolve().relative_to(root.resolve())
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        assert sha(target) == result["source_sha256"][str(path)]
    result["source_snapshot"] = str(snapshot.relative_to(out))
    (out / "build_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["artifacts"]), flush=True)


if __name__ == "__main__":
    main()
