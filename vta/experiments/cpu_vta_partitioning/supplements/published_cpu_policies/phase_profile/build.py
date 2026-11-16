"""Rebuild only the optional runner; reuse all runtime and module bytes."""
from pathlib import Path
import os
import shutil
import subprocess
import sys
import time

from cpu_vta_partitioning.common import read,sha256,write
from workload_model.common import ROOT,require_space
from .patch import patch,PINNED_SOURCE_SHA256

PARENT=ROOT/'artifacts/bundles/abbcccc62af230ee844745ac'
RUNTIME=ROOT/'artifacts/yolo_wait_diagnosis_01/capsule/runtime'
RECORD=ROOT/'artifacts/bundles/27b583640470adcc2bbd2194/plans/runtime_build.json'


def build(output):
    require_space();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    source_dir=PARENT/'sources/vta/apps/native_deploy'
    source=source_dir/'vta_stage_pipeline_runner.cc'
    if sha256(source)!=PINNED_SOURCE_SHA256:raise ValueError('pinned source changed')
    native=output/'sources';native.mkdir(exist_ok=True)
    for p in source_dir.glob('*.h'):shutil.copy2(p,native/p.name)
    header=Path(__file__).with_name('phase_probe.h');shutil.copy2(header,native/header.name)
    patched=native/'vta_stage_pipeline_runner_phase.cc';patched.write_text(patch(source.read_text()))
    record=read(RECORD);tc=record['toolchain']
    if sha256(tc['sdk_setup'])!=tc['sdk_setup_sha256']:raise ValueError('SDK setup changed')
    for tool in tc['tools'].values():
        if sha256(tool['path'])!=tool['sha256']:raise ValueError('SDK tool changed')
    env=dict(os.environ);env.pop('LD_LIBRARY_PATH',None)
    destination=output/'vta_stage_pipeline_runner_phase'
    # Existing authenticated recipe; no runtime build and no CMake tree.
    recipe=record['runner_compiler_recipe']
    repo=ROOT.parents[4]
    cmd=['bash','-c','source "$1" || exit $?; shift; exec "$@"','phase-sdk',tc['sdk_setup'],
         'bash','-c',recipe,'phase-compile',str(patched),str(destination),str(repo),str(RUNTIME)]
    started=time.monotonic();result=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=180)
    (output/'compile.log').write_text(result.stdout+result.stderr)
    if result.returncode:raise RuntimeError('phase runner build failed; see '+str(output/'compile.log'))
    proof=dict(runner_sha256=sha256(destination),source_original_sha256=sha256(source),
        source_patched_sha256=sha256(patched),header_sha256=sha256(header),toolchain=tc,
        compiler_recipe=recipe,runtime_rebuilt=False,stage_modules_recompiled=False,
        runtime_sha256=sha256(RUNTIME/'libtvm_runtime.so'),seconds=time.monotonic()-started)
    write(output/'BUILD.json',proof,replace=True)
    qemu=shutil.which('qemu-aarch64-static')
    if not qemu:raise RuntimeError('QEMU required before deploying optional runner')
    checked=subprocess.run([qemu,str(RUNTIME/'ld-linux-aarch64.so.1'),'--library-path',str(RUNTIME),
                            str(destination),'--help'],capture_output=True,text=True,timeout=30)
    (output/'qemu_help.log').write_text(checked.stdout+checked.stderr)
    if checked.returncode or '--phase-event-json' not in checked.stdout+checked.stderr:
        raise ValueError('phase runner QEMU loading/CLI failed')
    return proof
