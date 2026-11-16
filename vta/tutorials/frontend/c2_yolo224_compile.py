#!/usr/bin/env python3
"""Compile and freeze the unified C2 runner against the isolated runtime."""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
from c2_yolo224_prepare import ROOT,HERE,sha,write


def compile_runner(out):
    package=out/'package'
    for name,source in [('libvta.so',out/'runtime/libvta.so'),
                        ('libtvm_runtime.so',ROOT/'build_axu_aarch64/libtvm_runtime.so')]:
        if not (package/name).exists():shutil.copy2(source,package/name)
        elif sha(package/name)!=sha(source):raise ValueError('runtime drift before native compilation')
    compiler=shutil.which('aarch64-xilinx-linux-g++')
    if not compiler or not os.environ.get('SDKTARGETSYSROOT'):
        raise RuntimeError('source AXU SDK first')
    command=[compiler,'--sysroot='+os.environ['SDKTARGETSYSROOT'],'-std=c++17','-O2','-pthread',
             '-ffunction-sections','-fdata-sections','-DTVM_KALLOC_ALIGNMENT=256','-DTVM_INDEX_DEFAULT_I64=1',
             '-DDMLC_USE_FOPEN64=0','-DDMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>']
    command += ['-I'+str(ROOT/p) for p in ('include','3rdparty/dlpack/include','3rdparty/dmlc-core/include',
                '3rdparty/vta-hw/include','3rdparty/picojson')]
    command += [str(HERE/'c2_yolo224_runner.cc'),'-L'+str(package),'-Wl,--gc-sections',
                '-Wl,-rpath,$ORIGIN','-ltvm_runtime','-lvta','-ldl','-lrt','-o',str(package/'c2_yolo224_runner')]
    result=subprocess.run(command,capture_output=True,text=True)
    index=1
    while (out/f'compile_native_{index:02d}.json').exists():index+=1
    write(out/f'compile_native_{index:02d}.json',dict(command=command,returncode=result.returncode,
          stdout=result.stdout,stderr=result.stderr,source_sha256=sha(HERE/'c2_yolo224_runner.cc')))
    result.check_returncode()
    sources=[HERE/'c2_yolo224_runner.cc',HERE/'c3_input_early_release_runner.cc',HERE/'c2_yolo224_compile.py']
    snap=out/'native_source'/f'build{index:02d}'
    snap.mkdir(parents=True)
    for p in sources:shutil.copy2(p,snap/p.name)
    print(json.dumps({'runner_sha256':sha(package/'c2_yolo224_runner'),'build':index}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    compile_runner(p.parse_args().out.resolve())
