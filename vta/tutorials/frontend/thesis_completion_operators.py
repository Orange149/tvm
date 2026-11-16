#!/usr/bin/env python3
"""Compile stage01 operators and freeze independent LLVM diagnostic outputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import tvm
from tvm import te, topi
from tvm.contrib import cc
import vta

from thesis_supplement_run import RunStore, sha256, write_new_json
from run_vta_c3_reuse_supplement import instantiate
from c3_reuse_semantics import capture_lower

ROOT = Path(__file__).resolve().parents[3]


def arrays(run):
    definition = json.loads((run.directory / 'raw/candidate_definitions.json').read_text())[0]
    workload = definition['identity']['workload']
    shape = tuple(workload[1][1])
    weight_shape = tuple(workload[2][1])
    uniform = np.random.Generator(np.random.PCG64(59001)).random(shape).astype('float32')
    normal = np.random.Generator(np.random.PCG64(59002)).standard_normal(shape).astype('float32')
    weight = np.random.Generator(np.random.PCG64(59003)).integers(-8, 9, weight_shape, dtype=np.int8)
    inputs = {'zeros': np.zeros(shape, dtype=np.int8),
              'uniform': (np.floor(64 * uniform) - 32).astype(np.int8),
              'normal': np.clip(np.rint(8 * normal), -32, 31).astype(np.int8)}
    data = te.placeholder(shape, 'int8', 'data')
    kernel = te.placeholder(weight_shape, 'int8', 'weight')
    # Unpack to the independent generic NCHW LLVM computation.
    nchw = te.compute((1, 64, 28, 28), lambda n, c, h, w: data[n, c // 16, h, w, 0, c % 16], name='unpack_input')
    oihw = te.compute((64, 64, 3, 3), lambda o, i, h, w: kernel[o // 16, i // 16, h, w, o % 16, i % 16], name='unpack_weight')
    convolution = topi.nn.conv2d_nchw(nchw, oihw, workload[3], workload[4], workload[5], 'int32')
    packed = te.compute(shape, lambda n, c, h, w, bn, bc:
                        tvm.tir.Cast('int8', tvm.tir.min(tvm.tir.max(convolution[n, c * 16 + bc, h, w] >> 8, 0), 127)), name='pack_output')
    module = tvm.build(te.create_schedule(packed.op), [data, kernel, packed], target='llvm', name='reference')
    module.export_library(str(run.directory / 'raw/llvm_reference.so'))
    run.directory.joinpath('raw/llvm_reference.ll').write_text(module.get_source('ll'))
    np.save(run.directory / 'raw/uniform_rng_float32.npy', uniform, allow_pickle=False)
    np.save(run.directory / 'raw/normal_rng_float32.npy', normal, allow_pickle=False)
    np.save(run.directory / 'raw/weights.npy', weight, allow_pickle=False)
    results = {}
    for name, value in inputs.items():
        output = tvm.nd.empty(shape, 'int8')
        module['reference'](tvm.nd.array(value), tvm.nd.array(weight), output)
        expected = output.numpy()
        np.save(run.directory / f'raw/{name}_input.npy', value, allow_pickle=False)
        np.save(run.directory / f'raw/{name}_expected.npy', expected, allow_pickle=False)
        np.save(run.directory / f'raw/{name}_sentinel.npy', np.bitwise_not(expected), allow_pickle=False)
        results[name] = {'input_sha256': hashlib.sha256(value.tobytes()).hexdigest(),
                         'expected_sha256': hashlib.sha256(expected.tobytes()).hexdigest()}
    write_new_json(run.directory / 'analysis/frozen_diagnostic_inputs.json', {
        'shape': list(shape), 'weight_shape': list(weight_shape), 'diagnostics': results,
        'packing': 'input n,ci_outer,h,w,batch_inner,ci_inner; weight co_outer,ci_outer,kh,kw,co_inner,ci_inner',
        'llvm_reference_sha256': sha256(run.directory / 'raw/llvm_reference.so'),
        'weight_sha256': hashlib.sha256(weight.tobytes()).hexdigest(),
    })


def operator(run, mode):
    rows = json.loads((run.directory / 'raw/candidate_definitions.json').read_text())
    definition = next(row for row in rows if row['implementation_mode'] == mode)
    schedule, tensors = instantiate(definition)
    lowered, storage, arenas = capture_lower(schedule, tensors)
    (run.directory / f'raw/mode{mode}_tir.json').write_text(tvm.ir.save_json(lowered))
    (run.directory / f'raw/mode{mode}_tir.txt').write_text(str(lowered))
    with vta.build_config(debug_flag=vta.get_env().DEBUG_DUMP_INSN,
                          disabled_pass={'tir.CommonSubexprElimTIR'}):
        module = vta.build(schedule, tensors,
                           target=tvm.target.Target(vta.get_env().target, host=vta.get_env().target_host), name='main')
    sysroot = os.environ['SDKTARGETSYSROOT']
    module.export_library(str(run.directory / f'raw/mode{mode}.so'),
                          fcompile=cc.cross_compiler('aarch64-xilinx-linux-g++'),
                          options=['--sysroot=' + sysroot, '-Wl,-rpath-link,' + sysroot + '/lib', '-Wl,-rpath-link,' + sysroot + '/usr/lib'])
    write_new_json(run.directory / f'analysis/mode{mode}_build.json', {
        'actual_implementation_mode': mode, 'candidate': definition,
        'binary_sha256': sha256(run.directory / f'raw/mode{mode}.so'),
        'tir_sha256': sha256(run.directory / f'raw/mode{mode}_tir.json'),
        'storage_arenas': arenas,
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('arrays', 'operator'))
    parser.add_argument('run_dir')
    parser.add_argument('--mode', type=int)
    args = parser.parse_args()
    run = RunStore(args.run_dir)
    if args.operation == 'arrays':
        arrays(run)
    else:
        operator(run, args.mode)
