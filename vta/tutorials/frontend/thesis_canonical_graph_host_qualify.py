#!/usr/bin/env python3
"""Host-only synthetic qualification of common qIR and typed cut interfaces."""
import argparse
import json
from pathlib import Path
import numpy as np
import tvm
from tvm import autotvm, relay
from tvm.contrib import graph_executor
from thesis_canonical_graph_split import UnitGraph
from thesis_supplement_run import sha256, write_new_json


def build(mod, directory):
    directory.mkdir()
    with autotvm.apply_history_best([]), tvm.transform.PassContext(opt_level=3, disabled_pass=['AlterOpLayout']):
        factory = relay.build(mod, target='llvm')
    factory.get_lib().export_library(str(directory / 'module.so'))
    (directory / 'graph.json').open('x').write(factory.get_graph_json())
    (directory / 'params.bin').open('xb').write(relay.save_param_dict(factory.get_params()))
    return factory


def executor(factory):
    return graph_executor.GraphModule(factory['default'](tvm.cpu()))


def differences(actual, expected):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError('Interface dtype or shape changed')
    exact = np.array_equal(actual, expected)
    if actual.dtype.kind in 'iu':
        passed = exact
    else:
        passed = np.allclose(actual, expected, atol=1e-5, rtol=1e-4)
    result = {'dtype': str(actual.dtype), 'shape': list(actual.shape), 'exact': bool(exact),
              'mismatching_elements': int(np.count_nonzero(actual != expected)),
              'max_abs_error': float(np.max(np.abs(actual.astype('float64') - expected.astype('float64')))),
              'passed': bool(passed)}
    if not passed:
        raise ValueError('Numerical qualification failed: ' + json.dumps(result))
    return result


def qualify(source, canonical, output):
    output.mkdir(parents=True, exist_ok=False)
    graph = UnitGraph(tvm.ir.load_json(canonical.read_text()))
    original = tvm.ir.load_json(source.read_text())
    write_new_json(output / 'contract.json', {
        'scope': 'Host LLVM synthetic preparation only; no board/real images/accuracy or performance labels',
        'source_qir_sha256': sha256(source), 'canonical_qir_sha256': sha256(canonical),
        'uniform_seed': 59001, 'normal_seed': 59002, 'rng': 'NumPy PCG64',
        'input_shape': [1, 3, 224, 224], 'cases': ['zeros', 'uniform_float32_0_1', 'normal_float32'],
        'integer_boundary_check': 'exact', 'float_final_atol': 1e-5, 'float_final_rtol': 1e-4,
        'transports': ['copy', 'CPU set_input_zero_copy with poisoned original pools'],
        'passthrough_rule': 'Carry the actual incoming NDArray for live_input outputs; never read an unrebound input storage pool as a passthrough output',
        'stage02_gate_passed': False,
    })
    unique_edges = sorted({n for b in range(22) for n in graph.frontier(b)}, key=lambda n: graph.ids.get(n, -1))
    indices = {node: i for i, node in enumerate(unique_edges)}
    probe = relay.transform.InferType()(tvm.IRModule.from_expr(relay.Function(list(graph.mod['main'].params), relay.Tuple(unique_edges))))
    compiled = output / 'compiled'
    compiled.mkdir()
    source_exe = executor(build(original, compiled / 'original_qir'))
    common_exe = executor(build(graph.mod, compiled / 'canonical_qir'))
    probe_exe = executor(build(probe, compiled / 'canonical_boundary_probe'))
    programs = {}
    for kind, ranges in [('integrated', [(0, 3), (3, 18), (18, 21)]), ('units', [(i, i+1) for i in range(21)])]:
        programs[kind] = []
        for begin, end in ranges:
            mod, inputs, outputs = graph.stage(begin, end)
            factory = build(mod, compiled / f'{kind}_{begin:02}_{end-1:02}')
            programs[kind].append({'begin': begin, 'end': end, 'inputs': inputs, 'outputs': outputs,
                                   'executors': {transport: executor(factory) for transport in ('copy', 'shared')}})
            print(f'compiled {kind} {begin:02}..{end-1:02}', flush=True)
    shape = (1, 3, 224, 224)
    cases = {'zeros': np.zeros(shape, 'float32'),
             'uniform': np.random.Generator(np.random.PCG64(59001)).random(shape, dtype=np.float32),
             'normal': np.random.Generator(np.random.PCG64(59002)).standard_normal(shape, dtype=np.float32)}
    results = []
    for name, values in cases.items():
        directory = output / name
        directory.mkdir()
        np.save(directory / 'input.npy', values, allow_pickle=False)
        source_exe.set_input('data', values)
        source_exe.run()
        source_logits = source_exe.get_output(0).numpy()
        common_exe.set_input('data', values)
        common_exe.run()
        common_logits = common_exe.get_output(0).numpy()
        np.save(directory / 'original_logits.npy', source_logits, allow_pickle=False)
        np.save(directory / 'canonical_logits.npy', common_logits, allow_pickle=False)
        original_common = differences(common_logits, source_logits)
        probe_exe.set_input('data', values)
        probe_exe.run()
        references = {n: probe_exe.get_output(i).numpy() for n, i in indices.items()}
        np.savez(directory / 'canonical_boundaries.npz', **{f'node_{graph.ids.get(n, -1)}': a for n, a in references.items()})
        probe_final = differences(references[graph.mod['main'].body], common_logits)
        observations = {}
        for kind, stages in programs.items():
            for transport in ('copy', 'shared'):
                root_input = tvm.nd.array(values, tvm.cpu())
                state = [root_input]
                snapshots, checks, carries = {}, [], []
                for stage in stages:
                    exe = stage['executors'][transport]
                    if len(state) != len(stage['inputs']):
                        raise ValueError('Incorrect complete live frontier')
                    for i, array in enumerate(state):
                        parameter = f'edge{stage["begin"]}_{i}'
                        if transport == 'copy':
                            exe.set_input(parameter, array)
                        else:
                            pool = exe.get_input(parameter)
                            actual = array.numpy()
                            poison = np.bitwise_not(actual.view('uint8')).view(actual.dtype).reshape(actual.shape)
                            pool.copyfrom(poison)
                            if not np.array_equal(pool.numpy().view('uint8'), poison.view('uint8')):
                                raise ValueError('Original input pool poison failed')
                            exe.set_input_zero_copy(parameter, array)
                    exe.run()
                    incoming = state
                    state = []
                    for i, node in enumerate(stage['outputs']):
                        if node in stage['inputs']:
                            source_index = stage['inputs'].index(node)
                            array = incoming[source_index]
                            carries.append({'begin': stage['begin'], 'output_index': i, 'live_input_index': source_index})
                        else:
                            array = exe.get_output(i)
                        actual = array.numpy()
                        check = differences(actual, references[node])
                        check.update(begin=stage['begin'], end_exclusive=stage['end'], node_id=graph.ids.get(node, -1))
                        checks.append(check)
                        snapshots[f'after_{stage["end"]:02}_node_{graph.ids.get(node, -1)}'] = actual
                        state.append(array)
                logits = state[0].numpy()
                observations[kind + '_' + transport] = {'boundary_checks': checks, 'passthrough_routes': carries,
                                                        'final_vs_common': differences(logits, common_logits)}
                np.save(directory / (kind + '_' + transport + '_logits.npy'), logits, allow_pickle=False)
                np.savez(directory / (kind + '_' + transport + '_boundaries.npz'), **snapshots)
            copied = np.load(directory / (kind + '_copy_logits.npy'), allow_pickle=False)
            shared = np.load(directory / (kind + '_shared_logits.npy'), allow_pickle=False)
            if copied.tobytes() != shared.tobytes():
                raise ValueError('Host copy/shared final bytes differ')
        write_new_json(directory / 'checks.json', {'original_to_common': original_common,
                       'probe_vs_common': probe_final, 'programs': observations, 'copy_shared_final_byte_exact': True})
        results.append({'input': name, 'all_boundary_and_final_checks_passed': True})
        print(name + ': all boundary/final comparisons passed', flush=True)
    summary = {'status': 'HOST_SYNTHETIC_CANONICAL_SPLIT_PASS', 'cases': results,
               'units': 21, 'integrated_stages': 3, 'common_pass': 'EliminateCommonSubexpr; no extra quantize',
               'integer_boundaries_exact': True, 'copy_shared_final_bytes_exact': True,
               'real_images_or_accuracy': False, 'board_accessed': False, 'stage02_qualified': False}
    write_new_json(output / 'summary.json', summary)
    write_new_json(output / 'artifact_hashes.json', {str(p.relative_to(output)): {'sha256': sha256(p), 'bytes': p.stat().st_size}
                                                   for p in sorted(output.rglob('*')) if p.is_file()})
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-qir', type=Path, required=True)
    p.add_argument('--canonical-qir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(qualify(args.source_qir, args.canonical_qir, args.output), indent=2))
