#!/usr/bin/env python3
"""RPC diagnostic full-output checks; never used as native pipeline timings."""
import argparse
import json
from pathlib import Path
import numpy as np
import tvm
from tvm import rpc
from tvm.contrib import graph_executor
from thesis_supplement_run import RunStore, sha256, write_new_json


def numeric(actual, expected, contract):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError('Shape/dtype changed')
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError('Non-finite output')
    exact = np.array_equal(actual, expected)
    passed = exact if actual.dtype.kind in 'iu' else np.allclose(actual, expected,
        atol=contract['float_atol'], rtol=contract['float_rtol'])
    result = {'dtype': str(actual.dtype), 'shape': list(actual.shape), 'exact': bool(exact),
              'mismatching_elements': int(np.count_nonzero(actual != expected)),
              'max_abs_error': float(np.max(np.abs(actual.astype('float64') - expected.astype('float64')))),
              'passed': bool(passed)}
    if not passed:
        raise ValueError('Numeric failure: ' + json.dumps(result))
    return result


def check(run, condition):
    r = run.directory
    if (r / 'invalid_session.json').exists():
        raise ValueError('Invalid run cannot collect more samples')
    remote_dir = json.loads((r / 'analysis/board_directory.json').read_text())['path']
    manifest = json.loads((r / 'raw/board_programs/program_manifest.json').read_text())
    selected = [p for p in manifest['programs'] if (p['name'] == 'Q_board') == (condition == 'Qboard')]
    values = np.load(r / 'raw/inputs/input.npy', allow_pickle=False)
    with np.load(r / 'raw/host_reference/reference_boundaries.npz', allow_pickle=False) as data:
        references = {key: data[key] for key in data.files}
    final_reference = np.load(r / 'raw/host_reference/reference_logits.npy', allow_pickle=False)
    output = r / 'raw/board_numeric' / condition
    output.mkdir(parents=True, exist_ok=False)
    remote = rpc.connect('192.168.1.247', 9090, session_timeout=600)
    harness = remote.load_module(remote_dir + '/thesis_graph_snapshot.so')
    once = remote.get_function('thesis.graph_snapshot_once')
    loaded = []
    for record in selected:
        directory = r / 'raw/board_programs' / record['name']
        for name, expected in record['files'].items():
            if sha256(directory / name) != expected['sha256']:
                raise ValueError('Program artifact drift')
        module = remote.load_module(remote_dir + '/board_programs/' + record['name'] + '/module.so')
        devices = [remote.ext_dev(0), remote.cpu(0)] if record['device'] == 'vta' else [remote.cpu(0)]
        exe = graph_executor.create((directory / 'graph.json').read_text(), module, devices)
        exe.load_params((directory / 'params.bin').read_bytes())
        loaded.append((record, exe, module))
    records = []
    for repeat in range(10):
        state = [tvm.nd.array(values, remote.cpu(0))]
        final = None
        for stage_index, (record, exe, module) in enumerate(loaded):
            if len(state) != len(record['input_names']):
                raise ValueError('Complete frontier arity changed')
            for name, value in zip(record['input_names'], state):
                # Host NDArray CopyTo cannot treat RPC-encoded CPU (129) as CPU
                # when the graph input is RPC ext_dev (140). Stage diagnostics
                # explicitly materialize bytes through NumPy; native timing does
                # not use this transport path.
                exe.set_input(name, value.numpy())
            incoming = state
            state = []
            identity = f'b1_{condition}_r{repeat:02}_s{stage_index}'
            node_id = record['output_node_ids'][0]
            expected = final_reference if condition == 'Qboard' else references[f'node_{node_id}']
            immediate = tvm.nd.empty(expected.shape, str(expected.dtype), remote.cpu(0))
            delayed = tvm.nd.empty(expected.shape, str(expected.dtype), remote.cpu(0))
            started = run.event('model_call_start', full_call_id=identity, condition=condition,
                                stage_index=stage_index, cost_class='functional_qualification')
            observation = json.loads(once(exe.module, 0, immediate, delayed, identity))
            first, second = immediate.numpy(), delayed.numpy()
            np.save(output / (identity + '_immediate.npy'), first, allow_pickle=False)
            np.save(output / (identity + '_delayed.npy'), second, allow_pickle=False)
            result = {'full_call_id': identity, 'stage_index': stage_index, 'node_id': node_id,
                      'board_observation': observation, 'numeric': numeric(first, expected, run.contract['numeric'])}
            if not observation['stable'] or first.tobytes() != second.tobytes() or observation['delayed_read_ns'] - observation['immediate_read_ns'] < 10_000_000:
                raise ValueError('Board output not stable for 10ms: ' + identity)
            records.append(result)
            run.event('model_call_end', parent_event_id=started, cost_class='functional_qualification', **result)
            for i, source in enumerate(record['output_sources']):
                value = incoming[source['live_input_index']] if source['kind'] == 'live_input' else exe.get_output(i)
                state.append(tvm.nd.array(value.numpy(), remote.cpu(0)))
            final = first
        numeric(final, final_reference, run.contract['numeric'])
        print(condition + f' repeat{repeat:02}: complete outputs and 10ms stability PASS', flush=True)
    write_new_json(r / 'analysis' / (condition + '_numeric_checks.json'), {
        'status': 'SINGLE_IMAGE_BOARD_NUMERIC_PASS', 'condition': condition, 'records': records,
        'full_model_calls': 10, 'stage_calls': len(records), 'extra_quantize_calls': 0,
        'backend': 'board RPC graph executors; not native pipeline performance', 'accuracy_evidence': False})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run_dir', type=Path)
    p.add_argument('--condition', choices=('Qboard', 'A'), required=True)
    args = p.parse_args()
    check(RunStore(args.run_dir), args.condition)
