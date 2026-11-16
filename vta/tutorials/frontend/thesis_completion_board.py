#!/usr/bin/env python3
"""Execute one boot of the frozen stage01 matrix and preserve full outputs."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import tvm
from tvm import rpc
from thesis_supplement_run import RunStore, sha256, write_new_json


def run_boot(run, index, boot_id, board_directory, only_mode, binding_file='analysis/board_artifact_bindings.json'):
    bindings = json.loads((run.directory / binding_file).read_text())
    for item in bindings.values():
        if sha256(item['host_path']) != item['sha256']:
            raise ValueError('Artifact changed after deployment binding')
    if run.contract['completion'].get('execution_context_reset', {}).get('required_before_each_new_command_queue'):
        session = f'b{index}_{only_mode}'
        ready = [e for e in run.events() if e['event'] == 'context_reset_ready' and e.get('session_id') == session]
        if len(ready) != 1 or any(e['event'] == 'context_reset_claimed' and e.get('session_id') == session for e in run.events()):
            raise ValueError('Missing, duplicate or already consumed context reset receipt')
        event = ready[0]
        receipt_path = run.directory / event['receipt']
        if sha256(receipt_path) != event['receipt_sha256']:
            raise ValueError('Reset receipt changed')
        receipt = json.loads(receipt_path.read_text())
        if receipt['boot_id'] != boot_id or receipt['board_directory'] != board_directory or receipt['bitstream_sha256'] != bindings['vta_hpc.bit']['sha256']:
            raise ValueError('Reset receipt differs from the actual condition bindings')
        if sha256(run.directory / receipt['action_stdout']) != receipt['action_stdout_sha256']:
            raise ValueError('Reset raw evidence changed')
        run.event('context_reset_claimed', session_id=session, boot_id=boot_id,
                  reset_event_id=event['event_id'], reset_receipt=event['receipt'])
    prefix = run.directory / 'raw' / f'boot{index}'
    if only_mode == 'O':
        prefix.mkdir(exist_ok=False)
        write_new_json(prefix / 'identity.json', {'boot_index': index, 'boot_id': boot_id,
                                               'board_directory': board_directory})
    else:
        identity = json.loads((prefix / 'identity.json').read_text())
        if identity['boot_id'] != boot_id or identity['board_directory'] != board_directory:
            raise ValueError('Boot identity differs across serial condition processes')
    known_boots = list(run.directory.glob('raw/boot*/identity.json'))
    ids = [json.loads(path.read_text())['boot_id'] for path in known_boots]
    if len(ids) != len(set(ids)):
        raise ValueError('Independent boots must have distinct boot_id')
    weights = np.load(run.directory / 'raw/weights.npy', allow_pickle=False)
    records = prefix / 'calls.jsonl'
    try:
        for label, mode in [('O', 0), ('I', 1), ('W', 4), ('IW', 5)]:
            if label != only_mode:
                continue
            # Each RPC connection forks a fresh board child for this condition.
            remote = rpc.connect('192.168.1.247', 9090, session_timeout=300)
            keep_harness = remote.load_module(board_directory + '/thesis_completion_harness_shared.so')
            module = remote.load_module(board_directory + f'/mode{mode}.so')
            once = remote.get_function('thesis.completion_once')
            device, cpu = remote.ext_dev(0), remote.cpu(0)
            weight = tvm.nd.array(weights, device)
            for name in ('zeros', 'uniform', 'normal'):
                actual_input = np.load(run.directory / f'raw/{name}_input.npy', allow_pickle=False)
                expected = np.load(run.directory / f'raw/{name}_expected.npy', allow_pickle=False)
                sentinel_values = np.load(run.directory / f'raw/{name}_sentinel.npy', allow_pickle=False)
                inp = tvm.nd.array(actual_input, device)
                output = tvm.nd.empty(expected.shape, 'int8', device=device)
                reference = tvm.nd.array(expected, cpu)
                sentinel = tvm.nd.array(sentinel_values, cpu)
                immediate = tvm.nd.empty(expected.shape, 'int8', cpu)
                delayed = tvm.nd.empty(expected.shape, 'int8', cpu)
                count = 11 if name == 'zeros' else 10
                for position in range(count):
                    health = name == 'zeros' and position == 0
                    round_id = position - 1 if name == 'zeros' else position
                    identity = f'b{index}_{label}_{name}_' + ('health' if health else f'r{round_id:02}')
                    started = run.event('full_call_start', boot_id=boot_id, session_id=f'b{index}_{label}',
                                        full_call_id=identity, condition=label, input=name,
                                        scored=not health, cost_class='functional_qualification')
                    begin = time.monotonic()
                    value = json.loads(once(module, inp, weight, output, reference, immediate, delayed, sentinel, identity))
                    first, second = immediate.numpy(), delayed.numpy()
                    np.save(prefix / (identity + '_immediate.npy'), first, allow_pickle=False)
                    np.save(prefix / (identity + '_delayed.npy'), second, allow_pickle=False)
                    mismatch = int(np.count_nonzero(first != expected))
                    changes = int(np.count_nonzero(first != second))
                    record = {'full_call_id': identity, 'boot_id': boot_id, 'mode': label,
                              'implementation_mode': mode, 'input': name, 'round': round_id,
                              'scored': not health, 'host_wall_seconds': time.monotonic() - begin,
                              'board_observation': value, 'immediate_mismatches': mismatch,
                              'delayed_mismatches': int(np.count_nonzero(second != expected)),
                              'changed_bytes': changes,
                              'immediate_sha256': sha256(prefix / (identity + '_immediate.npy')),
                              'delayed_sha256': sha256(prefix / (identity + '_delayed.npy'))}
                    with records.open('a') as stream:
                        stream.write(json.dumps(record, sort_keys=True) + '\n')
                        stream.flush()
                    run.event('full_call_end', parent_event_id=started, cost_class='functional_qualification', **record)
                    if mismatch or changes or record['delayed_mismatches'] or not all(
                            value[key] for key in ('sentinel_visible', 'first_correct', 'delayed_correct', 'stable')):
                        raise ValueError('Completion/output failure: ' + identity)
                del inp, output, reference, sentinel, immediate, delayed
            del weight, once, module, keep_harness, device, cpu, remote
            print(f'boot{index} {label}: 30 scored calls and one health complete', flush=True)
        write_new_json(prefix / ('summary_' + only_mode + '.json'), {
            'status': 'CONDITION_COLLECTED_PENDING_TRACE_AUDIT', 'condition': only_mode,
            'scored_full_calls': 30, 'health_full_calls': 1, 'boot_id': boot_id})
        if only_mode == 'IW':
            rows = [json.loads(line) for line in records.read_text().splitlines()]
            if sum(item['scored'] for item in rows) != 120 or sum(not item['scored'] for item in rows) != 4:
                raise ValueError('Complete boot matrix missing conditions')
            write_new_json(prefix / 'summary.json', {'status': 'MATRIX_COLLECTED_PENDING_REGISTER_TRACE_AUDIT',
                                                   'scored_full_calls': 120, 'health_full_calls': 4,
                                                   'boot_id': boot_id})
    except BaseException as error:
        run.event('board_session_invalid', boot_id=boot_id, reason=str(error))
        write_new_json(run.directory / 'invalid_session.json', {'boot_id': boot_id, 'boot_index': index,
                                                              'reason': str(error), 'all_costs_retained': True})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir')
    parser.add_argument('--boot-index', type=int, required=True)
    parser.add_argument('--boot-id', required=True)
    parser.add_argument('--board-directory', required=True)
    parser.add_argument('--mode', choices=('O', 'I', 'W', 'IW'), required=True)
    parser.add_argument('--binding-file', default='analysis/board_artifact_bindings.json')
    args = parser.parse_args()
    run_boot(RunStore(args.run_dir), args.boot_index, args.boot_id, args.board_directory, args.mode, args.binding_file)
