#!/usr/bin/env python3
"""Audit observed completion events, instruction tails and full output files."""
import argparse
import json
from pathlib import Path
import re
import numpy as np
from thesis_supplement_run import sha256, write_new_json


def audit(run, index, log):
    directory = run / 'raw' / f'boot{index}'
    records = [json.loads(line) for line in (directory / 'calls.jsonl').read_text().splitlines()]
    observed, current, instruction_block = {}, None, None
    for line in log.read_text(errors='replace').splitlines():
        if line.startswith('[THESIS_CALL_BEGIN] '):
            if current is not None:
                raise ValueError('Overlapping full calls')
            current = line.split()[1]
            observed[current] = {'submissions': [], 'instruction_blocks': [], 'ended': False}
        elif line.startswith('[THESIS_CALL_END] '):
            if line.split()[1] != current:
                raise ValueError('Call end association failed')
            observed[current]['ended'] = True
            current = None
        elif line.startswith('There are '):
            if current is None:
                raise ValueError('Instructions outside a bounded call')
            count = int(re.fullmatch(r'There are (\d+) instructions', line).group(1))
            instruction_block = {'declared_count': count, 'instructions': []}
            observed[current]['instruction_blocks'].append(instruction_block)
        elif line.startswith('INSTRUCTION '):
            number, description = re.fullmatch(r'INSTRUCTION (\d+): (.*)', line).groups()
            instruction_block['instructions'].append({'index': int(number), 'description': description})
        elif '[C3_COMPLETION] ' in line:
            if current is None:
                raise ValueError('Completion outside a bounded call')
            value = json.loads(line.split('[C3_COMPLETION] ', 1)[1])
            if value['after_clear_valid'] != 0 or value['return_valid'] != 1 or value['return_done'] != 1 or value['timeout']:
                raise ValueError('Unqualified fresh completion')
            observed[current]['submissions'].append(value)
    if current is not None:
        raise ValueError('Unfinished full call')
    if set(observed) != {item['full_call_id'] for item in records}:
        raise ValueError('Call log/record coverage differs')
    single, multiple, scored_submits, health_submits = 0, 0, 0, 0
    for record in records:
        identity = record['full_call_id']
        evidence = observed[identity]
        if not evidence['ended'] or not evidence['submissions']:
            raise ValueError('Missing completion evidence')
        if len(evidence['submissions']) != len(evidence['instruction_blocks']):
            raise ValueError('Missing instruction blocks')
        for completion, block in zip(evidence['submissions'], evidence['instruction_blocks']):
            commands = block['instructions']
            if len(commands) != block['declared_count'] or len(commands) != completion['insn_count']:
                raise ValueError('Instruction count disagrees')
            if [command['index'] for command in commands] != list(range(len(commands))):
                raise ValueError('Instruction indexing incomplete')
            if commands[-1]['description'] != 'FINISH':
                raise ValueError('Actual submission does not end with FINISH')
        for kind in ('immediate', 'delayed'):
            path = directory / (identity + '_' + kind + '.npy')
            if sha256(path) != record[kind + '_sha256']:
                raise ValueError('Changed output artifact')
            output = np.load(path, allow_pickle=False)
            expected = np.load(run / f'raw/{record["input"]}_expected.npy', allow_pickle=False)
            if not np.array_equal(output, expected):
                raise ValueError('Output recomputation failed')
        times = record['board_observation']
        if times['delayed_read_ns'] - times['immediate_read_ns'] < 10_000_000:
            raise ValueError('Delay was not 10ms on board')
        if record['scored']:
            scored_submits += len(evidence['submissions'])
            single += len(evidence['submissions']) == 1
            multiple += len(evidence['submissions']) > 1
        else:
            health_submits += len(evidence['submissions'])
    write_new_json(directory / 'observed_submissions.json', observed)
    coverage = {(item['mode'], item['input']): sum(record['scored'] and record['mode'] == item['mode'] and record['input'] == item['input'] for record in records)
                for item in records}
    if len(coverage) != 12 or any(count != 10 for count in coverage.values()):
        raise ValueError('Missing 4×3×10 coverage')
    result = {'status': 'BOOT_COMPLETION_AUDIT_PASS', 'boot_id': records[0]['boot_id'],
              'scored_full_calls': sum(item['scored'] for item in records),
              'health_full_calls': sum(not item['scored'] for item in records),
              'scored_submissions': scored_submits, 'health_submissions': health_submits,
              'scored_calls_with_single_submission': single, 'scored_calls_with_multiple_submissions': multiple,
              'coverage': {mode + '/' + inp: count for (mode, inp), count in coverage.items()},
              'log_sha256': sha256(log), 'outputs_recomputed': True,
              'minimum_immediate_read_overhead_ns': min(item['board_observation']['immediate_read_ns'] - item['board_observation']['returned_ns'] for item in records),
              'maximum_immediate_read_overhead_ns': max(item['board_observation']['immediate_read_ns'] - item['board_observation']['returned_ns'] for item in records)}
    write_new_json(run / f'analysis/boot{index}_trace_audit.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--boot-index', type=int, required=True)
    parser.add_argument('--log', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.run_dir, args.boot_index, args.log), indent=2))
