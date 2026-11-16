#!/usr/bin/env python3
"""Reconcile immutable action evidence without erasing failed preparation."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from thesis_supplement_run import RunStore, sha256, write_new_json


def reconcile(directory):
    run = RunStore(directory)
    starts, ends = {}, {}
    for event in run.events():
        if event['event'] == 'action_start':
            if event['action_id'] in starts:
                raise ValueError('Duplicate action start')
            starts[event['action_id']] = event
        elif event['event'] == 'action_end':
            if event['action_id'] in ends:
                raise ValueError('Duplicate action end')
            ends[event['action_id']] = event
    costs = defaultdict(float)
    failed = []
    for identity, end in ends.items():
        start = starts[identity]
        if end['parent_event_id'] != start['event_id']:
            raise ValueError('Broken event association')
        if end['wall_seconds'] < 0:
            raise ValueError('Negative measured cost')
        for key in ('stdout', 'stderr'):
            if sha256(run.directory / end[key]) != end[key + '_sha256']:
                raise ValueError('Changed raw action output')
        costs[end['cost_class']] += end['wall_seconds']
        if end['status'] != 'ok':
            failed.append({k: end[k] for k in
                           ('action_id', 'action', 'status', 'returncode',
                            'cost_class', 'wall_seconds', 'stdout', 'stderr')})
    return {
        'matched_actions': len(ends),
        'unfinished_action_ids': sorted(set(starts) - set(ends)),
        'measured_action_cost_seconds': dict(costs),
        'measured_action_total_seconds': sum(costs.values()),
        'failed_actions_preserved': failed,
        'pre_run_and_interactive_development_cost_seconds': None,
        'unknown_cost_reason': 'Interactive discovery/development and unlogged shell-launch failures have no complete instrumented boundary.',
        'end_to_end_wall_seconds': None,
        'wall_boundary_note': 'Actions use individual host process monotonic clocks. Sum is accounted action cost, not online T0→T1 or campaign elapsed wall.',
    }


def seal(directory):
    directory = Path(directory)
    values = {
        str(path.relative_to(directory)): {'sha256': sha256(path), 'bytes': path.stat().st_size}
        for path in sorted(directory.rglob('*'))
        if path.is_file() and path.name != 'artifact_hashes.json'
    }
    write_new_json(directory / 'artifact_hashes.json', values)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    write_new_json(args.output, reconcile(args.run_dir))
