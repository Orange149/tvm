#!/usr/bin/env python3
"""Audit STORE acknowledgement ordinals with explicit device context boundaries."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
from thesis_supplement_run import sha256, write_new_json


def parse(log):
    blocks, current, block = [], None, None
    for line in log.read_text().splitlines():
        if line.startswith('[THESIS_CALL_BEGIN] '):
            current = line.split()[1]
        elif line.startswith('There are '):
            if current is None:
                raise ValueError('Unbounded submission')
            block = {'full_call_id': current, 'commands': [],
                     'declared_count': int(re.fullmatch(r'There are (\d+) instructions', line)[1])}
            blocks.append(block)
        elif line.startswith('INSTRUCTION '):
            index, desc = re.fullmatch(r'INSTRUCTION (\d+): (.*)', line).groups()
            block['commands'].append({'index': int(index), 'description': desc})
        elif line.startswith('\tdep - '):
            flags = list(map(int, re.findall(r': ([01])', line)))
            if len(flags) != 4:
                raise ValueError('Incomplete dependency flags')
            block['commands'][-1]['flags'] = flags
        elif line.startswith('\tl2g_queue'):
            block['dump_l2g_g2l'] = list(map(int, re.findall(r'= (-?\d+)', line)))
        elif line.startswith('\ts2g_queue'):
            block['dump_s2g_g2s'] = list(map(int, re.findall(r'= (-?\d+)', line)))
        elif line.startswith('[THESIS_CALL_END] '):
            if line.split()[1] != current:
                raise ValueError('Mismatched full-call boundary')
            current = None
    if current is not None:
        raise ValueError('Unfinished call')
    for block in blocks:
        commands = block['commands']
        if len(commands) != block['declared_count'] or [c['index'] for c in commands] != list(range(len(commands))):
            raise ValueError('Incomplete instruction dump')
        if commands[-1]['description'] != 'FINISH':
            raise ValueError('Missing terminal FINISH')
        balance, pushes, pops, last_store, last_ordinal = [0, 0, 0, 0], 0, 0, None, None
        for cmd in commands[:-1]:
            desc = cmd['description']
            stage = 1 if desc in ('LOAD INP', 'LOAD WGT', 'NOP-MEMORY-STAGE') else 3 if desc in ('STORE:', 'NOP-STORE-STAGE') else 2
            pp, pn, sp, sn = cmd['flags']
            if stage == 1:
                if pp or sp:
                    raise ValueError('Invalid LOAD dependency direction')
                balance[1] -= pn
                balance[0] += sn
            elif stage == 3:
                if pn or sn:
                    raise ValueError('Invalid STORE dependency direction')
                balance[3] -= pp
                balance[2] += sp
                pushes += sp
                if desc == 'STORE:':
                    last_store = cmd['index']
                    last_ordinal = pushes if sp else None
            else:
                balance[0] -= pp
                balance[2] -= pn
                balance[1] += sp
                balance[3] += sn
                pops += pn
        if balance[:2] != block['dump_l2g_g2l'] or balance[2:] != block['dump_s2g_g2s']:
            raise ValueError('Recomputed balance differs from actual dump')
        block.update(balance=balance, store_pushes=pushes, compute_store_pops=pops,
                     last_real_store_index=last_store, last_real_store_ack_ordinal=last_ordinal)
    return blocks


def inventory(blocks, reset_per_condition):
    # Producer/consumer streams each preserve instruction order. FINISH follows every
    # compute pop. A pop can consume an old acknowledgement; counting only net balance
    # is insufficient. Inventory is an upper bound including queued tail NOOP pushes,
    # which need not already have arrived at the host return timestamp.
    tokens, previous_condition, findings = [0, 0, 0, 0], None, []
    for index, block in enumerate(blocks):
        condition = block['full_call_id'].split('_')[1]
        context_change = condition != previous_condition
        if context_change and reset_per_condition:
            tokens = [0, 0, 0, 0]
        initial = list(tokens)
        final = [a + b for a, b in zip(tokens, block['balance'])]
        if any(value < 0 for value in final):
            raise ValueError('Negative inventory under the stated reset contract')
        ordinal = block['last_real_store_ack_ordinal']
        has_store = block['last_real_store_index'] is not None
        sufficient = not has_store or (ordinal is not None and block['compute_store_pops'] >= initial[2] + ordinal)
        findings.append({'submission_index': index, 'full_call_id': block['full_call_id'],
                         'context_change': context_change, 'initial_inventory_upper_bound': initial,
                         'final_inventory_upper_bound': final,
                         'last_real_store_index': block['last_real_store_index'],
                         'last_real_store_ack_ordinal': ordinal,
                         'compute_store_pops_before_finish': block['compute_store_pops'],
                         'store_ack_bound_sufficient': sufficient})
        tokens, previous_condition = final, condition
    return findings


def verify_resets(run, index):
    from thesis_supplement_run import RunStore
    store = RunStore(run)
    if not store.contract['completion'].get('execution_context_reset', {}).get('required_before_each_new_command_queue'):
        raise ValueError('Contract does not declare actual context resets')
    events = store.events()
    rows = [json.loads(line) for line in (run / f'raw/boot{index}/calls.jsonl').read_text().splitlines()]
    if len({row['boot_id'] for row in rows}) != 1:
        raise ValueError('Mixed boot identities')
    boot_id = rows[0]['boot_id']
    identity = json.loads((run / f'raw/boot{index}/identity.json').read_text())
    if identity['boot_id'] != boot_id:
        raise ValueError('Boot identity differs from actual call records')
    bindings = json.loads((run / 'analysis/board_artifact_bindings.json').read_text())
    verified = []
    for mode in ('O', 'I', 'W', 'IW'):
        session = f'b{index}_{mode}'
        def one(kind):
            matches = [(n, e) for n, e in enumerate(events) if e['event'] == kind and e.get('session_id') == session]
            if len(matches) != 1:
                raise ValueError(f'Expected one {kind} for {session}')
            return matches[0]
        ready_index, ready = one('context_reset_ready')
        claim_index, claim = one('context_reset_claimed')
        receipt_path = run / ready['receipt']
        if sha256(receipt_path) != ready['receipt_sha256']:
            raise ValueError('Reset receipt drift')
        receipt = json.loads(receipt_path.read_text())
        ends = [(n, e) for n, e in enumerate(events) if e['event'] == 'action_end' and e.get('action_id') == receipt['action_id']]
        starts = [e for e in events if e['event'] == 'action_start' and e.get('action_id') == receipt['action_id']]
        first_calls = [n for n, e in enumerate(events) if e['event'] == 'full_call_start' and e.get('session_id') == session]
        if len(ends) != 1 or len(starts) != 1 or not first_calls:
            raise ValueError('Missing reset action or actual condition calls')
        end_index, end = ends[0]
        if not end_index < ready_index < claim_index < min(first_calls) or end['status'] != 'ok' or end['returncode'] != 0:
            raise ValueError('Reset action failed or receipt order invalid')
        if claim['reset_event_id'] != ready['event_id'] or receipt['boot_id'] != boot_id or ready['boot_id'] != boot_id or claim['boot_id'] != boot_id:
            raise ValueError('Reset boot or claimed event differs')
        if receipt['board_directory'] != identity['board_directory']:
            raise ValueError('Reset board directory differs')
        if receipt['bitstream_sha256'] != bindings['vta_hpc.bit']['sha256']:
            raise ValueError('Reset bitstream differs')
        stdout = run / receipt['action_stdout']
        script = run / f'raw/context_reset_{session}.sh'
        if sha256(stdout) != receipt['action_stdout_sha256'] or end['stdout_sha256'] != receipt['action_stdout_sha256']:
            raise ValueError('Raw reset evidence drift')
        if sha256(script) != receipt['reset_script_sha256'] or starts[0]['stdin_sha256'] != receipt['reset_script_sha256']:
            raise ValueError('Executed reset script differs')
        expected = f'THESIS_CONTEXT_RESET_OK {session} {boot_id} {receipt["bitstream_sha256"]} operating'
        if expected not in stdout.read_text().splitlines():
            raise ValueError('Missing actual reset marker')
        verified.append({'session_id': session, 'boot_id': boot_id, 'receipt': ready['receipt'],
                         'receipt_sha256': ready['receipt_sha256']})
    return verified


def audit(log, output, run=None, index=None):
    blocks = parse(log)
    resets = verify_resets(run, index) if run is not None else []
    actual_bound = inventory(blocks, bool(resets))
    reset_bound = inventory(blocks, True)
    insufficient = [item for item in actual_bound if not item['store_ack_bound_sufficient']]
    result = {
        'status': 'CONTEXT_RESET_PRECONDITION_REQUIRED' if insufficient else 'STORE_ACK_BOUND_PASS',
        'log_sha256': sha256(log), 'submissions': len(blocks), 'verified_reset_receipts': resets,
        'descriptor_net_balance_counts': dict(Counter(str(b['balance']) for b in blocks)),
        'actual_context_contract_model': {
            'condition_processes_reset_hardware': bool(resets),
            'insufficient_store_ack_bound_submissions': len(insufficient),
            'first_insufficient_submission': insufficient[0] if insufficient else None,
            'findings': actual_bound,
        },
        'proposed_reset_before_each_fresh_condition_context': {
            'counterfactual_only_not_executed': not bool(resets),
            'insufficient_store_ack_bound_submissions': sum(not x['store_ack_bound_sufficient'] for x in reset_bound),
            'findings': reset_bound,
        },
        'interpretation': 'A terminal post-sync dep_pop intentionally retains an acknowledgement for the next call in the same CommandQueue. A new RPC child resets host pending flags; driver allocation/free only maps/unmaps registers. Without a proven hardware reset between contexts, older acknowledgements can satisfy later STORE pops. Any insufficient ordinal bound is a certification gap, not an observed numeric failure.',
        'limits': 'Descriptor-derived upper bounds, not FIFO-occupancy readback or proof of an actual late STORE. FINISH dependency flags are supported by current finalization source and are not printed by DumpInsn. The reset scenario is analysis of existing descriptors, never a replacement for a new actual two-boot run.',
    }
    write_new_json(output, result)
    return {key: value for key, value in result.items() if key not in ('actual_context_contract_model', 'proposed_reset_before_each_fresh_condition_context')}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--log', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--run-dir', type=Path)
    p.add_argument('--boot-index', type=int)
    args = p.parse_args()
    if (args.run_dir is None) != (args.boot_index is None):
        p.error('--run-dir and --boot-index must be provided together')
    print(json.dumps(audit(args.log, args.output, args.run_dir, args.boot_index), indent=2))
