#!/usr/bin/env python3
"""Run a frozen sequence with monotonic admission and charged deadline overruns.

Time starts at wrapper entry, before plan loading and child imports. This runner
does not predict correctness, read an oracle, or kill a running FPGA operation.
"""
import time

ENTRY_TIME = time.monotonic()

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(plan):
    if plan.get('schema') != 'c3_budgeted_process_plan_v1':
        raise ValueError('unsupported plan schema')
    budget = plan.get('budget_seconds')
    if isinstance(budget, bool) or not isinstance(budget, (int, float)) or not math.isfinite(budget) or budget <= 0:
        raise ValueError('budget must be finite and positive')
    if plan.get('schedule') != 'one_candidate_build_then_measure':
        raise ValueError('this executor requires sequential candidate builds')
    names, finished = set(), set()
    pending = None
    seen_candidate = False
    for step in plan['steps']:
        name = step['name']
        if not isinstance(name, str) or not name or name in names:
            raise ValueError('duplicate/empty step name')
        names.add(name)
        argv = step['argv']
        if not isinstance(argv, list) or not argv or not all(isinstance(v, str) and v for v in argv):
            raise ValueError('argv must be a nonempty string array')
        if not Path(step['cwd']).is_absolute():
            raise ValueError('cwd must be absolute')
        kind, cid = step['kind'], step.get('candidate_id')
        if kind == 'prepare':
            if seen_candidate or cid is not None:
                raise ValueError('shared preparation must precede all candidates')
        elif kind == 'build':
            if pending is not None or not isinstance(cid,str) or not cid or cid in finished:
                raise ValueError('build must follow previous candidate measurement')
            pending, seen_candidate = cid, True
        elif kind == 'measure':
            if not cid or pending != cid:
                raise ValueError('measurement must match its immediately preceding build')
            receipt = step['completion_receipt']
            if not Path(receipt['path']).is_absolute() or not receipt.get('required_values'):
                raise ValueError('measurement needs a bound completion receipt')
            if receipt['required_values'].get('candidate_id') != cid:
                raise ValueError('receipt must certify the same candidate identity')
            finished.add(cid)
            pending = None
        else:
            raise ValueError('unknown step kind')
    if pending is not None:
        raise ValueError('build without planned measurement')


def run_step(step, log):
    receipt = step.get('completion_receipt')
    if receipt and Path(receipt['path']).exists():
        raise ValueError('stale completion receipt exists before execution')
    with log.open('xb') as stream:
        result = subprocess.run(step['argv'], cwd=step['cwd'], stdout=stream,
                                stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        return dict(returncode=result.returncode, completion_verified=False)
    if receipt:
        document = json.loads(Path(receipt['path']).read_text())
        if any(document.get(k) != v for k,v in receipt['required_values'].items()):
            raise ValueError('measurement completion receipt mismatch')
        return dict(returncode=0, completion_verified=True,
                    receipt_sha256=sha256(receipt['path']))
    return dict(returncode=0, completion_verified=False)


def execute(plan, output, origin, clock=time.monotonic, action=run_step):
    validate(plan)
    rows, eligible = [], []
    status = 'completed'
    budget = plan['budget_seconds']
    for index, step in enumerate(plan['steps']):
        before = clock() - origin
        if before >= budget:
            status = 'budget_exhausted_before_next_step'
            break
        record = dict(index=index, name=step['name'], kind=step['kind'],
                      candidate_id=step.get('candidate_id'), started_seconds=before)
        try:
            record.update(action(step,output / f'{index:03d}.log'))
        except Exception as error:
            record.update(returncode=None, completion_verified=False,
                          error_type=type(error).__name__, error=str(error))
        ended = clock() - origin
        record.update(finished_seconds=ended, elapsed_seconds=ended-before,
                      finished_within_budget=ended <= budget,
                      overrun_seconds=max(0.,ended-budget))
        rows.append(record)
        # The event writer and between-step orchestration are charged by the
        # shared origin too. Late labels are recorded but not budget-eligible.
        with (output/'events.jsonl').open('a') as stream:
            stream.write(json.dumps(record,allow_nan=False)+'\n')
        if record['returncode'] != 0:
            status = 'failed_closed'
            break
        if step['kind']=='measure' and record['completion_verified'] and ended <= budget:
            eligible.append(step['candidate_id'])
        if ended > budget:
            status = 'budget_overrun_current_step_drained'
            break
    ended = clock() - origin
    return dict(schema='c3_budgeted_process_execution_v1', status=status,
                budget_seconds=budget, elapsed_seconds=ended,
                overrun_seconds=max(0.,ended-budget),
                budget_eligible_completed_candidates=eligible,
                steps=rows, unstarted_steps=len(plan['steps'])-len(rows),
                deadline_policy='soft admission; drain active step; charge overrun; exclude late label',
                clock_scope='wrapper entry through execution and receipt checks; excludes interpreter startup and final report serialization',
                oracle_used=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    raw=args.plan.read_bytes()
    plan=json.loads(raw)
    validate(plan)
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'plan.json').write_bytes(raw)
    result=execute(plan,args.output,ENTRY_TIME)
    result.update(plan_sha256=sha256(args.output/'plan.json'),executor_sha256=sha256(__file__))
    (args.output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    artifacts={p.name:sha256(p) for p in args.output.iterdir() if p.is_file()}
    (args.output/'artifact_hashes.json').write_text(json.dumps({'artifacts':artifacts},indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if result['status']=='failed_closed':
        sys.exit(1)


if __name__=='__main__':
    main()
