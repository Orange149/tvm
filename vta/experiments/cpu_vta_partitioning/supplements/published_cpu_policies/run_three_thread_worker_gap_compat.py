"""Validate the frozen runner's producer/collector events without rerunning jobs."""
from pathlib import Path
from types import FunctionType
import argparse
import json

import run_three_thread_worker_gap as original
from phase_profile import runtime as phase
from scheduler_gap_audit import runtime as queue_runtime
from cpu_vta_partitioning.common import read,sha256
from finite_service.common import freeze


def validate_control_tids(rows,affinity):
    roles={r['role']:r['tid'] for r in affinity['control_threads']}
    if set(roles)!={'producer','collector'}:raise ValueError('control roles incomplete')
    for r in rows:
        if r['stage'] in ('_producer','_collector') and r['tid']!=roles[r['stage'][1:]]:
            raise ValueError('control event TID differs from actual affinity')


def phase_summary(raw,doc,affinity):
    rows=[json.loads(s) for s in (Path(raw)/'phase_trace.jsonl').read_text().splitlines()]
    if doc['spec']['scope']=='component':
        queue_runtime.validate_queue_rows(rows,scope='stage')
        return phase.phase_summary(raw,doc,affinity)
    result=queue_runtime.phase_summary(raw,doc,affinity)
    validate_control_tids(rows,affinity)
    return result


def summary(raw,doc,bundle):
    events=[json.loads(s) for s in (Path(raw)/'wait_trace.jsonl').read_text().splitlines()]
    queue_runtime.validate_queue_rows(events,scope='worker')
    f=phase.summary
    return FunctionType(f.__code__,{**f.__globals__,'phase_summary':phase_summary})(raw,doc,bundle)


def accept(*args,**kwargs):
    f=original.accept
    return FunctionType(f.__code__,{**f.__globals__,'summary':summary})(*args,**kwargs)


def verified_receipts():
    f=original.verified_receipts
    return FunctionType(f.__code__,{**f.__globals__,'summary':summary})()


def proof():
    root=original.ROOT;build_path=root/'artifacts/scheduler_gap_audit_01/build/BUILD.json';build=read(build_path)
    for p,h in build['source_files'].items():
        if sha256(p)!=h:raise ValueError('frozen binary source changed')
    bundle=original.POOL/'yolov3_tiny416/capsule';m=read(bundle/'manifest.json')
    if m['phase_runner_sha256']!=build['runner_sha256'] or m['adapter_sha256']!=build['adapter_sha256']:
        raise ValueError('actual observed binary does not match queue/control recorder sources')
    for row in read(original.POOL/'COMMAND_AUDIT.json')['sessions']:
        task=original.make_task(bundle,row['spec']);argv,_=original.native_command(task,bundle,'/bundle','/job')
        if task['task_id']!=row['task_id'] or argv!=row['command']:raise ValueError('native task changed')
    return freeze(original.POOL/'VALIDATOR_COMPAT.json',dict(source_sha256=sha256(__file__),
        original_source_sha256=sha256(original.__file__),build_sha256=sha256(build_path),
        reused_validator_sha256=sha256(root/'scheduler_gap_audit/runtime.py'),
        existing_tasks_and_commands_unchanged=True,no_native_remeasurement=True,
        all_stage_and_control_events_required=True,control_TID_check=True,
        actual_adapter_includes_TASKSTATS_reads_enabled_by_default=True,
        no_scheduling_or_wait_policy_change=True))


def resume(*,execute_board=False):
    proof();f=original.board
    return FunctionType(f.__code__,{**f.__globals__,'accept':accept},f.__name__,f.__defaults__)(execute_board=execute_board)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('proof','resume','validate'))
    p.add_argument('--execute-board',action='store_true');a=p.parse_args()
    if a.command=='resume' and not a.execute_board:p.error('--execute-board is required')
    r=resume(execute_board=True) if a.command=='resume' else proof() if a.command=='proof' else dict(passed=True,receipts=len(verified_receipts()))
    print(json.dumps(r,ensure_ascii=False,indent=2))
