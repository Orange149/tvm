#!/usr/bin/env python3
"""Validate passive single-core worker mappings without adding performance probes.

The frozen runner emits active worker_slots only under --resource-affinity-probe.
Performance deliberately omits that flag. Existing _affinity already verifies a
bijective singleton-core mapping; use it to validate each traced logical worker.
All frozen source bytes, task hashes, windows and native outputs stay unchanged.
"""
from copy import deepcopy
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))


def passive_slots(document,affinity):
    result=deepcopy(affinity)
    for cfg,role in zip(document['stages'],result['stages']):
        if cfg['device']!='cpu' or role.get('worker_slots'):continue
        if document['qualification']:raise ValueError('qualification requires active worker slot proof')
        entries=role['effective_affinities'];mapping=cfg['worker_core_map']
        if (len(entries)!=cfg['threads'] or len(set(role['thread_ids']))!=cfg['threads']
                or {e['tid'] for e in entries}!=set(role['thread_ids'])
                or role['requested_cpus']!=cfg['mask'] or role['active_runtime_threads']!=cfg['threads']
                or any(len(e['cpus'])!=1 or e['cpus'][0] not in mapping for e in entries)):
            raise ValueError('passive worker ownership/core proof incomplete')
        by_core={e['cpus'][0]:e['tid'] for e in entries}
        if len(by_core)!=cfg['threads'] or by_core[mapping[0]]!=role['caller_tid']:
            raise ValueError('passive worker map is not bijective or caller differs')
        role['worker_slots']=[dict(slot=i,tid=by_core[core],cpus=[core]) for i,core in enumerate(mapping)]
    return result


def install():
    from wait_diagnosis import runtime
    original=runtime.trace_summary
    def compatible(raw,document,bundle,affinity):
        return original(raw,document,bundle,passive_slots(document,affinity))
    runtime.trace_summary=compatible


def main(argv=None):
    from wait_diagnosis.common import OUTPUT,RUN,frozen
    from cpu_vta_partitioning.common import read,sha256,write
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('audit','prepare','validate','board','status','analyze'))
    p.add_argument('action',nargs='?',choices=('run','resume'))
    p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--run-dir',type=Path,default=RUN)
    p.add_argument('--local-config',type=Path);p.add_argument('--execute-board',action='store_true')
    a=p.parse_args(argv)
    source=Path(__file__).resolve()
    proof=dict(controller=str(source),controller_sha256=sha256(source),
        scope='performance validation only: worker_slots reconstructed from already validated singleton affinities',
        native_programs_changed=False,task_hashes_changed=False,additional_probes=False,remeasurement=False)
    if a.command in ('board','analyze'):
        frozen(a.run_dir/'PASSIVE_WORKER_COMPATIBILITY.json',proof)
        saved=a.run_dir/'sources'/source.name
        if saved.exists() and sha256(saved)!=sha256(source):raise ValueError('compatibility snapshot changed')
        if not saved.exists():saved.parent.mkdir(parents=True,exist_ok=True);saved.write_bytes(source.read_bytes())
        failure=a.run_dir/'FAILURE_01_PASSIVE_SLOTS.json'
        if (a.run_dir/'state.json').exists() and read(a.run_dir/'state.json')['status']=='STOPPED_ON_ERROR' and not failure.exists():
            write(failure,read(a.run_dir/'state.json'))
    install()
    from run_yolo_wait_diagnosis import main as original_main
    args=[a.command]+([a.action] if a.action else [])+['--output',str(a.output),'--run-dir',str(a.run_dir)]
    if a.local_config:args+=['--local-config',str(a.local_config)]
    if a.execute_board:args+=['--execute-board']
    return original_main(args)


if __name__=='__main__':main()
