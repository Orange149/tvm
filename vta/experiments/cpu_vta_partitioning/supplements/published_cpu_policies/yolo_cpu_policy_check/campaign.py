"""Exactly one qualification and one long session for every frozen candidate."""
from copy import deepcopy
from pathlib import Path
import shlex
import time
from types import FunctionType
from cpu_vta_partitioning.common import digest, locked, now, read, sync_tree, write
from cpu_vta_partitioning.transport import SSH
from yolo_validation import pipeline as old_pipeline
from yolo_validation.components import hardware_check
from run_yolo_validation_gate_compat import adapted_summary
from .common import protocol, frozen, assert_prepared
from .bundle import validate_bundle


def make_task(bundle,name,phase):
    bundle=Path(bundle)
    if phase not in ('qualify','performance'):raise ValueError('unsupported phase')
    cfg=read(bundle/'plans/pool.json')[name];p=read(bundle/'plans/protocol.json');m=read(bundle/'manifest.json')
    window=p['qualification' if phase=='qualify' else 'performance']
    doc=dict(kind='qualify_pipeline' if phase=='qualify' else 'performance',phase=phase,
        candidate_id=name,configuration=deepcopy(cfg),stages=deepcopy(cfg['stages']),
        environment=dict(cores=4,frequency_khz=1066666,poll_sleep_ns=1000,post_start_sleep_ns=1000,
                         spin_count=0,exclude_worker0=1,bind_threads=1),
        **window,input_indices=[0],executor_sha256=m['runner_sha256'],bundle_id=m['bundle_id'],
        experiment=p['experiment'],output_mode='all_stage_dumps' if phase=='qualify' else 'all_final_hashes')
    doc['task_id']=digest(doc)[:24];return doc


native_command=FunctionType(old_pipeline.native_command.__code__,
    {**old_pipeline.native_command.__globals__,'make_task':make_task})


def summary(raw,document,bundle):
    function,_=adapted_summary();result=function(raw,document,bundle)
    if document['phase']!='qualify':
        import json
        rows=[json.loads(line) for line in (Path(raw)/'result.jsonl').read_text().splitlines()]
        result['window_work']=old_pipeline.work_boundaries(rows,document)
    return result


# Reuse the proven acceptance/acknowledgement path with this experiment's
# validator, without modifying any module globals in the archived controller.
accept=FunctionType(old_pipeline.accept.__code__,{**old_pipeline.accept.__globals__,'summary':summary})


def matched_ssh(config,*,execute_board=False):
    measure=FunctionType(SSH.measure.__code__,{**SSH.measure.__globals__,'native_command':native_command})
    return type('YOLOPolicyCheckSSH',(SSH,),{'measure':measure})(config,execute_board=execute_board)


def run(output,run_dir,local_config,*,execute_board=False):
    if not execute_board:raise ValueError('--execute-board is required')
    output,run_dir=Path(output).resolve(),Path(run_dir).resolve();assert_prepared(output)
    bundle=output/'pipeline_bundle';m=validate_bundle(bundle)
    ssh=matched_ssh(read(local_config)['board'],execute_board=True);boot=ssh.boot_id()
    with locked(run_dir):
        frozen(run_dir/'run.json',dict(boot_id=boot,bundle_id=m['bundle_id'],bundle=str(bundle)))
        frozen(run_dir/'selection_freeze.json',read(bundle/'plans/ranking_freeze.json'))
        frozen(run_dir/'GATE_VALIDATOR_COMPATIBILITY.json',adapted_summary()[1])
        started=time.monotonic()
        def state(stage,**fields):
            write(run_dir/'state.json',dict(status=stage,boot_id=boot,bundle=str(bundle),
                updated_at=now(),**fields),replace=True)
        try:
            remote=ssh.deploy(bundle,run_dir)
            ssh.qualify_environment(remote,boot,run_dir,bundle)
            write(run_dir/'hardware.json',hardware_check(ssh,bundle,boot),replace=True)
            order=read(bundle/'plans/execution_order.json');done=0
            state('MEASURING',accepted_sessions=done,candidate_count=len(order))
            for name in order:
                for phase in ('qualify','performance'):
                    item=accept(ssh,make_task(bundle,name,phase),bundle,remote,run_dir,boot);done+=1
                    state('MEASURING',accepted_sessions=done,candidate_count=len(order),last_candidate=name,last_phase=phase)
                    print('YOLO策略验证',done,'/',2*len(order),phase,name,item['summary'].get('fps','qualified'),flush=True)
            ssh.idle_command('cd '+shlex.quote(remote)+' && sha256sum -c board.sha256 > post_run_verification.log && sync',
                boot=boot,timeout=600)
            state('COMPLETE',accepted_sessions=done,candidate_count=len(order),
                complete_DNN_inferences=105*len(order),post_deployment_hashes_passed=True,
                execution_elapsed_seconds=time.monotonic()-started)
            sync_tree(run_dir)
        except Exception as error:
            state('STOPPED_ON_ERROR',error_type=type(error).__name__,error=str(error));raise
    return read(run_dir/'state.json')


def status(output,run_dir):
    output,run_dir=Path(output),Path(run_dir)
    return dict(prepared=(output/'pipeline_bundle/manifest.json').exists(),
                state=read(run_dir/'state.json') if (run_dir/'state.json').exists() else {'status':'NOT_STARTED'})
