"""Frozen bounded stage-one measurements and evidence-gated stage two."""
from pathlib import Path
import shlex
import time
from types import FunctionType
from cpu_vta_partitioning.common import digest,locked,now,read,sha256,sync_tree,write
from yolo_validation.pipeline import accept as old_accept
from yolo_validation.components import hardware_check
from .common import assert_prepared,frozen,protocol
from .prepare import validate_bundle
from .runtime import make_task,matched_ssh,summary

accept=FunctionType(old_accept.__code__,{**old_accept.__globals__,'summary':summary})


def key(spec):return (spec['scope'],spec['case'],spec['policy'],spec['probe'])


def decide(results):
    isolated=results[('component','overlap4','original',False)]['cost']['service_ms']
    clean=results[('pipeline','overlap4','original',False)]['stage_service_ms'][0]
    observed=results[('pipeline','overlap4','original',True)]
    measured=observed['stage_service_ms'][0];gap=clean-isolated;perturbation=abs(measured-clean)
    stage=observed['wait']['stages']['0']
    target_yields=stage['barrier_yield_calls']+stage['completion_yield_calls']
    conditions=dict(positive_slowdown=gap>0,target_wait_yields=target_yields>0,
                    probe_within_twenty_percent_of_gap=gap>0 and perturbation<=.2*gap)
    return dict(execute_blocking=all(conditions.values()),conditions=conditions,
        isolated_service_ms=isolated,pipeline_service_ms=clean,observed_pipeline_service_ms=measured,
        gap_ms=gap,probe_perturbation_ms=perturbation,target_wait_yield_calls=target_yields,
        decision_is_not_causal_proof=True,no_extra_measurements_on_failure=True)


def run(output,run_dir,local_config,*,execute_board=False):
    if not execute_board:raise ValueError('--execute-board required')
    output,run_dir=Path(output).resolve(),Path(run_dir).resolve();assert_prepared(output)
    bundle=output/'capsule';m=validate_bundle(bundle);ssh=matched_ssh(read(local_config)['board'],execute_board=True)
    boot=ssh.boot_id();started=time.monotonic()
    with locked(run_dir):
        frozen(run_dir/'run.json',dict(boot_id=boot,bundle_id=m['bundle_id'],bundle=str(bundle)))
        frozen(run_dir/'protocol_freeze.json',dict(protocol=protocol(),preparation_sha256=sha256(output/'PREPARATION.json'),
            task_order_sha256=sha256(bundle/'plans/task_order.json')))
        def state(status,**fields):
            write(run_dir/'state.json',dict(status=status,boot_id=boot,bundle=str(bundle),updated_at=now(),**fields),replace=True)
        try:
            remote=ssh.deploy(bundle,run_dir);ssh.qualify_environment(remote,boot,run_dir,bundle)
            write(run_dir/'hardware.json',hardware_check(ssh,bundle,boot),replace=True)
            kernel=ssh.idle_command('zcat /proc/config.gz | grep -E "CONFIG_(SCHEDSTATS|FTRACE|FUTEX|SCHED_INFO)"',boot=boot)
            if '# CONFIG_SCHEDSTATS is not set' not in kernel or '# CONFIG_FTRACE is not set' not in kernel:
                raise ValueError('kernel diagnostic assumptions differ; do not reinterpret counters')
            frozen(run_dir/'KERNEL_CAPABILITIES.json',dict(raw=kernel,schedstats=False,ftrace=False,
                zeros_are_not_no_wait=True,kernel_modified=False))
            results={};receipts={};done=full=components=0
            order=read(bundle/'plans/task_order.json');decision=None
            for spec in order:
                if spec['stage']==2 and decision is None:
                    decision=decide(results)
                    decision.update(first_stage_receipts=receipts.copy(),frozen_at=now())
                    # When resuming, preserve the initial decision time and receipt hashes.
                    path=run_dir/'BLOCKING_DECISION.json'
                    if path.exists():
                        original=read(path)
                        if any(original[k]!=v for k,v in decision.items() if k!='frozen_at'):
                            raise ValueError('blocking decision changed on resume')
                        decision=original
                    else:frozen(path,decision)
                    print('等待干预条件',decision,flush=True)
                if spec['stage']==2 and not decision['execute_blocking']:continue
                for qualify in (True,False):
                    doc=make_task(bundle,spec,qualify)
                    item=accept(ssh,doc,bundle,remote,run_dir,boot);done+=1
                    count=sum(doc[k] for k in ('warmup','scored','tail'))
                    if spec['scope']=='pipeline':full+=count
                    else:components+=count
                    receipt=run_dir/'accepted'/boot/(doc['task_id']+'.json');receipts[str(receipt)]=sha256(receipt)
                    if not qualify:results[key(spec)]=item['summary']
                    state('MEASURING',accepted_sessions=done,DNN_inferences=full,component_calls=components,last_spec=spec)
                    print('等待诊断',done,'/ 24上限',spec,'qualify' if qualify else 'measure',
                          item['summary'].get('fps',item['summary'].get('stage_service_ms','qualified')),flush=True)
            ssh.idle_command('cd '+shlex.quote(remote)+' && sha256sum -c board.sha256 > post_run_verification.log && sync',
                boot=boot,timeout=600)
            state('COMPLETE',accepted_sessions=done,DNN_inferences=full,component_calls=components,
                blocking_executed=decision['execute_blocking'],post_deployment_hashes_passed=True,
                execution_elapsed_seconds=time.monotonic()-started)
            sync_tree(run_dir)
        except Exception as error:
            state('STOPPED_ON_ERROR',error_type=type(error).__name__,error=str(error));raise
    return read(run_dir/'state.json')


def status(output,run_dir):
    output,run_dir=Path(output),Path(run_dir)
    return dict(prepared=(output/'capsule/manifest.json').exists(),
        state=read(run_dir/'state.json') if (run_dir/'state.json').exists() else {'status':'NOT_STARTED'})
