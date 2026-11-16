#!/usr/bin/env python3
"""Replay/run with the frozen runner's documented omitted-off gate field.

The runner emits cpu_stage_gate_mode only for non-off modes (native source
lines 2693ff). This adapter mirrors the existing ResNet validator's default.
No task, executable, score, ranking or measurement window is modified.
"""
import argparse
import hashlib
import inspect
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from cpu_vta_partitioning.common import locked,now,read,sha256,write
from yolo_validation.common import DEFAULT_OUTPUT,DEFAULT_RUN,LOCAL_CONFIG,assert_sources,frozen

def adapted_summary():
    from yolo_transfer.pipeline import summary
    source=inspect.getsource(summary)
    old="row.get('cpu_stage_gate_mode')"
    if source.count(old)!=1:raise ValueError('frozen gate validation anchor changed')
    patched=source.replace(old,"row.get('cpu_stage_gate_mode', 'off')")
    namespace=dict(summary.__globals__)
    exec(compile(patched,'<YOLO frozen-runner omitted-off compatibility>','exec'),namespace)
    proof=dict(original_summary_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        adapted_summary_source_sha256=hashlib.sha256(patched.encode()).hexdigest(),
        exact_change="row.get('cpu_stage_gate_mode') -> row.get('cpu_stage_gate_mode', 'off')",
        mirrors_existing_ResNet_validator=True,native_programs_changed=False,scoring_changed=False,
        missing_non_off_mode_still_rejected=True)
    return namespace['summary'],proof

def install():
    from yolo_validation import pipeline
    function,proof=adapted_summary()
    pipeline.original_summary=function
    return proof

def run(output,original,config,*,execute_board=False,analysis_only=False):
    if not analysis_only and not execute_board:raise ValueError('--execute-board is required')
    from yolo_validation import pipeline,analysis
    output,original=Path(output).resolve(),Path(original).resolve();assert_sources(output)
    active=read(original/'ACTIVE_EXECUTION.json');cohort=Path(active['run_dir'])
    saved=read(cohort/'state.json');proof=install()
    proof.update(controller_sha256=sha256(Path(__file__)),controller_file=str(Path(__file__).resolve()))
    frozen(cohort/'GATE_VALIDATOR_COMPATIBILITY.json',proof)
    source=cohort/'sources/run_yolo_validation_gate_compat.py'
    if source.exists() and sha256(source)!=sha256(Path(__file__)):raise ValueError('validator source snapshot changed')
    if not source.exists():
        source.parent.mkdir(parents=True,exist_ok=True);source.write_bytes(Path(__file__).read_bytes())
    # The original failed validation record remains available and is not counted
    # as an additional inference or a performance session.
    failure=cohort/'FAILURE_01_GATE_FIELD.json'
    if not failure.exists() and saved['status']=='STOPPED_ON_ERROR':write(failure,saved)
    started=time.monotonic()
    with locked(cohort):
        def state(stage,**fields):
            value=dict(status=stage,boot_id=active['boot_id'],active_execution=str(cohort),
                output=str(output),updated_at=now(),validator_compatibility=str(cohort/'GATE_VALIDATOR_COMPATIBILITY.json'),**fields)
            write(cohort/'state.json',value,replace=True);write(original/'state.json',value,replace=True)
        try:
            if not analysis_only:
                state('PIPELINE')
                result=pipeline.run(output/'pipeline_bundle',cohort/'pipeline',config,execute_board=True)
            else:result=read(cohort/'pipeline/state.json')
            state('ANALYZING',pipeline=result)
            report=analysis.analyze(output,cohort)
            state('COMPLETE',pipeline=result,report=report,execution_elapsed_seconds=time.monotonic()-started)
            return read(cohort/'state.json')
        except Exception as error:
            state('STOPPED_ON_ERROR',error_type=type(error).__name__,error=str(error));raise

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    p.add_argument('--run-dir',type=Path,default=DEFAULT_RUN);p.add_argument('--local-config',type=Path,default=LOCAL_CONFIG)
    p.add_argument('--execute-board',action='store_true');p.add_argument('--analysis-only',action='store_true');a=p.parse_args()
    if not a.analysis_only and not a.execute_board:p.error('--execute-board is required')
    print(__import__('json').dumps(run(a.output,a.run_dir,a.local_config,
        execute_board=a.execute_board,analysis_only=a.analysis_only),ensure_ascii=False))

if __name__=='__main__':main()
