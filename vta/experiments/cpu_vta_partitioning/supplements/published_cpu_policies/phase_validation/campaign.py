"""Bind existing strong measurement/resume machinery to the frozen holdouts."""
from pathlib import Path
import shutil
import subprocess
from types import FunctionType

from cpu_vta_partitioning.common import read,sha256,write
from phase_profile import campaign as base
from phase_profile.runtime import make_task,native_command,matched_ssh
from phase_profile.qemu import qemu_summary
from workload_model.common import OUTPUT,ROOT
from .prepare import validate


def specs(output,scope):
    output=Path(output)
    if scope=='components':return read(output/'COMPONENT_TASKS.json')
    seal=read(output/'PREDICTION_FREEZE.json')
    for name,expected in seal['files'].items():
        if sha256(output/name)!=expected:raise ValueError('holdout prediction changed')
    for name,expected in seal['model_sources'].items():
        if sha256(name)!=expected:raise ValueError('holdout predictor code changed')
    return read(output/'PIPELINE_TASKS.json')


def board(output,run_dir,local_config,*,scope,execute_board=False):
    if not execute_board:raise ValueError('--execute-board is required')
    output,run_dir=Path(output).resolve(),Path(run_dir).resolve()
    proof=read(output/'QEMU_QUALIFICATION.json')
    if not proof['passed'] or proof['capsule_ids']!={n:validate(output/n/'capsule')['bundle_id']
                                                  for n in ('resnet18','yolov3_tiny416')}:
        raise ValueError('ARM qualification does not bind current capsules')
    boot=read(ROOT/'runs/cross_network_phase_profile_01/run.json')['boot_id']
    def fixed_boot_ssh(config,*,execute_board=False):
        instance=matched_ssh(config,execute_board=execute_board)
        original=instance.boot_id
        def checked_boot():
            current=original()
            if current!=boot:raise ValueError('board boot changed; bounded holdout batch cannot proceed')
            return current
        instance.boot_id=checked_boot
        return instance
    run=FunctionType(base.run.__code__,{**base.run.__globals__,'specs':specs,'validate':validate,
                                      'matched_ssh':fixed_boot_ssh})
    return run(output,run_dir,local_config,scope=scope,execute_board=True)


def qemu(output):
    output=Path(output).resolve();executable=shutil.which('qemu-aarch64-static')
    if not executable:raise RuntimeError('QEMU required')
    results=[]
    for spec in specs(output,'components'):
        if spec['phase']!='qualify':continue
        bundle=output/spec['network']/'capsule';validate(bundle)
        doc=make_task(bundle,spec);raw=output/'qemu'/(spec['network']+'_'+spec['case'])
        raw.mkdir(parents=True,exist_ok=True)
        argv,inputs=native_command(doc,bundle,str(bundle),str(raw));(raw/'inputs.txt').write_text(inputs)
        at=argv.index(str(bundle)+'/runtime/ld-linux-aarch64.so.1')
        command=[executable,*[v for item in argv[2:at] for v in ('-E',item)],*argv[at:]]
        result=subprocess.run(command,capture_output=True,text=True,timeout=120)
        (raw/'native.log').write_text(result.stdout+result.stderr)
        if result.returncode:raise ValueError('ARM holdout component failed: '+str(raw))
        result=qemu_summary(raw,doc,bundle)
        results.append(dict(network=spec['network'],case=spec['case'],summary=result))
        print('ARM两线程资格通过',spec['network'],spec['case'],flush=True)
    if len(results)!=4:raise ValueError('four new CPU signatures must be qualified')
    write(output/'QEMU_QUALIFICATION.json',dict(passed=True,cases=results,
        capsule_ids={n:read(output/n/'capsule/manifest.json')['bundle_id'] for n in ('resnet18','yolov3_tiny416')},
        board_access=False),replace=True)
    return dict(passed=True,component_calls=4)
