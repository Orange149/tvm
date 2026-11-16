from pathlib import Path
from types import FunctionType
import os, shutil, subprocess, time
from cpu_vta_partitioning.common import digest,read,write,sha256,verify_inventory
from yolo_scheduler_trace import prepare as old
from .common import ROOT,ORIGINAL,OUTPUT,PARENT,protocol,specs,sources,freeze
from .header_patch import patched_header,once


def build(output):
    dest=Path(output)/'build';src=dest/'sources';src.mkdir(parents=True,exist_ok=True)
    previous=read(ORIGINAL/'build/BUILD.json')
    for p,h in previous['source_files'].items():
        if sha256(p)!=h:raise ValueError('frozen observer source differs')
        shutil.copy2(p,src/Path(p).name)
    header=src/'vta_session_profile.h';header.write_text(patched_header(header.read_text()))
    shutil.copy2(Path(__file__).with_name('ring_drain.h'),src/'ring_drain.h')
    runner=src/'vta_stage_pipeline_runner_phase.cc'
    runner.write_text(once(runner.read_text(),'        collector_phase.End();',
        '        collector_phase.End();\n        session_profile.Checkpoint();'))
    tc=previous['toolchain']
    if sha256(tc['sdk_setup'])!=tc['sdk_setup_sha256']:raise ValueError('SDK differs')
    for t in list(tc['tools'].values())+list(tc.get('setup_files',{}).values()):
        if sha256(t['path'])!=t['sha256']:raise ValueError('SDK dependency differs')
    recipe=read(ROOT/'artifacts/cross_network_model_01/phase_profile/build/BUILD.json')['compiler_recipe']
    env=dict(os.environ);env.pop('LD_LIBRARY_PATH',None);tick=time.monotonic()
    argv=['bash','-c','source "$1" || exit $?; shift; exec "$@"','drained-sdk',tc['sdk_setup'],
        'bash','-c',recipe,'drained-compile',str(runner),str(dest/'vta_stage_pipeline_runner_phase'),
        str(ROOT.parents[4]),str(PARENT/'yolov3_tiny416/capsule/runtime')]
    p=subprocess.run(argv,env=env,capture_output=True,text=True,timeout=180)
    (dest/'compile.log').write_text(p.stdout+p.stderr)
    if p.returncode:raise ValueError('drained recorder build failed: '+str(dest/'compile.log'))
    proof=dict(runner_sha256=sha256(dest/'vta_stage_pipeline_runner_phase'),
        old_runner_sha256=previous['old_runner_sha256'],adapter_sha256=previous['adapter_sha256'],
        source_files={str(p):sha256(p) for p in src.iterdir()},toolchain=tc,
        runtime_rebuilt=False,modules_recompiled=False,seconds=time.monotonic()-tick,
        parent_build_sha256=sha256(ORIGINAL/'build/BUILD.json'),
        runner_change='drain existing target rings on collector thread after each frame; retain in memory; reject capacity uncertainty')
    write(dest/'BUILD.json',proof);return proof


def validate(output):
    output=Path(output);bundle=output/'capsule';m=read(bundle/'manifest.json')
    if digest({k:v for k,v in m.items() if k!='bundle_id'})[:24]!=m['bundle_id']:
        raise ValueError('drained capsule seal differs')
    verify_inventory(bundle,m['files']);old.validate_input_paths(bundle,m)
    seal=read(output/'PREPARATION_FREEZE.json')
    if seal['sources']!=sources() or seal['protocol']!=protocol() or seal['bundle_id']!=m['bundle_id']:
        raise ValueError('drained recorder controller or protocol changed')
    build_info=read(output/'build/BUILD.json')
    for p,h in build_info['source_files'].items():
        if sha256(p)!=h:raise ValueError('drained build source differs')
    if sha256(output/'build/vta_stage_pipeline_runner_phase')!=build_info['runner_sha256']:
        raise ValueError('drained executable differs')
    return dict(validated=True,bundle_id=m['bundle_id'],native_sessions=3,DNN_inferences=97,
        optional_new_measurement_not_executed_by_prepare=True)


def prepare(output=OUTPUT):
    old.validate(ORIGINAL)
    fn=FunctionType(old.prepare.__code__,{**old.prepare.__globals__,
        'build':build,'protocol':protocol,'specs':specs,'sources':sources,'validate':validate})
    return fn(output)


def qemu(output=OUTPUT):
    from .runtime import make_task
    fn=FunctionType(old.qemu.__code__,{**old.qemu.__globals__,
        'validate':validate,'make_task':make_task,'specs':specs},closure=old.qemu.__closure__)
    # old.qemu has a local relative import of its make_task. Replacing its
    # globals cannot override that import, so use the new package as __package__.
    fn.__globals__['__package__']=__package__
    return fn(output)
