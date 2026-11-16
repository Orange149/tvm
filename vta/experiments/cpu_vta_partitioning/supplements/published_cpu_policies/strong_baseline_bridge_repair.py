"""Independent correction of qualification export mode; timed work is unchanged."""
from pathlib import Path
from types import FunctionType
import shutil
from cpu_vta_partitioning.common import read,write,sha256,digest,inventory
from cpu_vta_partitioning.measurement import frames,input_sequence,correctness
from cpu_policies.open_resources.runtime import _affinity
from phase_profile.prepare import link_checked
from finite_service.common import freeze
from strong_baseline import bridge as base

ROOT=base.ROOT
DEST=base.OUTPUT/'bridge02'
RUN=ROOT/'runs/resnet18_strong_baseline_bridge_02'
CONFIG=base.CONFIG
VARIANTS=base.VARIANTS
baseline=base.baseline
specs=base.specs


def sources():
    return {**base.sources(),**{str(p):sha256(p) for p in [Path(__file__),ROOT/'run_strong_baseline_bridge02.py',
        ROOT/'tests/test_strong_baseline_bridge02.py']}}


def protocol():
    p=base.protocol();p.update(experiment='resnet18_strong_baseline_bridge_02',
        qualification_output_mode='raw_all_stages',failed_parent_qualification_sessions=1,
        failed_parent_performance_sessions=0)
    return p


def prepare():
    target=DEST/'capsule'
    if (target/'manifest.json').exists():return validate(target)
    source=base.DEST/'capsule';base.validate(source);m=read(source/'manifest.json')
    for leaf,value in m['files'].items():link_checked(source/leaf,target/leaf,value['sha256'])
    write(target/'plans/bridge02.json',dict(sources=sources(),protocol=protocol(),
        original_plan_sha256=sha256(base.DEST/'BOARD_PLAN.json'),
        correction='qualifications export every stage; classification/raw performance modes unchanged'))
    new={k:v for k,v in m.items() if k not in ('bundle_id','files')}
    new.update(experiment=protocol()['experiment'],files=inventory(target))
    new['bundle_id']=digest(new)[:24];write(target/'manifest.json',new)
    original=read(base.DEST/'BOARD_PLAN.json')
    freeze(DEST/'BOARD_PLAN.json',dict(sources=sources(),protocol=protocol(),specs=specs(),
        baseline_sha256=sha256(base.OUTPUT/'BASELINE_FREEZE.json'),bundle_sha256=sha256(target/'manifest.json'),
        expected_boot_id=original['expected_boot_id'],sessions=6,DNN_inferences=87,
        classification_full_output_checked_in_qualification_only=True))
    return validate(target)


def validate(bundle):
    result=base.validate(bundle);p=read(Path(bundle)/'plans/bridge02.json')
    if p['sources']!=sources():raise ValueError('bridge02 sources changed')
    return result


def make_task(bundle,spec):
    doc=base.make_task(bundle,spec);doc.pop('task_id');doc['experiment']=protocol()['experiment']
    if doc['kind'].startswith('qualify'):doc['output_mode']='raw_all_stages'
    doc['task_id']=digest(doc)[:24];return doc


def native_command(doc,bundle,remote,job):
    fn=FunctionType(base.native_command.__code__,{**base.native_command.__globals__,'make_task':make_task})
    return fn(doc,bundle,remote,job)


def summary(raw,doc,bundle):
    if not doc['kind'].startswith('qualify'):return base.summary(raw,doc,bundle)
    raw,bundle=Path(raw),Path(bundle);rows=frames(raw/'result.jsonl',1);input_sequence(rows,doc)
    if rows[0].get('output_mode')!='raw_all_stages' or len(rows[0].get('stage_raw_outputs',[]))!=7:
        raise ValueError('bridge02 full stage outputs missing')
    ledger,idle=_affinity(read(raw/'affinity.json'),doc['configuration'],True)
    correctness(rows,doc,bundle,raw)
    outputs=rows[0].get('raw_outputs',[])
    if len(outputs)!=1:raise ValueError('bridge02 final output missing')
    record=read(bundle/'plans/inputs.json')['records'][0]
    import numpy as np
    expected=int(np.argmax(np.load(bundle/record['frontiers']['308']['npy'])))
    return dict(correct=True,output_fingerprints=[[o['index'],o['shape'],o['dtype'],o['fnv1a64']] for o in outputs],
        expected_top1=expected,full_tensor_output_observed=True,all_stage_outputs_checked=True,
        classification_only_when_requested=False,thread_ledger=ledger,idle_auxiliary_threads=idle)


def qemu():
    fn=FunctionType(base.qemu.__code__,{**base.qemu.__globals__,'DEST':DEST,'validate':validate,'make_task':make_task})
    return fn()


def board(*,execute_board=False):
    fn=FunctionType(base.board.__code__,{**base.board.__globals__,'DEST':DEST,'RUN':RUN,
        'sources':sources,'validate':validate,'make_task':make_task,'native_command':native_command,'summary':summary})
    return fn(execute_board=execute_board)
