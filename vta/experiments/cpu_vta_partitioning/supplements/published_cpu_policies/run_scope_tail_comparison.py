"""Prepared three-method, two-network comparison with the audited tail.

The88-layout campaign is an expansion. Preparation is local; board execution
requires a separate concrete approval bound to this selection's hash.
"""
from copy import deepcopy
from pathlib import Path
from types import FunctionType
import argparse
import inspect
import json

import run_scope_board as parent
from cpu_vta_partitioning.common import read,write,sha256
from run_scope_rank import freeze,assert_plan
from policy_branch_eval.common import isolated_inputs_only

ROOT=parent.ROOT
OLD=ROOT/'artifacts/run_scope_common_validation_01'
OUT=ROOT/'artifacts/run_scope_tail_comparison_20261112_01'
RUN=ROOT/'runs/run_scope_tail_comparison_20261112_01'
CONFIG=parent.CONFIG


def sources():
    return {**parent.sources(),str(Path(__file__)):sha256(__file__),
        str(ROOT/'analyze_run_scope_board.py'):sha256(ROOT/'analyze_run_scope_board.py'),
        str(ROOT/'tests/test_run_scope_tail_comparison.py'):sha256(ROOT/'tests/test_run_scope_tail_comparison.py')}


def bind(name):
    f=getattr(parent,name)
    return FunctionType(f.__code__,{**f.__globals__,'OUT':OUT,'RUN':RUN,
        'sources':sources,'selection':selection,'assert_sources':assert_sources,
        'validate':validate,'bind_run':bind_run},f.__name__,f.__defaults__)


def freeze_selection():
    assert_plan()
    old=read(OLD/'FREEZE.json')
    for leaf,h in old['files'].items():
        if sha256(OLD/leaf)!=h:raise ValueError('old full-domain static selection changed')
    source=read(OLD/'PLAN.json')
    for p,h in source['sources'].items():
        if sha256(p)!=h:raise ValueError('original complete-domain ranking evidence changed')
    with isolated_inputs_only():
        lists=read(OLD/'LISTS.json');pool=read(OLD/'POOL.json');order=read(OLD/'ORDER.json')
        for methods in lists.values():
            if set(methods)!={'tarnawski','original_complete','run_scope_complete'}:raise ValueError('three declared methods required')
            if any(len(rows)!=20 or len({r['id'] for r in rows})!=20 for rows in methods.values()):raise ValueError('full Top20 required')
        if {n:len(v) for n,v in pool.items()}!={'resnet18':40,'yolov3_tiny416':48}:raise ValueError('frozen common domain union differs')
        p=deepcopy(read(OLD/'PROTOCOL.json'))
        p.update(experiment='run_scope_tail_comparison_20261112_01',
            performance=dict(warmup=8,scored=16,tail=24),short_window=dict(warmup=8,scored=16,tail=24),
            current_state='PREPARED_PENDING_EXPANSION_APPROVAL',additional_native_campaign_requires_user_approval=True,
            no_new_temporal_model_in_comparison=True,
            purpose='Fair full-flow budgets1/5/20 for Tarnawski adaptation, strongest complete model, and isolated run-share input repair. Temporal replay remains diagnostic.')
        plan=dict(configurations=88,native_sessions=176,DNN_inferences=4312,
            network_counts={'resnet18':40,'yolov3_tiny416':48},methods=source['methods'],
            old_selection_sha256=sha256(OLD/'FREEZE.json'),static_lists_identical_to_old_full_domain_search=True,
            no_added_historical_layouts=True,no_new_ranking=True,no_new_component_profiles=True,
            no_new_runtime_or_executor=True,no_temporal_replay_promoted=True,
            candidate_pool_unchanged_only_measurement_tail_changed=True,
            expected_existing_boot=read(ROOT/'artifacts/current_pipeline_timeline_20261112_01/PLAN.json')['protocol']['expected_boot'],
            board_execution_approved=False,native_started=False)
        for leaf,v in [('PLAN.json',plan),('LISTS.json',lists),('POOL.json',pool),('ORDER.json',order),('PROTOCOL.json',p)]:
            freeze(OUT/leaf,v)
        freeze(OUT/'FREEZE.json',dict(files={leaf:sha256(OUT/leaf) for leaf in
            ('PLAN.json','LISTS.json','POOL.json','ORDER.json','PROTOCOL.json')},new_native_sessions=0,no_pipeline_FPS_input=True))
    return plan


def selection():return bind('selection')()
def assert_sources():return bind('assert_sources')()
def validate(bundle):return bind('validate')(bundle)


def bind_run(run_dir,boot,bundles):
    if boot!=read(OUT/'PLAN.json')['expected_existing_boot']:raise ValueError('boot changed; establish a separate authorized batch')
    return bind('bind_run')(run_dir,boot,bundles)


def prepare():
    freeze_selection();return bind('prepare')()


def board(run_dir=RUN,config=CONFIG,*,execute_board=False):
    if not execute_board:raise ValueError('--execute-board is required')
    auth=OUT/'AUTHORIZATION.json'
    if not auth.exists() or read(auth).get('approved') is not True:
        raise ValueError('88-layout expansion requires the explicit user approval described in goal section6')
    if read(auth)['selection_sha256']!=sha256(OUT/'FREEZE.json'):raise ValueError('approved scope differs')
    return bind('board')(run_dir,config,execute_board=True)


def analyze(run_dir=RUN):
    import analyze_run_scope_board as analysis
    source=inspect.getsource(analysis.analyze)
    before='8预热+16完成时间戳+4尾部'
    if source.count(before)!=1:raise ValueError('analysis protocol description anchor changed')
    updated=source.replace(before,'8预热+16完成时间戳+24尾部')
    namespace={**analysis.__dict__,'OUT':OUT,'RUN':RUN,'__file__':__file__,
        'selection':selection,'assert_sources':assert_sources,'validate':validate}
    exec(compile(updated,'<same common analysis with audited tail24>','exec'),namespace)
    return namespace['analyze'](run_dir)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','validate','qemu','board','status','analyze'))
    p.add_argument('--execute-board',action='store_true');a=p.parse_args()
    if a.command=='prepare':value=prepare()
    elif a.command=='validate':value={n:validate(OUT/n/'capsule')['bundle_id'] for n in assert_sources()['bundles']}
    elif a.command=='qemu':value=bind('qemu')()
    elif a.command=='board':value=board(execute_board=a.execute_board)
    elif a.command=='analyze':value=analyze()
    else:value=read(RUN/'state.json') if (RUN/'state.json').exists() else dict(status='NOT_STARTED',approval_pending=True)
    print(json.dumps(value,ensure_ascii=False,indent=2))
