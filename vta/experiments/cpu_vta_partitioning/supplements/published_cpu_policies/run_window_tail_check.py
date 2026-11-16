"""A prepared four-layout diagnostic; native execution requires approval.

Reuse the frozen executor, measurement validation and durable controller.
Keep eight warmup frames and sixteen completion timestamps; change only tail.
"""
from copy import deepcopy
from pathlib import Path
from types import FunctionType
import argparse
import json

import run_scope_board as parent
from cpu_vta_partitioning.common import read,write,sha256,verify_inventory
from common_top20.runtime import task,summary
from run_scope_rank import freeze
from audit_latest_dp_window import window_inventory

ROOT=parent.ROOT
OUT=ROOT/'artifacts/window_tail_check_20261112_01'
RUN=ROOT/'runs/window_tail_check_20261112_01'
OLD=ROOT/'artifacts/run_scope_retest_20261112_01'
CONFIG=parent.CONFIG


def sources():
    return {**parent.sources(),str(Path(__file__)):sha256(__file__),
        str(ROOT/'audit_latest_dp_window.py'):sha256(ROOT/'audit_latest_dp_window.py'),
        str(ROOT/'tests/test_window_tail_check.py'):sha256(ROOT/'tests/test_window_tail_check.py')}


def bind(name):
    original=getattr(parent,name)
    return FunctionType(original.__code__,{**original.__globals__,'OUT':OUT,'RUN':RUN,
        'sources':sources,'selection':selection,'assert_sources':assert_sources,
        'validate':validate,'bind_run':bind_run},original.__name__,original.__defaults__)


def representatives(old_result,old_pool):
    result={};lists={};order=[]
    for network in ('resnet18','yolov3_tiny416'):
        selected={}
        for budget in (1,20):
            rows=[r for r in old_result['budgets'] if r['network']==network
                  and r['method']=='original_complete' and r['budget']==budget]
            if len(rows)!=1:raise ValueError('frozen diagnostic representative ambiguous')
            selected['static_first' if budget==1 else 'short_window_winner']=deepcopy(old_pool[network][rows[0]['configuration']])
        lists[network]={k:[v] for k,v in selected.items()}
        result[network]={v['id']:v for v in selected.values()}
        for key in sorted(result[network]):order.append(dict(network=network,configuration=key))
    if sum(map(len,result.values()))!=4:raise ValueError('four distinct representatives expected')
    return result,lists,order


def freeze_plan():
    results_path=OLD/'analysis/RESULTS.json'
    for path,expected in read(OLD/'analysis/MANIFEST.json')['files'].items():
        if sha256(path)!=expected:raise ValueError('old accepted report changed')
    old=read(results_path)
    for leaf,expected in read(OLD/'FREEZE.json')['files'].items():
        if sha256(OLD/leaf)!=expected:raise ValueError('old candidate selection changed')
    pool,lists,order=representatives(old,read(OLD/'POOL.json'))
    protocol=deepcopy(read(OLD/'PROTOCOL.json'))
    protocol.update(experiment='window_tail_check_20261112_01',
        short_window=dict(warmup=8,scored=16,tail=24),
        performance=dict(warmup=8,scored=16,tail=24),
        maximum_distinct_configurations=4,native_sessions=8,DNN_inferences=196,
        current_state='PREPARED_TAIL_DIAGNOSTIC_AWAITING_APPROVAL',
        additional_native_campaign_requires_user_approval=True,
        user_authorized_scope='Pending explicit approval of the tail change; local preparation only.',
        diagnostic_only=True,not_a_new_Top20_or_independent_confirmation=True)
    plan=dict(configurations=4,native_sessions=8,DNN_inferences=196,
        short_original_window=dict(warmup=8,scored=16,tail=4),
        new_window=protocol['short_window'],expected_existing_boot=old['boot_id'],
        old_result_sha256=sha256(results_path),old_result_path=str(results_path),
        small_diagnostic_not_full_domain_retest=True,
        old_short_measurement_not_repeated=True,no_new_ranking=True,
        no_new_profiles_or_programs=True,no_reboot=True,no_reprogram=True,
        hypothesis='Stage-source exhaustion inside the short completion window changes contention; extra tail should remove that boundary condition.',
        decision='Compare stage exhaustion and inventory first; do not label any observed FPS difference as a pure tail causal effect or stable gain.',
        estimated_board_minutes=[3,5],requires_explicit_window_change_approval=True)
    for leaf,value in [('PLAN.json',plan),('POOL.json',pool),('LISTS.json',lists),('ORDER.json',order),('PROTOCOL.json',protocol)]:
        freeze(OUT/leaf,value)
    freeze(OUT/'FREEZE.json',dict(files={leaf:sha256(OUT/leaf) for leaf in
        ('PLAN.json','POOL.json','LISTS.json','ORDER.json','PROTOCOL.json')}))
    return plan


def selection():return bind('selection')()
def assert_sources():return bind('assert_sources')()
def validate(bundle):return bind('validate')(bundle)


def bind_run(run_dir,boot,bundles):
    if boot!=read(OUT/'PLAN.json')['expected_existing_boot']:
        raise ValueError('board boot changed; do not mix this window check with the previous short data')
    return bind('bind_run')(run_dir,boot,bundles)


def prepare():
    freeze_plan()
    return bind('prepare')()


def board(run_dir=RUN,config=CONFIG,*,execute_board=False):
    approval=OUT/'AUTHORIZATION.json'
    if not approval.exists() or read(approval).get('approved') is not True:
        raise ValueError('the user has not approved the planned tail-window change')
    if read(approval)['selection_sha256']!=sha256(OUT/'FREEZE.json'):
        raise ValueError('approved native scope changed')
    return bind('board')(run_dir,config,execute_board=execute_board)


def analyze(run_dir=RUN):
    run_dir=Path(run_dir);state=read(run_dir/'state.json');prepared=assert_sources()
    if state['status']!='COMPLETE' or state['native_sessions']!=8 or state['DNN_inferences']!=196:
        raise ValueError('four-layout native diagnostic incomplete')
    if read(run_dir/'run.json')['bundles']!=prepared['bundles']:
        raise ValueError('capsule identity changed')
    plan,pool,_=selection();rows=[]
    old=read(plan['old_result_path'])
    if sha256(plan['old_result_path'])!=plan['old_result_sha256']:
        raise ValueError('old comparison source changed')
    for network,candidates in pool.items():
        bundle=OUT/network/'capsule';validate(bundle)
        for key,cfg in candidates.items():
            fingerprints=None
            for phase in ('qualify','short'):
                document=task(bundle,cfg,phase)
                path=run_dir/'accepted'/state['boot_id']/(document['task_id']+'.json');receipt=read(path)
                if receipt['task']!=document or receipt['boot_id']!=plan['expected_existing_boot']:
                    raise ValueError('native task or boot mismatch')
                verify_inventory(receipt['raw'],receipt['files'])
                checked=summary(receipt['raw'],document,bundle)
                if checked!=receipt['summary']:raise ValueError('native acceptance changed')
                if phase=='qualify':fingerprints=checked['output_fingerprints']
                else:
                    if checked['output_fingerprints']!=fingerprints:raise ValueError('native output changed')
                    raw=[json.loads(s) for s in (Path(receipt['raw'])/'result.jsonl').read_text().splitlines()]
                    inventory=window_inventory(raw,cfg['stages'],8,16)
                    previous=next(r for r in old['rows'] if r['network']==network and r['configuration']==key)
                    rows.append(dict(network=network,configuration=key,old_short_FPS=previous['FPS'],
                        new_tail_FPS=checked['FPS'],inventory=inventory,receipt=str(path),receipt_sha256=sha256(path)))
    value=dict(rows=rows,model_unchanged=True,diagnostic_only=True,same_boot=state['boot_id'],
               old_short_and_new_tail_in_separate_processes_and_time_blocks=True)
    freeze(OUT/'analysis/RESULTS.json',value)
    return value


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('prepare','validate','qemu','board','status','analyze'))
    p.add_argument('--execute-board',action='store_true');p.add_argument('--run-dir',type=Path,default=RUN)
    p.add_argument('--config',type=Path,default=CONFIG);a=p.parse_args()
    if a.command=='prepare':r=prepare()
    elif a.command=='validate':r={n:validate(OUT/n/'capsule')['bundle_id'] for n in assert_sources()['bundles']}
    elif a.command=='qemu':r=bind('qemu')()
    elif a.command=='board':r=board(a.run_dir,a.config,execute_board=a.execute_board)
    elif a.command=='analyze':r=analyze(a.run_dir)
    else:r=read(a.run_dir/'state.json') if (a.run_dir/'state.json').exists() else dict(status='NOT_STARTED')
    print(json.dumps(r,ensure_ascii=False,indent=2))
