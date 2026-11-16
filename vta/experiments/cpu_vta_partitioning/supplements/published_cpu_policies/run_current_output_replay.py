"""Existing isolated output work on current cases, no scheduler substitution.

The four extra offline replays have a frozen primary ring1 and ring4 sensitivity.
They do not read native pipeline results, change profiles, or add cost to II.
"""
from pathlib import Path
import argparse
import json
import time

import run_current_host_lock_replay as parent
from cpu_vta_partitioning.common import read,write,sha256,digest
from run_scope_rank import freeze
from isolated_output_control_model import IsolatedOutputFinite,validate_cost
from policy_branch_eval.common import isolated_inputs_only

ROOT=parent.ROOT
OUT=ROOT/'artifacts/current_output_replay_20261112_01'
COST=ROOT/'artifacts/isolated_output_replay_01/ISOLATED_COSTS.json'


def prepare():
    if (OUT/'PLAN.json').exists():return check()
    parent.check();seal=read(parent.OUT/'REPLAY_FREEZE.json')
    for leaf,h in seal['files'].items():
        if sha256(parent.OUT/leaf)!=h:raise ValueError('baseline replay changed')
    with isolated_inputs_only():
        inputs=read(parent.OUT/'ISOLATED_INPUTS.json');raw=read(COST)
        if not raw['no_DNN_or_pipeline_measurement_used']:raise ValueError('isolated output cost required')
        costs={str(r):validate_cost(raw['costs']['yolov3_tiny416_ring'+str(r)]) for r in (1,4)}
        if any(v['runtime_sha256']!='724d15640bc5f75548725048f9b6adb1750bce768aa6ea39fe3fbaeb6726af98' for v in costs.values()):
            raise ValueError('output/runtime identity differs')
        freeze(OUT/'ISOLATED_INPUTS.json',dict(**inputs,output_costs=costs))
    names=['run_current_output_replay.py','isolated_output_control_model.py','compiled_sleep_model.py']
    return freeze(OUT/'PLAN.json',dict(input_sha256=sha256(OUT/'ISOLATED_INPUTS.json'),
        parent_plan_sha256=sha256(parent.OUT/'PLAN.json'),baseline_seal_sha256=sha256(parent.OUT/'REPLAY_FREEZE.json'),
        isolated_output_source_sha256=sha256(COST),sources={str(ROOT/n):sha256(ROOT/n) for n in names},
        replays=4,new_native_sessions=0,new_DNN=0,primary_ring=1,sensitivity_ring=4,
        wake_substitution=False,sleeper_substitution=False,other_scheduler_rules_unchanged=True,
        output_cost_added_as_collector_actor_not_direct_II_penalty=True,
        hypothesis='Does independently measured output serialization materially affect current host-lock coupling?',
        no_choice_by_known_FPS=True,no_automatic_full_domain_promotion=True))


def check():
    plan=read(OUT/'PLAN.json');parent.check()
    if plan['input_sha256']!=sha256(OUT/'ISOLATED_INPUTS.json') or plan['isolated_output_source_sha256']!=sha256(COST):
        raise ValueError('control replay inputs changed')
    for p,h in plan['sources'].items():
        if sha256(p)!=h:raise ValueError('control replay source changed')
    if (plan['parent_plan_sha256']!=sha256(parent.OUT/'PLAN.json') or
        plan['baseline_seal_sha256']!=sha256(parent.OUT/'REPLAY_FREEZE.json')):
        raise ValueError('baseline changed')
    return plan


def replay():
    check();inputs=read(OUT/'ISOLATED_INPUTS.json');files={};started=time.monotonic()
    for key,item in inputs['cases'].items():
        for ring in (1,4):
            spec=dict(configuration=key,ring=ring);path=OUT/'replays'/(digest(spec)[:24]+'.json')
            if not path.exists():
                with isolated_inputs_only():
                    value=IsolatedOutputFinite(item['stages'],quanta=inputs['quanta_ns'],
                        wake=False,sleeper=False,isolated_output_cost=inputs['output_costs'][str(ring)],
                        collector_mask=item['configuration']['control_mask'],placement='wake_boundary',
                        completion_model='independent',completion_point='midpoint',runnable_completion_wait=True,
                        controls=None,gate=item['configuration']['execution_mode'],host_order='before',**inputs['window']).run()
                if value['compiled_constant_substitution']['replacements']:raise ValueError('unexpected scheduler rule substitution')
                freeze(path,dict(spec=spec,result=value,plan_sha256=sha256(OUT/'PLAN.json'),no_pipeline_performance_read=True))
            elif read(path)['plan_sha256']!=sha256(OUT/'PLAN.json'):raise ValueError('cached replay changed')
            files[str(path.relative_to(OUT))]=sha256(path)
            print('Existing isolated collector work',key,ring,flush=True)
    freeze(OUT/'REPLAY_FREEZE.json',dict(files=files,no_pipeline_performance_read=True))
    freeze(OUT/'COST.json',dict(elapsed_seconds=time.monotonic()-started,replays=len(files),new_DNN=0))
    return dict(replays=len(files),new_native_sessions=0)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','replay','validate'))
    a=p.parse_args();print(json.dumps(prepare() if a.command=='prepare' else replay() if a.command=='replay' else check(),ensure_ascii=False,indent=2))
