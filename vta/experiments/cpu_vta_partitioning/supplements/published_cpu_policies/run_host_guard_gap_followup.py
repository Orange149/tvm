"""Reuse the existing complete formula after the CPU-exclusion diagnosis.

No new formula or coefficient: full-domain YOLO replay of the historical four
enabled factors. Separate destination; stopped baseline work remains read-only.
This entry has no board commands and never resumes the failed ResNet bridge.
"""
from pathlib import Path
from types import FunctionType
import argparse
import json
import shutil

import strong_baseline.ranking as original
from cpu_vta_partitioning.common import read,sha256
from finite_service.common import freeze

ROOT=original.ROOT
DEST=ROOT/'artifacts/host_guard_gap_followup_01/yolo_complete'


def sources():return {**original.sources(),str(Path(__file__)):sha256(__file__)}


def bind(name):
    f=getattr(original,name)
    return FunctionType(f.__code__,{**f.__globals__,'DEST':DEST,'sources':sources,
        'assert_inputs':assert_inputs,'engine':engine},f.__name__,f.__defaults__)


def prepare():
    result=bind('prepare')()
    freeze(DEST/'PURPOSE.json',dict(kind='existing_complete_formula_followup',
        exact_existing_switches=['thread_weights','host_CPU_demand','team_service','host_concentration'],
        no_new_formula_or_parameter=True,no_domain_restriction=True,no_component_refresh=True,
        evidence='CPU exclusion reduces some service gaps but leaves a large YOLO front gap; host/control/DDR remain concurrent',
        no_claim_guard_is_exact_response_time=True,
        source_diagnostic_results_sha256=sha256(ROOT/'artifacts/cpu_exclusion_gap_01/RESULTS.json'),
        rank_inputs_are_only_existing_isolated_costs_modules_resources_and_DDR=True,
        new_board_sessions=0,baseline_bridge_not_resumed=True))
    return result


def assert_inputs():return bind('assert_inputs')()
def engine(p):return bind('engine')(p)


def copy_verified_checkpoints():
    p=assert_inputs();module=engine(p)
    programs=read(DEST/'inputs/programs.json');records=read(DEST/'inputs/isolated_costs.json');ddr=read(DEST/'inputs/ddr.json')
    costs=original.Costs(records,programs=programs,ddr=ddr,distribution='session_weights')
    graphs=module._topologies(programs,14,3);copied={}
    old=ROOT/'artifacts/strong_baseline_01/yolo_complete/search'
    for ctx in module.domain.contexts():
        source=old/(ctx['id']+'.json')
        if not source.exists():continue
        cp=read(source);payload=dict(programs=module._prepare(programs,costs,ctx,ddr),topologies=graphs,
            context=ctx,k=20,score_scale=module.SCORE_SCALE,implementation=p['legacy_implementation'])
        fields=('counts','gate_counts','rankings','topology_strata','resource_strata')
        if (cp['input_sha256']!=module._digest(payload)
            or cp['checkpoint_sha256']!=module._digest({k:v for k,v in cp.items() if k!='checkpoint_sha256'})
            or not cp['verification']['passed']
            or cp['verification']['source_sha256']!=p['legacy_implementation']['oracle.cc']
            or cp['verification']['result_sha256']!=module._digest({k:cp[k] for k in fields})):
            raise ValueError('old static checkpoint does not exactly match current complete search')
        target=DEST/'search'/source.name;target.parent.mkdir(exist_ok=True)
        if target.exists() and sha256(target)!=sha256(source):raise ValueError('copied static checkpoint differs')
        if not target.exists():shutil.copy2(source,target)
        copied[str(source)]=sha256(source)
    freeze(DEST/'READ_ONLY_CHECKPOINT_REUSE.json',dict(copied=copied,original_files_modified=False,
        only_complete_hash_and_CPP_verified_checkpoints=True))
    return len(copied)


def rank():
    prepare();count=copy_verified_checkpoints();result=bind('rank')()
    return {**result,'verified_context_checkpoints_reused':count,'no_board_execution':True}


def validate():return bind('validate')()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare','rank','validate'))
    result=globals()[parser.parse_args().command]();print(json.dumps(result,ensure_ascii=False,indent=2))
