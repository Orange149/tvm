from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_scope_inputs import RunScopeCosts,SCOPE,OLD_ASSUMPTION,NEW_ASSUMPTION
from cpu_policies.open_resources.costs import allocate_core_ns
from cpu_policies.open_resources.domain import stage,component,component_key
from cpu_vta_partitioning.common import digest


def fixture():
    row=stage('cpu',0,1,2,[0,1]);cfg=component(row);key=component_key(cfg)
    records={key:dict(configuration=cfg,kind='measured',session_thread_weights=[.6,.4],
        cost=dict(service_ms=10,run_core_ms=12,boundary_core_ms=2))}
    entry=dict(status='reconstruction_available',original_weights=[.6,.4],
        estimated_thread_CPU_ns=[30,70],all_recorded_run_CPU_ns=100,
        threads=2,source_receipt='isolated.fixture',source_boot='original_boot',scope=SCOPE)
    patches=dict(original_records_digest=digest(records),CPU_multithread_entries={key:entry},scope=SCOPE)
    return row,key,records,patches


def test_only_share_changes_with_exact_total_work_and_provenance():
    row,key,records,patches=fixture();before=deepcopy(records)
    value=RunScopeCosts(records,patches).get(row)
    assert value['cpu_core_weights']==[.3,.7]
    assert [value[k] for k in ('service_ms','run_core_ms','boundary_core_ms')]==[10,12,2]
    assert sum(allocate_core_ns(12_000_000,value['cpu_core_weights']))==12_000_000
    assert OLD_ASSUMPTION not in value['assumptions']
    assert NEW_ASSUMPTION in value['assumptions']
    assert records==before


def test_core_alias_keeps_same_scoped_estimate_and_explicit_homogeneous_assumption():
    row,key,records,patches=fixture()
    other=stage('cpu',0,1,2,[2,3]);value=RunScopeCosts(records,patches).get(other)
    assert value['source_key']==key and value['cpu_core_weights']==[.3,.7]
    assert 'homogeneous_core_permutation_not_per_mask_measurement' in value['assumptions']


def test_unobservable_worker_preserves_original_share_and_its_scope_warning():
    row,key,records,patches=fixture();entry=patches['CPU_multithread_entries'][key]
    entry['status']='retain_original_unobservable_noncaller';entry['estimated_thread_CPU_ns']=None
    value=RunScopeCosts(records,patches).get(row)
    assert value['cpu_core_weights']==[.6,.4]
    assert OLD_ASSUMPTION in value['assumptions']


@pytest.mark.parametrize('corruption',['missing','nonconserved','different_original','FPS'])
def test_inconsistent_or_performance_contaminated_inputs_rejected(corruption):
    row,key,records,patches=fixture()
    if corruption=='missing':
        patches['CPU_multithread_entries'].clear()
    elif corruption=='nonconserved':
        patches['CPU_multithread_entries'][key]['estimated_thread_CPU_ns']=[30,80]
    elif corruption=='different_original':
        patches['CPU_multithread_entries'][key]['original_weights']=[.5,.5]
    else:
        patches['CPU_multithread_entries'][key]['pipeline_FPS']=42
    with pytest.raises(ValueError):
        RunScopeCosts(records,patches)
