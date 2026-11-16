from copy import deepcopy
import json
from pathlib import Path
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT.parents[1]/'src')]
from run_yolo_wait_diagnosis_compat import passive_slots
from cpu_vta_partitioning.common import read
from wait_diagnosis.common import OUTPUT,RUN


@pytest.fixture
def actual():
    path=next(p for p in (RUN/'raw').glob('*/*/task.json') if not read(p)['qualification'])
    doc=read(path);raw=sorted(path.parent.glob('transfers/*/collected'))[-1]
    return raw,doc,read(raw/'affinity.json')


def test_actual_passive_worker_map_and_original_data_replay(actual):
    from wait_diagnosis.runtime import trace_summary
    raw,doc,a=actual;assert not a['stages'][0]['worker_slots']
    result=trace_summary(raw,doc,OUTPUT/'capsule',passive_slots(doc,a))
    assert result['stages']['0']['kernel_launches']>0
    assert not a['stages'][0]['worker_slots']


def test_missing_active_qualification_still_rejected(actual):
    raw,doc,a=actual;doc=deepcopy(doc);doc['qualification']=True
    with pytest.raises(ValueError,match='active'):passive_slots(doc,a)


def test_wrong_core_caller_and_duplicate_workers_rejected(actual):
    raw,doc,a=actual
    for change in ('core','caller','duplicate'):
        bad=deepcopy(a);stage=bad['stages'][0]
        if change=='core':stage['effective_affinities'][0]['cpus']=[99]
        elif change=='caller':stage['caller_tid']=999
        else:stage['effective_affinities'][1]['cpus']=stage['effective_affinities'][0]['cpus']
        with pytest.raises(ValueError):passive_slots(doc,bad)
