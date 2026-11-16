"""New-cost controls and frozen-before-performance guards."""
from copy import deepcopy
from pathlib import Path
import json
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT.parents[1]/'src')]
from cpu_policies.open_resources.domain import component,component_key,stage
from phase_validation.predict import fresh_records
from phase_validation.project import thread_weights
from phase_validation.campaign import specs,board


def test_profile_refresh_is_readonly_and_network_scoped():
    row=stage('cpu',0,3,2,[0,1]);key=component_key(component(row))
    records={key:dict(cost={'service_ms':99},session_thread_weights=[.5,.5])}
    original=deepcopy(records)
    profile=dict(identity=dict(network='resnet18',program='cpu_00_02'),
                 configuration=component(row),clean_cost={'service_ms':10},
                 fresh_session_thread_weights=[.8,.2],provenance={'clean_receipt':'isolated_cpu'})
    foreign={**profile,'identity':{'network':'yolov3_tiny416','program':'cpu_00_02'},'clean_cost':{'service_ms':999}}
    fresh,replaced=fresh_records(records,'resnet18',{'profiles':{'one':profile,'other':foreign}},{'profiles':{}})
    assert records==original and fresh[key]['cost']['service_ms']==10
    assert fresh[key]['session_thread_weights']==[.8,.2] and len(replaced)==1


def test_whole_session_weights_are_projected_by_actual_logical_core():
    clean={'summary':{'thread_ledger':[dict(stage='cpu_00_02',actual_mask=[1],cpu_time_delta_ms=20),
                                      dict(stage='cpu_00_02',actual_mask=[0],cpu_time_delta_ms=80)]}}
    profile={'identity':dict(threads=2,program='cpu_00_02',mask=[0,1])}
    assert thread_weights(clean,profile)==[.8,.2]
    clean['summary']['thread_ledger'].append(clean['summary']['thread_ledger'][0])
    with pytest.raises(ValueError):thread_weights(clean,profile)


def test_pipeline_tasks_cannot_precede_or_outlive_prediction_seal(tmp_path):
    with pytest.raises(FileNotFoundError):specs(tmp_path,'pipelines')
    (tmp_path/'PREDICTION_FREEZE.json').write_text(json.dumps({'files':{'PREDICTIONS.json':'wrong'},'model_sources':{}}))
    (tmp_path/'PREDICTIONS.json').write_text('{}')
    with pytest.raises(ValueError,match='prediction changed'):specs(tmp_path,'pipelines')


def test_board_permission_flag_is_required_before_any_local_or_remote_io(tmp_path):
    with pytest.raises(ValueError,match='execute-board'):board(tmp_path,tmp_path,tmp_path/'missing',scope='components')
