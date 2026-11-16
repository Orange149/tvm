"""Identity-only holdout selection cannot depend on observed throughput."""
from pathlib import Path
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT.parents[1]/'src')]
from phase_model.validation_plan import plan


def known(path,ids):
    (path/'KNOWN_CONFIGURATION_IDS.json').write_text(json.dumps({'ids':ids}))


def test_six_resource_holdouts_share_frozen_cuts_and_need_only_four_new_signatures(tmp_path):
    known(tmp_path,{'resnet18':[],'yolov3_tiny416':[]});plan(tmp_path)
    data=json.loads((tmp_path/'phase_validation/PLAN.json').read_text())
    assert data['pipeline_FPS_read'] is False
    assert data['new_component_signatures']==4 and data['new_pipeline_configurations']==6
    for cases in data['networks'].values():
        assert len({c['id'] for c in cases.values()})==3
        for cfg in cases.values():
            assert [s['threads'] for s in cfg['stages'] if s['device']=='cpu'] in [[3,2,1],[4,2,1],[2,2,1]]
            assert cfg['execution_mode']=='off' and cfg['host_mask']==[0,1,2,3]


def test_known_identity_is_replaced_by_rotation_without_using_performance(tmp_path):
    first=tmp_path/'first';first.mkdir();known(first,{'resnet18':[],'yolov3_tiny416':[]});plan(first)
    original=json.loads((first/'phase_validation/PLAN.json').read_text())['networks']
    second=tmp_path/'second';second.mkdir()
    excluded={n:[c['id'] for c in cases.values()] for n,cases in original.items()}
    known(second,excluded);plan(second)
    data=json.loads((second/'phase_validation/PLAN.json').read_text())['networks']
    for n,cases in data.items():
        assert all(c['id'] not in excluded[n] for c in cases.values())
