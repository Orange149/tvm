"""Preparation may reuse old boots; pipeline execution never disguises them."""
from copy import deepcopy
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT.parents[1]/'src')]
import run_yolo_validation_resume as resume
from yolo_validation.common import DEFAULT_OUTPUT,DEFAULT_RUN,PROFILE,assert_sources
from cpu_vta_partitioning.common import read

def old_record():
    return next(iter(read(DEFAULT_RUN/'traffic/traffic.json')['records'].values()))

def test_existing_DMA_replays_with_original_boot():
    record=old_record();result=resume.replay_record(PROFILE,record,DEFAULT_RUN/'traffic')
    assert result['source']['boot_id']=='a1380ccd-26b4-4ccf-afa8-a6671f986e35'
    assert result['source']['task_id']==record['source']['task_id']
    assert Path(result['source']['receipt']).is_file()

@pytest.mark.parametrize('fault',['hash','boot'])
def test_wrong_hash_or_relabelled_boot_rejected(fault):
    record=deepcopy(resume.replay_record(PROFILE,old_record(),DEFAULT_RUN/'traffic'))
    record['source']['receipt_sha256' if fault=='hash' else 'boot_id']='wrong'
    with pytest.raises(ValueError):resume.replay_record(PROFILE,record,DEFAULT_RUN/'traffic')

def test_partial_DMA_cannot_be_used_for_ranking():
    with pytest.raises(ValueError,match='complete'):resume.replay_index(PROFILE,DEFAULT_RUN/'traffic')

def test_no_board_work_without_explicit_gate(monkeypatch):
    monkeypatch.setattr(resume,'SSH',lambda *a,**k:(_ for _ in ()).throw(AssertionError('board touched')))
    with pytest.raises(ValueError,match='execute-board'):resume.run(DEFAULT_OUTPUT,DEFAULT_RUN,'local')

def test_frozen_model_and_controller_sources_still_match():
    assert assert_sources(DEFAULT_OUTPUT)['old_results_read_only']
