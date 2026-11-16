"""Actual runner omits the gate JSON key only in off mode."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT.parents[1]/'src')]
from cpu_vta_partitioning.common import read
from yolo_validation.common import DEFAULT_OUTPUT,DEFAULT_RUN,assert_sources
from run_yolo_validation_gate_compat import adapted_summary

@pytest.fixture
def actual():
    cohort=Path(read(DEFAULT_RUN/'ACTIVE_EXECUTION.json')['run_dir'])
    task=next(p for p in sorted((cohort/'pipeline/raw').glob('*/*/task.json'))
        if read(p)['phase']=='qualify' and read(p)['configuration']['execution_mode']=='off')
    raw=next(task.parent.glob('transfers/*/collected'))
    return raw,read(task),DEFAULT_OUTPUT/'pipeline_bundle'

def test_exact_one_token_semantics_and_actual_off_qualification(actual):
    raw,doc,bundle=actual;function,proof=adapted_summary()
    assert 'cpu_stage_gate_mode' not in json.loads((raw/'result.jsonl').read_text().splitlines()[0])
    result=function(raw,doc,bundle)
    assert result['correct'] and proof['native_programs_changed'] is False
    assert assert_sources(DEFAULT_OUTPUT)['old_results_read_only']

def test_absent_field_does_not_authorize_non_off_mode(actual):
    raw,doc,bundle=actual;function,_=adapted_summary();bad=deepcopy(doc)
    bad['configuration']['execution_mode']='resource_group'
    with pytest.raises(ValueError,match='gate'):function(raw,bad,bundle)

@pytest.mark.parametrize('mode',[None,'exclusive','invalid'])
def test_explicit_wrong_or_invalid_mode_still_rejected(actual,tmp_path,mode):
    raw,doc,bundle=actual;clone=tmp_path/'raw';shutil.copytree(raw,clone)
    rows=[json.loads(x) for x in (clone/'result.jsonl').read_text().splitlines()]
    rows[0]['cpu_stage_gate_mode']=mode
    (clone/'result.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    function,_=adapted_summary()
    with pytest.raises(ValueError,match='gate'):function(clone,doc,bundle)
