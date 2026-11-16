import json
from pathlib import Path
import subprocess
import sys

import pytest

from cpu_policies.external_packrat import run_original_methods

ROOT=Path(__file__).resolve().parents[1]


def invoke(*args):
    p=subprocess.run([sys.executable,str(ROOT/'run.py'),*args],cwd=ROOT,text=True,capture_output=True,check=True)
    return json.loads(p.stdout)


def test_workflow_cli_reports_ms_as_frames_per_second():
    r=invoke('pipeit','--input','examples/pipeit_workflow.json')
    assert r['layers']==[[0,1,2],[3]]
    assert r['reproduction_level']=='paper_algorithms_1_and_2'
    assert r['interval']==3
    assert r['predicted_fps']==pytest.approx(1000/3)


def test_merge_interpretation_is_not_claimed_as_verified_author_dse():
    r=invoke('pipeit','--input','examples/pipeit.json')
    assert r['eligible_as_official_author_dse'] is False
    assert r['merge_traversal_is_declared_interpretation'] is True
    assert r['interval']==pytest.approx(6.6)


def test_offline_verification_preserves_method_scope_and_output(tmp_path):
    target=tmp_path/'verification.json'
    r=invoke('verify','--output',str(target))
    assert r==json.loads(target.read_text())
    assert r['board_jobs_started']==0
    assert r['packrat']['allocations']==[{'threads':2,'batch':2,'latency_ms':8.0}]*2
    assert r['runtime']['caller_included_in_threads']


def test_external_author_code_must_match_the_reviewed_source(tmp_path):
    source=tmp_path/'optimizer.py';source.write_text('raise RuntimeError("must not execute")\n')
    with pytest.raises(ValueError,match='reviewed commit'):
        run_original_methods(source,{1:{1:1}},1,1)
