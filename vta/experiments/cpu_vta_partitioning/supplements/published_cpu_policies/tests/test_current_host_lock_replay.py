from copy import deepcopy
from pathlib import Path
from fractions import Fraction
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_current_host_lock_replay import normalize_device,exact
from phase_model.simulator import rational
from policy_branch_eval.common import isolated_inputs_only


def test_isolated_normalization_conserves_CPU_and_preserves_independent_completion():
    p=dict(clean_cost={'run_core_ms':.000040},stages=dict(host_serial_CPU=[10,20],devices=[
        dict(CPU=20,wall=100,submit_mmio_ns=2,completion_lower_ns=75,completion_upper_ns=85)]))
    original=deepcopy(p);s,factor=normalize_device(p)
    assert rational(factor)==Fraction(4,5)
    assert sum(rational(v) for v in s['host_serial_CPU'])+rational(s['devices'][0]['CPU'])==40
    assert rational(s['devices'][0]['wall'])-rational(s['devices'][0]['CPU'])==80
    assert s['devices'][0]['completion_lower_ns']==75 and p==original


def test_invalid_CPU_work_or_native_performance_input_is_rejected(tmp_path):
    with pytest.raises(ValueError):normalize_device(dict(clean_cost={'run_core_ms':1},stages=dict(host_serial_CPU=[0],devices=[])))
    forbidden=tmp_path/'runs/raw/result.jsonl';forbidden.parent.mkdir(parents=True);forbidden.write_text('FPS data')
    with isolated_inputs_only():
        with pytest.raises(RuntimeError,match='pipeline performance'):forbidden.read_text()
