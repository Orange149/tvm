from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'src'))
from staged_ablation.common import settings
from strong_baseline.audit import ROOT
from policy_branch_eval.common import isolated_inputs_only
import pytest


def test_complete_baseline_switches_are_not_the_later_reduced_main_model():
    full=settings('concentrated');reduced=settings('fine_host')
    assert full['host_guard'] and not reduced['host_guard']
    assert all(full[k] for k in ('thread_weights','host_cpu_demand','team_guard'))


def test_ranking_guard_refuses_pipeline_performance_files():
    with isolated_inputs_only():
        with pytest.raises((ValueError,RuntimeError,PermissionError)):
            (ROOT/'artifacts/device_completion_holdout_01/analysis/RESULTS.json').read_text()
