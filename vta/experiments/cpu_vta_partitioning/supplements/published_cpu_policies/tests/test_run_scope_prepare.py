"""Protect frozen baseline identity and the exact small diagnostic scope."""
import json
from pathlib import Path

import pytest

from cpu_policies.run_scope.common import DEFAULT_PARENT, DEFAULT_SOURCE_RUN, api, protocol
from cpu_policies.run_scope.domain import EXPECTED_IDS, PROGRAMS, configurations
from cpu_policies.run_scope.prepare import BASELINE_FILES, _source_identity


def test_run_scope_two_cases_match_completed_first_diagnosis():
    common = api()
    programs = common.read(DEFAULT_PARENT / 'plans/programs.json')
    original = common.read(DEFAULT_PARENT / 'plans/diagnosis_configurations.json')
    actual = configurations(programs)
    assert list(actual['pipelines']) == ['static_first', 'history_host2']
    assert len(actual['components']) == 4
    for section, entries in actual.items():
        for name, value in entries.items():
            assert value == original[section][name]
    assert {name: cfg['id'] for name, cfg in actual['pipelines'].items()} == EXPECTED_IDS
    used = {stage['program'] for entries in actual.values() for cfg in entries.values()
            for stage in cfg['stages']}
    assert used == set(PROGRAMS) == set(programs)


def test_run_scope_small_measurement_budget_and_matched_windows():
    value = protocol()
    assert len(value['component_cases']) == 4
    assert len(value['pipeline_cases']) == 2
    assert value['performance_blocks'] == ['old_plain', 'new_plain_before', 'run_probe', 'new_plain_after']
    assert value['component_blocks'] == ['component_off', 'component_probe']
    assert 8 * 10 + 6 + 8 * 28 == value['expected_forwards'] == 310
    assert 8 + 6 + 8 == value['expected_tasks'] == 22
    assert value['component_probe_frames'] == list(range(2, 10))
    assert value['performance_probe_frames'] == list(range(8, 24))
    assert value['qualification_probe_frames'] == [0]
    assert value['detailed_session_profile'] is False
    assert value['kernel_settings_mutated'] is False
    assert value['automatic_window_extension'] is False
    assert value['performance_rejection_threshold'] is None


def test_run_scope_baseline_costs_are_original_parent_passthrough():
    manifest, source = _source_identity(DEFAULT_PARENT, DEFAULT_SOURCE_RUN)
    assert manifest['bundle_id'] == 'ffbf5c23b09f93fb8f51a6d6'
    assert 'not_new_diagnosis_calibration' in source['baseline_cost_table_origin']
    assert set(source['baseline_files']) == set(BASELINE_FILES)
    for name, expected in source['baseline_files'].items():
        assert api().sha256(DEFAULT_PARENT / 'plans' / name) == expected
    old = api().read(DEFAULT_PARENT / 'plans/executors.json')['policy']
    assert old['sha256'] == source['source_policy_executor_sha256']
    assert source['source_pinned_native_sha256'] == '672710ad702b187225834949d233bf8ae917dae0750c230c60728d4646c87ca8'


def test_run_scope_refuses_an_incomplete_or_wrong_parent_diagnosis(tmp_path):
    (tmp_path / 'state.json').write_text(json.dumps({'status': 'INTERRUPTED', 'bundle_id': 'ffbf5c23b09f93fb8f51a6d6'}))
    with pytest.raises(ValueError, match='completed run'):
        _source_identity(DEFAULT_PARENT, tmp_path)
    (tmp_path / 'state.json').write_text(json.dumps({'status': 'COMPLETE', 'bundle_id': 'other_bundle'}))
    with pytest.raises(ValueError, match='completed run'):
        _source_identity(DEFAULT_PARENT, tmp_path)
