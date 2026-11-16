"""Preparation scope and immutable provenance of the per-core validation."""
import json

import pytest

from cpu_policies.core_model.common import DEFAULT_PARENT, DEFAULT_SOURCE_RUN, api, protocol
from cpu_policies.core_model.domain import PROGRAMS, baseline_candidate, requirements
from cpu_policies.core_model.prepare import BASELINE_FILES, _source_identity


def test_core_model_fixed_five_modules_and_eleven_new_components():
    programs = api().read(DEFAULT_PARENT / 'plans/programs.json')
    required = requirements(programs)
    assert len(required) == 11
    assert {stage['program'] for cfg in required.values() for stage in cfg['stages']} == set(PROGRAMS)
    assert len(PROGRAMS) == 5
    assert sum(cfg['stages'][0]['device'] == 'cpu' and cfg['stages'][0]['threads'] > 1
               for cfg in required.values()) == 6
    assert baseline_candidate() == api().read(DEFAULT_PARENT / 'plans/diagnosis_configurations.json')['pipelines']['static_first']


def test_core_model_budget_is_a_maximum_not_a_forced_pool_size():
    p = protocol()
    assert p['maximum_pipeline_count'] == 12
    assert 11 + 11 + 1 + 12 * 2 == p['maximum_tasks'] == 47
    assert 11 * 10 + 11 + 28 + 12 * 29 == p['maximum_forwards'] == 497
    assert p['performance_rejection_threshold'] is None
    assert p['automatic_window_extension'] is False
    assert p['formal_pipeline_probe'] is False
    assert p['worker_map_policy'] == 'caller_any_selected_core_helpers_remaining_ascending'


def test_core_model_parent_pass_through_costs_are_reference_only():
    manifest, provenance = _source_identity(DEFAULT_PARENT, DEFAULT_SOURCE_RUN)
    assert manifest['bundle_id'] == 'abbcccc62af230ee844745ac'
    assert provenance['source_state_sha256'] == api().sha256(DEFAULT_SOURCE_RUN / 'state.json')
    assert provenance['static_profile_input_policy'] == 'all_11_costs_are_new_this_batch_old_costs_reference_only'
    assert set(provenance['baseline_files']) == set(BASELINE_FILES)
    for name, expected in provenance['baseline_files'].items():
        assert api().sha256(DEFAULT_PARENT / 'plans' / name) == expected


def test_core_model_refuses_incomplete_or_other_source_batch(tmp_path):
    for state in ({'status': 'INTERRUPTED', 'bundle_id': 'abbcccc62af230ee844745ac'},
                  {'status': 'COMPLETE', 'bundle_id': 'other-bundle'}):
        (tmp_path / 'state.json').write_text(json.dumps(state))
        with pytest.raises(ValueError, match='completed run'):
            _source_identity(DEFAULT_PARENT, tmp_path)
