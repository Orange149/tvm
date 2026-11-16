"""The diagnosis must reproduce execution identities without touching old batches."""
from hashlib import sha256
from pathlib import Path

import pytest

from cpu_policies.diagnosis.domain import EXPECTED_IDS, PROGRAMS, configurations
from cpu_policies.diagnosis.prepare import _copy
from cpu_policies.diagnosis.common import DEFAULT_PARENT, api, protocol


def test_diagnosis_seven_cases_and_eight_module_closure():
    common = api()
    programs = common.read(DEFAULT_PARENT / 'plans/programs.json')
    values = configurations(programs)
    assert len(values['components']) == 4
    assert len(values['pipelines']) == 3
    assert {name: cfg['id'] for name, cfg in values['pipelines'].items()} == EXPECTED_IDS
    used = {s['program'] for section in values.values() for cfg in section.values()
            for s in cfg['stages']}
    assert used == set(PROGRAMS)
    assert len(PROGRAMS) == 8


def test_diagnosis_one_thread_is_measured_on_actual_core_three():
    cfg = configurations()['components']['cpu05_t1']
    assert cfg['stages'][0]['threads'] == 1
    assert cfg['stages'][0]['mask'] == [3]
    assert cfg['stages'][0]['worker_core_map'] == [3]
    assert cfg['stages'][0]['caller_core'] == 3


def test_diagnosis_only_host_placement_differs_in_matched_history_pair():
    values = configurations()['pipelines']
    pinned, shared = values['history_host2'], values['history_shared']
    for a, b in zip(pinned['stages'], shared['stages']):
        if a['device'] == 'cpu':
            assert a == b
        else:
            assert (a['begin'], a['end']) == (b['begin'], b['end'])
            assert a['mask'] == [2]
            assert b['mask'] == [0, 1, 2, 3]
    assert pinned['control_mask'] == shared['control_mask'] == [0, 1, 2, 3]
    assert pinned['execution_mode'] == shared['execution_mode'] == 'resource_group'


def test_diagnosis_protocol_counts_and_no_automatic_extra_measurements():
    value = protocol()
    cal = value['calibration']
    perf = value['performance']
    assert 4 * sum(cal[k] for k in ('warmup', 'scored', 'tail')) + 7 + 9 * sum(
        perf[k] for k in ('warmup', 'scored', 'tail')) == value['expected_forwards'] == 299
    assert 4 + 7 + 9 == value['expected_tasks'] == 20
    assert value['performance_rejection_threshold'] is None
    assert value['automatic_window_extension'] is False
    assert value['anchors'] is False
    assert value['kernel_settings_mutated'] is False


def test_diagnosis_copy_rejects_hash_mismatch_before_publication(tmp_path):
    source, target = tmp_path / 'source', tmp_path / 'target'
    source.write_bytes(b'original')
    with pytest.raises(ValueError, match='hash differs'):
        _copy(source, target, sha256(b'changed').hexdigest())
    assert not target.exists()
    _copy(source, target, sha256(b'original').hexdigest())
    assert target.read_bytes() == b'original'


def test_diagnosis_copy_rejects_symlink(tmp_path):
    source, linked = tmp_path / 'source', tmp_path / 'linked'
    source.write_bytes(b'original')
    linked.symlink_to(source)
    with pytest.raises(ValueError, match='linked frozen source'):
        _copy(linked, tmp_path / 'target')


def test_diagnosis_reference_proof_documents_recovered_by_exact_hash(tmp_path):
    from cpu_policies.diagnosis.prepare import _reference_documents
    records = _reference_documents(DEFAULT_PARENT, tmp_path)
    inputs = api().read(DEFAULT_PARENT / 'plans/inputs.json')
    assert set(records) == set(inputs['reference_identity']['documents'])
    for name, expected in inputs['reference_identity']['documents'].items():
        assert sha256((tmp_path / name).read_bytes()).hexdigest() == expected
        assert records[name]['sha256'] == expected
        assert records[name]['source_bundle_id'] == '62a5cff7a2439cf3a7828476'
