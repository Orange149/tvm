"""Pilot timing follows original imported receipts and keeps missing clocks unknown."""
from cpu_vta_partitioning.common import sha256, write
from resnet50_validation import components


def test_pilot_uses_imported_head_receipt_and_excludes_unknown_duration(tmp_path, monkeypatch):
    output, run = tmp_path / 'output', tmp_path / 'run'
    keys = ['signature_' + str(i) for i in range(39)]
    write(output / 'profile_capsule/plans/profile_order.json', keys + ['not_in_pilot'])
    records, docs = {}, {}
    for index, key in enumerate(keys):
        device = 'cpu' if index < 24 else 'vta'
        current = dict(task_id='new_task_' + str(index), component_key=key, bundle_id='new',
                       stages=[dict(device=device)])
        docs[key] = current
        original = dict(current)
        if index == 0:
            original.update(task_id='original_archived_task', bundle_id='old')
        # The second receipt was recovered without a measured collection clock.
        value = None if index == 1 else (10. if index == 0 else 2.)
        path = tmp_path / ('archived' if index == 0 else 'new') / (key + '.json')
        write(path, dict(task=original, boot_id='source_boot', measurement_and_collection_seconds=value))
        source = dict(receipt=str(path), receipt_sha256=sha256(path),
                      task_id=original['task_id'], boot_id='source_boot')
        if index == 0:
            source['archived_output'] = 'old preparation archive'
        records[key] = dict(source=source)
    monkeypatch.setattr(components, 'replay', lambda output, run, require_complete: records)
    monkeypatch.setattr(components.profile, 'make_task', lambda bundle, key: docs[key])
    result = components.pilot_cost(output, run)
    cpu, vta = result['component_types']['cpu'], result['component_types']['vta']
    assert result['pilot_sessions'] == 39 and result['extra_stage_calls'] == 0
    assert cpu['observed_sessions'] == 24 and cpu['timed_sessions'] == 23
    assert cpu['unknown_duration_sessions'] == 1 and cpu['measured_seconds'] is None
    assert cpu['known_timed_seconds'] == 54. and cpu['session_median_seconds'] == 2.
    assert cpu['imported_sessions'] == 1 and cpu['new_sessions'] == 23
    assert cpu['sources'][0]['task_id'] == 'original_archived_task'
    assert cpu['sources'][0]['receipt'].endswith('archived/signature_0.json')
    assert vta['observed_sessions'] == 15 and vta['timed_sessions'] == 15
    assert vta['imported_sessions'] == 0 and vta['measured_seconds'] == 30.
    assert cpu['total_median_projection_seconds'] == 2512.


def test_pilot_with_no_timed_group_does_not_fill_zero(tmp_path, monkeypatch):
    output = tmp_path / 'output'; keys = ['key' + str(i) for i in range(39)]
    write(output / 'profile_capsule/plans/profile_order.json', keys)
    records, docs = {}, {}
    for i, key in enumerate(keys):
        doc = dict(task_id=key, component_key=key, bundle_id='new', stages=[dict(device='cpu' if i < 24 else 'vta')])
        path = tmp_path / 'receipts' / (key + '.json'); write(path, dict(task=doc, boot_id='boot'))
        records[key] = dict(source=dict(receipt=str(path), receipt_sha256=sha256(path), task_id=key, boot_id='boot'))
        docs[key] = doc
    monkeypatch.setattr(components, 'replay', lambda output, run, require_complete: records)
    monkeypatch.setattr(components.profile, 'make_task', lambda bundle, key: docs[key])
    result = components.pilot_cost(output, tmp_path / 'run')
    for row in result['component_types'].values():
        assert row['timed_sessions'] == 0 and row['unknown_duration_sessions'] == row['observed_sessions']
        assert row['known_timed_seconds'] is None and row['session_median_seconds'] is None
        assert row['total_median_projection_seconds'] is None
