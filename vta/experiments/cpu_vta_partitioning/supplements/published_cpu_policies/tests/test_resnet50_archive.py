"""Archived evidence is read-only; only exact CPU signatures enter repaired costs."""
from copy import deepcopy
from pathlib import Path
import shutil

import pytest

from cpu_vta_partitioning.common import digest, inventory, read, sha256, write
from cpu_policies.open_resources.domain import component, component_key, stage
from resnet50_validation import archive_audit as archive
from resnet50_validation import components, prepare
from test_resnet50_resume import LocalSSH, resumed_campaign


def test_snapshot_uses_archived_compiler_sources_and_rejects_modified_snapshot(tmp_path):
    shutil.copytree(archive.OLD_OUTPUT / 'controller_snapshot', tmp_path / 'controller_snapshot')
    shutil.copyfile(archive.OLD_OUTPUT / 'CONTROLLERS.json', tmp_path / 'CONTROLLERS.json')
    doc = archive._snapshot(tmp_path)
    assert len(doc['files']) == 41
    compiler = next(row for name, row in doc['files'].items() if name.endswith('/resnet50_validation/compiler.py'))
    path = tmp_path / 'controller_snapshot' / compiler['path']
    path.write_text(path.read_text() + '\n# snapshot corruption\n')
    with pytest.raises(ValueError, match='archived source hash'):
        archive._snapshot(tmp_path)


@pytest.fixture
def matching_archive(tmp_path, monkeypatch):
    old_output, new_output = tmp_path / 'old', tmp_path / 'new'
    old_run, new_run = tmp_path / 'old_run', tmp_path / 'new_run'
    cfgs = [component(stage('cpu', 0, 1, 1, [0])),
            component(stage('vta', 1, 2, 1, [0, 1, 2, 3]))]
    monkeypatch.setattr(archive, 'assert_controllers', lambda output: {})
    monkeypatch.setattr(archive, 'audit', lambda output, run: {'status': 'ARCHIVED_EVIDENCE_VERIFIED', 'records': []})
    monkeypatch.setattr(archive.profile, 'validate', lambda bundle: read(Path(bundle) / 'manifest.json'))
    monkeypatch.setattr(prepare, 'validate_profile', lambda bundle: read(Path(bundle) / 'manifest.json'))
    calls = []
    def fake_summary(raw, doc, bundle):
        calls.append(Path(bundle))
        return {'correct': True}
    monkeypatch.setattr(archive.profile, 'summary', fake_summary)
    for root in (old_output, new_output):
        bundle = root / 'profile_capsule'
        configs = {component_key(cfg): cfg for cfg in cfgs}
        write(bundle / 'plans/component_configurations.json', configs)
        write(bundle / 'plans/inputs.json', {'records': [{'index': 0}]})
        write(bundle / 'plans/expected_outputs.json', {'one': 'correct'})
        programs = {}
        for cfg in cfgs:
            row = cfg['stages'][0]; name = row['program']
            programs[name] = dict(input_names=['data'], input_node_ids=[row['begin']], output_node_ids=[row['end']],
                                  input_types=[{'shape': [1], 'dtype': 'int8'}], output_types=[{'shape': [1], 'dtype': 'int8'}])
            for leaf in archive.MODULE_FILES:
                path = bundle / 'programs' / name / leaf
                path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b'identical executable fixture')
        write(bundle / 'plans/programs.json', programs)
        (bundle / 'runtime').mkdir(); (bundle / 'runtime/libtvm_runtime.so').write_bytes(b'same runtime')
        value = dict(network='resnet50_v2', unit_count=38, canonical_sha256='same graph',
                     compiler_revision=root.name, runner_sha256='runner', files=inventory(bundle))
        value['bundle_id'] = digest(value)[:24]; write(bundle / 'manifest.json', value)
    old_bundle = old_output / 'profile_capsule'
    write(old_run / 'components/run.json', dict(boot_id='old-boot', bundle_id=read(old_bundle / 'manifest.json')['bundle_id']))
    for cfg in cfgs:
        doc = archive.profile.make_task(old_bundle, component_key(cfg))
        raw = old_run / 'components/raw' / doc['task_id']; raw.mkdir(parents=True)
        (raw / 'boot_id').write_text('old-boot'); (raw / 'task_hash').write_text(digest(doc))
        path = old_run / 'components/accepted/old-boot' / (doc['task_id'] + '.json')
        write(path, dict(task=doc, boot_id='old-boot', raw=str(raw), files=inventory(raw), summary={'correct': True}))
    return old_output, old_run, new_output, new_run, cfgs, calls


def test_original_matcher_reused_but_only_exact_CPU_costs_imported(matching_archive):
    old, old_run, new, run, cfgs, calls = matching_archive
    result = archive.import_components(new, run, archived_output=old, archived_run=old_run)
    assert result['imported'] == 1 and result['matcher_exact_matches'] == 2
    assert result['excluded_matching_VTA_components'] == 1
    path = run / 'components/imported_components.json'; doc = read(path)
    assert set(doc['records']) == {component_key(cfgs[0])}
    item = next(iter(doc['records'].values()))
    assert item['source']['boot_id'] == 'old-boot' and item['task']['bundle_id'] != doc['bundle_id']
    assert item['source']['receipt'].startswith(str(old_run / 'components/accepted'))
    assert new / 'profile_capsule' in calls
    assert not (run / 'components/run.json').exists()  # actual new boot remains unset until board execution
    assert archive.imported_records(new / 'profile_capsule', run / 'components')
    assert archive.import_components(new, run, archived_output=old, archived_run=old_run) == result


def test_repaired_outputs_are_revalidated_not_just_old_references(matching_archive, monkeypatch):
    old, old_run, new, run, cfgs, _ = matching_archive
    archive.import_components(new, run, archived_output=old, archived_run=old_run)
    doc = read(run / 'components/imported_components.json')
    item = next(iter(doc['records'].values()))
    monkeypatch.setattr(archive.profile, 'summary', lambda raw, task, bundle: {'correct': False})
    with pytest.raises(ValueError, match='repaired reference'):
        archive.imported_record(new / 'profile_capsule', component_key(cfgs[0]), item)


def test_changed_module_hash_and_original_raw_identity_are_rejected(matching_archive):
    old, old_run, new, run, cfgs, _ = matching_archive
    archive.import_components(new, run, archived_output=old, archived_run=old_run)
    item = next(iter(read(run / 'components/imported_components.json')['records'].values()))
    module = new / 'profile_capsule/programs' / cfgs[0]['stages'][0]['program'] / 'module.so'
    module.write_bytes(b'different executable')
    with pytest.raises(ValueError, match='executable bytes'):
        archive.imported_record(new / 'profile_capsule', component_key(cfgs[0]), item)


def test_complete_and_replay_skip_imported_costs_without_remeasuring(resumed_campaign, monkeypatch):
    invoke, existing, tasks = resumed_campaign('cpu')
    old = read(existing); key = old['task']['component_key']; existing.unlink()
    imported = {'correct': True, 'configuration': old['task']['configuration'], 'source': {'boot_id': 'archived-boot'}}
    monkeypatch.setattr(archive, 'imported_records', lambda bundle, run: {key: imported})
    result = invoke()
    assert result['accepted'] == 2 and result['newly_measured'] == 1 and result['imported_components'] == 1
    assert LocalSSH.measured == [tasks[1]]
    output = existing.parents[3]; run = existing.parents[2]
    replay = components.replay(output, run, require_complete=False)
    assert replay[key] == imported and len(replay) == 2
    invoke()
    assert LocalSSH.measured == [tasks[1]]


def test_archived_batch_is_never_executed(tmp_path):
    write(tmp_path / 'COMPILER_REGRESSION_STOP.json', {'status': 'STOPPED_COMPILER_REGRESSION'})
    with pytest.raises(ValueError, match='archived'):
        components.complete(tmp_path, tmp_path / 'run', tmp_path / 'local.json', execute_board=True)
    with pytest.raises(ValueError, match='archived'):
        components.measure_traffic(tmp_path, tmp_path / 'run', tmp_path / 'local.json', execute_board=True)
