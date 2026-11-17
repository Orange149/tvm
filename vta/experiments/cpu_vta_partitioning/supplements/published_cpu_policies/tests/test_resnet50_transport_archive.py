"""A payload-verification timeout preserves imports but claims no native execution."""
from pathlib import Path

import pytest

from cpu_vta_partitioning.common import read, sha256, write
from resnet50_validation import archive_audit as archive


@pytest.fixture
def transport_archive(tmp_path, monkeypatch):
    output, run = tmp_path / 'output', tmp_path / 'run'
    snapshot_path = output / 'controller_snapshot/manifest.json'
    write(snapshot_path, {'fixture': 'forty-three frozen files'})
    snapshot = {'controller_sha256': 'controller', 'files': {str(i): {} for i in range(43)}}
    monkeypatch.setattr(archive, '_snapshot', lambda output: snapshot)
    monkeypatch.setattr(archive, '_manifest', lambda bundle: {'bundle_id': 'capsule'})
    write(output / 'profile_capsule/manifest.json', {'bundle_id': 'capsule'})
    write(run / 'run.json', {'boot_id': 'boot', 'controller_sha256': 'controller',
        'profile_capsule_id': 'capsule', 'output': str(output)})
    write(run / 'components/run.json', {'boot_id': 'boot', 'bundle_id': 'capsule'})
    write(run / 'state.json', {'boot_id': 'boot', 'status': 'STOPPED_ON_ERROR', 'error_type': 'TransportError'})
    origin = tmp_path / 'compiler_archive'
    write(origin / 'CONTROLLERS.json', {'old': 'controller'})
    write(origin / 'controller_snapshot/manifest.json', {'old': 'snapshot'})
    write(run / 'components/imported_components.json', {'source_boot_ids': ['old-source-boot'],
        'archived_output': str(origin), 'archive_audit': {'archived_output': str(origin),
        'boot_id': 'old-source-boot', 'archived_controller_sha256': sha256(origin / 'CONTROLLERS.json'),
        'source_snapshot_sha256': sha256(origin / 'controller_snapshot/manifest.json')}})
    imported = {str(i): {'source': {'boot_id': 'old-source-boot'}} for i in range(6)}
    monkeypatch.setattr(archive, 'imported_records', lambda bundle, run: imported)
    write(run / 'components/deployments/0000.json', {'manifest_sha256': sha256(output / 'profile_capsule/manifest.json'),
        'status': 'FAILED', 'elapsed_seconds': 11.})
    write(output / 'DEPLOYMENT_PREFLIGHT_STOP.json', {'status': 'STOPPED_DEPLOYMENT_VERIFICATION_TIMEOUT',
        'boot_id': 'boot', 'controller_sha256': 'controller', 'source_snapshot': str(snapshot_path),
        'state_sha256': sha256(run / 'state.json'), 'imported_CPU_costs': 6,
        'new_native_component_sessions': 0, 'new_DMA_sessions': 0, 'new_complete_DNN_inferences': 0})
    return output, run


def test_transport_archive_keeps_imports_separate_from_native_calls(transport_archive):
    output, run = transport_archive
    result = archive.audit(output, run)
    assert result['transport_preflight_validated'] and result['source_files'] == 43
    assert result['imported_CPU_costs'] == 6 and result['new_native_component_sessions'] == 0
    assert result['new_component_stage_calls'] == result['new_DMA_sessions'] == result['full_DNN_inferences'] == 0
    assert result['deployment_and_verification_seconds'] == 11.
    assert not result['executable'] and not result['board_accessed']


def test_transport_archive_rejects_modified_state_binding(transport_archive):
    output, run = transport_archive
    state = read(run / 'state.json'); state['boot_id'] = 'wrong-boot'
    write(run / 'state.json', state, replace=True)
    with pytest.raises(ValueError, match='state identity'):
        archive.audit(output, run)


@pytest.mark.parametrize('relative', ['components/accepted/boot/task.json', 'traffic/raw/boot/task/task.json'])
def test_transport_archive_rejects_any_own_native_task(transport_archive, relative):
    output, run = transport_archive
    write(run / relative, {'task': 'should not exist in a transfer-only batch'})
    with pytest.raises(ValueError, match='own native task'):
        archive.audit(output, run)
