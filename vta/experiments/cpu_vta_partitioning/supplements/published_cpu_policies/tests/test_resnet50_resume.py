"""Pilot resume boundaries and raw identity checks using an entirely local SSH fake."""
from pathlib import Path

import pytest

from cpu_vta_partitioning.common import digest, inventory, read, sha256, write
from cpu_policies.open_resources.domain import component, component_key, stage
from resnet50_validation import components


BOOT = '11111111-2222-3333-4444-555555555555'


class LocalSSH:
    measured = []
    acknowledged = []
    corrupt_new_boot = False

    def __init__(self, config=None, *, execute_board=False):
        assert execute_board

    def boot_id(self):
        return BOOT

    def deploy(self, bundle, run_dir):
        return '/local-fake-bundle'

    def idle_command(self, script, **kwargs):
        return ''

    def measure(self, document, bundle, remote, run_dir, boot):
        self.measured.append(document['task_id'])
        raw = Path(run_dir) / 'raw' / boot / document['task_id']
        raw.mkdir(parents=True, exist_ok=True)
        (raw / 'boot_id').write_text('different-boot' if self.corrupt_new_boot else boot)
        (raw / 'task_hash').write_text(digest(document))
        return raw

    def acknowledge(self, document, run_dir, boot):
        self.acknowledged.append(document['task_id'])


@pytest.fixture
def resumed_campaign(tmp_path, monkeypatch):
    from cpu_vta_partitioning import transport
    monkeypatch.setattr(components, 'assert_controllers', lambda output: {})
    monkeypatch.setattr(components, 'timed_deploy', lambda ssh, bundle, run_dir: ssh.deploy(bundle, run_dir))
    monkeypatch.setattr(components, 'check_hardware', lambda ssh, bundle, boot: {'boot_id': boot})
    monkeypatch.setattr(components.profile, 'validate', lambda bundle: read(Path(bundle) / 'manifest.json'))
    monkeypatch.setattr(components.profile, 'summary', lambda raw, doc, bundle: {'correct': True})
    monkeypatch.setattr(components.traffic, 'summary', lambda raw, doc, bundle: {'correct': True})
    monkeypatch.setattr(components.profile, 'matched_ssh', lambda config, execute_board=False: LocalSSH(config, execute_board=execute_board))
    monkeypatch.setattr(transport, 'SSH', LocalSSH)
    monkeypatch.setattr(LocalSSH, 'measured', [])
    monkeypatch.setattr(LocalSSH, 'acknowledged', [])
    monkeypatch.setattr(LocalSSH, 'corrupt_new_boot', False)
    local = tmp_path / 'local.json'; write(local, {'board': {}})

    def prepare(kind):
        output = tmp_path / kind; bundle = output / 'profile_capsule'; run = output / 'run'
        configurations, keys = {}, []
        for index in range(3):
            row = stage('cpu' if kind == 'cpu' else 'vta', index, index + 1, 1,
                        [0] if kind == 'cpu' else [0, 1, 2, 3])
            cfg = component(row); key = component_key(cfg)
            configurations[key] = cfg; keys.append(key)
            for name in ('module.so', 'graph.json', 'params.bin'):
                path = bundle / 'programs' / row['program'] / name
                path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b'local fixture')
        write(bundle / 'manifest.json', {'bundle_id': 'fixture', 'runner_sha256': 'runner'})
        write(bundle / 'plans/component_configurations.json', configurations)
        write(bundle / 'plans/profile_order.json', keys)
        write(bundle / 'plans/pilot.json', {'vta': [configurations[k]['stages'][0]['program'] for k in keys]})
        identity = dict(boot_id=BOOT, bundle_id='fixture')
        if kind == 'dma':
            identity['controller_sha256'] = sha256(Path(components.traffic.__file__))
        write(run / 'run.json', identity)
        make_task = components.profile.make_task if kind == 'cpu' else components.traffic.task
        first = make_task(bundle, keys[0])
        fake = LocalSSH(execute_board=True)
        raw = fake.measure(first, bundle, '/local-fake-bundle', run, BOOT)
        existing = run / 'accepted' / BOOT / (first['task_id'] + '.json')
        write(existing, dict(task=first, boot_id=BOOT, raw=str(raw), files=inventory(raw), summary={'correct': True}))
        LocalSSH.measured.clear()
        operation = components.complete if kind == 'cpu' else components.measure_traffic
        def invoke():
            return operation(output, run, local, execute_board=True, limit=2, expected_boot=BOOT)
        return invoke, existing, [make_task(bundle, key)['task_id'] for key in keys]
    return prepare


@pytest.mark.parametrize('kind', ['cpu', 'dma'])
def test_resume_pilot_limit_is_total_target_and_does_not_remeasure(resumed_campaign, kind):
    invoke, existing, tasks = resumed_campaign(kind)
    result = invoke()
    assert result['accepted'] == 2
    if kind == 'cpu':
        assert result['newly_measured'] == 1
    assert LocalSSH.measured == [tasks[1]]
    assert LocalSSH.acknowledged == [tasks[1]]
    assert read(existing)['task']['task_id'] == tasks[0]
    again = invoke()
    assert again['accepted'] == 2
    assert LocalSSH.measured == [tasks[1]] and LocalSSH.acknowledged == [tasks[1]]


@pytest.mark.parametrize('kind', ['cpu', 'dma'])
@pytest.mark.parametrize('field', ['boot_id', 'task_hash'])
def test_resumed_receipt_still_rejects_wrong_raw_identity(resumed_campaign, kind, field):
    invoke, path, _ = resumed_campaign(kind)
    item = read(path)
    (Path(item['raw']) / field).write_text('wrong-identity')
    item['files'] = inventory(item['raw']); write(path, item, replace=True)
    with pytest.raises(ValueError, match='raw.*(identity|boot|task)'):
        invoke()
    assert LocalSSH.measured == [] and LocalSSH.acknowledged == []


@pytest.mark.parametrize('kind', ['cpu', 'dma'])
def test_new_wrong_boot_is_never_acknowledged(resumed_campaign, kind):
    invoke, _, tasks = resumed_campaign(kind)
    LocalSSH.corrupt_new_boot = True
    with pytest.raises(ValueError, match='raw.*(identity|boot|task)'):
        invoke()
    assert LocalSSH.measured == [tasks[1]] and LocalSSH.acknowledged == []
