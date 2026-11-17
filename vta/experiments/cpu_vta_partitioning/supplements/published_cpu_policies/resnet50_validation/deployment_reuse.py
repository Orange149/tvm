"""Reuse immutable deployed bytes; retain the original SSH deployment lifecycle.

This module does not connect by itself. The supplied, explicitly enabled SSH
transport provides locking, boot checks, transfer, full SHA verification and the
final deployment marker. A proxy only unlinks NEW destinations before a tar
rewrite, so a shared inode can never overwrite the source bundle.
"""
from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
import re
import shlex
import tarfile
import time

from cpu_vta_partitioning.common import digest, now, read, sha256, write
from cpu_vta_partitioning.transport import SSH, TransportError

_RESERVED = {'board.sha256', '.deployed', 'verification.log'}


def _name(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_.+@/-]+', value):
        raise ValueError('unsafe deployment inventory path')
    path = PurePosixPath(value)
    if path.is_absolute() or str(path) != value or any(x in ('.', '..') for x in value.split('/')):
        raise ValueError('deployment inventory path escapes its bundle')
    if value in _RESERVED:
        raise ValueError('deployment inventory collides with transport metadata')
    return value


def _item(value):
    if (not isinstance(value, dict) or not re.fullmatch(r'[0-9a-f]{64}', str(value.get('sha256', '')))
            or type(value.get('bytes')) is not int or value['bytes'] < 0):
        raise ValueError('invalid deployment file identity')
    return {'sha256': value['sha256'], 'bytes': value['bytes']}


def _bundle(path):
    path = Path(path)
    manifest = read(path / 'manifest.json')
    identifier = manifest.get('bundle_id')
    if not isinstance(identifier, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', identifier):
        raise ValueError('invalid deployment bundle identity')
    if digest({k: v for k, v in manifest.items() if k != 'bundle_id'})[:24] != identifier:
        raise ValueError('deployment manifest identity differs')
    declared = {_name(name): _item(value) for name, value in manifest['files'].items()}
    plan_path = path / 'plans/board_payload.json'
    if declared.get('plans/board_payload.json') != dict(sha256=sha256(plan_path), bytes=plan_path.stat().st_size):
        raise ValueError('deployment payload inventory changed')
    plan = read(plan_path)
    files = {}
    for name, value in plan['files'].items():
        name, value = _name(name), _item(value)
        if declared.get(name) != value:
            raise ValueError('deployment payload differs from its manifest')
        local = path / name
        if local.is_symlink() or not local.is_file() or any(p.is_symlink() for p in local.parents if p != path.parent):
            raise ValueError('local deployment payload is not a regular bundle file')
        if local.stat().st_size != value['bytes']:
            raise ValueError('local deployment payload size differs')
        files[name] = value
    for name in ('plans/board_payload.json', 'manifest.json'):
        local = path / name
        if local.is_symlink() or not local.is_file():
            raise ValueError('local deployment metadata is not a regular file')
        files[name] = dict(sha256=sha256(local), bytes=local.stat().st_size)
    return manifest, files


def _base(ssh):
    raw = ssh.config['remote_base']
    p = PurePosixPath(raw)
    if (not p.is_absolute() or str(p) != raw.rstrip('/') or str(p) == '/'
            or any(x in ('.', '..') for x in raw.split('/')[1:])
            or '\n' in raw or '\x00' in raw):
        raise ValueError('unsafe remote deployment root')
    return str(p)


def _parents(path):
    p = PurePosixPath(path)
    return [str(x) for x in reversed(p.parents) if str(x) != '/']


def _guard(path):
    """Reject symlink/file ancestors before mkdir, unlink or tar extraction."""
    q = shlex.quote
    return '\n'.join('test ! -L ' + q(p) + ' && { test ! -e ' + q(p)
        + ' || test -d ' + q(p) + '; } || { echo unsafe_deployment_parent >&2; exit 97; }'
        for p in _parents(path))


def _detach(path):
    q = shlex.quote
    return (_guard(path) + '\n'
        + 'if test -d ' + q(path) + ' && test ! -L ' + q(path)
        + '; then echo unexpected_directory_at_file >&2; exit 97; fi\n'
        + 'rm -f ' + q(path))


class _SafeTransfer:
    """Run SSH.deploy unchanged, detaching members in its existing locked tar step."""
    def __init__(self, ssh, remote, files, report):
        self._ssh, self._remote, self._files, self._report = ssh, remote, files, report

    def __getattr__(self, name):
        return getattr(self._ssh, name)

    def boot_id(self):
        actual = self._ssh.boot_id()
        expected = self._report.get('boot_id')
        if expected is not None and actual != expected:
            raise TransportError('boot changed during asset reuse deployment')
        self._report['boot_id'] = actual
        return actual

    def idle_command(self, script, **kwargs):
        # ResNet50's 5GB inventory can legitimately require over ten minutes
        # on the board SD card. Keep the original complete checks unchanged.
        if 'sha256sum -c board.sha256' in script:
            original = kwargs.get('timeout', 60)
            if original < 1800:
                kwargs['timeout'] = 1800
                self._report.setdefault('hash_timeout_overrides', []).append(dict(
                    original_timeout_seconds=original, effective_timeout_seconds=1800,
                    operation='missing_scan' if '2>/dev/null || true' in script else 'full_verification'))
        transfer = None
        expected = 'cd ' + shlex.quote(self._remote) + ' && tar -xzf -'
        if script == expected and kwargs.get('input_path') is not None:
            archive = Path(kwargs['input_path'])
            members = []
            with tarfile.open(archive, 'r:*') as tar:
                for member in tar.getmembers():
                    name = member.name
                    if name != 'board.sha256': _name(name)
                    if name != 'board.sha256' and name not in self._files:
                        raise ValueError('transfer archive contains an undeclared member')
                    if not (member.isfile() or member.islnk()):
                        raise ValueError('transfer archive contains a non-file member')
                    if member.islnk():
                        _name(member.linkname)
                        if member.linkname not in self._files:
                            raise ValueError('transfer hardlink escapes its authenticated inventory')
                    if name in members:
                        raise ValueError('duplicate transfer archive member')
                    members.append(name)
            # Same maintenance lock as extraction: no window between unlink and
            # rewriting. Always unlink every member, even if a formerly-good
            # hardlink became corrupt since preflight.
            detach = '\n'.join(_detach(self._remote + '/' + name) for name in members)
            transfer = dict(status='STARTED',
                archive_sha256=sha256(archive), archive_bytes=archive.stat().st_size,
                members=members,
                payload_bytes=sum(self._files[n]['bytes'] for n in members if n != 'board.sha256'),
                destinations_unlinked_before_rewrite=True)
            # A full profile inventory can exceed the SSH command argv limit.
            # Upload a tiny control script through the same idle lock, then
            # authenticate and execute it under the extraction's maintenance
            # lock. Its final unlink and tar rewrite remain one atomic session.
            helper = archive.parent / ('deployment-detach-' + sha256(archive)[:20] + '.sh')
            helper.write_text('set -eu\n' + detach + '\n')
            remote_helper = self._remote + '/.deployment-detach-' + sha256(helper)[:20] + '.sh'
            self._ssh.idle_command(_detach(remote_helper) + '\ncat > ' + shlex.quote(remote_helper),
                boot=kwargs.get('boot'), input_path=helper, timeout=600)
            self._report.setdefault('transfer_control_bytes', 0)
            self._report['transfer_control_bytes'] += helper.stat().st_size
            script = ('test "$(sha256sum ' + shlex.quote(remote_helper)
                + ' | cut -d " " -f 1)" = ' + shlex.quote(sha256(helper))
                + ' && sh ' + shlex.quote(remote_helper) + ' && ' + script
                + ' && rm -f ' + shlex.quote(remote_helper))
            self._report.setdefault('transfers', []).append(transfer)
        began = time.monotonic()
        try:
            result = self._ssh.idle_command(script, **kwargs)
            if transfer is not None: transfer['status'] = 'COMPLETE'
            return result
        except Exception:
            if transfer is not None:
                transfer.update(status='FAILED', actual_wire_bytes='unknown_partial_stream')
            raise
        finally:
            if transfer is not None: transfer['elapsed_seconds'] = time.monotonic()-began



def _require_mount(ssh):
    mount = ssh.config.get('storage_mount')
    if mount:
        ssh.command('awk -v wanted=' + shlex.quote(mount)
            + " '$2 == wanted && $4 ~ /(^|,)rw(,|$)/ {ok=1} END {exit !ok}' /proc/mounts && test -w "
            + shlex.quote(mount))


def _target_status(ssh, remote, manifest_hash, inventory_hash, report):
    """Probe metadata only; neither marker nor inventory bypasses payload hashes."""
    _require_mount(ssh)
    boot = ssh.boot_id(); report['boot_id'] = boot; ssh.wait_idle(boot)
    marker, inventory = remote + '/.deployed', remote + '/board.sha256'
    q = shlex.quote
    raw = ssh.idle_command(_guard(marker) + '\n'
        + 'test ! -L ' + q(marker)
        + ' || { echo unsafe_deployment_marker >&2; exit 97; }\n'
        + 'if test -f ' + q(marker) + ' && test "$(cat ' + q(marker) + ')" = ' + q(manifest_hash)
        + '; then printf "DEPLOYED\\n";\n'
        + 'elif test ! -L ' + q(inventory) + ' && test -f ' + q(inventory)
        + ' && test "$(sha256sum ' + q(inventory) + ' | cut -d " " -f 1)" = ' + q(inventory_hash)
        + '; then printf "INVENTORY\\n"; else printf "MISSING\\n"; fi', boot=boot).strip()
    if raw not in ('DEPLOYED', 'INVENTORY', 'MISSING'):
        raise TransportError('unexpected deployment metadata probe')
    return raw


def _preseed(ssh, bundle, run_dir, source, target_files, source_files, mapping, report):
    base, q = _base(ssh), shlex.quote
    target_remote = base + '/bundles/' + report['target_bundle_id']
    source_remote = base + '/bundles/' + report['source_bundle_id']
    _require_mount(ssh)
    boot = ssh.boot_id()
    if report.get('boot_id') is not None and report['boot_id'] != boot:
        raise TransportError('boot changed before asset reuse preseed')
    report['boot_id'] = boot; ssh.wait_idle(boot)
    checksums = ''.join(v['sha256'] + '  ' + k + '\n' for k, v in sorted(target_files.items()))
    script = ['set -eu', _guard(target_remote + '/placeholder'), 'mkdir -p ' + q(target_remote),
        'source_ok=0', 'if test ! -L ' + q(source_remote) + ' && test -f ' + q(source_remote + '/.deployed')
        + ' && test ! -L ' + q(source_remote + '/.deployed')
        + ' && test "$(cat ' + q(source_remote + '/.deployed') + ')" = ' + q(sha256(source / 'manifest.json'))
        + '; then source_ok=1; else printf "SOURCE_UNAVAILABLE\\n"; fi']
    for name, item in sorted(target_files.items()):
        target = target_remote + '/' + name
        script += [_guard(target), 'mkdir -p ' + q(str(PurePosixPath(target).parent)),
            'if test ! -L ' + q(target) + ' && test -f ' + q(target)
            + ' && test "$(sha256sum ' + q(target) + ' | cut -d " " -f 1)" = ' + q(item['sha256'])
            + '; then printf "PRESENT\\t%s\\n" ' + q(name) + '; else', _detach(target)]
        if name in mapping:
            oldname = mapping[name]; old = source_remote + '/' + oldname
            safe = ' && '.join('test ! -L ' + q(p) for p in [*_parents(old), old])
            script += ['if test "$source_ok" = 1 && ' + safe + ' && test -f ' + q(old)
                + ' && test "$(sha256sum ' + q(old) + ' | cut -d " " -f 1)" = ' + q(item['sha256']) + '; then',
                'if ln ' + q(old) + ' ' + q(target) + '; then printf "LINKED\\t%s\\t%s\\n" '
                + q(name) + ' ' + q(oldname) + '; else printf "LINK_FAILED\\t%s\\n" ' + q(name) + '; fi',
                'else printf "TRANSFER\\t%s\\n" ' + q(name) + '; fi']
        else:
            script.append('printf "TRANSFER\\t%s\\n" ' + q(name))
        script.append('fi')
    # Authenticate the inventory BEFORE old deploy computes its missing set.
    # Publishing uses a detached file; no old inode is opened for writing.
    for name in ('board.sha256', '.deployed', 'verification.log'):
        script.append(_detach(target_remote + '/' + name))
    script += ['cat > ' + q(target_remote + '/board.sha256') + " <<'CPU_VTA_REUSE_CHECKSUMS'", checksums.rstrip('\n'),
        'CPU_VTA_REUSE_CHECKSUMS']
    path = run_dir / 'reuse-preseed.sh'; path.write_text('\n'.join(script) + '\n')
    report['preseed_script_sha256'] = sha256(path)
    report['preseed_control_bytes'] = path.stat().st_size
    raw = ssh.idle_command('sh -s', boot=boot, input_path=path, timeout=1800)
    report['preseed_raw'] = raw
    linked, present = [], []
    for line in raw.splitlines():
        fields = line.split('\t')
        if fields[0] == 'LINKED':
            if len(fields) != 3 or mapping.get(fields[1]) != fields[2]:
                raise TransportError('unexpected asset reuse receipt')
            linked.append(dict(target=fields[1], source=fields[2], **target_files[fields[1]]))
        elif fields[0] == 'PRESENT':
            if len(fields) != 2 or fields[1] not in target_files:
                raise TransportError('unexpected existing-file receipt')
            present.append(fields[1])
    report.update(reused_files=linked, reused_bytes=sum(x['bytes'] for x in linked),
        already_present_files=present, already_present_bytes=sum(target_files[x]['bytes'] for x in present),
        source_deployment_available='SOURCE_UNAVAILABLE' not in raw.splitlines())


def deploy(ssh, bundle, run_dir, source_bundle=None):
    """Return remote bundle path after original full deployment verification.

    Source/target manifests are immutable and can have different relative paths.
    SHA+byte-equal files are linked only when the old deployed-marker and actual
    old bytes agree. Missing sources and cross-filesystem link failures fall back
    to the original transfer. No source-side chmod, unlink or write is performed.
    """
    bundle, run_dir = Path(bundle), Path(run_dir)
    manifest, files = _bundle(bundle)
    remote = _base(ssh) + '/bundles/' + manifest['bundle_id']
    run_dir.mkdir(parents=True, exist_ok=True)
    receipt_dir = run_dir / 'deployments/reuse'; receipt_dir.mkdir(parents=True, exist_ok=True)
    receipt = receipt_dir / (str(time.time_ns()) + '.json')
    started = time.monotonic()
    report = dict(status='STARTED', started_at=now(), target_bundle_id=manifest['bundle_id'],
        target_manifest_sha256=sha256(bundle / 'manifest.json'), source_bundle_id=None,
        source_manifest_sha256=None, planned_reuse_files=0, planned_reuse_bytes=0,
        reused_files=[], reused_bytes=0, transfers=[], fallback_reason=None, hash_timeout_overrides=[],
        source_modified=False, full_original_manifest_verification=True,
        implementation_sha256=sha256(Path(__file__)))
    write(receipt, report)
    try:
        inventory = ''.join(v['sha256'] + '  ' + k + '\n' for k, v in sorted(files.items()))
        inventory_hash = hashlib.sha256(inventory.encode()).hexdigest()
        report['target_inventory_sha256'] = inventory_hash
        state = _target_status(ssh, remote, report['target_manifest_sha256'], inventory_hash, report)
        current, partial = state == 'DEPLOYED', state == 'INVENTORY'
        report['target_already_deployed'] = current
        report['authenticated_inventory_resume'] = partial
        report['preseed_skipped'] = current or partial
        if current:
            report['fallback_reason'] = 'target_current_deployment_use_original_full_verification'
        elif partial:
            report['fallback_reason'] = 'authenticated_inventory_use_original_missing_transfer_and_full_verification'
        else:
            source = Path(source_bundle) if source_bundle is not None else None
            if source is None or not (source / 'manifest.json').is_file():
                report['fallback_reason'] = 'no_local_source_bundle'
            else:
                old, old_files = _bundle(source)
                if old['bundle_id'] == manifest['bundle_id']:
                    raise ValueError('reuse source and target bundle identities must differ')
                report.update(source_bundle_id=old['bundle_id'], source_manifest_sha256=sha256(source / 'manifest.json'))
                by_content = {}
                for name, item in sorted(old_files.items()):
                    by_content.setdefault((item['sha256'], item['bytes']), name)
                mapping = {name: by_content[(item['sha256'], item['bytes'])]
                    for name, item in files.items() if (item['sha256'], item['bytes']) in by_content}
                report.update(planned_reuse_files=len(mapping), planned_reuse_bytes=sum(files[n]['bytes'] for n in mapping))
                if not mapping:
                    report['fallback_reason'] = 'no_exact_content_matches'
                else:
                    _preseed(ssh, bundle, run_dir, source, files, old_files, mapping, report)
                    if not report['source_deployment_available']:
                        report['fallback_reason'] = 'source_not_deployed_or_missing'
        # Exact existing implementation; proxy changes only safe overwrite
        # preparation inside its own maintenance-locked extraction operation.
        result = SSH.deploy(_SafeTransfer(ssh, remote, files, report), bundle, run_dir)
        if ssh.boot_id() != report['boot_id']:
            raise TransportError('boot changed before asset reuse deployment acceptance')
        if result != remote:
            raise TransportError('deployment returned an unexpected destination')
        if current:
            report.update(already_present_files=sorted(files),
                already_present_bytes=sum(item['bytes'] for item in files.values()))
        report.update(status='COMPLETE', remote=result)
        return result
    except Exception as error:
        report.update(status='FAILED', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report.update(elapsed_seconds=time.monotonic()-started,
            attempted_archive_bytes=sum(x['archive_bytes'] for x in report['transfers']),
            transferred_archive_bytes=sum(x['archive_bytes'] for x in report['transfers'] if x['status']=='COMPLETE'),
            transferred_payload_bytes=sum(x['payload_bytes'] for x in report['transfers'] if x['status']=='COMPLETE'),
            failed_transfer_wire_bytes_unknown=any(x['status']=='FAILED' for x in report['transfers']))
        write(receipt, report, replace=True)
