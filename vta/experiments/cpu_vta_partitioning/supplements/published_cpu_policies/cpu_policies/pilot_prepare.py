"""Assemble an isolated pilot using verified existing binaries; no SSH here."""
from copy import deepcopy
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .pilot_common import ROOT, DEFAULT_BUNDLE, identity, load_legacy, protocol


def _copy(source, target, expected=None):
    from cpu_vta_partitioning.common import sha256
    if source.is_symlink() or not source.is_file():
        raise ValueError('missing or linked frozen source: ' + str(source))
    if expected is not None and sha256(source) != expected:
        raise ValueError('frozen source hash differs: ' + str(source))
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def validate_bundle(bundle):
    bundle = load_legacy(bundle)
    from cpu_vta_partitioning.common import read, digest, verify_inventory, sha256
    from cpu_vta_partitioning.prepare.payload import execution_files
    m = read(bundle / 'manifest.json')
    if digest({k: v for k, v in m.items() if k != 'bundle_id'})[:24] != m['bundle_id']:
        raise ValueError('pilot manifest identity differs')
    if m['experiment'] != protocol()['experiment'] or m['controller_identity'] != identity():
        raise ValueError('pilot sources changed; use the frozen controller snapshot')
    verify_inventory(bundle, m['files'])
    if read(bundle / 'plans/pilot_protocol.json') != protocol():
        raise ValueError('pilot protocol differs')
    programs = read(bundle / 'plans/programs.json')
    if len(programs) != 172 or sum(p['device'] == 'cpu' for p in programs.values()) != 85:
        raise ValueError('full stage domain missing')
    executor = read(bundle / 'plans/executors.json')['policy']
    if sha256(bundle / executor['path']) != executor['sha256']:
        raise ValueError('pinned executor changed')
    inputs = read(bundle / 'plans/inputs.json')
    if inputs['available_indices'] != [0] or len(inputs['records']) != 1 or inputs['records'][0]['index'] != 0:
        raise ValueError('pilot must contain only input zero')
    references = {inputs['records'][0]['input']}
    for row in inputs['records'][0]['frontiers'].values():
        references.update(row[k] for k in ('raw', 'npy') if k in row)
    if references != set(inputs['files']) or any(n not in m['files'] for n in references):
        raise ValueError('single-input reference closure differs')
    verify_inventory(bundle, inputs['files'])
    if read(bundle / 'plans/board_payload.json')['files'] != execution_files(bundle):
        raise ValueError('pilot execution closure differs')
    ready = read(bundle / 'OFFLINE_READINESS.json')
    if ready['status'] != 'PREPARED_OFFLINE' or not ready['tests']['passed']:
        raise ValueError('offline tests incomplete')
    return m


def prepare(parent=None, *, run_tests=True):
    parent = load_legacy(parent or DEFAULT_BUNDLE)
    from cpu_vta_partitioning.common import read, write, sha256, digest, inventory, verify_inventory, locked
    from cpu_vta_partitioning.prepare.payload import execution_files
    from cpu_vta_partitioning.prepare.package import elf
    from .pilot_search import requirements
    original = read(parent / 'manifest.json')
    if digest({k: v for k, v in original.items() if k != 'bundle_id'})[:24] != original['bundle_id']:
        raise ValueError('parent manifest identity differs')
    verify_inventory(parent, original['files'])
    code = identity()
    work = ROOT / 'artifacts/pilot_builds' / digest({'code': code, 'parent': original['bundle_id']})[:24]
    with locked(work):
        for folder in ('programs', 'runtime', 'firmware', 'runner'):
            for name, expected in inventory(parent / folder).items():
                _copy(parent / folder / name, work / folder / name, expected['sha256'])
        programs = read(parent / 'plans/programs.json')
        inputs = deepcopy(read(parent / 'plans/inputs.json'))
        record = inputs['records'][0]
        wanted = {record['input']}
        for frontier in record['frontiers'].values():
            wanted.update(frontier[k] for k in ('raw', 'npy') if k in frontier)
        inputs['records'] = [record]
        inputs['available_indices'] = [0]
        inputs['files'] = {n: original['files'][n] for n in sorted(wanted)}
        for name in sorted(wanted):
            _copy(parent / name, work / name, original['files'][name]['sha256'])
        for filename in ('executors.json', 'compiler_identity.json', 'runtime_build.json',
                         'sdk_runtime.json', 'policy_native_acceptance.json', 'policy_runner_patch.json'):
            _copy(parent / 'plans' / filename, work / 'plans' / filename)
        for folder in ('base_controller', 'cpu_strategy', 'native_source', 'vta'):
            shutil.copytree(parent / 'sources' / folder, work / 'sources' / folder,
                            dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache'))
        snapshot = work / 'sources/published_cpu_policies'
        for filename in ('run.py', 'pyproject.toml', 'README.md'):
            _copy(ROOT / filename, snapshot / filename)
        for folder in ('cpu_policies', 'configs', 'tests', 'examples', 'docs', 'third_party'):
            shutil.copytree(ROOT / folder, snapshot / folder, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache', 'local.json'))
        write(work / 'plans/programs.json', programs, replace=True)
        write(work / 'plans/inputs.json', inputs, replace=True)
        write(work / 'plans/pilot_protocol.json', protocol(), replace=True)
        needed = requirements(programs, parent)
        if len(needed) != 514:
            raise ValueError('unexpected component coverage')
        write(work / 'plans/pilot_components.json', needed, replace=True)
        write(work / 'plans/board_execution.json', {
            'revision': 'published-cpu-policies-pilot-1', 'udmabuf_bytes': 201326592,
            'storage_margin_bytes': 256 * 1024**2, 'boot_policy': protocol()['boot_policy']}, replace=True)
        payload = execution_files(work)
        write(work / 'plans/board_payload.json', {'revision': 1, 'files': payload,
              'unpacked_bytes': sum(v['bytes'] for v in payload.values()),
              'transfer': 'gzip_tar_stream_no_remote_archive'}, replace=True)
        closure = {}
        for path in [*(work / 'runner').iterdir(), *(work / 'runtime').iterdir(),
                     *(work / 'programs' / name / 'module.so' for name in programs)]:
            if not path.is_file():
                continue
            info = elf(path)
            if any(not (work / 'runtime' / name).is_file() for name in info['needed']):
                raise ValueError('unresolved packaged shared library')
            closure[str(path.relative_to(work))] = info
        write(work / 'plans/elf_dependencies.json', closure, replace=True)
        emulator = shutil.which('qemu-aarch64-static')
        if not emulator:
            raise RuntimeError('QEMU unavailable for packaged loader check')
        checked = subprocess.run([emulator, str(work / 'runtime/ld-linux-aarch64.so.1'),
            '--library-path', str(work / 'runtime'), str(work / 'runner/vta_stage_pipeline_runner_pinned'), '--help'],
            capture_output=True, text=True, timeout=45)
        if checked.returncode or '--cpu-worker-pinning' not in checked.stdout + checked.stderr:
            raise RuntimeError('packaged loader/executor check failed')
        if not run_tests:
            raise ValueError('a deployable bundle requires local tests')
        env = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
        test = subprocess.run([sys.executable, '-m', 'pytest', '-c', 'pyproject.toml', '-q'],
                              cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
        write(work / 'plans/offline_tests.json', {'passed': test.returncode == 0,
              'stdout': test.stdout, 'stderr': test.stderr}, replace=True)
        if test.returncode:
            raise RuntimeError('pilot local tests failed:\n' + test.stdout + test.stderr)
        if identity() != code:
            raise RuntimeError('controller changed during preparation; prepare again after edits finish')
        write(work / 'OFFLINE_READINESS.json', {
            'status': 'PREPARED_OFFLINE', 'controller_identity': code, 'tests': {'passed': True},
            'programs': {'cpu': 85, 'vta': 87}, 'component_count': len(needed),
            'new_model_compilations': 0, 'new_executor_compilations': 0,
            'costs_reused_as_new_measurements': False,
            'board_validation_pending': ['actual affinity', 'FPGA numerics', 'all measurements'],
            'assumptions': ['homogeneous core permutation', 'control work not separately modeled'],
            'input_indices': [0], 'formal_independent_confirmations': 0}, replace=True)
        files = inventory(work, exclude=('manifest.json', '.lock'))
        # Cache files are never part of a frozen executable controller identity.
        files = {n: v for n, v in files.items() if '__pycache__/' not in n and '.pytest_cache/' not in n}
        m = {'experiment': protocol()['experiment'], 'controller_identity': code,
             'parent_bundle_id': original['bundle_id'], 'files': files}
        m['bundle_id'] = digest(m)[:24]
        write(work / 'manifest.json', m, replace=True)
        target = ROOT / 'artifacts/bundles' / m['bundle_id']
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copytree(work, target, ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache', '.lock'))
        validate_bundle(target)
        return target
