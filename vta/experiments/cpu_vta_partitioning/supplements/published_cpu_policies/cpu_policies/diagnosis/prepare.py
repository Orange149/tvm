"""Prepare an eight-module diagnosis capsule locally, without SSH or compilation."""
from copy import deepcopy
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from .common import (ROOT, PROJECT, DEFAULT_PARENT, DEFAULT_SOURCE_RUN, api, identity,
                     load_legacy, protocol)
from .domain import PROGRAMS, configurations, cases


def _copy(source, target, expected=None):
    common = api()
    source, target = Path(source), Path(target)
    if source.is_symlink() or not source.is_file():
        raise ValueError('missing or linked frozen source: ' + str(source))
    if expected is not None and common.sha256(source) != expected:
        raise ValueError('frozen source hash differs: ' + str(source))
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)



def _reference_documents(parent, target=None):
    """Restore proof documents lost by an ancestor's capsule reduction.

    The retained input plan names exact hashes. A missing proof can therefore
    come from the original complete bundle only when those same bytes match;
    an existing corrupt file is always rejected rather than silently replaced.
    """
    common = api(parent)
    documents = common.read(parent / 'plans/inputs.json').get('reference_identity', {}).get('documents', {})
    required = {'sources/reference/inputs_manifest.json',
                'sources/reference/prior_inputs_manifest.json',
                'sources/reference/reference_proof.json'}
    if set(documents) != required:
        raise ValueError('reference proof document closure is incomplete')
    original_bundle = PROJECT / 'artifacts/bundles/62a5cff7a2439cf3a7828476'
    result = {}
    for name, expected in sorted(documents.items()):
        source_root = parent if common.relative_file(parent, name).is_file() else original_bundle
        source = common.relative_file(source_root, name)
        manifest = common.read(source_root / 'manifest.json')
        if common.digest({k: v for k, v in manifest.items() if k != 'bundle_id'})[:24] != manifest['bundle_id']:
            raise ValueError('reference source bundle manifest identity differs')
        if manifest.get('files', {}).get(name, {}).get('sha256') != expected:
            raise ValueError('reference proof source manifest differs: ' + name)
        if not source.is_file() or common.sha256(source) != expected:
            raise ValueError('reference proof source missing or changed: ' + name)
        if target is not None:
            _copy(source, target / name, expected)
        result[name] = {'sha256': expected, 'source_bundle_id': manifest['bundle_id'],
                        'source_manifest_sha256': common.sha256(source_root / 'manifest.json'),
                        'source_path': str(source)}
    return result


def _source_identity(parent, source_run):
    common = api(parent)
    manifest = common.read(parent / 'manifest.json')
    if common.digest({k: v for k, v in manifest.items() if k != 'bundle_id'})[:24] != manifest['bundle_id']:
        raise ValueError('parent manifest identity differs')
    common.verify_inventory(parent, manifest['files'])
    state = common.read(source_run / 'state.json')
    if state.get('status') != 'COMPLETE' or state.get('bundle_id') != manifest['bundle_id']:
        raise ValueError('diagnostic baseline is not a completed run of the source bundle')
    common.verify_inventory(source_run, state['frozen_files'])
    configs = configurations(common.read(parent / 'plans/programs.json'))
    pool = common.read(source_run / 'search_pool.json')
    for case, cfg in configs['pipelines'].items():
        if pool.get(cfg['id']) != cfg:
            # Some pool entries contain selection metadata outside configuration.
            entry = pool.get(cfg['id'], {})
            if entry.get('configuration') != cfg:
                from ..open_resources.domain import configuration
                if any(entry.get(k) != v for k, v in configuration(cfg).items()):
                    raise ValueError('baseline case differs from its frozen pipeline: ' + case)
    return manifest, {
        'source_bundle_id': manifest['bundle_id'],
        'source_bundle_manifest_sha256': common.sha256(parent / 'manifest.json'),
        'source_run': str(source_run), 'source_run_boot_id': state.get('boot_id'),
        'source_state_sha256': common.sha256(source_run / 'state.json'),
        'source_costs_sha256': common.sha256(source_run / 'costs.json'),
        'source_pool_sha256': common.sha256(source_run / 'search_pool.json'),
        'source_ddr_sha256': common.sha256(source_run / 'ddr.json'),
        'baseline_is_reference_only_not_new_measurement': True,
    }


def validate_bundle(bundle):
    bundle = load_legacy(bundle)
    common = api(bundle)
    from cpu_vta_partitioning.prepare.payload import execution_files
    manifest = common.read(bundle / 'manifest.json')
    if common.digest({k: v for k, v in manifest.items() if k != 'bundle_id'})[:24] != manifest.get('bundle_id'):
        raise ValueError('diagnostic manifest identity differs')
    if manifest.get('experiment') != protocol()['experiment'] or manifest.get('controller_identity') != identity():
        raise ValueError('diagnostic controller changed; use its frozen snapshot')
    common.verify_inventory(bundle, manifest['files'])
    if common.read(bundle / 'plans/diagnosis_protocol.json') != protocol():
        raise ValueError('diagnostic protocol differs')
    programs = common.read(bundle / 'plans/programs.json')
    if set(programs) != set(PROGRAMS):
        raise ValueError('diagnostic module closure differs')
    cases(bundle)
    inputs = common.read(bundle / 'plans/inputs.json')
    if (inputs['available_indices'] != [0] or len(inputs['records']) != 1
            or inputs['records'][0]['index'] != 0):
        raise ValueError('only input zero is allowed')
    common.verify_inventory(bundle, inputs['files'])
    proof_documents = inputs.get('reference_identity', {}).get('documents', {})
    if set(proof_documents) != {'sources/reference/inputs_manifest.json',
                                'sources/reference/prior_inputs_manifest.json',
                                'sources/reference/reference_proof.json'}:
        raise ValueError('reference proof document closure is incomplete')
    for name, expected in proof_documents.items():
        path = common.relative_file(bundle, name)
        if not path.is_file() or common.sha256(path) != expected:
            raise ValueError('reference proof document missing or changed: ' + name)
    executor = common.read(bundle / 'plans/executors.json')['policy']
    if common.sha256(bundle / executor['path']) != executor['sha256']:
        raise ValueError('frozen diagnostic executor changed')
    payload = common.read(bundle / 'plans/board_payload.json')
    if payload['files'] != execution_files(bundle) or payload['unpacked_bytes'] != sum(v['bytes'] for v in payload['files'].values()):
        raise ValueError('diagnostic execution closure differs')
    provenance = common.read(bundle / 'plans/source_provenance.json')
    if common.sha256(bundle / 'plans/baseline_costs.json') != provenance['source_costs_sha256']:
        raise ValueError('baseline costs differ from frozen source')
    if common.sha256(bundle / 'plans/baseline_ddr.json') != provenance['source_ddr_sha256']:
        raise ValueError('baseline DDR costs differ from frozen source')
    from ..open_resources.costs import Costs, score_configuration
    old_costs = Costs(common.read(bundle / 'plans/baseline_costs.json'), programs=programs,
                      ddr=common.read(bundle / 'plans/baseline_ddr.json'))
    expected_predictions = {cfg['id']: {'diagnosis_case': name, **score_configuration(cfg, old_costs)}
                            for name, cfg in cases(bundle)['pipelines'].items()}
    if common.read(bundle / 'plans/baseline_predictions.json') != expected_predictions:
        raise ValueError('baseline diagnostic predictions differ from frozen costs')
    ready = common.read(bundle / 'OFFLINE_READINESS.json')
    if ready.get('status') != 'PREPARED_OFFLINE' or ready.get('tests', {}).get('passed') is not True:
        raise ValueError('diagnostic offline acceptance incomplete')
    return manifest


def prepare(parent=None, source_run=None, *, run_tests=True):
    parent = load_legacy(parent or DEFAULT_PARENT)
    source_run = Path(source_run or DEFAULT_SOURCE_RUN).resolve()
    common = api(parent)
    from cpu_vta_partitioning.prepare.payload import execution_files
    from cpu_vta_partitioning.prepare.package import elf
    if not run_tests:
        raise ValueError('a deployable diagnosis capsule requires passing tests')
    original, provenance = _source_identity(parent, source_run)
    code = identity()
    builds = ROOT / 'artifacts/diagnosis_builds'
    builds.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='prepare-', dir=builds))
    # Every copied executable retains its source hash. Existing bundles/runs are read-only.
    for folder in ('runtime', 'runner', 'firmware', 'inputs', 'references'):
        for name, expected in common.inventory(parent / folder).items():
            _copy(parent / folder / name, work / folder / name, expected['sha256'])
    full_programs = common.read(parent / 'plans/programs.json')
    programs = {name: deepcopy(full_programs[name]) for name in sorted(PROGRAMS)}
    for name, item in programs.items():
        for leaf, expected in item['files'].items():
            _copy(parent / 'programs' / name / leaf, work / 'programs' / name / leaf, expected['sha256'])
    for name in ('inputs.json', 'executors.json', 'compiler_identity.json', 'runtime_build.json',
                 'sdk_runtime.json', 'policy_native_acceptance.json', 'policy_runner_patch.json'):
        _copy(parent / 'plans' / name, work / 'plans' / name)
    # Include original native sources and reusable controller primitives, not their old results.
    for name in ('base_controller', 'cpu_strategy', 'native_source', 'vta'):
        shutil.copytree(parent / 'sources' / name, work / 'sources' / name,
                        ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache'))
    common.write(work / 'plans/reference_document_provenance.json', _reference_documents(parent, work))
    snapshot = work / 'sources/diagnosis_controller'
    for name in ('run_diagnosis.py', 'pyproject.toml'):
        _copy(ROOT / name, snapshot / name)
    documentation = ROOT / 'docs/prediction_gap_diagnosis.md'
    if documentation.is_file():
        _copy(documentation, snapshot / 'docs/prediction_gap_diagnosis.md')
    _copy(ROOT / 'cpu_policies/__init__.py', snapshot / 'cpu_policies/__init__.py')
    for name in ('diagnosis', 'open_resources'):
        shutil.copytree(ROOT / 'cpu_policies' / name, snapshot / 'cpu_policies' / name,
                        ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache'))
    for name in ('diagnosis', 'open_resources'):
        shutil.copytree(ROOT / 'configs' / name, snapshot / 'configs' / name)
    for path in sorted((ROOT / 'tests').glob('*diagnosis*.py')):
        _copy(path, snapshot / 'tests' / path.name)
    _copy(source_run / 'costs.json', work / 'plans/baseline_costs.json', provenance['source_costs_sha256'])
    _copy(source_run / 'ddr.json', work / 'plans/baseline_ddr.json', provenance['source_ddr_sha256'])
    from ..open_resources.costs import Costs, score_configuration
    old_costs = Costs(common.read(work / 'plans/baseline_costs.json'), programs=full_programs,
                      ddr=common.read(work / 'plans/baseline_ddr.json'))
    predictions = {cfg['id']: {'diagnosis_case': name, **score_configuration(cfg, old_costs)}
                   for name, cfg in configurations(full_programs)['pipelines'].items()}
    common.write(work / 'plans/baseline_predictions.json', predictions)
    _copy(source_run / 'state.json', work / 'provenance/baseline_state.json', provenance['source_state_sha256'])
    common.write(work / 'plans/programs.json', programs)
    common.write(work / 'plans/diagnosis_configurations.json', configurations(programs))
    common.write(work / 'plans/diagnosis_protocol.json', protocol())
    common.write(work / 'plans/pilot_protocol.json', protocol())
    common.write(work / 'plans/source_provenance.json', provenance)
    common.write(work / 'plans/source_parent_manifest.json', original)
    common.write(work / 'plans/board_execution.json', {
        'revision': 'prediction-gap-diagnosis-1', 'udmabuf_bytes': 201326592,
        'storage_margin_bytes': 256 * 1024**2,
        'boot_policy': protocol()['boot_policy']})
    payload = execution_files(work)
    common.write(work / 'plans/board_payload.json', {
        'revision': 1, 'files': payload, 'unpacked_bytes': sum(v['bytes'] for v in payload.values()),
        'transfer': 'gzip_tar_stream_no_remote_archive'})
    closure = {}
    for path in [*(work / 'runner').iterdir(), *(work / 'runtime').iterdir(),
                 *(work / 'programs' / name / 'module.so' for name in programs)]:
        if not path.is_file():
            continue
        info = elf(path)
        if any(not (work / 'runtime' / name).is_file() for name in info['needed']):
            raise ValueError('unresolved packaged shared library')
        closure[str(path.relative_to(work))] = info
    common.write(work / 'plans/elf_dependencies.json', closure)
    emulator = shutil.which('qemu-aarch64-static')
    if not emulator:
        raise RuntimeError('QEMU unavailable for packaged loader validation')
    checked = subprocess.run([emulator, str(work / 'runtime/ld-linux-aarch64.so.1'),
                              '--library-path', str(work / 'runtime'),
                              str(work / 'runner/vta_stage_pipeline_runner_pinned'), '--help'],
                             capture_output=True, text=True, timeout=45)
    help_text = checked.stdout + checked.stderr
    required_flags = ('--cpu-worker-pinning', '--queue-timing', '--session-profile-json',
                      '--session-profile-window', '--resource-affinity-probe')
    if checked.returncode or any(flag not in help_text for flag in required_flags):
        raise RuntimeError('packaged loader or diagnostic parameters unsupported')
    env = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
    testfiles = sorted(str(path.relative_to(ROOT)) for path in (ROOT / 'tests').glob('*diagnosis*.py'))
    if not testfiles:
        raise ValueError('diagnostic acceptance tests missing')
    tests = subprocess.run([sys.executable, '-m', 'pytest', '-c', 'pyproject.toml', '-q', *testfiles],
                           cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
    common.write(work / 'plans/offline_tests.json', {
        'passed': tests.returncode == 0, 'stdout': tests.stdout, 'stderr': tests.stderr,
        'testfiles': testfiles})
    if tests.returncode:
        raise RuntimeError('diagnosis tests failed:\n' + tests.stdout + tests.stderr)
    if identity() != code:
        raise RuntimeError('controller changed during preparation; prepare after edits finish')
    common.write(work / 'OFFLINE_READINESS.json', {
        'status': 'PREPARED_OFFLINE', 'controller_identity': code, 'tests': {'passed': True},
        'programs': {'cpu': 4, 'vta': 4}, 'component_cases': 4, 'pipeline_cases': 3,
        'expected_tasks': 20, 'expected_forwards': 299,
        'new_model_compilations': 0, 'new_executor_compilations': 0,
        'executable_payload_bytes': sum(v['bytes'] for v in payload.values()),
        'board_validation_pending': ['actual thread/role affinity', 'complete output correctness',
                                     'session profile artifact and counter availability'],
        'unsupported_counters_are_unknown_not_zero': True,
        'source_state_sha256': provenance['source_state_sha256']})
    files = {n: v for n, v in common.inventory(work).items()
             if '__pycache__/' not in n and '.pytest_cache/' not in n}
    manifest = {'experiment': protocol()['experiment'], 'controller_identity': code,
                'parent_bundle_id': original['bundle_id'], 'files': files}
    manifest['bundle_id'] = common.digest(manifest)[:24]
    common.write(work / 'manifest.json', manifest)
    target = ROOT / 'artifacts/bundles' / manifest['bundle_id']
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        # This only removes this function's unsealed temporary preparation directory.
        validate_bundle(target)
        shutil.rmtree(work)
    else:
        work.rename(target)
    validate_bundle(target)
    return target
