"""Local-only construction of an independent core-model diagnosis capsule.

The original runner and five stage modules are copied byte for byte. Only a
new opt-in diagnostic executable is compiled; TVM and VTA runtimes are not built.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile

from .common import (ROOT, REPO, DEFAULT_PARENT, DEFAULT_SOURCE_RUN, api, identity,
                     load_legacy, protocol)
from .domain import PROGRAMS, configurations, requirements, baseline_candidate
from ..diagnosis.prepare import _copy, _reference_documents


BASELINE_FILES = ('baseline_costs.json', 'baseline_ddr.json', 'baseline_predictions.json')


def _source_identity(parent, source_run):
    common = api(parent)
    manifest = common.read(parent / 'manifest.json')
    if common.digest({k: v for k, v in manifest.items() if k != 'bundle_id'})[:24] != manifest['bundle_id']:
        raise ValueError('core-model parent manifest identity differs')
    common.verify_inventory(parent, manifest['files'])
    state = common.read(source_run / 'state.json')
    if state.get('status') != 'COMPLETE' or state.get('bundle_id') != manifest['bundle_id']:
        raise ValueError('source diagnosis must be a completed run of the parent bundle')
    common.verify_inventory(source_run, state['frozen_files'])
    old_cases = common.read(parent / 'plans/diagnosis_configurations.json')
    baseline = baseline_candidate()
    if old_cases['pipelines'].get('static_first') != baseline:
        raise ValueError('fixed baseline differs from the source experiment')
    policy = common.read(parent / 'plans/executors.json')['policy']
    native = parent / 'sources/vta/apps/native_deploy/vta_stage_pipeline_runner.cc'
    return manifest, {
        'source_bundle_id': manifest['bundle_id'],
        'source_bundle_manifest_sha256': common.sha256(parent / 'manifest.json'),
        'source_run': str(source_run), 'source_run_boot_id': state.get('boot_id'),
        'source_state_sha256': common.sha256(source_run / 'state.json'),
        'source_policy_executor_sha256': policy['sha256'],
        'source_pinned_native_sha256': common.sha256(native),
        'baseline_cost_table_origin': 'pass_through_parent_frozen_original_cost_table_not_new_diagnosis_calibration',
        'baseline_files': {name: common.sha256(parent / 'plans' / name) for name in BASELINE_FILES},
        'source_costs_sha256': common.sha256(parent / 'plans/baseline_costs.json'),
        'source_ddr_sha256': common.sha256(parent / 'plans/baseline_ddr.json'),
        'baseline_is_reference_only_not_new_measurement': True,
        'static_profile_input_policy': 'all_11_costs_are_new_this_batch_old_costs_reference_only',
    }


def _sdk_checked(parent, work):
    """Use the recorded SDK only when all effective compiler identities match."""
    common = api(parent)
    from cpu_vta_partitioning.prepare.toolchain import PROBE
    record = common.read(parent / 'plans/runtime_build.json')
    expected = record['toolchain']
    setup = Path(expected['sdk_setup'])
    if common.sha256(setup) != expected['sdk_setup_sha256']:
        raise ValueError('recorded SDK setup changed')
    env = dict(os.environ)
    env.pop('LD_LIBRARY_PATH', None)
    checked = subprocess.run(['bash', '-c', 'source "$1" || exit $?; shift; exec "$@"',
                              'core-model-sdk-probe', str(setup), sys.executable, '-c', PROBE],
                             capture_output=True, text=True, env=env, timeout=60)
    log = work / 'build_logs/sdk_probe.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(checked.stdout + checked.stderr)
    if checked.returncode:
        raise RuntimeError('SDK identity probe failed; see ' + str(log))
    actual = json.loads(checked.stdout.splitlines()[-1])
    actual.update(sdk_setup=str(setup), sdk_setup_sha256=common.sha256(setup))
    if actual != expected:
        raise ValueError('effective SDK compiler, flags or setup files differ from frozen toolchain')
    return record, env


def _compile_runner(parent, work):
    common = api(parent)
    from .patch_runner import prepare_source
    native = prepare_source(parent, work / 'sources/core_model_native')
    source = Path(native['source_path'])
    output = work / 'runner/vta_stage_pipeline_runner_core_model'
    record, env = _sdk_checked(parent, work)
    # The additional dependency file records the actual non-system compile inputs.
    original_recipe = record['runner_compiler_recipe']
    if not original_recipe.startswith('exec $CXX '):
        raise ValueError('unrecognized frozen runner compiler recipe')
    recipe = original_recipe.replace('exec $CXX ', 'exec $CXX -MMD -MF "$5" ', 1)
    dependency_file = work / 'build_logs/core_model_runner.d'
    command = ['bash', '-c', 'source "$1" || exit $?; shift; exec "$@"',
               'core-model-sdk', record['toolchain']['sdk_setup'], 'bash', '-c', recipe,
               'core-model-runner', str(source), str(output), str(REPO),
               str(work / 'runtime'), str(dependency_file)]
    result = subprocess.run(command, capture_output=True, text=True, env=env, timeout=300)
    (work / 'build_logs/core_model_compile.log').write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError('core-model runner compilation failed; see ' + str(work / 'build_logs/core_model_compile.log'))
    # Copy the used repository headers so later relocation has no source dependency.
    text = dependency_file.read_text().replace('\\\n', '')
    dependencies = shlex.split(text.split(':', 1)[1])
    headers = {}
    for item in dependencies:
        path = Path(item).resolve()
        if path.is_relative_to(REPO) and not path.is_relative_to(work):
            name = 'sources/build_headers/' + str(path.relative_to(REPO))
            _copy(path, work / name, common.sha256(path))
            headers[name] = {'original_path': str(path), 'sha256': common.sha256(path)}
    executors = deepcopy(common.read(work / 'plans/executors.json'))
    executors['core_model'] = {'path': 'runner/vta_stage_pipeline_runner_core_model',
                              'sha256': common.sha256(output)}
    common.write(work / 'plans/executors.json', executors, replace=True)
    native.update({'runner_path': executors['core_model']['path'], 'runner_sha256': common.sha256(output),
                   'source_path_in_bundle': str(source.relative_to(work)),
                   'compiler_recipe_original': original_recipe, 'compiler_recipe_used': recipe,
                   'toolchain': record['toolchain'], 'repository_headers': headers,
                   'runtime_rebuilt': False, 'stage_modules_recompiled': False})
    # Absolute staging paths are provenance only; run-time resolution uses the relative paths above.
    common.write(work / 'plans/core_model_native_build.json', native)
    return native


def _validate_native(bundle):
    common = api(bundle)
    specs = common.read(bundle / 'plans/executors.json')
    if not {'policy', 'core_model'} <= set(specs):
        raise ValueError('old and new core-model executors are both required')
    for key in ('policy', 'core_model'):
        entry = specs[key]
        if common.sha256(common.relative_file(bundle, entry['path'])) != entry['sha256']:
            raise ValueError('executor changed: ' + key)
    build = common.read(bundle / 'plans/core_model_native_build.json')
    if (build['runner_sha256'] != specs['core_model']['sha256']
            or build.get('runtime_rebuilt') is not False or build.get('stage_modules_recompiled') is not False):
        raise ValueError('core-model build identity differs')
    if common.sha256(bundle / build['source_path_in_bundle']) != build['source_patched_sha256']:
        raise ValueError('core-model patched source changed')
    for name, value in build.get('repository_headers', {}).items():
        if common.sha256(common.relative_file(bundle, name)) != value['sha256']:
            raise ValueError('recorded compile input changed')


def validate_bundle(bundle):
    bundle = load_legacy(bundle)
    common = api(bundle)
    from cpu_vta_partitioning.prepare.payload import execution_files
    manifest = common.read(bundle / 'manifest.json')
    if common.digest({k: v for k, v in manifest.items() if k != 'bundle_id'})[:24] != manifest.get('bundle_id'):
        raise ValueError('core-model manifest identity differs')
    if manifest.get('experiment') != protocol()['experiment'] or manifest.get('controller_identity') != identity():
        raise ValueError('core-model controller changed; use its frozen snapshot')
    common.verify_inventory(bundle, manifest['files'])
    if common.read(bundle / 'plans/core_model_protocol.json') != protocol():
        raise ValueError('core-model protocol differs')
    programs = common.read(bundle / 'plans/programs.json')
    if set(programs) != set(PROGRAMS):
        raise ValueError('core-model module closure differs')
    expected = configurations(programs)
    if common.read(bundle / 'plans/diagnosis_configurations.json') != expected:
        raise ValueError('fixed core-model cases changed')
    if common.read(bundle / 'plans/component_requirements.json') != requirements(programs):
        raise ValueError('eleven component requirements changed')
    if common.read(bundle / 'plans/fixed_baseline.json') != baseline_candidate():
        raise ValueError('fixed baseline identity changed')
    inputs = common.read(bundle / 'plans/inputs.json')
    if inputs['available_indices'] != [0] or len(inputs['records']) != 1 or inputs['records'][0]['index'] != 0:
        raise ValueError('core-model input identity differs')
    common.verify_inventory(bundle, inputs['files'])
    documents = inputs.get('reference_identity', {}).get('documents', {})
    if set(documents) != {'sources/reference/inputs_manifest.json', 'sources/reference/prior_inputs_manifest.json',
                          'sources/reference/reference_proof.json'}:
        raise ValueError('core-model reference proof closure is incomplete')
    for name, expected in documents.items():
        if common.sha256(common.relative_file(bundle, name)) != expected:
            raise ValueError('core-model reference proof changed')
    _validate_native(bundle)
    provenance = common.read(bundle / 'plans/source_provenance.json')
    for name in BASELINE_FILES:
        if common.sha256(bundle / 'plans' / name) != provenance['baseline_files'][name]:
            raise ValueError('pass-through baseline source changed')
    if common.sha256(bundle / 'runner/vta_stage_pipeline_runner_pinned') != provenance['source_policy_executor_sha256']:
        raise ValueError('legacy bridge runner differs from the completed first diagnosis')
    payload = common.read(bundle / 'plans/board_payload.json')
    if payload['files'] != execution_files(bundle) or payload['unpacked_bytes'] != sum(v['bytes'] for v in payload['files'].values()):
        raise ValueError('core-model payload closure differs')
    ready = common.read(bundle / 'OFFLINE_READINESS.json')
    if ready.get('status') != 'PREPARED_OFFLINE' or ready.get('tests', {}).get('passed') is not True:
        raise ValueError('core-model readiness incomplete')
    if ready.get('maximum_tasks') != 47 or ready.get('maximum_forwards') != 497:
        raise ValueError('core-model maximum window coverage differs')
    return manifest


def prepare(parent=None, source_run=None, *, run_tests=True):
    parent = load_legacy(parent or DEFAULT_PARENT)
    source_run = Path(source_run or DEFAULT_SOURCE_RUN).resolve()
    common = api(parent)
    from cpu_vta_partitioning.prepare.payload import execution_files
    from cpu_vta_partitioning.prepare.package import elf
    if not run_tests:
        raise ValueError('a deployable core-model capsule requires passing local tests')
    original, provenance = _source_identity(parent, source_run)
    code = identity()
    builds = ROOT / 'artifacts/core_model_builds'
    builds.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='prepare-', dir=builds))
    for folder in ('runtime', 'firmware', 'inputs', 'references'):
        for name, item in common.inventory(parent / folder).items():
            _copy(parent / folder / name, work / folder / name, item['sha256'])
    old_policy = common.read(parent / 'plans/executors.json')['policy']
    _copy(parent / old_policy['path'], work / old_policy['path'], old_policy['sha256'])
    all_programs = common.read(parent / 'plans/programs.json')
    if not set(PROGRAMS) <= set(all_programs):
        raise ValueError('source lacks five required verified modules')
    programs = {name: deepcopy(all_programs[name]) for name in PROGRAMS}
    for name, program in programs.items():
        for leaf, item in program['files'].items():
            _copy(parent / 'programs' / name / leaf, work / 'programs' / name / leaf, item['sha256'])
    for name in ('inputs.json', 'executors.json', 'compiler_identity.json', 'runtime_build.json',
                 'sdk_runtime.json', 'policy_native_acceptance.json', 'policy_runner_patch.json', *BASELINE_FILES):
        _copy(parent / 'plans' / name, work / 'plans' / name, original['files']['plans/' + name]['sha256'])
    common.write(work / 'plans/executors.json', {'policy': old_policy}, replace=True)
    for name in ('base_controller', 'cpu_strategy', 'native_source', 'vta'):
        shutil.copytree(parent / 'sources' / name, work / 'sources' / name,
                        ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache'))
    common.write(work / 'plans/reference_document_provenance.json', _reference_documents(parent, work))
    _copy(source_run / 'state.json', work / 'provenance/source_diagnosis_state.json', provenance['source_state_sha256'])
    common.write(work / 'plans/source_provenance.json', provenance)
    common.write(work / 'plans/source_parent_manifest.json', original)
    common.write(work / 'plans/programs.json', programs)
    common.write(work / 'plans/diagnosis_configurations.json', configurations(programs))
    common.write(work / 'plans/component_requirements.json', requirements(programs))
    common.write(work / 'plans/fixed_baseline.json', baseline_candidate())
    common.write(work / 'plans/core_model_protocol.json', protocol())
    common.write(work / 'plans/pilot_protocol.json', protocol())
    common.write(work / 'plans/board_execution.json', {
        'revision': 'prediction-gap-core-model-1', 'udmabuf_bytes': 201326592,
        'storage_margin_bytes': 256 * 1024**2, 'boot_policy': protocol()['boot_policy']})
    native = _compile_runner(parent, work)
    snapshot = work / 'sources/core_model_controller'
    for name in ('run_core_model.py', 'run_run_scope.py', 'run_diagnosis.py', 'pyproject.toml'):
        _copy(ROOT / name, snapshot / name)
    for filename in ('prediction_gap_diagnosis.md', 'RUN_SCOPE_DIAGNOSIS_zh.md', 'CORE_MODEL_VALIDATION_zh.md'):
        if (ROOT / 'docs' / filename).is_file():
            _copy(ROOT / 'docs' / filename, snapshot / 'docs' / filename)
    _copy(ROOT / 'cpu_policies/__init__.py', snapshot / 'cpu_policies/__init__.py')
    for name in ('core_model', 'run_scope', 'diagnosis', 'open_resources'):
        shutil.copytree(ROOT / 'cpu_policies' / name, snapshot / 'cpu_policies' / name,
                        ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache'))
        shutil.copytree(ROOT / 'configs' / name, snapshot / 'configs' / name)
    testfiles = sorted({*list((ROOT / 'tests').glob('*diagnosis*.py')),
                        *list((ROOT / 'tests').glob('*run_scope*.py')),
                        *list((ROOT / 'tests').glob('*core_model*.py'))})
    for path in testfiles:
        _copy(path, snapshot / 'tests' / path.name)
    closure = {}
    for path in [*(work / 'runner').iterdir(), *(work / 'runtime').iterdir(),
                 *(work / 'programs' / name / 'module.so' for name in programs)]:
        if path.is_file():
            info = elf(path)
            if any(not (work / 'runtime' / name).is_file() for name in info['needed']):
                raise ValueError('unresolved core-model ELF shared library')
            closure[str(path.relative_to(work))] = info
    common.write(work / 'plans/elf_dependencies.json', closure)
    emulator = shutil.which('qemu-aarch64-static')
    if not emulator:
        raise RuntimeError('QEMU required for core-model loader acceptance')
    for runner, flags in (('vta_stage_pipeline_runner_pinned', ('--cpu-worker-pinning', '--resource-affinity-probe')),
                          ('vta_stage_pipeline_runner_core_model', ('--run-thread-counters-json', '--run-thread-counters-frames', 'cpu-worker-map'))):
        checked = subprocess.run([emulator, str(work / 'runtime/ld-linux-aarch64.so.1'),
                                  '--library-path', str(work / 'runtime'), str(work / 'runner' / runner), '--help'],
                                 capture_output=True, text=True, timeout=45)
        if checked.returncode or any(flag not in checked.stdout + checked.stderr for flag in flags):
            raise RuntimeError('core-model runner loader or explicit parameter check failed: ' + runner)
    env = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
    tests = subprocess.run([sys.executable, '-m', 'pytest', '-c', 'pyproject.toml', '-q',
                            *(str(p.relative_to(ROOT)) for p in testfiles)],
                           cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
    common.write(work / 'plans/offline_tests.json', {'passed': tests.returncode == 0,
                 'stdout': tests.stdout, 'stderr': tests.stderr,
                 'testfiles': [str(p.relative_to(ROOT)) for p in testfiles]})
    if tests.returncode:
        raise RuntimeError('core-model tests failed:\n' + tests.stdout + tests.stderr)
    if identity() != code:
        raise RuntimeError('core-model controller changed during preparation; freeze after edits finish')
    payload = execution_files(work)
    common.write(work / 'plans/board_payload.json', {'revision': 1, 'files': payload,
                 'unpacked_bytes': sum(v['bytes'] for v in payload.values()),
                 'transfer': 'gzip_tar_stream_no_remote_archive'})
    common.write(work / 'OFFLINE_READINESS.json', {
        'status': 'PREPARED_OFFLINE', 'controller_identity': code, 'tests': {'passed': True},
        'programs': {'cpu': 3, 'vta': 2}, 'component_cases': 11, 'maximum_pipeline_cases': 12,
        'maximum_tasks': 47, 'maximum_forwards': 497,
        'new_model_compilations': 0, 'new_executor_compilations': 1,
        'old_policy_runner_sha256': provenance['source_policy_executor_sha256'],
        'new_core_model_runner_sha256': native['runner_sha256'],
        'executable_payload_bytes': sum(v['bytes'] for v in payload.values()),
        'unsupported_counters_are_unknown_not_zero': True,
        'source_state_sha256': provenance['source_state_sha256'],
        'board_validation_pending': ['actual core-model TID/role coverage', 'complete output correctness',
                                     'per-thread counter availability and fresh-snapshot descriptor safety']})
    files = {name: value for name, value in common.inventory(work).items()
             if '__pycache__/' not in name and '.pytest_cache/' not in name}
    manifest = {'experiment': protocol()['experiment'], 'controller_identity': code,
                'parent_bundle_id': original['bundle_id'], 'files': files}
    manifest['bundle_id'] = common.digest(manifest)[:24]
    common.write(work / 'manifest.json', manifest)
    target = ROOT / 'artifacts/bundles' / manifest['bundle_id']
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        validate_bundle(target)
        shutil.rmtree(work)
    else:
        work.rename(target)
    validate_bundle(target)
    return target
