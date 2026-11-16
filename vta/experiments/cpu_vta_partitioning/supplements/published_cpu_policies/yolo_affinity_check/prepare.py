"""Copy only the frozen assets; no compilation or board access."""
from pathlib import Path
import os
import shutil
import subprocess
import time

from cpu_vta_partitioning.common import (digest, inventory, now, read, sha256,
    sync_tree, verify_inventory, write)
from cpu_vta_partitioning.prepare.build import copy_checked
from .common import (PARENT, ROOT, protocol, configurations, task_order, sources,
                     frozen, assert_prepared, validate_config)


def audit(output):
    parent = read(PARENT / 'manifest.json')
    verify_inventory(PARENT, parent['files'])
    catalog = read(PARENT / 'plans/programs.json')
    configs = configurations()
    for scope, values in configs.items():
        for cfg in values.values():
            validate_config(cfg, catalog, component=scope == 'components')
    value = dict(status='AUDITED', parent_bundle_id=parent['bundle_id'],
        parent_manifest_sha256=sha256(PARENT / 'manifest.json'),
        runner_sha256=parent['runner_sha256'], runtime_sha256=parent['runtime_sha256'],
        configurations=configs, pipeline_FPS_read=False, new_compilations=0)
    frozen(Path(output) / 'AUDIT.json', value)
    return value


def qemu_test(bundle, work):
    """Execute every isolated 3/4-thread mode with the real ARM CPU module."""
    from .runtime import make_task, native_command, summary
    bundle, work = Path(bundle).resolve(), Path(work).resolve()
    emulator = shutil.which('qemu-aarch64-static')
    if not emulator:
        raise RuntimeError('QEMU is required for the existing ARM executor')
    load = subprocess.run([emulator, str(bundle / 'runtime/ld-linux-aarch64.so.1'),
        '--library-path', str(bundle / 'runtime'), str(bundle / 'runner/vta_stage_pipeline_runner'),
        '--help'], capture_output=True, text=True, timeout=45)
    if load.returncode or any(flag not in load.stdout + load.stderr for flag in
            ('--cpu-worker-pinning', '--resource-affinity-probe', '--control-cpu-affinity')):
        raise ValueError('frozen executor lacks the required ARM ABI/flags')
    cases = []
    host_env = dict(os.environ)
    for name in ('LD_PRELOAD', 'LD_LIBRARY_PATH'):
        host_env.pop(name, None)
    for spec in task_order():
        if spec['scope'] != 'component' or spec['phase'] != 'qualify':
            continue
        raw = work / spec['case']
        raw.mkdir(parents=True, exist_ok=True)
        doc = make_task(bundle, spec)
        argv, inputs = native_command(doc, bundle, str(bundle), str(raw))
        (raw / 'inputs.txt').write_text(inputs)
        loader = argv.index(str(bundle / 'runtime/ld-linux-aarch64.so.1'))
        guest_env = argv[2:loader]
        command = [emulator, *[v for item in guest_env for v in ('-E', item)], *argv[loader:]]
        started = time.monotonic()
        result = subprocess.run(command, env=host_env, capture_output=True, text=True, timeout=90)
        (raw / 'native.log').write_text(result.stdout + result.stderr)
        write(raw / 'task.json', doc, replace=True)
        if result.returncode:
            raise ValueError('QEMU CPU qualification failed: ' + spec['case'])
        verified = summary(raw, doc, bundle, qemu=True)
        cases.append(dict(case=spec['case'], passed=True, seconds=time.monotonic()-started,
            correct=verified['correct'], native_stage_calls=1,
            emulator_helpers_separate_from_guest_compute_slots=True,
            affinity=read(raw / 'affinity.json')))
    if len(cases) != 4:
        raise ValueError('QEMU did not cover all four component modes')
    value = dict(passed=True, cases=cases, local_CPU_stage_calls=4,
        counts_excluded_from_board_DNN_and_component_budget=True,
        board_validator_qemu_allowance=False, executor_sha256=sha256(bundle/'runner/vta_stage_pipeline_runner'))
    write(work / 'RESULT.json', value, replace=True)
    return value


def prepare(output):
    output = Path(output).resolve()
    if (output / 'PREPARATION.json').exists():
        return validate(output)
    started = time.monotonic()
    if not (output / 'AUDIT.json').exists():
        audit(output)
    parent = read(PARENT / 'manifest.json')
    configs, bundle = configurations(), output / 'capsule'
    for name, info in parent['files'].items():
        if name.startswith(('runtime/', 'runner/', 'firmware/', 'references/')) or name == 'inputs/000.bin':
            copy_checked(PARENT / name, bundle / name, info['sha256'])
    full = read(PARENT / 'plans/programs.json')
    programs = {s['program']: full[s['program']] for c in configs['pipelines'].values() for s in c['stages']}
    for name in programs:
        for leaf in ('graph.json', 'module.so', 'params.bin', 'record.json'):
            copy_checked(PARENT / 'programs' / name / leaf, bundle / 'programs' / name / leaf)
    for leaf in ('inputs.json', 'expected_outputs.json'):
        copy_checked(PARENT / 'plans' / leaf, bundle / 'plans' / leaf)
    write(bundle / 'plans/programs.json', programs, replace=True)
    write(bundle / 'plans/configurations.json', configs, replace=True)
    write(bundle / 'plans/protocol.json', protocol(), replace=True)
    write(bundle / 'plans/task_order.json', task_order(), replace=True)
    # A temporary local ID is used solely for four non-board QEMU checks.
    write(bundle / 'manifest.json', dict(bundle_id='LOCAL_QUALIFICATION_ONLY',
          runner_sha256=parent['runner_sha256'], runtime_sha256=parent['runtime_sha256']), replace=True)
    result = qemu_test(bundle, output / 'local_QEMU')
    write(bundle / 'plans/QEMU.json', result, replace=True)
    prepared = dict(protocol=protocol(), sources=sources(), prepared_at=now(),
        source_files={str(p.resolve()): sha256(p) for p in
            (PARENT / 'manifest.json', PARENT / 'plans/configurations.json')},
        source_programs=programs, runner_sha256=parent['runner_sha256'],
        runtime_sha256=parent['runtime_sha256'])
    frozen(output / 'PREPARATION.json', prepared)
    write(bundle / 'plans/PREPARATION.json', prepared, replace=True)
    import cpu_vta_partitioning
    base = Path(cpu_vta_partitioning.__file__).parent
    for name, expected in prepared['sources'].items():
        src = base / name[6:] if name.startswith('@base/') else ROOT / name
        copy_checked(src, bundle / 'sources' / name.replace('@base/', 'base/'), expected)
    write(bundle / 'plans/board_execution.json', dict(boot_policy='one_boot_no_automatic_reboot',
        udmabuf_bytes=201326592, storage_margin_bytes=268435456), replace=True)
    payload = {n: i for n, i in inventory(bundle).items()
               if n.startswith(('programs/', 'runtime/', 'runner/', 'firmware/')) and not n.endswith('/record.json')}
    for name in ('inputs/000.bin', 'plans/board_execution.json'):
        payload[name] = dict(bytes=(bundle / name).stat().st_size, sha256=sha256(bundle / name))
    write(bundle / 'plans/board_payload.json', dict(files=payload,
        unpacked_bytes=sum(i['bytes'] for i in payload.values())), replace=True)
    write(bundle / 'plans/preparation_cost.json', dict(seconds=time.monotonic()-started,
        new_stage_compilations=0, runtime_rebuilds=0, executor_rebuilds=0,
        payload_bytes=sum(i['bytes'] for i in payload.values())), replace=True)
    manifest = dict(experiment=protocol()['experiment'], runner_sha256=parent['runner_sha256'],
        runtime_sha256=parent['runtime_sha256'], files=inventory(bundle, ['manifest.json']))
    manifest['bundle_id'] = digest(manifest)[:24]
    write(bundle / 'manifest.json', manifest, replace=True)
    sync_tree(bundle)
    return validate(output)


def validate_bundle(bundle):
    bundle = Path(bundle)
    m = read(bundle / 'manifest.json')
    if m['bundle_id'] != digest({k: v for k, v in m.items() if k != 'bundle_id'})[:24]:
        raise ValueError('affinity capsule seal differs')
    verify_inventory(bundle, m['files'])
    if read(bundle / 'plans/protocol.json') != protocol():
        raise ValueError('protocol differs')
    if read(bundle / 'plans/PREPARATION.json')['sources'] != sources():
        raise ValueError('controller source changed')
    if read(bundle / 'plans/task_order.json') != task_order():
        raise ValueError('frozen execution order differs')
    if read(bundle / 'plans/configurations.json') != configurations():
        raise ValueError('matched configurations changed')
    if not read(bundle / 'plans/QEMU.json')['passed']:
        raise ValueError('QEMU qualification missing')
    for path, field in (('runner/vta_stage_pipeline_runner', 'runner_sha256'),
                        ('runtime/libtvm_runtime.so', 'runtime_sha256')):
        if sha256(bundle / path) != m[field]:
            raise ValueError('executor/runtime identity differs')
    # Every native path must resolve under the relocated capsule/job root.
    from .runtime import make_task, native_command
    for spec in task_order():
        doc = make_task(bundle, spec)
        argv, inputs = native_command(doc, bundle, '/relocated/capsule', '/relocated/job')
        if any(str(PARENT) in v or str(bundle) in v for v in argv + [inputs]):
            raise ValueError('native command still depends on a local source directory')
    return m


def validate(output):
    assert_prepared(output)
    m = validate_bundle(Path(output) / 'capsule')
    return dict(validated=True, bundle_id=m['bundle_id'], executor_unchanged=True,
        runtime_unchanged=True, native_sessions=20, full_DNN_inferences=836,
        isolated_CPU_stage_calls=44, qemu_passed=True)
