from pathlib import Path
import json
import os
import shlex
import shutil
import subprocess

from cpu_vta_partitioning.common import now, read, sha256, write
from cpu_vta_partitioning.transport import SSH
from scheduler_event_audit.decode import validate_stream, intersection

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'artifacts/yield_semantics_01'
CONFIG = ROOT / 'artifacts/cross_network_model_01/board.local.json'


def freeze(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError('Yield frozen evidence differs: ' + str(path))
    else:
        write(path, value)
    return value


def sources():
    paths = [ROOT / 'run_yield_semantics.py', *sorted(Path(__file__).parent.glob('*.py')),
             Path(__file__).parent / 'probe_body.cc', ROOT / 'tests/test_yield_semantics.py',
             ROOT / 'scheduler_event_audit/switch_probe.cc',
             ROOT / 'scheduler_event_audit/decode.py']
    return {str(p): sha256(p) for p in paths}


def protocol():
    return dict(experiment='yield_semantics_01', max_native_sessions=1, DNN_inferences=0,
        FPGA_access=False, system_changes=False, reboot=False, automatic_repeat=False,
        cases=['busy_same', 'yield_same', 'yield_separate'],
        workload_threads=6, target_busy_CPU_ns_per_case=80_000_000,
        target_busy_CPU_ns_total=240_000_000, per_thread_wall_limit_ns=3_000_000_000,
        parent_core=2, caller_core=0, separate_target_core=1,
        hypotheses=[
            'Yield can reduce a waiting caller service share when another task shares its core',
            'a Yield waiter can still consume CPU when no other task shares its core'],
        observation='compare waiter CPU per common wall window; retain no-change and mixed outcomes',
        directional_support='same-core yielding waiter CPU fraction below busy-wait fraction; separate-core waiter remains CPU-active',
        no_DNN_slowdown_fitting=True,
        no_automatic_model_change=True,
        source_basis='TVM binary Yield -> sched_yield; Linux v5.4 fair.c skip buddy with fairness guard; vendor details unknown')


def prepare(output=OUTPUT):
    output = Path(output)
    build = output / 'build'
    build.mkdir(parents=True, exist_ok=True)
    prepared = build / 'PREPARED.json'
    if prepared.exists():
        doc = read(prepared)
        if doc['sources'] != sources() or sha256(build / 'probe') != doc['binary_sha256']:
            raise ValueError('Yield preparation identity changed')
        return dict(status='CACHED', binary_sha256=doc['binary_sha256'])
    original = ROOT / 'scheduler_event_audit/switch_probe.cc'
    previous = read(ROOT / 'artifacts/finite_service_01/build/PREPARED.json')
    if previous['sources'][str(original)] != sha256(original):
        raise ValueError('validated event helper source changed')
    prefix = original.read_text().split('struct Segment {', 1)[0]
    code = prefix + '\n' + (Path(__file__).parent / 'probe_body.cc').read_text()
    source = build / 'probe.cc'
    source.write_text(code)
    current_boot = read(ROOT / 'artifacts/finite_service_01/PLAN.json')['expected_boot_id']
    runtime = ROOT / 'artifacts/scheduler_gap_audit_01/yolov3_tiny416/capsule/runtime'
    library = runtime / 'libtvm_runtime.so'
    objdump = shutil.which('aarch64-linux-gnu-objdump')
    if not objdump:
        raise ValueError('AArch64 binary disassembler unavailable')
    dis = subprocess.run([objdump, '-d', '--disassemble=_ZN3tvm7runtime9threading5YieldEv', str(library)],
                         capture_output=True, text=True, check=True)
    if '<sched_yield@plt>' not in dis.stdout:
        raise ValueError('deployed runtime Yield implementation unproven')
    (output / 'Yield.disassembly.txt').write_text(dis.stdout)
    freeze(output / 'PLAN.json', dict(protocol=protocol(), sources=sources(),
        expected_boot_id=current_boot, source_sha256=sha256(source),
        runtime_sha256=sha256(library), disassembly_sha256=sha256(output / 'Yield.disassembly.txt'),
        hypothesis_frozen_before_measurement=True))
    tc = previous['toolchain']
    if sha256(tc['sdk_setup']) != tc['sdk_setup_sha256']:
        raise ValueError('SDK setup changed')
    for tool in list(tc['tools'].values()) + list(tc.get('setup_files', {}).values()):
        if sha256(tool['path']) != tool['sha256']:
            raise ValueError('SDK tool changed')
    env = dict(os.environ)
    env.pop('LD_LIBRARY_PATH', None)
    command = ['bash', '-c', 'source "$1" || exit $?; shift; exec "$@"', 'yield-sdk', tc['sdk_setup'],
               'bash', '-c', 'exec $CXX -std=c++17 -O2 "$1" -o "$2" -pthread', 'yield-compile',
               str(source.resolve()), str((build / 'probe').resolve())]
    p = subprocess.run(command, env=env, capture_output=True, text=True, timeout=120)
    (build / 'compile.log').write_text(p.stdout + p.stderr)
    if p.returncode:
        raise ValueError('Yield probe compilation failed')
    qemu = shutil.which('qemu-aarch64-static')
    if not qemu:
        raise ValueError('QEMU unavailable')
    p = subprocess.run([qemu, str(runtime / 'ld-linux-aarch64.so.1'), '--library-path', str(runtime),
                        str((build / 'probe').resolve()), '--describe'],
                       capture_output=True, text=True, timeout=30, check=True)
    description = json.loads(p.stdout)
    if description['DNN_inferences'] or description['workload_threads'] != 6:
        raise ValueError('native scope differs')
    freeze(prepared, dict(sources=sources(), source_sha256=sha256(source),
        binary_sha256=sha256(build / 'probe'), toolchain=tc, QEMU_load_passed=True,
        QEMU_not_native_counter_validation=True, description=description))
    return dict(status='PREPARED', binary_sha256=sha256(build / 'probe'), DNN_inferences=0)


def summarize(rows):
    comparisons = []
    for name in protocol()['cases']:
        pair = [r for r in rows if r['case'] == name]
        if sorted(r['index'] for r in pair) != [0, 1]:
            raise ValueError('case pair incomplete')
        by = {r['index']: r for r in pair}
        segments = {i: by[i]['segments'][0] for i in (0, 1)}
        left = max(s['wall_start_ns'] for s in segments.values())
        right = min(s['wall_end_ns'] for s in segments.values())
        if right <= left:
            raise ValueError('controlled workers had no common work window')
        overlap_CPU = {i: sum(intersection(left, right, x['start_ns'], x['end_ns'])
                             for x in by[i]['on_CPU']) for i in (0, 1)}
        comparisons.append(dict(case=name, common_wall_ns=right-left,
            target_CPU_in_common_window_ns=overlap_CPU[0],
            waiter_CPU_in_common_window_ns=overlap_CPU[1],
            waiter_CPU_fraction=overlap_CPU[1]/(right-left),
            target_total_CPU_ns=segments[0]['CPU_clock_ns'],
            target_wall_ns=segments[0]['wall_end_ns']-segments[0]['wall_start_ns'],
            waiter_total_CPU_ns=segments[1]['CPU_clock_ns'],
            waiter_switches=by[1]['switches'], waiter_nonpreempt_out=by[1]['nonpreempt_out']))
    by = {r['case']: r for r in comparisons}
    return dict(cases=comparisons,
        same_core_Yield_reduces_waiter_fraction=(by['yield_same']['waiter_CPU_fraction'] < by['busy_same']['waiter_CPU_fraction']),
        isolated_core_Yield_waiter_active=by['yield_separate']['waiter_CPU_in_common_window_ns'] > 0,
        not_DNN_prediction_validation=True, vendor_CFS_reproduced=False)


def check(document):
    if document.get('status') != 'MEASURED' or document.get('DNN_inferences') != 0 or document.get('periodic_sampling') is not False:
        raise ValueError('Yield experiment unavailable; cannot infer zero waiting')
    streams = document['streams']
    expected = [(name, i) for name in protocol()['cases'] for i in range(2)]
    if [(s['case'], s['index']) for s in streams] != expected:
        raise ValueError('Yield controlled cases incomplete')
    if len({s['tid'] for s in streams}) != 6:
        raise ValueError('Yield task identity ambiguous')
    rows = []
    for stream in streams:
        if stream['data_size']-(stream['head']-stream['tail']) < 256:
            raise ValueError('Yield event ring near full; pending loss cannot be excluded')
        if stream['nice'] != 0 or len(stream['segments']) != 1:
            raise ValueError('Yield task settings differ')
        row = validate_stream(stream)
        segment = row['segments'][0]
        core = 1 if row['case'] == 'yield_separate' and row['index'] == 0 else 0
        if segment['requested_core'] != core:
            raise ValueError('Yield work core changed')
        if segment['wall_end_ns']-segment['wall_start_ns'] > 3_100_000_000:
            raise ValueError('finite task exceeded wall limit')
        if row['index'] == 0 and not 80_000_000 <= segment['CPU_clock_ns'] <= 85_000_000:
            raise ValueError('target busy CPU work budget differs')
        should_yield = row['index'] == 1 and row['case'].startswith('yield')
        if should_yield != (stream['yield_calls'] > 0):
            raise ValueError('requested wait mechanism did not execute')
        row['yield_calls'] = stream['yield_calls']
        rows.append(row)
    # Reject inconsistent simultaneous execution on a physical core.
    for name in protocol()['cases']:
        pair = [r for r in rows if r['case'] == name]
        for a in pair[0]['on_CPU']:
            for b in pair[1]['on_CPU']:
                if a['cpu'] == b['cpu'] and intersection(a['start_ns'], a['end_ns'], b['start_ns'], b['end_ns']) > 100_000:
                    raise ValueError('two controlled tasks execute on one core')
    return dict(passed=True, rows=rows, summary=summarize(rows),
                records=sum(len(r['events']) for r in rows), no_lost_records=True)


def board(output=OUTPUT, *, execute_board=False):
    if not execute_board:
        raise ValueError('--execute-board is required')
    output = Path(output)
    plan, proof = read(output / 'PLAN.json'), read(output / 'build/PREPARED.json')
    if plan['sources'] != sources() or proof['sources'] != sources() or sha256(output / 'build/probe') != proof['binary_sha256']:
        raise ValueError('Yield native identity changed')
    receipt = output / 'RECEIPT.json'
    if receipt.exists():
        old = read(receipt)
        if sha256(old['raw']) != old['raw_sha256'] or check(read(old['raw'])) != old['validation']:
            raise ValueError('Yield accepted evidence changed')
        return dict(status='CACHED', new_native_sessions=0, summary=old['validation']['summary'])
    ssh = SSH(read(CONFIG)['board'], execute_board=True)
    boot = ssh.boot_id()
    if boot != plan['expected_boot_id']:
        raise ValueError('boot differs; stop without mixing batches')
    remote = ssh.config['remote_base'].rstrip('/')+'/diagnostics/yield_semantics_01/'+boot+'/'+proof['binary_sha256'][:24]
    quote = shlex.quote
    ssh.idle_command('mkdir -p '+quote(remote), boot=boot)
    ssh.upload(output / 'build/probe', remote+'/probe')
    script = '\n'.join(['cd '+quote(remote),
        'test "$(sha256sum probe | cut -d " " -f 1)" = '+quote(proof['binary_sha256']),
        'chmod 755 probe',
        'if test -f exit_code; then :; elif test -d .attempt; then exit 86; else',
        'mkdir .attempt',
        'if ./probe > raw.json.tmp 2> native.log; then code=0; else code=$?; fi',
        'mv raw.json.tmp raw.json', 'printf "%s\\n" "$code" > exit_code', 'fi', 'cat raw.json'])
    raw = ssh.idle_command(script, boot=boot, timeout=30)
    path = output / 'raw.json'
    path.write_text(raw)
    validation = check(read(path))
    if ssh.boot_id() != boot:
        raise ValueError('boot changed during Yield check')
    freeze(receipt, dict(boot_id=boot, raw=str(path.resolve()), raw_sha256=sha256(path),
        binary_sha256=proof['binary_sha256'], sources=sources(), validation=validation,
        native_sessions=1, DNN_inferences=0, FPGA_access=False, system_changes=False, recorded_at=now()))
    return dict(status='COMPLETE', new_native_sessions=1, DNN_inferences=0,
                records=validation['records'], summary=validation['summary'])
