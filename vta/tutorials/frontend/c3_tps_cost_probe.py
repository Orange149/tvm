"""Independent R boundary-timing preparation and local qualification; no board execution.

Only sealed generated runtime and helper sources are instrumented. Removing the
explicit observation additions must recover their source text byte for byte.
"""
import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

from c3_tps_sync_release_local import HERE, REPO, HOLD, OFFICIAL, PYTHON, read, write, sha, require, verify

LOCAL = HOLD / '20261016_c3_tps_sync_release_local_run04'
BOARD = HOLD / '20261016_c3_tps_sync_release_board_run01'
HEADER = HERE / 'c3_tps_cost_boundary.h'
UNITS = [('off', 'detailed'), ('off', 'minimal'), ('on', 'detailed'), ('on', 'minimal')]


def observation(code):
    return '// C3_COST_BEGIN\n' + code + '\n// C3_COST_END\n'


def strip_observation(text):
    return re.sub(r'// C3_COST_BEGIN\n.*?\n// C3_COST_END\n', '', text, flags=re.S)


def instrument_runtime(source):
    anchor = '    int timeout = VTADeviceRun(device_, insn_queue_.dram_phy_addr(), insn_count, wait_cycles);'
    require(source.count(anchor) == 1, 'DRIVER_SEAM_NOT_UNIQUE')
    source_copy = observation('#include "c3_tps_cost_boundary.h"') + source
    source_copy = source_copy.replace(anchor + '\n', observation('    c3_tps_cost::BeginDriver();') +
                                     anchor + '\n' + observation('    c3_tps_cost::EndDriver(timeout);'))
    clear = '  c3_official_window::ClearLedger();\n'
    require(source_copy.count(clear) == 1, 'CLEAR_SEAM_NOT_UNIQUE')
    source_copy = source_copy.replace(clear, clear + observation('  c3_tps_cost::Clear();'))
    source_copy += observation('TVM_REGISTER_GLOBAL("vta.runtime.tps_cost_boundary").set_body_typed([]() {\n'
                               '  return c3_tps_cost::JSON();\n});')
    require(strip_observation(source_copy) == source, 'RUNTIME_EXECUTION_SOURCE_CHANGED')
    return source_copy


def instrument_helper(source):
    prefix = '''namespace {
int64_t c3_cost_begin_ns = 0, c3_cost_end_ns = 0;
void C3CostSave(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
  c3_cost_begin_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(a.time_since_epoch()).count();
  c3_cost_end_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(b.time_since_epoch()).count();
}
}  // namespace'''
    anchor = 'namespace {\n'
    require(source.count(anchor) == 1, 'HELPER_NAMESPACE_SEAM')
    out = source.replace(anchor, observation(prefix) + anchor, 1)
    anchor = '      const auto end = std::chrono::steady_clock::now();\n'
    require(out.count(anchor) == 1, 'HELPER_TIMER_SEAM')
    out = out.replace(anchor, anchor + observation('      C3CostSave(begin, end);'))
    # Qualification dumps keep their existing semantics; their time is never a performance sample.
    anchor = '        target.function.CallPacked(target.parameters, &ignored);\n'
    require(out.count(anchor) == 1, 'HELPER_DIAGNOSTIC_SEAM')
    out = out.replace(anchor, observation('        const auto c3_diag_begin = std::chrono::steady_clock::now();') +
                      anchor + observation('        C3CostSave(c3_diag_begin, std::chrono::steady_clock::now());'))
    anchor = '      started.store(0); completed.store(0);\n'
    require(out.count(anchor) == 1, 'HELPER_RESET_SEAM')
    out = out.replace(anchor, anchor + observation('      c3_cost_begin_ns = c3_cost_end_ns = 0;'))
    out += observation('''TVM_REGISTER_GLOBAL("c3.tps_cost.helper_boundary").set_body_typed([]() {
  std::ostringstream out;
  out << "{\\"begin_ns\\":" << c3_cost_begin_ns << ",\\"end_ns\\":" << c3_cost_end_ns
      << ",\\"started_calls\\":" << started.load() << ",\\"completed_calls\\":" << completed.load() << '}';
  return out.str();
});''')
    require(strip_observation(out) == source, 'HELPER_EXECUTION_SOURCE_CHANGED')
    return out


def split_boundary(helper, driver, mode, index, elapsed_ms=None):
    require(mode in ('off', 'on') and driver['mode'] == mode, 'BOUNDARY_MODE_CHANGED')
    require(helper['started_calls'] == helper['completed_calls'] == index, 'HELPER_CALL_COUNT')
    require(driver['begins'] == driver['ends'] == 1 and driver['timeout'] == 0, 'DRIVER_CALL_COUNT_OR_TIMEOUT')
    a, z = helper['begin_ns'], helper['end_ns']
    require(type(a) is int and type(z) is int and 0 < a < z, 'INVALID_HELPER_CLOCK')
    total = (z-a)/1e6
    if elapsed_ms is not None:
        require(math.isfinite(elapsed_ms) and elapsed_ms > 0 and abs(elapsed_ms-total) < 1e-6,
                'HELPER_TIME_DOES_NOT_CLOSE')
    if mode == 'off':
        require(driver['clock_reads'] == driver['driver_begin_ns'] == driver['driver_end_ns'] == 0,
                'OFF_MODE_READS_CLOCK')
        return {'complete_ms': total, 'split_available': False}
    b, c = driver['driver_begin_ns'], driver['driver_end_ns']
    require(type(b) is int and type(c) is int and driver['clock_reads'] == 2 and a <= b < c <= z,
            'BOUNDARY_ORDER_OR_CLOCK_COUNT')
    return {'complete_ms': total, 'before_driver_ms': (b-a)/1e6,
            'driver_run_and_wait_ms': (c-b)/1e6, 'after_driver_ms': (z-c)/1e6,
            'split_available': True, 'pure_device_compute_time': False}


def sealed_sources():
    bindings = {}
    for root in (LOCAL, BOARD):
        manifest = root / 'completion_manifest.json'
        bindings[str(manifest)] = sha(manifest)
        for name, digest in read(manifest)['artifact_sha256'].items():
            bindings[str(root/name)] = digest
    # Original source/compiler/helper/runtime identities; no recompilation of the target module.
    bindings.update(read(LOCAL/'freeze.json')['bindings'])
    for target in ('sim', 'arm'):
        folder = OFFICIAL / ('runtime_local_final' if target == 'sim' else 'runtime_arm')
        record = read(folder/'build.json')
        # The old build record predates the generator already adopted by run04.
        # We compile the sealed generated C++, never rerun that old generator.
        # All other original build inputs must still match; the active launcher
        # remains bound to run04's accepted source identity.
        historical_inputs = dict(record['source_bindings'])
        generator = str(HERE/'c3_official_window_runtime.py')
        historical_inputs.pop(generator)
        require(sha(generator) == read(LOCAL/'freeze.json')['bindings'][generator],
                'RUN04_ADOPTED_BUILD_LAUNCHER_CHANGED')
        bindings.update(historical_inputs)
        bindings[str(folder/'c3_official_window_runtime_generated.cc')] = record['runtime_overlay_sha256']
        bindings[str(folder/'c3_official_window_runtime_driver_generated.cc')] = record['driver_overlay_sha256']
        bindings[str(folder/'build.json')] = sha(folder/'build.json')
        for name in ('c3_official_window_runtime_collection.h', 'c3_official_window_runtime_uop_queue.h'):
            bindings[str(folder/name)] = record['source_bindings'][str(HERE/name)]
    for p in (Path(__file__), HEADER, HERE/'test_c3_tps_cost_probe.py'):
        bindings[str(p)] = sha(p)
    verify(bindings)
    return bindings


def build(root, target):
    from c3_official_window_runtime import run_compiler
    old = OFFICIAL / ('runtime_local_final' if target == 'sim' else 'runtime_arm')
    out = root / ('runtime_' + target)
    out.mkdir()
    original = old/'c3_official_window_runtime_generated.cc'
    (out/original.name).write_text(instrument_runtime(original.read_text()))
    for name in ('c3_official_window_runtime_driver_generated.cc',
                 'c3_official_window_runtime_collection.h', 'c3_official_window_runtime_uop_queue.h'):
        shutil.copyfile(old/name, out/name)
    shutil.copyfile(HEADER, out/HEADER.name)
    record = read(old/'build.json')
    command = [s.replace(str(old), str(out)) for s in record['command']]
    result = run_compiler(command, target)
    (out/'build.log').write_text(result.stdout+result.stderr)
    write(out/'build.json', {'returncode': result.returncode, 'command': command,
                            'execution_source_equal_after_removing_observation': True})
    require(result.returncode == 0, 'RUNTIME_BUILD_FAILED')
    old_helper = OFFICIAL/'helper'/('local_final02' if target == 'sim' else 'arm_attempt02')
    hrec = read(old_helper/'build.json')
    source = Path(hrec['source'])
    require(sha(source) == hrec['source_sha256'], 'HELPER_SOURCE_CHANGED')
    hdir = root/('helper_'+target); hdir.mkdir()
    new_source = hdir/source.name
    new_source.write_text(instrument_helper(source.read_text()))
    command = [str(new_source) if s == str(source) else s.replace(str(old_helper), str(hdir))
               for s in hrec['command']]
    result = run_compiler(command, target)
    (hdir/'build.log').write_text(result.stdout+result.stderr)
    write(hdir/'build.json', {'returncode': result.returncode, 'command': command,
                             'execution_source_equal_after_removing_observation': True})
    require(result.returncode == 0, 'HELPER_BUILD_FAILED')


def prepare(root):
    require(not root.exists(), 'ATTEMPT_EXISTS_NO_OVERWRITE')
    bindings = sealed_sources()
    root.mkdir()
    write(root/'source_adoption.json', {'bindings': bindings, 'new_target_calls': 0})
    generator = HERE/'c3_official_window_runtime.py'
    write(root/'generator_provenance.json', {
        'old_build_record_sha': read(OFFICIAL/'runtime_local_final/build.json')['source_bindings'][str(generator)],
        'run04_adopted_current_sha': sha(generator),
        'historical_generator_reexecuted': False,
        'runtime_input': 'sealed generated C++; observation removal restores exact original text',
        'initial_tool_precheck': 'rejected old generator SHA against current file; no research attempt or target call existed',
        'resolution': 'retain old provenance; verify actual compile inputs and run04-adopted launcher separately'})
    write(root/'contract.json', {
        'program': 'R only; unchanged sealed module', 'local_complete_calls': 28,
        'units': UNITS, 'calls_per_unit': 7, 'tests': 'three seeds, W0-W1-W0, directed',
        'clock': 'std::chrono::steady_clock, nanoseconds, common host clock domain',
        'driver_inner_reads_on': 2, 'driver_inner_reads_off': 0,
        'detailed_profile_in_minimal': False, 'per_instruction_observation_in_minimal': False,
        'new_target_compilation': False, 'performance_evidence_from_FSim': False,
        'worker_limit_seconds': 600, 'worker_limit_bytes': 8*1024**3,
        'stop_on_first_failure': True, 'no_retry': True,
        'stage_end': 'local qualification; board environment must be restored and rechecked separately'})
    try:
        for target in ('sim', 'arm'):
            build(root, target)
        verify(bindings)
        for mode, collection in UNITS:
            unit = root / (mode+'_'+collection); unit.mkdir()
            (unit/'prepare/R').mkdir(parents=True)
            shutil.copyfile(LOCAL/'prepare/R/descriptor.json', unit/'prepare/R/descriptor.json')
            write(unit/'prepare/result.json', {'programs': {'R': read(LOCAL/'prepare/R/descriptor.json')}})
            shutil.copyfile(LOCAL/'hardware_archive.json', unit/'hardware_archive.json')
            from c3_tps_sync_release_local import ledger
            write(unit/'call_ledger.json', [r for r in ledger() if r['program']=='R' and r['mode']==collection])
            ub = {**bindings, **{str(p):sha(p) for p in root.rglob('*') if p.is_file()}}
            write(unit/'freeze.json', {'bindings': ub})
        bindings.update({str(p): sha(p) for p in root.rglob('*') if p.is_file()})
        write(root/'freeze.json', {'bindings': bindings})
        write(root/'prepare_complete.json', {'status':'PREPARED', 'new_target_calls':0})
    except BaseException as error:
        write(root/'stop.json', {'phase':'prepare', 'error':str(error), 'new_target_calls':0})
        raise


def worker(root, mode, collection):
    # Reuse the qualified worker unchanged; observation callbacks run only after each invocation.
    import c3_tps_sync_release_local as previous
    previous.RUNTIME = root/'runtime_sim'
    previous.HELPER = root/'helper_sim/c3_official_window_runtime_helper.so'
    unit = root/(mode+'_'+collection)
    os.environ['C3_TPS_COST_MODE'] = mode
    original_capture = previous.capture_helper_call
    def capture(index, invoke, counts, metadata, submissions, persist):
        value, execution = original_capture(index, invoke, counts, metadata, submissions, persist)
        import tvm
        h = json.loads(tvm.get_global_func('c3.tps_cost.helper_boundary')())
        d = json.loads(tvm.get_global_func('vta.runtime.tps_cost_boundary')())
        split = split_boundary(h, d, mode, index)
        write(unit/f'boundary_{index:02d}.json', {'helper':h,'driver':d,'split':split,
                                                'local_duration_not_performance':True})
        return value, execution
    previous.capture_helper_call = capture
    previous.qualification_worker(unit, 'R', collection)


def qualify(root):
    from c3_official_window_workflow import _tree_rss
    require((root/'prepare_complete.json').exists() and not (root/'stop.json').exists(), 'NOT_PREPARED')
    require(not list(root.glob('*_worker.json')), 'QUALIFICATION_ALREADY_STARTED')
    bindings = read(root/'freeze.json')['bindings']; verify(bindings)
    env = dict(os.environ, PYTHONPATH=':'.join(str(REPO/p) for p in ('python','vta/python','vta/tutorials/frontend')),
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', TVM_NUM_THREADS='1')
    try:
        for mode, collection in UNITS:
            name = mode+'_'+collection
            argv = [str(PYTHON), str(Path(__file__).resolve()), '_worker', '--output',str(root),
                    '--mode',mode,'--collection',collection]
            started, peak, limit = time.monotonic(), 0, None
            with (root/(name+'.log')).open('x') as log:
                proc = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    while proc.poll() is None:
                        peak=max(peak,_tree_rss(proc.pid))
                        if peak>8*1024**3 or time.monotonic()-started>600:
                            limit='WORKER_RESOURCE_LIMIT';os.killpg(proc.pid,9);break
                        time.sleep(.2)
                    code=proc.wait()
                finally:
                    if proc.poll() is None: os.killpg(proc.pid,9);proc.wait()
            write(root/(name+'_worker.json'), {'returncode':code,'limit':limit,'peak_tree_rss':peak,
                  'elapsed_seconds_resource_control_only':time.monotonic()-started})
            require(code==0 and limit is None, 'QUALIFICATION_WORKER_FAILED:'+name)
            verify(bindings)
        compare(root)
        write(root/'qualification_complete.json', {'status':'FOUR_MODES_R_QUALIFIED','new_FSim_calls':28,
                                                  'new_FPGA_calls':0,'performance_conclusion':False})
    except BaseException as error:
        write(root/'stop.json', {'phase':'qualify','error':str(error),'retry_allowed':False})
        raise


def compare(root):
    from c3_five_tradeoff_commands import decode_submissions
    from c3_tps_notification_probe import canonical_payload
    comparisons=[]
    for index in range(1,8):
        paths=[LOCAL/'R_minimal'/f'native_call_{index:02d}.txt']+[
            root/(m+'_'+c)/('R_'+c)/f'native_call_{index:02d}.txt' for m,c in UNITS]
        payloads=[]
        for p in paths:
            blocks=decode_submissions(p.read_text());require(len(blocks)==1,'UNEXPECTED_MULTI_SUBMIT')
            payloads.append([(r['stage'],r['bits'],canonical_payload(r)) for r in blocks[0]])
        require(all(p==payloads[0] for p in payloads[1:]),'OBSERVATION_CHANGED_EXECUTION')
        comparisons.append({'case':index,'five_paths_identical':True})
    write(root/'execution_equivalence.json', {'status':'SEALED_R_AND_FOUR_MODES_EQUAL','cases':comparisons})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('prepare','qualify','_worker'))
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--mode',choices=('off','on'))
    parser.add_argument('--collection',choices=('detailed','minimal'))
    args=parser.parse_args();root=args.output.resolve()
    if args.phase=='_worker':worker(root,args.mode,args.collection)
    else:globals()[args.phase](root)


if __name__=='__main__':main()
