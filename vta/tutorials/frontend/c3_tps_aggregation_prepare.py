#!/usr/bin/env python3
"""Adopt the sealed TPS-only local qualification and cross-build its three modules.

This module never contacts a board or calls a fused target function. The failed
TPS/official attempt remains failed; only its six already successful TPS workers
are adopted, without changing their sources, IR, runtime, or helper.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import struct
import subprocess
import time
import traceback

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
HOLD = HERE / 'report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout'
QUALIFIED = HOLD / '20261014_c3_tps_window_complete_run03'
PAUSE = HOLD / '20261014_c3_tps_window_pause_checkpoint01'
OFFICIAL = HOLD / '20261014_c3_official_valid_window_run01'
STORAGE = HOLD / '20261014_c3_tps_window_storage_helper_run01'
SDK = Path('/home/orange/alinx_plsdk/install/environment-setup-aarch64-xilinx-linux')
PYTHON = Path('/home/orange/miniconda3/envs/vta-resnet/bin/python')
VARIANTS = ('B', 'N', 'G')
PAUSE_MANIFEST_SHA = '93af7b7459703ed7bef012f4230fe39b9406ef03662ed451e5e5cf6aa43ad3c2'
PAUSE_CHECKPOINT_SHA = 'aa00b4ebdc5aa16e7bfedba296e8cdc3556b0183151dee14acd0bc639868db24'


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def bound_file(path, expected=None):
    path = Path(path).resolve()
    digest = sha(path)
    require(expected is None or digest == expected, 'FROZEN_FILE_CHANGED:' + str(path))
    return {'path': str(path), 'sha256': digest}


def verify_bindings(bindings):
    for path, digest in bindings.items():
        bound_file(path, digest)


def common_runtime_bindings(common):
    """Retain the already documented builder-only revision, never execution edits."""
    result = {}
    seal = read(OFFICIAL / 'completion_manifest.json')
    paths = [OFFICIAL / 'runtime_contract.json']
    for runtime in common['runtime']:
        build_path = Path(runtime['directory']) / 'build.json'
        paths.append(build_path)
        build = read(build_path)
        for path, digest in build['source_bindings'].items():
            current = sha(path)
            if current != digest:
                revision = runtime.get('post_build_generator_changes', {}).get(path, {})
                require(path == str(HERE / 'c3_official_window_runtime.py') and
                        revision.get('at_compile_sha256') == digest and
                        revision.get('current_sha256') == current and
                        runtime['current_generator_reproduces_compiled_sources'] is True,
                        'RUNTIME_BUILD_SOURCE_CHANGED:' + path)
            result[path] = current
    for path in paths:
        bound_file(path, seal['artifact_sha256'][str(path.relative_to(OFFICIAL))])
        result[str(path)] = sha(path)
    return result


def compare_initializer_bindings(sim_llvm, arm_llvm, sim_contract, arm_contract):
    """Compare corresponding TIR scopes, never an unordered body/hash multiset."""
    from c3_valid_window_cross import initializers
    sim_bodies = {row['initializer']: row for row in initializers(sim_llvm)}
    arm_bodies = {row['initializer']: row for row in initializers(arm_llvm)}
    sim = sim_contract['bindings']
    arm = arm_contract['bindings']
    require(len(sim) == len(arm) > 0, 'ARM_FSIM_SCOPE_CENSUS')
    rows = []
    fields = ('source_scope_ordinal', 'dispatch', 'signature_bytes',
              'dispatch_call_sites', 'mode_reset_opcode_use_imm_imm')
    for index, (left, right) in enumerate(zip(sim, arm)):
        require(left['source_scope_ordinal'] == right['source_scope_ordinal'] == index,
                'ARM_FSIM_SCOPE_ORDER')
        require(all(left[key] == right[key] for key in fields),
                'ARM_FSIM_SCOPE_CACHE_CONTRACT:' + str(index))
        require(left['initializer'] in sim_bodies and right['initializer'] in arm_bodies,
                'ARM_FSIM_INITIALIZER_MISSING:' + str(index))
        first = sim_bodies[left['initializer']]
        second = arm_bodies[right['initializer']]
        require(first['body'] == second['body'], 'ARM_FSIM_BOUND_INITIALIZER_BODY:' + str(index))
        rows.append({'source_scope_ordinal': index, 'fsim_handle': left['handle'],
                     'arm_handle': right['handle'], 'fsim_initializer': left['initializer'],
                     'arm_initializer': right['initializer'], 'signature_bytes': left['signature_bytes'],
                     'dispatch': left['dispatch'], 'dispatch_call_sites': left['dispatch_call_sites'],
                     'canonical_body_sha256': first['sha256'], 'canonical_body_equal': True})
    # Equal bodies do not permit changing which source scopes share an identity.
    for i in range(len(rows)):
        for j in range(len(rows)):
            for field in ('handle', 'initializer'):
                require((sim[i][field] == sim[j][field]) == (arm[i][field] == arm[j][field]),
                        'ARM_FSIM_CACHE_SHARING_RELATION:' + field)
    require(set(sim_bodies) == {row['initializer'] for row in sim} and
            set(arm_bodies) == {row['initializer'] for row in arm}, 'UNBOUND_INITIALIZER_BODY')
    return {'status': 'PER_TIR_SCOPE_INITIALIZER_AND_CACHE_SHARING_EQUAL', 'bindings': rows,
            'canonicalization': 'local SSA/labels and metadata reference IDs/comments only; integer operations, operands, calls and CFG retained',
            'unordered_multiset_used_as_proof': False}


def _verify_worker(directory, variant, mode, runtime, helper):
    summary = read(directory / 'summary.json')
    require(summary['status'] == 'COMPLETE_WINDOW_LOCAL_LOGICAL_NATIVE_AND_FSIM_PASSED' and
            summary['object_id'] == 'TPS' and summary['variant'] == variant and
            summary['collection'] == mode and summary['local_complete_calls'] == 7 and
            summary['hardware_pipeline_qualified'] is True and summary['board_qualified'] is False,
            'TPS_WORKER_NOT_QUALIFIED:' + str(directory))
    require(summary['runtime_sha256'] == runtime['sha256'] and
            summary['helper_sha256'] == helper['sha256'] and
            summary['module_sha256'] == sha(directory / 'fsim_module.so'), 'TPS_WORKER_BINARY_BINDING')
    worker = read(directory.with_name(directory.name + '_worker.json'))
    require(worker['status'] == 'PASS' and worker['returncode'] == 0, 'TPS_WORKER_FAILED')
    logical = read(directory / 'logical_certificate.json')
    require(logical['status'] == 'LOCAL_LOGICAL_GATES_PASSED' and
            logical['output_products_and_tail_exactly_equal'] is True and
            logical['tensor_requests_exactly_equal'] is True and
            logical['acc_spacing']['status'] == 'ACC_DISTANCE_PASSED', 'TPS_LOGICAL_CERTIFICATE')
    require(variant != 'N' or logical['N_kernel_and_cache_identity_equal'] is True,
            'N_CACHE_IDENTITY_CHANGED')
    calls = read(directory / 'calls.json')
    require(len(calls) == 7, 'TPS_CALL_CENSUS')
    for index, call in enumerate(calls, 1):
        require(call == read(directory / f'call_{index:02d}_end.json'), 'TPS_CALL_END_DIFFERS')
        require(call['call_index'] == index and call['correct'] is True and
                call['target_calls'] == 1 and call['performance_sample'] is False and
                call['board_contacted'] is False and call['output_sha256'] == call['reference_sha256'],
                'TPS_CALL_QUALIFICATION_FAILED')
        meta = call['helper_metadata']
        require(meta['started_calls'] == meta['completed_calls'] == index and
                meta['packed_calls_per_invocation'] == 1 and meta['adds_synchronize'] is False and
                meta['parameter_count'] == 4, 'TPS_HELPER_CALL_CONTRACT')
        require(call['buffer_handles'] == calls[0]['buffer_handles'] and
                call['actual_storage'] == calls[0]['actual_storage'] and
                call['same_weight_storage'] == calls[0]['same_weight_storage'], 'TPS_STORAGE_CHANGED')
        dump = bound_file(directory / f'native_call_{index:02d}.txt', call['native_dump_sha256'])
        native = read(directory / f'native_certificate_{index:02d}.json')
        require(native['dump_sha256'] == dump['sha256'] and
                native['status'] == 'ACTUAL_BOUNDED_NATIVE_FIFO_TOKEN_UOP_WITH_NOP_BARRIERS_CERTIFIED',
                'TPS_NATIVE_CERTIFICATE')
        ledger = call['submit_ledger']
        require(ledger['mode'] == mode and ledger['overflow'] is False and
                ledger['attempted_submits'] == ledger['recorded_submits'] == native['submission_count'],
                'TPS_SUBMIT_CENSUS')
        require(all(row['completion'] is True and row['timeout'] == 0 for row in ledger['submits']),
                'TPS_SUBMIT_NOT_COMPLETED')
        require(all(row['status'] == 'FINITE_FIFO_PROGRESS_PROVED'
                    for row in read(directory / f'progress_{index:02d}.json')), 'TPS_FINITE_PROGRESS')
    require([row['seed'] for row in calls[:3]] == [0, 20250901, 20260910] and
            all(row['kind'] == 'three_seed' for row in calls[:3]), 'TPS_SEED_QUALIFICATION')
    first, changed, last = calls[3:6]
    require(first['input_sha256'] == last['input_sha256'] and
            first['input_sha256'][0] == changed['input_sha256'][0] and
            first['input_sha256'][2:] == changed['input_sha256'][2:] and
            first['input_sha256'][1] != changed['input_sha256'][1] and
            first['output_sha256'] == last['output_sha256'] != changed['output_sha256'],
            'TPS_WEIGHT_RETURN_CHECK')
    require(calls[6]['kind'] == 'directed' and
            calls[6]['output_distribution']['nonzero_unsaturated'] > 0, 'TPS_DIRECTED_INPUT')
    return summary


def adopt_qualification():
    """Read-only adoption; no rebuilding or fused-function invocation."""
    import tvm
    import vta  # Register the frozen VTA intrinsics before loading saved TIR.
    from c3_official_window_baseline import normalize_llvm
    from c3_tps_window_hardware import audit_archive
    bound_file(PAUSE / 'artifact_manifest.json', PAUSE_MANIFEST_SHA)
    bound_file(PAUSE / 'checkpoint.json', PAUSE_CHECKPOINT_SHA)
    seal = read(PAUSE / 'artifact_manifest.json')
    require(Path(seal['base_directory']).resolve() == HOLD.resolve(), 'PAUSE_SEAL_ROOT')
    sealed = {str(HOLD / name): digest for name, digest in seal['artifact_sha256'].items()}
    verify_bindings(sealed)
    verify_bindings(seal['source_sha256'])
    identity = read(QUALIFIED / 'identity.json')
    require(len(identity['source_bindings']) == 54, 'QUALIFICATION_SOURCE_CENSUS_CHANGED')
    verify_bindings(identity['source_bindings'])
    stage = read(QUALIFIED / 'stage_result.json')
    require(stage['status'] == 'STOP_LOCAL_GATE_FAILED', 'HISTORICAL_ATTEMPT_STATUS_CHANGED')
    failure = read(QUALIFIED / 'failure_diagnosis.json')
    require(failure['object_id'] == 'official' and failure['variant'] == 'G' and
            failure['target_calls'] == 0, 'HISTORICAL_FAILURE_NOT_SEPARATE_FROM_TPS')
    hardware = read(QUALIFIED / 'hardware_archive.json')
    require(hardware == audit_archive(REPO), 'HARDWARE_ARCHIVE_OR_FEEDBACK_CHANGED')
    common = read(OFFICIAL / 'runtime_contract.json')
    runtime_bindings = common_runtime_bindings(common)
    local_runtime = common['runtime'][0]
    runtime = bound_file(local_runtime['library'], local_runtime['binary_sha256'])
    helper = bound_file(common['helper']['local']['binary'], common['helper']['local']['binary_sha256'])
    adopted = {}
    for variant in VARIANTS:
        detailed = QUALIFIED / f'TPS_{variant}_detailed'
        minimal = QUALIFIED / f'TPS_{variant}_minimal'
        a = _verify_worker(detailed, variant, 'detailed', runtime, helper)
        b = _verify_worker(minimal, variant, 'minimal', runtime, helper)
        require(a['native_submissions'] == b['native_submissions'] and
                a['native_uop_values'] == b['native_uop_values'], 'TPS_MODES_CHANGE_NATIVE_EXECUTION')
        require(normalize_llvm((detailed / 'module.ll').read_text()) ==
                normalize_llvm((minimal / 'module.ll').read_text()), 'TPS_MODES_CHANGE_LLVM')
        require(tvm.ir.structural_equal(tvm.ir.load_json((detailed / 'final.json').read_text()),
                                       tvm.ir.load_json((minimal / 'final.json').read_text()),
                                       map_free_vars=False), 'TPS_MODES_CHANGE_FINAL_IR')
        adopted[variant] = {'detailed': str(detailed), 'minimal': str(minimal),
                            'adopted_FSim_calls': 14,
                            'summary_files': [bound_file(detailed / 'summary.json'),
                                              bound_file(minimal / 'summary.json')]}
    return {'schema': 'c3_tps_aggregation_adoption_v1', 'status': 'TPS_SUBSET_LOCAL_QUALIFICATION_ADOPTED',
            'source_attempt': str(QUALIFIED), 'source_attempt_status': stage['status'],
            'source_attempt_status_preserved': True, 'pause_manifest': bound_file(PAUSE / 'artifact_manifest.json'),
            'pause_checkpoint': bound_file(PAUSE / 'checkpoint.json'),
            'verified_sealed_artifacts': len(sealed), 'verified_original_source_bindings': 54,
            'source_bindings': {**identity['source_bindings'], **runtime_bindings},
            'hardware_archive': bound_file(QUALIFIED / 'hardware_archive.json'),
            'programs': adopted, 'adopted_FSim_calls': 42, 'new_FSim_calls': 0,
            'board_contacted': False, 'FPGA_qualified': False, 'performance_conclusion': False}


def arm_worker(output, variant, adoption_path):
    """Cross-build only; do not load or execute the generated operator module."""
    import tvm
    import vta
    from tvm.contrib import cc
    from c3_official_window_baseline import buffer_abi
    from c3_official_window_llvm import llvm_alu_contract
    from build_vta_yolov3_tiny_relay_residency_dispatch_pair import cross_compiler
    output.mkdir(parents=True, exist_ok=False)
    adoption = read(adoption_path)
    verify_bindings(adoption['source_bindings'])
    source = Path(adoption['programs'][variant]['detailed'])
    final = tvm.ir.load_json((source / 'final.json').read_text())
    require(isinstance(final, tvm.IRModule) and len(final.functions) == 1, 'FINAL_NOT_SINGLE_FUNCTION_MODULE')
    global_var = list(final.functions)[0]
    function = final[global_var]
    symbol = str(function.attrs['global_symbol'])
    abi = buffer_abi(function)
    require(len(abi) == 4, 'TPS_ABI_NOT_FOUR_TENSORS')
    (output / 'final.json').write_text(tvm.ir.save_json(final))
    fsim_llvm = (source / 'module.ll').read_text()
    sim_contract = llvm_alu_contract(function, fsim_llvm)
    require(sim_contract == read(source / 'alu_initializer_contract.json'), 'SEALED_FSIM_LLVM_CONTRACT_CHANGED')
    cfg = dict(vta.get_env().cfg_dict, TARGET='axu5evb')
    frozen_cfg = read(REPO / '3rdparty/vta-hw/config/vta_config.json')
    require(all(cfg.get(key) == value for key, value in frozen_cfg.items()), 'VTA_STRUCTURE_CONFIGURATION_CHANGED')
    with vta.Environment(cfg):
        target = tvm.target.Target(vta.get_env().target, host=vta.get_env().target_host)
        with tvm.transform.PassContext(opt_level=3, disabled_pass={'AlterOpLayout', 'tir.CommonSubexprElimTIR'}):
            lib = tvm.build(final, target=target)
        llvm = lib.get_source('ll')
        require('target triple = "aarch64-linux-gnu"' in llvm, 'ARM_TARGET_TRIPLE')
        (output / 'module.ll').write_text(llvm)
        arm_contract = llvm_alu_contract(function, llvm)
        comparison = compare_initializer_bindings(fsim_llvm, llvm, sim_contract, arm_contract)
        records = []
        original_compile = cc._linux_compile
        def record_compile(destination, objects, options, compile_cmd, compile_shared=False):
            inputs = [objects] if isinstance(objects, str) else list(objects)
            retained = []
            folder = output / 'link_inputs'
            folder.mkdir(exist_ok=True)
            for index, name in enumerate(inputs):
                copied = folder / (str(index) + '_' + Path(name).name)
                shutil.copyfile(name, copied)
                retained.append(bound_file(copied))
            records.append({'compiler': compile_cmd, 'destination': destination,
                            'options': options, 'compile_shared': compile_shared, 'inputs': retained})
            return original_compile(destination, objects, options, compile_cmd, compile_shared)
        cc._linux_compile = record_compile
        try:
            lib.export_library(str(output / 'module.so'), fcompile=cross_compiler())
        finally:
            cc._linux_compile = original_compile
        write(output / 'compiler_invocations.json', records)
    blob = (output / 'module.so').read_bytes()
    require(blob[:4] == b'\x7fELF' and blob[4:6] == bytes((2, 1)) and
            struct.unpack_from('<H', blob, 18)[0] == 183, 'ARM_ELF_MACHINE')
    compiler_argv = shlex.split(os.environ['CXX'])
    compiler = Path(shutil.which(compiler_argv[0])).resolve()
    version = subprocess.check_output([str(compiler), '--version'], text=True)
    readelf = shutil.which(os.environ.get('READELF', 'aarch64-xilinx-linux-readelf'))
    symbols = subprocess.check_output([readelf, '--dyn-syms', '--wide', str(output / 'module.so')], text=True)
    require(symbol in symbols, 'TPS_SYMBOL_NOT_EXPORTED')
    (output / 'dynamic_symbols.txt').write_text(symbols)
    verify_bindings(adoption['source_bindings'])
    result = {'schema': 'c3_tps_aggregation_arm_contract_v1',
              'status': 'ARM_MATCHES_BOUND_FSIM_INITIALIZERS_FPGA_PENDING', 'variant': variant,
              'symbol': symbol, 'abi': abi, 'source_directory': str(source),
              'source_final_tir': bound_file(source / 'final.json'), 'saved_final_tir': bound_file(output / 'final.json'),
              'final_tir_structurally_unchanged': tvm.ir.structural_equal(final,
                  tvm.ir.load_json((output / 'final.json').read_text()), map_free_vars=False),
              'fsim_llvm': bound_file(source / 'module.ll'), 'arm_llvm': bound_file(output / 'module.ll'),
              'arm_module': bound_file(output / 'module.so'), 'fsim_cache_contract': sim_contract,
              'arm_cache_contract': arm_contract, 'initializer_comparison': comparison,
              'hardware_cfg': cfg, 'target': str(target), 'opt_level': 3,
              'disabled_pass': ['AlterOpLayout', 'tir.CommonSubexprElimTIR'],
              'sdk': bound_file(SDK), 'compiler': bound_file(compiler), 'compiler_argv': compiler_argv,
              'compiler_version': version, 'source': bound_file(__file__),
              'new_FSim_calls': 0, 'board_contacted': False, 'FPGA_qualified': False}
    require(result['final_tir_structurally_unchanged'], 'SAVED_ARM_FINAL_IR_CHANGED')
    write(output / 'arm_contract.json', result)


def _launch_arm(output, variant, adoption_path):
    from c3_official_window_workflow import _tree_rss
    argv = [str(PYTHON), str(Path(__file__).resolve()), '--arm-worker', variant,
            '--output', str(output), '--adoption', str(adoption_path)]
    shell = 'set -e\nunset LD_LIBRARY_PATH\nsource ' + shlex.quote(str(SDK)) + '\nexec ' + shlex.join(argv)
    environment = dict(os.environ, PYTHONPATH=':'.join(str(REPO / p) for p in
                       ('python', 'vta/python', 'vta/tutorials/frontend')),
                       OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    started, peak, limit = time.monotonic(), 0, None
    with output.with_suffix('.log').open('x') as stream:
        process = subprocess.Popen(['bash', '-c', shell], env=environment,
                                   stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while process.poll() is None:
                peak = max(peak, _tree_rss(process.pid))
                if peak > 8 * 1024**3:
                    limit = 'ARM_WORKER_MEMORY_LIMIT'
                elif time.monotonic() - started > 600:
                    limit = 'ARM_WORKER_TIME_LIMIT'
                if limit:
                    os.killpg(process.pid, signal.SIGKILL)
                    break
                time.sleep(0.2)
            code = process.wait()
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    write(output.with_name(output.name + '_worker.json'),
          {'variant': variant, 'returncode': code, 'resource_limit': limit,
           'peak_process_tree_bytes': peak, 'seconds': time.monotonic() - started,
           'command': argv, 'sdk': bound_file(SDK), 'board_contacted': False})
    require(code == 0 and limit is None, 'ARM_WORKER_FAILED:' + str(output))


def prepare_artifacts(output: Path) -> dict:
    """Build independent TPS-only preparation artifacts in a fresh directory."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    try:
        script_at_start = sha(__file__)
        adoption = adopt_qualification()
        write(output / 'adoption.json', adoption)
        common = read(OFFICIAL / 'runtime_contract.json')
        arm_runtime = common['runtime'][1]
        runtime = bound_file(arm_runtime['library'], arm_runtime['binary_sha256'])
        helper = bound_file(common['helper']['arm']['binary'], common['helper']['arm']['binary_sha256'])
        storage_build = read(STORAGE / 'arm_build.json')
        require(storage_build['returncode'] == 0, 'STORAGE_HELPER_BUILD_FAILED')
        storage_helper = bound_file(STORAGE / 'storage_arm.so', storage_build['binary_sha256'])
        hardware = read(QUALIFIED / 'hardware_archive.json')
        verify_bindings(hardware['source_bindings'])
        bit = hardware['bitstreams'][0]
        bitstream = bound_file(bit['path'], bit['whole_sha256'])
        programs = {}
        for variant in VARIANTS:
            arm = output / variant / 'arm'
            arm.parent.mkdir()
            _launch_arm(arm, variant, output / 'adoption.json')
            contract = read(arm / 'arm_contract.json')
            qualified = Path(adoption['programs'][variant]['detailed'])
            paths = {'final_tir': qualified / 'final.json', 'fsim_module': qualified / 'fsim_module.so',
                     'fsim_llvm': qualified / 'module.ll', 'arm_module': arm / 'module.so',
                     'arm_llvm': arm / 'module.ll', 'arm_contract': arm / 'arm_contract.json',
                     'reference_contract': qualified / 'reference_contract.json'}
            programs[variant] = {'variant': variant, 'symbol': contract['symbol'], 'abi': contract['abi'],
                                 'qualified_directory': str(qualified),
                                 **{key: str(path) for key, path in paths.items()},
                                 'sha256': {key: sha(path) for key, path in paths.items()}}
        verify_bindings(adoption['source_bindings'])
        require(script_at_start == sha(__file__), 'PREPARATION_SOURCE_CHANGED_DURING_RUN')
        result = {'schema': 'c3_tps_aggregation_prepared_v1', 'status': 'TPS_ARM_ARTIFACTS_PREPARED_FPGA_PENDING',
                  'adoption': str(output / 'adoption.json'), 'programs': programs,
                  'runtime': runtime, 'helper': helper, 'storage_helper': storage_helper,
                  'runtime_tvm': bound_file(REPO / 'build_axu_aarch64/libtvm_runtime.so'),
                  'rpc_server': bound_file(REPO / 'build_axu_aarch64/tvm_rpc'),
                  'bitstream': bitstream, 'hardware_archive': adoption['hardware_archive'],
                  'source_bindings': {**adoption['source_bindings'], str(Path(__file__).resolve()): script_at_start,
                                      str(HERE / 'c3_valid_window_cross.py'): sha(HERE / 'c3_valid_window_cross.py')},
                  'adopted_FSim_calls': 42, 'new_FSim_calls': 0, 'board_contacted': False}
        write(output / 'prepared.json', result)
        return result
    except Exception as error:
        write(output / 'failure.json', {'status': 'TPS_PREPARATION_FAILED', 'error': str(error),
                                        'traceback': traceback.format_exc(), 'board_contacted': False,
                                        'new_FSim_calls': 0})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--arm-worker', choices=VARIANTS)
    parser.add_argument('--adoption', type=Path)
    args = parser.parse_args()
    if args.arm_worker:
        require(args.adoption is not None, 'ARM_WORKER_ADOPTION_REQUIRED')
        arm_worker(args.output.resolve(), args.arm_worker, args.adoption.resolve())
    else:
        print(json.dumps(prepare_artifacts(args.output), sort_keys=True))
