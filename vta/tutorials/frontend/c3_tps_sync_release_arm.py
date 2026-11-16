"""ARM-only preparation for the sealed run04 R/S comparison; no target invocation."""
import argparse
import os
from pathlib import Path
import shlex
import shutil
import signal
import struct
import subprocess
import time
from c3_tps_aggregation_prepare import (HERE, REPO, HOLD, OFFICIAL, STORAGE, SDK, PYTHON,
    sha, read, write, require, bound_file, verify_bindings, common_runtime_bindings,
    compare_initializer_bindings)
LOCAL = HOLD/'20261016_c3_tps_sync_release_local_run04'
VARIANTS = ('R', 'S')


def adopt():
    seal = read(LOCAL/'completion_manifest.json')
    require(seal['status'] == 'R_S_LOCAL_QUALIFIED', 'LOCAL_NOT_QUALIFIED')
    bindings = {str(LOCAL/k): v for k,v in seal['artifact_sha256'].items()}
    bindings[str(LOCAL/'completion_manifest.json')] = sha(LOCAL/'completion_manifest.json')
    bindings.update(read(LOCAL/'freeze.json')['bindings'])
    verify_bindings(bindings)
    stage = read(LOCAL/'stage_result.json')
    account = stage['call_accounting']
    require(account['fully_certified'] == 28 and account['completed_calls']['exact'] == 28
            and account['started_calls']['exact'] == 28, 'LOCAL_CALL_ACCOUNTING')
    for program in VARIANTS:
        for mode in ('detailed', 'minimal'):
            calls = read(LOCAL/f'{program}_{mode}/summary.json')['calls']
            require(len(calls) == 7 and all(c['correct'] and c['status']=='PASS' for c in calls),
                    'LOCAL_UNIT_NOT_QUALIFIED')
    common = read(OFFICIAL/'runtime_contract.json')
    bindings.update(common_runtime_bindings(common))
    return {'status': 'RUN04_R_S_28_CALL_QUALIFICATION_ADOPTED',
            'source_bindings': bindings, 'source_attempt': str(LOCAL),
            'adopted_FSim_calls':28, 'new_FSim_calls':0,
            'programs': {p: {'detailed':str(LOCAL/('prepare/R' if p=='R' else 'S_program'))}
                         for p in VARIANTS},
            'hardware_archive':bound_file(LOCAL/'hardware_archive.json')}

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
    # Exact source LLVM is bound by run04 manifest; S has no separate ALU contract file.
    if (source / 'alu_initializer_contract.json').exists():
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



def prepare_artifacts(output):
    output.mkdir(parents=True, exist_ok=False)
    adoption = adopt()
    write(output/'adoption.json', adoption)
    common = read(OFFICIAL/'runtime_contract.json')
    arm_runtime = common['runtime'][1]
    storage_build = read(STORAGE/'arm_build.json')
    require(storage_build['returncode']==0, 'STORAGE_HELPER_BUILD_FAILED')
    hardware = read(LOCAL/'hardware_archive.json')
    verify_bindings(hardware['source_bindings'])
    bit = hardware['bitstreams'][0]
    programs = {}
    for p in VARIANTS:
        arm = output/p/'arm'
        arm.parent.mkdir()
        _launch_arm(arm, p, output/'adoption.json')
        contract = read(arm/'arm_contract.json')
        source = Path(adoption['programs'][p]['detailed'])
        descriptor = read(source/'descriptor.json')
        require(contract['symbol']==descriptor['symbol'] and contract['abi']==descriptor['abi'],
                'ARM_ABI_OR_SYMBOL_CHANGED')
        paths = {'final_tir':source/'final.json', 'fsim_module':source/'fsim_module.so',
                 'fsim_llvm':source/'module.ll', 'arm_module':arm/'module.so',
                 'arm_llvm':arm/'module.ll', 'arm_contract':arm/'arm_contract.json',
                 'reference_contract':LOCAL/'prepare/R/reference_contract.json'}
        programs[p] = {'variant':p, 'symbol':contract['symbol'], 'abi':contract['abi'],
            'qualified_directory':str(source), 'local_calls':str(LOCAL/f'{p}_detailed/summary.json'),
            **{k:str(v) for k,v in paths.items()}, 'sha256':{k:sha(v) for k,v in paths.items()}}
    result = {'schema':'c3_tps_sync_release_prepared_v1', 'status':'R_S_ARM_PREPARED_FPGA_PENDING',
        'programs':programs, 'adoption':str(output/'adoption.json'),
        'runtime':bound_file(arm_runtime['library'],arm_runtime['binary_sha256']),
        'helper':bound_file(common['helper']['arm']['binary'],common['helper']['arm']['binary_sha256']),
        'storage_helper':bound_file(STORAGE/'storage_arm.so',storage_build['binary_sha256']),
        'runtime_tvm':bound_file(REPO/'build_axu_aarch64/libtvm_runtime.so'),
        'rpc_server':bound_file(REPO/'build_axu_aarch64/tvm_rpc'),
        'bitstream':bound_file(bit['path'],bit['whole_sha256']),
        'hardware_archive':adoption['hardware_archive'], 'source_bindings':adoption['source_bindings'],
        'adopted_FSim_calls':28, 'new_FSim_calls':0, 'board_contacted':False}
    verify_bindings(adoption['source_bindings'])
    write(output/'prepared.json',result)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--arm-worker',choices=VARIANTS)
    parser.add_argument('--adoption',type=Path)
    args=parser.parse_args()
    if args.arm_worker:
        arm_worker(args.output.resolve(),args.arm_worker,args.adoption.resolve())
    else:
        prepare_artifacts(args.output.resolve())
