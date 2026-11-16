"""Build the fixed YOLO224 graph pair with only node34/46 B -> A routing.

Keep the exact original ARM graph library, JSON, parameters, CPU functions and
all other VTA functions. No target call or board entry exists in this module.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

from c3_tps_sync_release_local import HERE, REPO, HOLD, PYTHON, read, write, sha, verify, require, env_for_worker

BASE = HOLD/'20261008_c3_yolo224_incremental_prepare_run03/G2'
OFFICIAL = HOLD/'20261014_c3_official_valid_window_run01'
ARM_COMMON = HOLD/'20261017_c3_yolo416_arm_prepare_run01'
SYMBOLS = {34: 'tvmgen_default_fused_nn_conv2d_add_right_shift_clip_cast_5',
           46: 'tvmgen_default_fused_nn_conv2d_add_right_shift_clip_cast_8'}
ENTRY = 'tvmgen_default_fused_nn_conv2d_add_right_shift_clip_cast'


def experimental_collection(source):
    require(source.count('std::array<Submit, 16> submits{};') == 1, 'LEDGER_SOURCE_CHANGED')
    result = source.replace('std::array<Submit, 16> submits{};',
                            'std::array<Submit, 32> submits{};\n  bool graph_dump{false};')
    return result


def experimental_runtime(source):
    result = source
    for name, count in (('VTA_DEBUG_DUMP_INSN', 1), ('VTA_DEBUG_DUMP_UOP', 2)):
        old = 'if (debug_flag_ & '+name+') {'
        require(result.count(old) == count, 'DUMP_SOURCE_CHANGED:'+name)
        result = result.replace(old, 'if ((debug_flag_ & '+name+') || c3_official_window::State().graph_dump) {')
    result += '\nextern "C" void C3Yolo224SetGraphDump(int enabled) { c3_official_window::State().graph_dump = enabled != 0; }\n'
    return result


def graph_helper(source):
    old = 'debug(handle, 6);  // Dumps, without FORCE_SERIAL.'
    require(source.count(old) == 1 and source.count('debug(handle, 0);') == 1, 'GRAPH_HELPER_CAPTURE_SOURCE')
    source = source.replace(old, 'reinterpret_cast<void (*)(int)>(dlsym(RTLD_DEFAULT, "C3Yolo224SetGraphDump"))(1);')
    return source.replace('debug(handle, 0);',
        'reinterpret_cast<void (*)(int)>(dlsym(RTLD_DEFAULT, "C3Yolo224SetGraphDump"))(0);')


def build_runtime(root, target):
    from c3_official_window_runtime import run_compiler
    old = OFFICIAL/('runtime_local_final' if target == 'sim' else 'runtime_arm')
    out = root/('runtime_'+target)
    out.mkdir()
    for p in old.glob('*.h'): shutil.copyfile(p, out/p.name)
    name = 'c3_official_window_runtime_collection.h'
    (out/name).write_text(experimental_collection((old/name).read_text()))
    name = 'c3_official_window_runtime_generated.cc'
    (out/name).write_text(experimental_runtime((old/name).read_text()))
    name = 'c3_official_window_runtime_driver_generated.cc'
    shutil.copyfile(old/name, out/name)
    command = [value.replace(str(old), str(out)) for value in read(old/'build.json')['command']]
    result = run_compiler(command, target)
    (out/'build.log').write_text(result.stdout+'\n'+result.stderr)
    require(result.returncode == 0, 'COLLECTION_RUNTIME_BUILD:'+target)
    binary = out/('libvta_fsim.so' if target == 'sim' else 'libvta.so')
    write(out/'build.json', {'command': command, 'binary_sha256': sha(binary),
          'source_runtime': str(old), 'execution_policy_changed': False,
          'changes': ['submit observation capacity 16 to 32', 'diagnostic dump independent of original debug setters'],
          'production_runtime_modified': False})


def precheck_sources():
    record = read(BASE/'build.json')
    for filename, field in (('graph.json','graph_sha256'), ('params.bin','params_sha256'), ('graphlib.so','graphlib_sha256')):
        require(sha(BASE/filename) == record[field], 'LEGACY_GRAPH_ARTIFACT_CHANGED:'+filename)
    graph = read(BASE/'graph.json')
    require(len(graph['heads']) == 8, 'EIGHT_RAW_HEADS_REQUIRED')
    for node, symbol in SYMBOLS.items():
        require(graph['nodes'][node]['attrs']['func_name'] == symbol and
                sum(n.get('attrs',{}).get('func_name') == symbol for n in graph['nodes']) == 1,
                'EXACT_TWO_UNIQUE_GRAPH_CALLS_REQUIRED')
    return graph


def prepare(root):
    import tvm, vta
    from tvm.contrib import cc
    from c3_official_window_baseline import buffer_abi, normalize_llvm, is_vta_mac
    from c3_yolo416_graph_cpu_reference import cpu_instruction_gate
    from c3_yolo416_graph_host_reference import raw_pointer_gate
    from c3_yolo224_sync_structure import project
    from c3_valid_window import certify_trace
    from c3_official_window_reference import from_workload, postprocess_from_output_terms, semantic_gates
    from c3_tps_sync_release_prepare import _hardware
    root.mkdir(exist_ok=False)
    try:
        graph = precheck_sources()
        with (root/'preflight.log').open('x') as log:
            result = subprocess.run([str(PYTHON), '-m', 'unittest', 'test_c3_yolo224_two_node_prepare', '-v'],
                                    env=env_for_worker(), stdout=log, stderr=subprocess.STDOUT)
        require(result.returncode == 0, 'ZERO_TARGET_TOOL_PREFLIGHT')
        bindings = {str(p): sha(p) for p in BASE.iterdir() if p.is_file()}
        paths = [*HERE.glob('c3_*.py'), HERE/'c3_yolo224_two_node_route.cc',
                 HERE/'c3_yolo416_graph_once.cc', HERE/'c3_yolo416_plugin_lifetime.cc',
                 HERE/'test_c3_yolo224_two_node_prepare.py']
        bindings.update({str(p): sha(p) for p in paths})
        models = {}
        for variant in ('B','A'):
            models[variant] = {}
            for node in (34,46):
                old = HOLD/('20261017_c3_node%d_uop_identity_board_run01' % node)
                seal = read(old/'completion_manifest.json')
                verify({str(old/k): value for k,value in seal['artifact_sha256'].items()})
                bindings.update({str(old/k): value for k,value in seal['artifact_sha256'].items()})
                descriptor = read(old/'artifacts/prepared.json')['programs'][variant]
                verify({descriptor[k]: value for k,value in descriptor['sha256'].items()})
                models[variant][str(node)] = descriptor
        write(root/'replacements.json', models)
        module = tvm.IRModule(); cpu, vta_rows = {}, {}
        original_ids = read(BASE/'functions.json')
        archived_files = read(BASE.parent/'artifact_hashes.json')['artifacts']
        for symbol in original_ids:
            path = BASE/(symbol+'.tir.json')
            require(sha(path) == archived_files['G2/'+path.name], 'ARCHIVED_FUNCTION_CHANGED:'+symbol)
            fn = tvm.ir.load_json(path.read_text())
            module[symbol] = fn
            if is_vta_mac(fn): vta_rows[symbol] = fn
            else:
                cpu[symbol] = {'abi': buffer_abi(fn), 'instruction_gate': cpu_instruction_gate(fn),
                               'raw_pointer_gate': raw_pointer_gate(fn)}
        require(len(vta_rows) == 11 and len(cpu) == 21, 'FIXED_GRAPH_FUNCTION_COVERAGE')
        (root/'base_final.json').write_text(tvm.ir.save_json(module))
        write(root/'CPU_manifest.json', cpu)
        functions = {}
        for symbol, fn in vta_rows.items():
            if symbol in SYMBOLS.values():
                node = next(n for n,s in SYMBOLS.items() if s == symbol)
                desc = models['B'][str(node)]
                reference = read(desc['reference_contract'])
                require(reference['abi'] == buffer_abi(fn), 'TARGET_GRAPH_ABI')
                functions[symbol] = {'symbol': symbol, 'abi': buffer_abi(fn), 'workload':
                    read(Path(desc['qualified_directory']).parents[1]/'prepare/result.json')['programs']['R']['workload'],
                    'reference': reference, 'runtime_flags': [], 'source_TIR': str(BASE/(symbol+'.tir.json')),
                    'adopted_independent_reference_from_qualified_B': True}
                continue
            events, counts, flags = project(fn)
            require(flags in ([], [32,0]), 'UNKNOWN_ORIGINAL_RUNTIME_MODE')
            require(counts['expanded_uop_steps'] <= 5000000, 'PER_FUNCTION_EXPANSION_LIMIT')
            ancestors = [(1 << i)-1 for i in range(len(events))] if flags else None
            semantic = certify_trace(events, ancestors=ancestors)
            abi = buffer_abi(fn)
            kh, kw = abi[1]['shape'][2:4]
            require(kh == kw and kh in (1,3) and abi[0]['shape'][2:4] == abi[3]['shape'][2:4], 'FROZEN_CONV_GEOMETRY')
            pad = 1 if kh == 3 else 0
            workload = ['conv2d_packed.vta', ['TENSOR',abi[0]['shape'],'int8'], ['TENSOR',abi[1]['shape'],'int8'],
                        [1,1], [pad]*4, [1,1], 'NCHW1n16c', 'int32']
            tail = postprocess_from_output_terms(semantic['output_terms'], abi)
            spec = from_workload(workload, abi, tail)
            proof = semantic_gates(spec, semantic['output_terms'])
            functions[symbol] = {'symbol': symbol, 'abi': abi, 'workload': workload,
                                 'reference': spec.contract(), 'runtime_flags': flags,
                                 'counts': counts, 'independent_semantics': proof,
                                 'source_TIR': str(BASE/(symbol+'.tir.json'))}
        write(root/'functions.json', functions)
        with vta.Environment(dict(vta.get_env().cfg_dict, TARGET='sim')):
            target = tvm.target.Target(vta.get_env().target, host=vta.get_env().target_host)
            with tvm.transform.PassContext(opt_level=3, disabled_pass={'AlterOpLayout','tir.CommonSubexprElimTIR'}):
                lib = tvm.build(module, target=target)
                repeated = tvm.build(tvm.ir.load_json(tvm.ir.save_json(module)), target=target)
            require(normalize_llvm(lib.get_source('ll')) == normalize_llvm(repeated.get_source('ll')), 'BASE_FSIM_ROUNDTRIP')
            (root/'base_fsim.ll').write_text(lib.get_source('ll'))
            lib.export_library(str(root/'base_fsim.so'))
        shutil.copyfile(BASE/'graphlib.so', root/'base_arm.so')
        require(sha(root/'base_arm.so') == sha(BASE/'graphlib.so'), 'UNCHANGED_ARM_BASE_BINARY')
        for name in ('graph.json','params.bin'): shutil.copyfile(BASE/name, root/name)
        for variant in ('B','A'):
            for node in (34,46):
                descriptor = models[variant][str(node)]
                require(descriptor['abi'] == functions[SYMBOLS[node]]['abi'], 'REPLACEMENT_GRAPH_ABI')
                require(read(descriptor['reference_contract']) == functions[SYMBOLS[node]]['reference'], 'REPLACEMENT_FULL_SEMANTICS')
        hardware = Path(models['B']['34']['qualified_directory']).parents[1]/'hardware_archive.json'
        shutil.copyfile(hardware, root/'hardware_archive.json');_hardware(root)
        write(root/'freeze.json', {'bindings': bindings, 'new_target_calls': 0})
        build_runtime(root, 'sim');build_runtime(root, 'arm')
        helper_source = root/'graph_once.cc'
        helper_source.write_text(graph_helper((HERE/'c3_yolo416_graph_once.cc').read_text()))
        include = ['-I'+str(REPO/p) for p in ('include','3rdparty/dmlc-core/include','3rdparty/dlpack/include')]
        options = ['-std=c++17','-O2','-ldl',*include]
        lifetime = str(HERE/'c3_yolo416_plugin_lifetime.cc')
        cc.create_shared(str(root/'graph_once_fsim.so'), [str(helper_source), lifetime], options=options)
        cc.create_shared(str(root/'route_fsim.so'), [str(HERE/'c3_yolo224_two_node_route.cc'), lifetime], options=options)
        common_helpers = {'observer_fsim': HOLD/'20261017_c3_yolo416_graph_prepare_run02/graph_observer_fsim.so'}
        for name,p in common_helpers.items(): bindings[str(p)] = sha(p)
        verify(bindings)
        write(root/'contract.json', {'network': 'yolov3_tiny_224', 'base': str(BASE),
              'only_graph_call_replacements': SYMBOLS, 'variants': ['B','A'],
              'unchanged_ARM_base_library_sha256': sha(root/'base_arm.so'),
              'unchanged_graph_and_parameters': True, 'all_CPU_and_other_VTA_functions_unchanged': True,
              'not_a_full_network_TPS_or_TopHub_comparison': True,
              'expected_submissions': 17, 'original_force_serial_functions': 5,
              'collection_capacity': 32, 'runtime_execution_policy_changed': False,
              'formal_route_has_no_per_function_wrapper': True,
              'new_target_calls': 0, 'FPGA_calls': 0})
        write(root/'prepare_complete.json', {'status': 'YOLO224_TWO_NODE_GRAPH_PAIR_PREPARED_LOCAL_PENDING'})
        status = 'YOLO224_TWO_NODE_GRAPH_PAIR_PREPARED_LOCAL_PENDING'
    except Exception as error:
        status = 'STOPPED_GRAPH_PREPARATION'
        write(root/'failure.json', {'type': type(error).__name__, 'message': str(error), 'new_target_calls': 0})
        traceback.print_exc()
    write(root/'status.json', {'status': status, 'new_target_calls': 0, 'board_contacted': False})
    write(root/'completion_manifest.json', {'status': status,
          'artifact_sha256': {str(p.relative_to(root)): sha(p) for p in root.rglob('*') if p.is_file()}})
    print(status, flush=True)
    return 0 if status.endswith('LOCAL_PENDING') else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    sys.exit(prepare(parser.parse_args().output.resolve()))
