"""Two preselected YOLO224 TPS baselines in the broader thread domain.

No tuning, identity optimization, production changes, board execution, or timing.
The existing author schedule and adapters are retained; a single-thread function
does not enter the two-context ImprovedDoubleBuffer adapter.
"""
import argparse
import importlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

from c3_tps_sync_release_local import (
    HERE, REPO, HOLD, PYTHON, RUNTIME, HELPER, KINDS, SEEDS,
    read, write, sha, require, verify, env_for_worker, call_accounting, compare_modes,
)
from c3_yolo416_tps_model import select

HARDWARE = HOLD/'20261016_c3_tps_sync_release_local_run04/hardware_archive.json'
EXPECTED = {34: 'b1h7w7ci1co32oc1hth1', 46: 'b1h14w14ci1co8oc1hth1'}


def sources(node):
    require(node in EXPECTED, 'FIXED_TWO_NODE_SCOPE')
    return importlib.import_module('c3_node%d_tps_local' % node)


def selection(node):
    cfg = read(REPO/'3rdparty/vta-hw/config/vta_config.json')
    capacities = {k: 1 << cfg['LOG_'+k+'_BUFF_SIZE'] for k in ('INP', 'WGT', 'ACC')}
    chosen = select(sources(node).WORKLOAD, capacities)
    historical = read(HOLD/('20261017_c3_node%d_tps_applicability01' % node)/'audit.json')
    old = historical['prospective']['paper_minimum_all_supported_threads'][0]
    require(chosen['selected'] == old, 'BROADER_SELECTION_DIFFERS_FROM_ARCHIVED_RULE')
    require(chosen['selected']['id'] == EXPECTED[node], 'PRESELECTED_CONFIGURATION_CHANGED')
    return chosen


def route(node, choice):
    return {'candidate_id': 'node%d-paper-TPS-broader-domain-baseline' % node,
            'identity': {'workload': sources(node).WORKLOAD, 'public_mode': 'original',
                         'implementation_mode': 0,
                         'complete_config_entity': choice['complete_config_entity']}}


def compile_worker(unit, node):
    import tvm
    import vta
    from vta.testing import simulator
    from c3_banerjee_local import author_namespace
    from c3_banerjee_completion import CloseCallBoundary, SplitPaddedInputChannels
    from c3_yolo416_tps_dma import SplitStridedOutputChannels
    from c3_tps_sync_release_prepare import complete_function, compiled_module_contract, canonical_config, _hardware
    from c3_official_window_baseline import buffer_abi, normalize_llvm, plain
    from c3_official_window_method import audit_function
    from c3_official_window_reference import from_workload, postprocess_from_output_terms, semantic_gates
    from c3_valid_window import certify_trace
    from c3_tps_window_method import acc_spacing
    from c3_uop_identity_applicability import inspect_function
    from vta.top.residency_dispatch import ExplicitResidencyDispatch

    source = sources(node)
    verify(read(unit/'freeze.json')['bindings'])
    require(Path(tvm._ffi.base._LIB._name).resolve() == (REPO/'build/libtvm.so').resolve(), 'COMPILER_DSO')
    require(Path(simulator.LIBS[0]._name).resolve() == (RUNTIME/'libvta_fsim.so').resolve(), 'FSIM_DSO')
    choice = read(unit/'selection.json')
    require(choice == selection(node), 'FROZEN_SELECTION_CHANGED')
    chosen_route = route(node, choice)
    require(choice['selected']['tile']['oc_nthread'] == choice['selected']['tile']['h_nthread'] == 1,
            'THIS_PREPARATION_EXPECTS_SINGLE_THREAD_BASELINE')
    out = unit/'prepare/R'
    out.mkdir(parents=True, exist_ok=False)
    text = source.extract_relay((source.GRAPH/'relay_000.txt').read_text())
    (out/'source_relay.txt').write_text(text)
    relay = tvm.relay.parse(text)
    captures, order, actual = [], [], []

    @tvm.instrument.pass_instrument
    class Archive:
        def run_before_pass(self, mod, info):
            order.append(info.name)
            self.save(mod, info.name, 'before')
        def run_after_pass(self, mod, info):
            self.save(mod, info.name, 'after')
        def save(self, mod, name, when):
            if any(s in name for s in ('LiftAttrScope', 'InjectVirtualThread', 'InjectCoProcSync',
                                       'FoldUopLoop', 'CPUAccessRewrite', 'CloseCallBoundary',
                                       'SplitPaddedInputChannels', 'SplitStridedOutputChannels')):
                stem = '%03d_%s_%s' % (len(captures), when, name.replace('.', '_'))
                (out/(stem+'.json')).write_text(tvm.ir.save_json(mod))
                captures.append({'stem': stem, 'name': name, 'when': when})

    with vta.Environment(dict(vta.get_env().cfg_dict, TARGET='sim')):
        namespace = author_namespace(source.AUTHOR, safe_uop_order=False)
        source.check_schedule_environment(namespace)
        original = namespace['schedule_conv2d_packed']
        def schedule(cfg, outs):
            source.check_schedule_environment(namespace)
            require(not cfg.is_fallback, 'FALLBACK_FORBIDDEN')
            used = plain(cfg.to_json_dict())
            require(canonical_config(used) == canonical_config(choice['complete_config_entity']), 'ACTUAL_CONFIG')
            actual.append(used)
            return original(cfg, outs)
        namespace['schedule_conv2d_packed'] = schedule
        env = vta.get_env()
        config = dict(vta.build_config().config)
        passes = list(config['tir.add_lower_pass'])
        at = next(i for i, (_, p) in enumerate(passes) if p.info.name == 'tir.InjectCopyIntrin')
        passes[at:at] = [(1, SplitPaddedInputChannels()), (1, SplitStridedOutputChannels())]
        at = next(i for i, (phase, _) in enumerate(passes) if phase >= 2)
        passes.insert(at, (2, CloseCallBoundary()))
        config['tir.add_lower_pass'] = passes
        disabled = {'AlterOpLayout', 'tir.CommonSubexprElimTIR'}
        target = tvm.target.Target(env.target, host=env.target_host)
        tvm.relay.backend.te_compiler.get().clear()
        schedule_context = getattr(source, 'node%d_author_schedule' % node)
        with schedule_context(namespace), ExplicitResidencyDispatch([chosen_route]) as dispatch:
            with tvm.transform.PassContext(opt_level=3, config=config, instruments=[Archive()], disabled_pass=disabled):
                graph, raw_lib, params = tvm.relay.build(relay, target=target)
        del raw_lib
        require(not params and actual and dispatch.summary()['all_routes_scheduled'], 'FUSION_DISPATCH')
        def captured(suffix, when):
            found = [c for c in captures if c['name'].endswith(suffix) and c['when'] == when]
            require(len(found) == 1, 'NONUNIQUE_CAPTURE:'+suffix)
            return tvm.ir.load_json((out/(found[0]['stem']+'.json')).read_text())
        parent = captured('FoldUopLoop', 'after')
        direct = captured('CPUAccessRewrite', 'after')
        require(tvm.ir.structural_equal(parent, captured('CPUAccessRewrite', 'before')), 'PARENT_PASS_GAP')
        restored = tvm.ir.load_json(tvm.ir.save_json(parent))
        require(tvm.ir.structural_equal(parent, restored), 'PARENT_ROUNDTRIP')
        final = vta.transform.CPUAccessRewrite()(restored)
        require(tvm.ir.structural_equal(final, direct), 'FINAL_ROUNDTRIP')
        gv, fn = complete_function(parent, source.SYMBOL)
        abi = buffer_abi(fn)
        require([r['shape'] for r in abi] == source.SHAPES and
                [r['dtype'] for r in abi] == ['int8', 'int8', 'int32', 'int8'], 'REAL_FUSION_ABI')
        audit = audit_function(fn)
        require(audit['counts']['expanded_uop_steps'] <= 5000000, 'EXPANSION_LIMIT')
        semantic = certify_trace(audit['events'])
        tail = postprocess_from_output_terms(semantic['output_terms'], abi)
        spec = from_workload(source.WORKLOAD, abi, tail)
        old_reference = read(unit/'historical_reference.json')
        require(spec.contract() == old_reference, 'FULL_FUSION_REFERENCE_CHANGED')
        hardware = _hardware(unit)
        write(out/'independent_semantics.json', semantic_gates(spec, semantic['output_terms']))
        write(out/'logical_certificate.json', {'status': 'LOGICAL_PASSED_NATIVE_PENDING',
              'counts': audit['counts'], 'acc_spacing': acc_spacing(audit['events'], hardware['verified_acc_interval'])})
        write(out/'reference_contract.json', spec.contract())
        write(out/'events.json', audit['events'])
        traffic = source.tensor_traffic(audit['events'])
        require(traffic['WGT']['multiplicities'] == [1], 'TPS_WEIGHT_ONCE_NOT_ESTABLISHED')
        write(out/'tensor_requests.json', traffic)
        write(out/'identity_census.json', inspect_function(fn))
        for name, mod in (('parent', parent), ('final', final)):
            (out/(name+'.json')).write_text(tvm.ir.save_json(mod))
            (out/(name+'.tir')).write_text(mod.script())
        with tvm.transform.PassContext(opt_level=3, disabled_pass=disabled):
            lib = tvm.build(final, target=target)
            other = tvm.build(direct, target=target)
        llvm = lib.get_source('ll')
        require(normalize_llvm(llvm) == normalize_llvm(other.get_source('ll')), 'LLVM_ROUNDTRIP')
        (out/'module.ll').write_text(llvm)
        lib.export_library(str(out/'fsim_module.so'))
        compiled = compiled_module_contract(final, source.SYMBOL, llvm)
        require(compiled['abi'] == abi, 'COMPILED_ABI')
        write(out/'compiled_cache_contract.json', compiled['compiled_cache_contract'])
        write(out/'alu_initializer_contract.json', compiled['alu_initializer_contract'])
        write(out/'compilation_context.json', {'route': chosen_route, 'actual_configs': actual,
              'dispatch': dispatch.summary(), 'safe_uop_order': False, 'pass_order': order,
              'captures': captures, 'target': str(target), 'environment': env.cfg_dict,
              'extra_passes': [[int(i), p.info.name] for i, p in passes],
              'disabled_passes': sorted(disabled), 'ImprovedDoubleBuffer_not_applicable_to_oc1_h1': True,
              'new_optimization': False, 'not_exact_author_stack': True})
        write(out/'graph.json', json.loads(graph))
    descriptor = {'program': 'R', 'variant': 'C', 'node': node, 'symbol': source.SYMBOL,
                  'global_var': gv.name_hint, 'abi': abi, 'workload': source.WORKLOAD,
                  'parent_module': str(out/'parent.json'), 'final_module': str(out/'final.json'),
                  'module_path': str(out/'fsim_module.so'), 'reference_contract': str(out/'reference_contract.json'),
                  'artifact_bindings': {str(p): sha(p) for p in out.iterdir() if p.is_file()}}
    write(out/'descriptor.json', descriptor)
    write(unit/'prepare/result.json', {'status': 'BROADER_TPS_PREPARED', 'programs': {'R': descriptor}})
    write(unit/'prepare_complete.json', {'status': 'BROADER_TPS_PREPARED', 'result_sha256': sha(unit/'prepare/result.json')})
    verify(read(unit/'freeze.json')['bindings'])


def worker(root, node, phase, mode=None):
    import psutil
    unit = root/('node%d' % node)
    label = '%d_%s%s' % (node, phase, '_'+mode if mode else '')
    argv = [str(PYTHON), str(Path(__file__).resolve()), '_worker', str(root), '--node', str(node), '--phase', phase]
    if mode:
        argv += ['--mode', mode]
    env = env_for_worker()
    for key in ('VTA_UOP_BUFFER_BYTES', 'VTA_INSN_BUFFER_BYTES', 'VTA_INSN_SUBMIT_THRESHOLD_BYTES'):
        env[key] = '33554432'
    started, peak, limit = time.monotonic(), 0, None
    with (root/(label+'.log')).open('x') as log:
        child = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        while child.poll() is None:
            try:
                proc = psutil.Process(child.pid)
                peak = max(peak, sum(p.memory_info().rss for p in [proc, *proc.children(recursive=True)]))
            except psutil.Error:
                pass
            if peak > 8*1024**3 or time.monotonic()-started > 600:
                limit = 'MEMORY' if peak > 8*1024**3 else 'TIME'
                os.killpg(child.pid, signal.SIGKILL)
                break
            time.sleep(.1)
        code = child.wait()
    record = {'returncode': code, 'resource_limit': limit, 'sampled_tree_peak_rss': peak,
              'resource_control_seconds': time.monotonic()-started, 'argv': argv}
    write(root/(label+'_worker.json'), record)
    return code == 0 and limit is None


def freeze(root):
    from c3_official_window_baseline import compiler_sources
    hardware = read(HARDWARE)
    verify(hardware['source_bindings'])
    paths = [*HERE.glob('c3_*.py'), HERE/'test_c3_tps_broader_baseline_local.py',
             HELPER, RUNTIME/'libvta_fsim.so', HARDWARE,
             REPO/'3rdparty/vta-hw/config/vta_config.json', *list((REPO/'vta/python').rglob('*.py'))]
    bindings = {str(p.resolve()): sha(p) for p in paths if p.is_file()}
    bindings.update(compiler_sources())
    bindings.update(hardware['source_bindings'])
    for node in EXPECTED:
        source = sources(node)
        paths = [source.GRAPH/'relay_000.txt', *source.AUTHOR.glob('*')]
        for p in paths:
            if p.is_file(): bindings[str(p.resolve())] = sha(p)
        unit = root/('node%d' % node)
        unit.mkdir()
        old_root = HOLD/('20261017_c3_node34_tps_local_run02' if node == 34 else '20261017_c3_node46_tps_local_run01')
        seal = read(old_root/'completion_manifest.json')
        reference = old_root/'prepare/R/reference_contract.json'
        require(seal['artifact_sha256']['prepare/R/reference_contract.json'] == sha(reference), 'HISTORICAL_REFERENCE_CHANGED')
        bindings[str(reference)] = sha(reference)
        bindings[str(old_root/'completion_manifest.json')] = sha(old_root/'completion_manifest.json')
        shutil.copyfile(reference, unit/'historical_reference.json')
        write(unit/'selection.json', selection(node))
        shutil.copyfile(HARDWARE, unit/'hardware_archive.json')
        write(unit/'call_ledger.json', [{'program': 'R', 'mode': mode, 'call_index': i, 'kind': kind,
              'seed': seed, 'target_calls': 1, 'performance_sample': False}
              for mode in ('detailed', 'minimal') for i, (kind, seed) in enumerate(zip(KINDS, SEEDS), 1)])
        for p in unit.iterdir(): bindings[str(p)] = sha(p)
    for node in EXPECTED:
        write(root/('node%d' % node)/'freeze.json', {'bindings': bindings, 'max_calls': 14})
    write(root/'freeze.json', {'bindings': bindings, 'max_calls': 28, 'board_contacted': False})
    write(root/'contract.json', {'programs': EXPECTED, 'max_complete_FSim_calls': 28,
          'sequence': ['prepare34', 'prepare46', '34_detailed', '34_minimal', '46_detailed', '46_minimal'],
          'comparison_role': 'stronger existing-method baseline, not new optimization',
          'stop_on_failure': True, 'no_retry_or_configuration_substitution': True,
          'local_duration_is_not_performance': True, 'board_entry_point': False})


def run(root):
    root.mkdir(exist_ok=False)
    status, error = 'STOPPED_PREFLIGHT', None
    try:
        with (root/'preflight.log').open('x') as log:
            code = subprocess.run([str(PYTHON), '-m', 'unittest', 'test_c3_tps_broader_baseline_local',
                  'test_c3_yolo416_qualification_v2', '-v'], env=env_for_worker(), stdout=log, stderr=subprocess.STDOUT).returncode
        require(code == 0, 'ZERO_TARGET_PREFLIGHT_FAILED')
        freeze(root)
        status = 'STOPPED_PREPARATION'
        for node in EXPECTED:
            require(worker(root, node, 'prepare'), 'PREPARATION_WORKER_FAILED:%d' % node)
            unit = root/('node%d' % node)
            ready = read(unit/'prepare_complete.json')
            require(ready['status'] == 'BROADER_TPS_PREPARED' and
                    ready['result_sha256'] == sha(unit/'prepare/result.json') and
                    read(unit/'prepare/result.json')['status'] == 'BROADER_TPS_PREPARED', 'PREPARE_CONTRACT')
        status = 'STOPPED_QUALIFICATION'
        for node in EXPECTED:
            for mode in ('detailed', 'minimal'):
                require(worker(root, node, 'qualify', mode), 'QUALIFICATION_WORKER_FAILED:%d:%s' % (node, mode))
            compare_modes(root/('node%d' % node), 'R')
        verify(read(root/'freeze.json')['bindings'])
        status = 'TWO_BROADER_TPS_BASELINES_LOCAL_QUALIFIED'
    except Exception as exc:
        error = {'type': type(exc).__name__, 'message': str(exc)}
        traceback.print_exc()
    accounts = {str(n): call_accounting(root/('node%d' % n)) for n in EXPECTED}
    write(root/'call_accounting.json', accounts)
    write(root/'status.json', {'status': status, 'error': error, 'paused': True,
                              'FPGA_calls': 0, 'performance_conclusion': None})
    write(root/'completion_manifest.json', {'status': status, 'paused': True,
          'artifact_sha256': {str(p.relative_to(root)): sha(p) for p in root.rglob('*') if p.is_file()}})
    print(json.dumps({'status': status, 'error': error,
                      'fully_certified': sum(a['fully_certified'] for a in accounts.values())}), flush=True)
    return 0 if status == 'TWO_BROADER_TPS_BASELINES_LOCAL_QUALIFIED' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['run', '_worker'])
    parser.add_argument('output', type=Path)
    parser.add_argument('--node', type=int, choices=[34, 46])
    parser.add_argument('--phase', choices=['prepare', 'qualify'])
    parser.add_argument('--mode', choices=['detailed', 'minimal'])
    args = parser.parse_args()
    if args.action == 'run':
        sys.exit(run(args.output.resolve()))
    unit = args.output.resolve()/('node%d' % args.node)
    if args.phase == 'prepare':
        compile_worker(unit, args.node)
    else:
        from c3_yolo416_qualification_v2 import qualification_worker
        qualification_worker(unit, 'R', args.mode)
