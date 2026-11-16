"""Certify the real local AXU5EVB fallback for YOLO224 nodes 34 and 46.

The installed fallback setter and original schedule are observed, never replaced
with a manually supplied configuration. No history records or TPS route is used.
Only existing common correctness/DMA adapters are retained. Local FSim timings
are not performance evidence; board measurement is a separate frozen attempt.
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
from contextlib import contextmanager

PRIOR_ZERO_CALL = HOLD/'20261019_c3_fallback_identity_local_run01'
TOOLFIX = HOLD/'20261019_c3_fallback_dispatch_toolfix01'
PHYSICAL_TOOLFIX = HOLD/'20261019_c3_fallback_physical_toolfix01'
HARDWARE = HOLD/'20261016_c3_tps_sync_release_local_run04/hardware_archive.json'
EXPECTED = {34: (1, 7, 1, 1, 1, 2, 1), 46: (1, 14, 1, 1, 1, 2, 1)}


def sources(node):
    require(node in EXPECTED, 'FIXED_TWO_NODE_SCOPE')
    return importlib.import_module('c3_node%d_tps_local' % node)


def config_values(config):
    entity = {name: (kind, value) for name, kind, value in config['entity']}
    names = ('tile_b', 'tile_h', 'tile_w', 'tile_ci', 'tile_co', 'oc_nthread', 'h_nthread')
    require(set(entity) == set(names), 'FALLBACK_CONFIG_FIELDS')
    return tuple(entity[n][1][-1] if entity[n][0] == 'sp' else entity[n][1] for n in names)


def fallback_snapshot(cfg):
    # FallbackConfigEntity has no to_json_dict in this pinned compiler.
    # Read every actual schedule field; never construct a replacement dispatch config.
    from tvm.autotvm.task.space import SplitEntity, OtherOptionEntity
    require(cfg.is_fallback, 'NOT_A_FALLBACK_SNAPSHOT')
    entries = []
    for name, entity in cfg._entity_map.items():
        if isinstance(entity, SplitEntity):
            entries.append([name, 'sp', [int(v) for v in entity.size]])
        elif isinstance(entity, OtherOptionEntity):
            entries.append([name, 'ot', int(entity.val)])
        else:
            raise ValueError('UNKNOWN_FALLBACK_ENTITY:'+name)
    return {'code_hash': getattr(cfg, 'code_hash', None), 'entity': entries}


@contextmanager
def observe_fallback(node, actual, queries):
    import tvm
    import vta
    from tvm.autotvm.task.dispatcher import FallbackContext, DispatchContext
    from vta.top import vta_conv2d
    from vta.top.residency_dispatch import active_context
    from c3_official_window_baseline import plain
    require(vta.get_env().TARGET == 'axu5evb', 'AXU_FALLBACK_POLICY_REQUIRED')
    require(active_context() is None, 'EXPLICIT_ROUTE_FORBIDDEN')
    original = vta_conv2d._schedule_conv2d_packed_impl

    class RecordedFallback(DispatchContext):
        def __init__(self):
            super().__init__()
            self.fallback = FallbackContext()

        def _query_inside(self, target, workload):
            # Ordinary DispatchContext prevents Relay from opening TopHub.
            # The actual config is still created by the installed fallback context.
            cfg = self.fallback._query_inside(target, workload)
            require(cfg.is_fallback, 'NON_FALLBACK_QUERY')
            queries.append({'target': str(target), 'workload': plain(workload),
                            'is_fallback': bool(cfg.is_fallback), 'config_object': id(cfg)})
            return cfg

    def schedule(cfg, outs, residency_mode='original', **kwargs):
        require(cfg.is_fallback and residency_mode in ('original', 0) and not kwargs,
                'NON_FALLBACK_OR_RESIDENCY_ROUTE')
        require(active_context() is None, 'EXPLICIT_ROUTE_FORBIDDEN')
        result = original(cfg, outs, residency_mode=residency_mode)
        require(cfg.is_fallback, 'FALLBACK_FLAG_CHANGED')
        used = fallback_snapshot(cfg)
        require(config_values(used) == EXPECTED[node], 'OBSERVED_LOCAL_FALLBACK_CHANGED')
        actual.append({'config': used, 'is_fallback': True, 'config_object': id(cfg),
                       'target_policy': vta.get_env().TARGET, 'residency_mode': residency_mode})
        return result

    vta_conv2d._schedule_conv2d_packed_impl = schedule
    try:
        with RecordedFallback() as ctx:
            yield ctx
    finally:
        vta_conv2d._schedule_conv2d_packed_impl = original


def compile_worker(unit, node):
    import tvm
    import vta
    from vta.testing import simulator
    from c3_banerjee_completion import CloseCallBoundary, SplitPaddedInputChannels
    from c3_yolo416_tps_dma import SplitStridedOutputChannels
    from c3_tps_sync_release_prepare import complete_function, compiled_module_contract, canonical_config, _hardware
    from c3_official_window_baseline import buffer_abi, normalize_llvm, plain
    from c3_official_window_method import audit_function
    from c3_official_window_reference import from_workload, postprocess_from_output_terms, semantic_gates
    from c3_valid_window import certify_trace
    from c3_tps_window_method import acc_spacing
    from c3_uop_identity_applicability import inspect_function

    source = sources(node)
    verify(read(unit/'freeze.json')['bindings'])
    require(Path(tvm._ffi.base._LIB._name).resolve() == (REPO/'build/libtvm.so').resolve(), 'COMPILER_DSO')
    require(Path(simulator.LIBS[0]._name).resolve() == (RUNTIME/'libvta_fsim.so').resolve(), 'FSIM_DSO')
    out = unit/'prepare/R'
    out.mkdir(parents=True, exist_ok=False)
    text = source.extract_relay((source.GRAPH/'relay_000.txt').read_text())
    (out/'source_relay.txt').write_text(text)
    relay = tvm.relay.parse(text)
    captures, order, actual, queries = [], [], [], []

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

    from unittest.mock import patch
    with vta.Environment(dict(vta.get_env().cfg_dict, TARGET='axu5evb')):
        env = vta.get_env()
        config = dict(vta.build_config().config)
        passes = list(config['tir.add_lower_pass'])
        at = next(i for i, (_, p) in enumerate(passes) if p.info.name == 'tir.InjectCopyIntrin')
        passes[at:at] = [(1, SplitPaddedInputChannels()), (1, SplitStridedOutputChannels())]
        at = next(i for i, (phase, _) in enumerate(passes) if phase >= 2)
        passes.insert(at, (2, CloseCallBoundary()))
        config['tir.add_lower_pass'] = passes
        disabled = {'AlterOpLayout', 'tir.CommonSubexprElimTIR'}
        target = tvm.target.Target(env.target, host='llvm')
        tvm.relay.backend.te_compiler.get().clear()
        with patch('tvm.autotvm.tophub.context', side_effect=ValueError('HISTORY_CONTEXT_FORBIDDEN')) as history_guard, observe_fallback(node, actual, queries) as dispatch:
            require(not isinstance(dispatch, tvm.autotvm.FallbackContext), 'RELAY_TOPHUB_AUTOLOAD_MUST_BE_DISABLED')
            with tvm.transform.PassContext(opt_level=3, config=config, instruments=[Archive()], disabled_pass=disabled):
                graph, raw_lib, params = tvm.relay.build(relay, target=target)
        del raw_lib
        require(history_guard.call_count == 0, 'HISTORY_CONTEXT_WAS_QUERIED')
        require(not params and len(actual) == 1, 'EXACT_FALLBACK_FUSION_REQUIRED')
        matched = [q for q in queries if q['workload'] == source.WORKLOAD]
        require(matched and all(q['is_fallback'] for q in queries), 'FALLBACK_QUERY_NOT_ESTABLISHED')
        require(any(q['config_object'] == actual[0]['config_object'] for q in matched), 'ACTUAL_QUERY_CONFIG_BINDING')
        write(out/'fallback_dispatch.json', {'actual_configs': actual, 'queries': queries,
              'history_used': False, 'TopHub_context_calls': history_guard.call_count, 'explicit_residency_used': False})
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
        old=PRIOR_ZERO_CALL/('node%d/prepare/R'%node)
        require(tvm.ir.structural_equal(parent,tvm.ir.load_json((old/'parent.json').read_text())) and
                tvm.ir.structural_equal(final,tvm.ir.load_json((old/'final.json').read_text())),
                'HISTORY_BYPASS_CHANGED_EXECUTION_IR')
        require(normalize_llvm(llvm)==normalize_llvm((old/'module.ll').read_text()), 'HISTORY_BYPASS_CHANGED_LLVM')
        write(out/'history_bypass_representation_check.json', {'status':'PARENT_FINAL_LLVM_UNCHANGED',
              'prior_attempt':str(PRIOR_ZERO_CALL),'old_target_calls_adopted':0,'history_context_calls':history_guard.call_count})
        write(out/'compiled_cache_contract.json', compiled['compiled_cache_contract'])
        write(out/'alu_initializer_contract.json', compiled['alu_initializer_contract'])
        write(out/'compilation_context.json', {'actual_configs': actual, 'dispatch_queries': queries,
              'fallback_policy_source': str(REPO/'vta/python/vta/top/vta_conv2d.py'),
              'fallback_policy_sha256': sha(REPO/'vta/python/vta/top/vta_conv2d.py'),
              'fallback_name': 'frozen locally adapted AXU5EVB default safe fallback',
              'history_or_TopHub_used': False, 'explicit_residency_route_used': False,
              'pass_order': order, 'captures': captures, 'target': str(target),
              'environment': env.cfg_dict, 'extra_passes': [[int(i), p.info.name] for i, p in passes],
              'disabled_passes': sorted(disabled), 'author_schedule_used': False,
              'common_correctness_adapters': ['SplitPaddedInputChannels', 'SplitStridedOutputChannels', 'CloseCallBoundary'],
              'new_optimization': False, 'not_upstream_unmodified_fallback': True})
        write(out/'graph.json', json.loads(graph))
    descriptor = {'program': 'R', 'variant': 'F', 'node': node, 'symbol': source.SYMBOL,
                  'global_var': gv.name_hint, 'abi': abi, 'workload': source.WORKLOAD,
                  'parent_module': str(out/'parent.json'), 'final_module': str(out/'final.json'),
                  'module_path': str(out/'fsim_module.so'), 'reference_contract': str(out/'reference_contract.json'),
                  'artifact_bindings': {str(p): sha(p) for p in out.iterdir() if p.is_file()}}
    write(out/'descriptor.json', descriptor)
    write(unit/'prepare/result.json', {'status': 'AXU_FALLBACK_PREPARED', 'programs': {'R': descriptor}})
    write(unit/'prepare_complete.json', {'status': 'AXU_FALLBACK_PREPARED', 'result_sha256': sha(unit/'prepare/result.json')})
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
    prior_manifest=read(PRIOR_ZERO_CALL/'completion_manifest.json')
    require(all(a['registered_attempts']==a['fully_certified']==0 for a in read(PRIOR_ZERO_CALL/'call_accounting.json').values()), 'OLD_ATTEMPT_NOT_ZERO_CALL')
    prior_paths=[PRIOR_ZERO_CALL/'completion_manifest.json', PRIOR_ZERO_CALL/'call_accounting.json', TOOLFIX/'completion_manifest.json', PHYSICAL_TOOLFIX/'completion_manifest.json']
    for node in EXPECTED:
        for name in ('parent.json','final.json','module.ll'):
            path=PRIOR_ZERO_CALL/('node%d/prepare/R'%node)/name
            require(sha(path)==prior_manifest['artifact_sha256'][str(path.relative_to(PRIOR_ZERO_CALL))], 'PRIOR_ZERO_CALL_ARTIFACT_CHANGED')
            prior_paths.append(path)
    paths = [*prior_paths, *HERE.glob('c3_*.py'), HERE/'test_c3_fallback_identity_local.py',
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
          'comparison_role': 'actual local AXU fallback F; no TPS transform or history',
          'stop_on_failure': True, 'no_retry_or_configuration_substitution': True,
          'local_duration_is_not_performance': True, 'board_entry_point': False})


def run(root):
    root.mkdir(exist_ok=False)
    status, error = 'STOPPED_PREFLIGHT', None
    try:
        with (root/'preflight.log').open('x') as log:
            code = subprocess.run([str(PYTHON), '-m', 'unittest', 'test_c3_fallback_identity_local',
                  'test_c3_yolo416_qualification_v2', 'test_c3_fallback_physical', '-v'], env=env_for_worker(), stdout=log, stderr=subprocess.STDOUT).returncode
        require(code == 0, 'ZERO_TARGET_PREFLIGHT_FAILED')
        freeze(root)
        status = 'STOPPED_PREPARATION'
        for node in EXPECTED:
            require(worker(root, node, 'prepare'), 'PREPARATION_WORKER_FAILED:%d' % node)
            unit = root/('node%d' % node)
            ready = read(unit/'prepare_complete.json')
            require(ready['status'] == 'AXU_FALLBACK_PREPARED' and
                    ready['result_sha256'] == sha(unit/'prepare/result.json') and
                    read(unit/'prepare/result.json')['status'] == 'AXU_FALLBACK_PREPARED', 'PREPARE_CONTRACT')
        status = 'STOPPED_QUALIFICATION'
        for node in EXPECTED:
            for mode in ('detailed', 'minimal'):
                require(worker(root, node, 'qualify', mode), 'QUALIFICATION_WORKER_FAILED:%d:%s' % (node, mode))
            compare_modes(root/('node%d' % node), 'R')
        verify(read(root/'freeze.json')['bindings'])
        status = 'TWO_AXU_FALLBACKS_LOCAL_QUALIFIED'
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
    return 0 if status == 'TWO_AXU_FALLBACKS_LOCAL_QUALIFIED' else 1


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
        from c3_fallback_qualification import qualification_worker
        qualification_worker(unit, 'R', args.mode)
