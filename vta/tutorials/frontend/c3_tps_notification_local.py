#!/usr/bin/env python3
"""One bounded local carrier experiment, 42 complete FSim calls, no board API.

Reuse sealed B/N/G modules. Build a NEW FSim-only runtime overlay that removes
six certified carrier NOPs after queue generation. Do not recombine UOP LOADs.
All original experiment paths remain read-only. No latency conclusion.
"""
import argparse
from copy import deepcopy
import ctypes
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import time
import traceback

from c3_tps_aggregation_prepare import HERE, REPO, HOLD, QUALIFIED, OFFICIAL, PYTHON, sha, read, write, verify_bindings
from c3_tps_notification_probe import (require, propose, certify_rewrite, organization,
                                      material, relations, finite_batch_progress, canonical_payload)
from c3_five_tradeoff_commands import decode_submissions

BOARD = HOLD / '20261015_c3_tps_aggregation_run02'
BASE_RUNTIME = OFFICIAL / 'runtime_local_final'
HEADER = HERE / 'c3_tps_notification_carrier.h'
HELPER = OFFICIAL / 'helper/local_final02/c3_official_window_runtime_helper.so'
BEGIN = 'C3_CARRIER_PRE_ENCODING_BEGIN\n'
END = 'C3_CARRIER_PRE_ENCODING_END\n'


def depths():
    h = read(QUALIFIED / 'hardware_archive.json')['fifo_depths']
    return ({i: h[k] for i, k in enumerate(('load_queue', 'gemm_queue', 'store_queue'), 1)},
            {lane: h[k] for lane, k in zip(((1, 2), (2, 1), (2, 3), (3, 2)),
                ('l2g_queue', 'g2l_queue', 'g2s_queue', 's2g_queue'))})


def build(out):
    out.mkdir()
    source = BASE_RUNTIME / 'c3_official_window_runtime_generated.cc'
    original = source.read_text()
    anchor = '  // Commit all pending pop of corresponding stage\n'
    require(original.count(anchor) == 1, 'INSN_QUEUE_INSERTION_CHANGED')
    text = original.replace(anchor, HEADER.read_text() + '\n' + anchor)
    anchor = '    CHECK(!insn_queue_.PendingPop());\n'
    require(text.count(anchor) == 1, 'FINALIZED_QUEUE_ANCHOR_CHANGED')
    hook = '''    CHECK_EQ(debug_flag_ & VTA_DEBUG_FORCE_SERIAL, 0);
    CHECK(!queue_capacity::BoundaryAuditEnabled());
    if (debug_flag_ & VTA_DEBUG_DUMP_INSN) {
      printf("C3_CARRIER_PRE_ENCODING_BEGIN\\n");
      insn_queue_.DumpInsn();
      printf("C3_CARRIER_PRE_ENCODING_END\\n");
    }
    insn_queue_.C3TPSNotificationCarriers();
'''
    text = text.replace(anchor, anchor + hook)
    generated = out / source.name
    generated.write_text(text)
    for name in ('c3_official_window_runtime_collection.h', 'c3_official_window_runtime_uop_queue.h'):
        (out / name).write_bytes((BASE_RUNTIME / name).read_bytes())
    command = [s.replace(str(BASE_RUNTIME), str(out)) for s in read(BASE_RUNTIME / 'build.json')['command']]
    result = subprocess.run(command, capture_output=True, text=True, timeout=600)
    (out / 'build.log').write_text(result.stdout + result.stderr)
    write(out / 'build.json', {'returncode': result.returncode, 'command': command,
        'parent_generated_sha256': sha(source), 'generated_sha256': sha(generated),
        'carrier_header_sha256': sha(HEADER), 'target': 'FSim only',
        'library_sha256': sha(out / 'libvta_fsim.so') if result.returncode == 0 else None,
        'insertion': 'after normal drain and FINISH, before serialization/barriers',
        'UOP_queue_and_greedy_generation_changed': False, 'board_contacted': False})
    require(result.returncode == 0, 'LOCAL_RUNTIME_BUILD_FAILED')


def structure(out):
    command, token = depths()
    original, rows = {}, {}
    for variant in 'BNG':
        path = QUALIFIED / f'TPS_{variant}_detailed/native_call_02.txt'
        before = decode_submissions(path.read_text())[0]
        original[variant] = before
        after, _ = propose(before)
        proof = certify_rewrite(before, after, command, token)
        rows[variant] = {'source': str(path), 'source_sha256': sha(path),
                         'before': organization(before), 'proposal': organization(after), 'proof': proof}
    b, n = original['B'], original['N']
    require(len(material(b)) == len(material(n)), 'B_N_MATERIAL_CENSUS')
    added = sorted(relations(n) - relations(b))
    removed = sorted(relations(b) - relations(n))
    changed = []
    for i, j in added:
        require((j, i) not in relations(b), 'CHANGE_IS_NOT_PREVIOUSLY_UNORDERED')
        changed.append({'source_material': i, 'target_material': j,
            'source_description': material(n)[i]['description'],
            'target_description': material(n)[j]['description'],
            'B_instructions': [material(b)[i]['index'], material(b)[j]['index']],
            'N_instructions': [material(n)[i]['index'], material(n)[j]['index']]})
    require(len(added) == 14 and not removed, 'UNEXPECTED_B_N_DEPENDENCY_DELTA')
    write(out / 'structure.json', {'variants': rows, 'B_to_N_added_material_dependencies': changed,
        'B_to_N_removed_material_dependencies': removed,
        'entry_wait_windows': 4, 'exit_release_windows': 2,
        'potential_overlap_only_not_device_cycle_measurement': True,
        'proposal_retains_all_N_material_constraints': True})


def worker(root, variant, mode):
    os.environ['TVM_LIBRARY_PATH'] = str(root / 'runtime')
    os.environ['C3_OFFICIAL_WINDOW_COLLECTION'] = mode
    os.environ['VTA_REPLAY_POLICY'] = 'disabled'
    os.environ.pop('VTA_COMMAND_MANIFEST_ID', None)
    import numpy as np
    import tvm
    import vta
    from vta.testing import simulator
    from c3_valid_window import certify_trace
    from c3_official_window_method import audit_function
    from c3_official_window_reference import from_workload, array_sha, semantic_gates
    from c3_official_window_native_binding import native_binding_details
    from c3_official_window_workflow import native_certificate_for_storage
    from c3_tps_window_native import certify_native

    verify_bindings(read(root / 'freeze.json')['bindings'])
    out = root / f'{variant}_{mode}'
    out.mkdir()
    source = QUALIFIED / f'TPS_{variant}_detailed'
    actual_runtime = Path(simulator.LIBS[0]._name).resolve()
    require(actual_runtime == (root / 'runtime/libvta_fsim.so').resolve(), 'WRONG_RUNTIME_PATH')
    require(sha(actual_runtime) == read(root / 'runtime/build.json')['library_sha256'], 'WRONG_RUNTIME_BINARY')
    parent = tvm.ir.load_json((source / 'parent.json').read_text())
    require(len(parent.functions) == 1, 'EXPECTED_ONE_COMPLETE_FUSED_FUNCTION')
    function = next(iter(parent.functions.values()))
    events = audit_function(function)['events']
    sem = certify_trace(events)
    contract = read(source / 'reference_contract.json')
    workload = ['conv2d_packed.vta', ['TENSOR', contract['input_shape'], 'int8'],
                ['TENSOR', contract['weight_shape'], 'int8'], contract['strides'],
                contract['padding'], contract['dilation'], 'NCHW1n16c', 'int32']
    spec = from_workload(workload, contract['abi'], contract['postprocess'])
    require(spec.contract() == contract, 'REFERENCE_CONTRACT_CHANGED')
    write(out / 'logical_certificate.json', semantic_gates(spec, sem['output_terms']))
    lib = tvm.runtime.load_module(str(source / 'fsim_module.so'))
    helper = tvm.runtime.load_module(str(HELPER))
    diagnostic = tvm.get_global_func('c3.official_window.fused_diagnostic_once')
    reset = tvm.get_global_func('c3.official_window.reset_counts')
    counts = tvm.get_global_func('c3.official_window.counts')
    metadata = tvm.get_global_func('c3.official_window.metadata')
    clear = tvm.get_global_func('vta.runtime.official_submit_ledger_clear')
    ledger_func = tvm.get_global_func('vta.runtime.official_submit_ledger')
    collection = tvm.get_global_func('vta.runtime.official_collection_status')
    require(json.loads(collection())['mode'] == mode, 'WRONG_MODE')
    command, token = depths()
    cases = spec.qualification_cases()
    require(len(cases) == 7, 'FROZEN_LOCAL_BUDGET_CHANGED')
    buffers = tuple(tvm.nd.array(a, tvm.device('ext_dev', 0)) for a in cases[0]['arrays'])
    handles = [int(ctypes.cast(b.handle, ctypes.c_void_p).value) for b in buffers]
    storage = [int(b.handle.contents.data) for b in buffers]
    rows, definitions = [], None
    reset()
    for index, case in enumerate(cases, 1):
        require(np.all(case['arrays'][-1] != case['expected']), 'OUTPUT_SENTINEL_MISSING')
        for b, a in zip(buffers, case['arrays']): b.copyfrom(a)
        clear()
        write(out / f'call_{index:02d}_begin.json', {'variant': variant, 'mode': mode,
            'call_index': index, 'kind': case['kind'], 'seed': case['seed'],
            'target_calls_planned': 1, 'performance_sample': False, 'board_contacted': False})
        dump = diagnostic(lib, str(function.attrs['global_symbol']), *buffers)
        ledger = json.loads(ledger_func())
        (out / f'call_{index:02d}_capture.txt').write_text(dump)
        require(dump.count(BEGIN) == dump.count(END) == 1, 'EXPECTED_ONE_EXPLICIT_SUBMIT')
        prefix, rest = dump.split(BEGIN)
        pre, post = rest.split(END)
        pre_dump = prefix + pre
        before = decode_submissions(pre_dump)[0]
        after = decode_submissions(post)[0]
        require(ledger['recorded_submits'] == ledger['attempted_submits'] == 1 and
                ledger['submits'][0]['insns'] == len(after) and ledger['submits'][0]['completion'],
                'ACTUAL_SUBMIT_LEDGER_MISMATCH')
        proof = certify_rewrite(before, after, command, token)
        # The before queue was captured from this call but NEVER executed.
        # Adapt only its instruction census for a structural queue certificate.
        # Actual execution/driver completion evidence stays in ledger unchanged.
        pre_ledger = deepcopy(ledger)
        pre_ledger['submits'][0]['insns'] = len(before)
        if 'insn_bytes' in pre_ledger['submits'][0]:
            pre_ledger['submits'][0]['insn_bytes'] = len(before) * 16
        native = certify_native(events, pre_dump, pre_ledger, definitions)
        definitions = native['cache_definitions']
        pre_sem = certify_trace(events, ancestors=native['ancestors'])
        require(pre_sem['output_terms'] == sem['output_terms'], 'PRE_ENCODING_SEMANTICS_CHANGED')
        actual_binding = native_binding_details(events, post)
        require(actual_binding['ancestors'] == native['ancestors'], 'ACTUAL_MATERIAL_ANCESTORS_CHANGED')
        # Non-NOP payload equality includes every UOP LOAD address/length/source,
        # every GEMM/ALU range and ordering. The native per-reference live-value
        # proof therefore transfers exactly; NOPs never read/write UOP values.
        proof['uop_live_value_proof'] = 'exact_non_NOP_compute_fifo_preservation_from_pre_encoding_certificate'
        write(out / f'pre_encoding_certificate_{index:02d}.json', {
            'scope': 'captured queue before rewrite; not an executed submission',
            'certificate': native_certificate_for_storage(native)})
        write(out / f'actual_certificate_{index:02d}.json', {
            'rewrite': proof, 'binding': {k: v for k, v in actual_binding.items() if k != 'ancestors'},
            'actual_submit_ledger': ledger, 'material_ancestors_exactly_equal': True})
        actual = buffers[-1].numpy()
        require(np.array_equal(actual, case['expected']), 'FULL_FUSION_MISMATCH')
        require(all(np.array_equal(b.numpy(), a) for b, a in zip(buffers[:-1], case['arrays'][:-1])),
                'READONLY_PARAMETER_CHANGED')
        require(storage == [int(b.handle.contents.data) for b in buffers] and
                handles == [int(ctypes.cast(b.handle, ctypes.c_void_p).value) for b in buffers],
                'PARAMETER_STORAGE_CHANGED')
        meta, collected = json.loads(metadata()), json.loads(collection())
        require(int(counts()) == meta['started_calls'] == meta['completed_calls'] == index, 'UNEXPECTED_TARGET_CALL')
        require(collected['mode'] == mode and collected['debug_flag'] == 0, 'COLLECTION_CHANGED')
        require(mode != 'minimal' or collected['detailed_clock_reads'] == 0, 'MINIMAL_CLOCK_READS')
        row = {'index': index, 'variant': variant, 'mode': mode, 'kind': case['kind'], 'seed': case['seed'],
               'correct': True, 'helper': meta, 'submit_ledger': ledger,
               'buffer_handles': handles, 'actual_storage': storage,
               'input_sha256': [array_sha(a) for a in case['arrays'][:-1]],
               'output_sha256': array_sha(actual), 'reference_sha256': array_sha(case['expected']),
               'organization': organization(after), 'removed_nops': len(proof['edits']),
               'pre_encoding_wraps': native['uop_values']['uop_sram_wraps'],
               'performance_sample': False, 'board_contacted': False}
        if case['kind'] == 'directed':
            row['output_distribution'] = {'zero': int(np.count_nonzero(actual == 0)),
                'clip_endpoints': int(np.count_nonzero((actual == 127) | (actual == -127))),
                'nonzero_unsaturated': int(np.count_nonzero((actual != 0) & (actual != 127) & (actual != -127))),
                'total': int(actual.size)}
        write(out / f'call_{index:02d}_end.json', row)
        rows.append(row)
        print(json.dumps({'variant': variant, 'mode': mode, 'call': index, 'status': 'PASS'}), flush=True)
    require(helper is not None, 'HELPER_LIFETIME')
    write(out / 'summary.json', {'status': 'LOCAL_CARRIER_CERTIFICATES_AND_FSIM_PASSED',
        'variant': variant, 'mode': mode, 'calls': rows, 'target_calls': len(rows),
        'runtime_sha256': sha(actual_runtime), 'helper_sha256': sha(HELPER),
        'unchanged_module_sha256': sha(source / 'fsim_module.so'), 'board_qualified': False})


def run(root):
    root.mkdir(exist_ok=False)
    (root / 'source_snapshot').mkdir()
    for path in list(HERE.glob('c3_tps_notification*')) + [HERE / 'test_c3_tps_notification_probe.py']:
        if path.is_file(): shutil.copyfile(path, root / 'source_snapshot' / path.name)
    # Verify the completed FPGA result and every preexisting frozen binding.
    frozen = read(BOARD / 'freeze.json')['source_and_artifact_bindings']
    verify_bindings(frozen)
    seal = read(BOARD / 'completion_manifest.json')
    verify_bindings({str(BOARD / p): d for p, d in seal['artifact_sha256'].items()})
    bindings = dict(frozen)
    for path in list(HERE.glob('c3_tps_notification*')) + [HERE / 'test_c3_tps_notification_probe.py',
            QUALIFIED / 'hardware_archive.json', BASE_RUNTIME / 'build.json',
            BASE_RUNTIME / 'c3_official_window_runtime_generated.cc',
            BASE_RUNTIME / 'c3_official_window_runtime_collection.h']:
        bindings[str(path)] = sha(path)
    # Keep the exact Python modules/certificates used by the six workers frozen.
    for variant in 'BNG':
        source = QUALIFIED / f'TPS_{variant}_detailed'
        for path in source.iterdir():
            if path.is_file(): bindings[str(path)] = sha(path)
    write(root / 'freeze.json', {'scope': 'TPS224 B/Nprime/Gprime notification carriers; local only',
        'bindings': bindings, 'new_FSim_calls_planned': 42, 'FPGA_calls_planned': 0,
        'variants': ['B', 'Nprime', 'Gprime'], 'seeds': [0, 20250901, 20260910],
        'modes': ['detailed', 'minimal'], 'per_worker_cases': 'three seeds, W0-W1-W0, directed',
        'limits': {'process_tree_bytes': 8*1024**3, 'worker_seconds': 600},
        'UOP_load_recombination': False, 'cutpoint_search': False,
        'prior_complete_board_artifacts_verified': len(seal['artifact_sha256']),
        'board_contacted': False})
    structure(root)
    build(root / 'runtime')
    from c3_official_window_workflow import _tree_rss
    results = []
    for variant in 'BNG':
        for mode in ('detailed', 'minimal'):
            command = [str(PYTHON), str(Path(__file__).resolve()), '--worker', variant,
                       '--mode', mode, '--output', str(root)]
            env = dict(os.environ, PYTHONPATH=':'.join(str(REPO / p) for p in
                ('python', 'vta/python', 'vta/tutorials/frontend')), OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
            start, peak, limit = time.monotonic(), 0, None
            with (root / f'{variant}_{mode}.log').open('x') as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                         env=env, start_new_session=True)
                while child.poll() is None:
                    peak = max(peak, _tree_rss(child.pid))
                    if peak > 8*1024**3: limit = 'MEMORY_LIMIT'
                    if time.monotonic() - start > 600: limit = 'TIME_LIMIT'
                    if limit:
                        os.killpg(child.pid, signal.SIGKILL)
                        break
                    time.sleep(.2)
                code = child.wait()
            result = {'variant': variant, 'mode': mode, 'returncode': code, 'limit': limit,
                      'wall_seconds': time.monotonic()-start, 'peak_tree_rss': peak}
            write(root / f'{variant}_{mode}_worker.json', result)
            results.append(result)
            print(json.dumps(result), flush=True)
            if code != 0:
                write(root / 'stage_result.json', {'status': 'STOP_LOCAL_GATE_FAILED',
                    'workers': results, 'board_contacted': False, 'automatic_retry': False})
                return
    verify_bindings(bindings)
    # Both observation modes must generate the same descriptors, dependency bits,
    # and relative DRAM addresses. Compare actual outputs after decoding.
    mode_rows = []
    for variant in 'BNG':
        for index in range(1, 8):
            records = []
            for mode in ('detailed', 'minimal'):
                text = (root / f'{variant}_{mode}/call_{index:02d}_capture.txt').read_text().split(END)[1]
                records.append(decode_submissions(text)[0])
            # FSim allocates deterministic virtual physical addresses in fresh
            # processes; require exact fields instead of normalizing real data.
            require([(r['bits'], canonical_payload(r)) for r in records[0]] ==
                    [(r['bits'], canonical_payload(r)) for r in records[1]], 'MODE_ENCODING_CHANGED')
            mode_rows.append({'variant': variant, 'call_index': index, 'exact_encoding_equal': True})
    write(root / 'mode_equivalence.json', mode_rows)
    write(root / 'stage_result.json', {'status': 'LOCAL_CARRIER_PREEXPERIMENT_COMPLETE',
        'workers': results, 'new_FSim_calls': 42, 'FPGA_calls': 0, 'board_contacted': False,
        'performance_conclusion': None, 'paused': True})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', choices=tuple('BNG'))
    parser.add_argument('--mode', choices=('detailed', 'minimal'))
    args = parser.parse_args()
    try:
        if args.worker: worker(args.output.resolve(), args.worker, args.mode)
        else: run(args.output.resolve())
    except Exception:
        traceback.print_exc()
        if not args.worker and args.output.exists() and not (args.output / 'stage_result.json').exists():
            write(args.output / 'stage_result.json', {'status': 'STOP_LOCAL_EXCEPTION',
                'traceback': traceback.format_exc(), 'board_contacted': False, 'automatic_retry': False})
        raise
