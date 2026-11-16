"""Prepare a fixed A/C comparison using the qualified operator executor.

A is the sealed identity-reuse program; C is the preselected broader-domain
TPS baseline. The reused two-slot ledger calls its slots TH/P internally:
TH=C and P=A here. Neither slot is a TopHub program in this experiment.
No historical performance sample participates in the new comparison.
"""
import argparse
import json
from pathlib import Path
import struct
import sys
import traceback

from c3_tps_sync_release_local import HERE, REPO, HOLD, read, write, sha, verify, require
from c3_yolo416_comparison_protocol import manifest, analyse
from c3_yolo416_board_transport import transport_contract

LOCAL = HOLD/'20261017_c3_tps_broader_baseline_local_run01'
SLOTS = {'TH': 'C', 'P': 'A'}


def bound(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path)}


def bind_seal(root, bindings):
    seal = read(root/'completion_manifest.json')
    values = {str(root/k): v for k, v in seal['artifact_sha256'].items()}
    verify(values)
    bindings.update(values)
    bindings[str(root/'completion_manifest.json')] = sha(root/'completion_manifest.json')
    return seal


def pool_entry(node, programs):
    require(set(programs) == {'A', 'C'}, 'EXACT_A_C_PAIR_REQUIRED')
    require(all(p['actual_variant'] == key for key, p in programs.items()), 'VARIANT_IDENTITY_MISMATCH')
    return {'object_id': 'yolo224_node%d' % node, 'kind': 'operator',
            'programs': {slot: programs[variant] for slot, variant in SLOTS.items()},
            'representatives': {'TH': 'TH', 'P': 'P'}}


def arm_build(source, output):
    import tvm, vta
    from c3_tps_sync_release_prepare import complete_function
    from c3_official_window_llvm import llvm_alu_contract
    from c3_tps_aggregation_prepare import compare_initializer_bindings
    from build_vta_yolov3_tiny_relay_residency_dispatch_pair import cross_compiler
    output.mkdir(parents=True, exist_ok=False)
    final = tvm.ir.load_json((source/'final.json').read_text())
    gv = list(final.functions)[0]
    symbol = str(final[gv].attrs['global_symbol'])
    _, fn = complete_function(final, symbol)
    llvm_fsim = (source/'module.ll').read_text()
    sim = llvm_alu_contract(fn, llvm_fsim)
    require(sim == read(source/'alu_initializer_contract.json'), 'QUALIFIED_FSIM_INITIALIZERS_CHANGED')
    with vta.Environment(dict(vta.get_env().cfg_dict, TARGET='axu5evb')):
        target = tvm.target.Target(vta.get_env().target, host=vta.get_env().target_host)
        with tvm.transform.PassContext(opt_level=3, disabled_pass={'AlterOpLayout', 'tir.CommonSubexprElimTIR'}):
            lib = tvm.build(final, target=target)
        llvm = lib.get_source('ll')
        require('target triple = "aarch64-linux-gnu"' in llvm, 'ARM_TARGET_TRIPLE')
        arm = llvm_alu_contract(fn, llvm)
        comparison = compare_initializer_bindings(llvm_fsim, llvm, sim, arm)
        lib.export_library(str(output/'module.so'), fcompile=cross_compiler())
    blob = (output/'module.so').read_bytes()
    require(blob[:6] == b'\x7fELF\x02\x01' and struct.unpack_from('<H', blob, 18)[0] == 183, 'AARCH64_ELF')
    (output/'module.ll').write_text(llvm)
    write(output/'contract.json', {'status': 'ARM_INITIALIZERS_MATCH_QUALIFIED_FSIM',
          'initializer_comparison': comparison, 'source_final': bound(source/'final.json'),
          'module': bound(output/'module.so'), 'llvm': bound(output/'module.ll')})
    return output/'module.so'


def prepare(root, host, known_hosts):
    import numpy as np
    from c3_tps_aggregation_board import make_cases
    from c3_official_window_reference import array_sha
    root.mkdir(exist_ok=False)
    try:
        bindings = {}
        require(bind_seal(LOCAL, bindings)['status'] == 'TWO_BROADER_TPS_BASELINES_LOCAL_QUALIFIED',
                'BOTH_C_BASELINES_MUST_BE_QUALIFIED')
        bindings.update(read(LOCAL/'freeze.json')['bindings'])
        programs, pool, common = {}, [], None
        for node in (34, 46):
            old = HOLD/('20261017_c3_node%d_uop_identity_board_run01' % node)
            bind_seal(old, bindings)
            old_artifacts = read(old/'artifacts/prepared.json')
            keys = ('runtime', 'helper', 'storage_helper', 'runtime_tvm', 'rpc_server', 'bitstream', 'hardware_archive')
            current_common = {k: old_artifacts[k] for k in keys}
            if common is None: common = current_common
            require(all(common[k]['sha256'] == current_common[k]['sha256'] for k in keys),
                    'HISTORICAL_A_COMMON_RUNTIME_OR_HELPER_DIFFERS')
            a = old_artifacts['programs']['A']
            verify({a[k]: v for k, v in a['sha256'].items()})
            for p, value in {a[k]: v for k, v in a['sha256'].items()}.items(): bindings[p] = value
            a_unit = Path(a['qualified_directory']).parents[1]
            bind_seal(a_unit, bindings)
            c_unit = LOCAL/('node%d' % node)
            d = read(c_unit/'prepare/result.json')['programs']['R']
            verify(d['artifact_bindings'])
            reference = read(d['reference_contract'])
            require(reference == read(a['reference_contract']) and d['abi'] == a['abi'], 'A_C_FUSION_SEMANTICS_DIFFER')
            arm = arm_build(c_unit/'prepare/R', root/('node%d_C_arm' % node))
            cases, ids = make_cases(reference)
            require(ids['0:W0']['reference'] != ids['0:W1']['reference'], 'WEIGHT_CHECK_NOT_OBSERVABLE')
            folder = root/('node%d_cases' % node)
            folder.mkdir()
            case_rows = []
            for (seed, weight), (arrays, expected) in cases.items():
                path = folder/('%d_%s.npz' % (seed, weight))
                require(np.all(arrays[-1] != expected), 'SENTINEL_EQUALS_REFERENCE')
                np.savez(path, **{'a%d' % i: a for i, a in enumerate(arrays)}, expected=expected)
                case_rows.append({'seed': seed, 'weight': weight, 'path': str(path),
                                  'input_sha256': [array_sha(a) for a in arrays[:3]],
                                  'expected_sha256': array_sha(expected)})
            model_cases = {'cases': case_rows, 'origin': 'same independent seeded cases as historical A/B protocol; not trained graph intermediates'}
            object_id = 'yolo224_node%d' % node
            identities = {}
            for slot, variant in SLOTS.items():
                is_a = variant == 'A'
                module = Path(a['arm_module']) if is_a else arm
                unit = a_unit if is_a else c_unit
                parent = unit/'prepare/R/parent.json'
                final = unit/'prepare/R/final.json'
                descriptor = {'kind': 'operator', 'variant': slot, 'actual_variant': variant,
                    'object_id': object_id, 'symbol': d['symbol'], 'abi': d['abi'],
                    'arm_module': str(module), 'parent_module': str(parent), 'local_unit': str(unit),
                    'model_cases': model_cases, 'execution_identity': {'module_sha256': sha(module),
                                                                        'final_IR_sha256': sha(final)}}
                programs[object_id+'/'+slot] = descriptor
                identities[variant] = {'actual_variant': variant, 'module': bound(module),
                    'source_final': bound(final), 'source_local': str(unit), 'abi': d['abi'],
                    'semantic_reference_sha256': sha(d['reference_contract']),
                    'execution_identity': descriptor['execution_identity']}
            pool.append(pool_entry(node, identities))
        for value in common.values():
            require(sha(value['path']) == value['sha256'], 'COMMON_RUNTIME_HARDWARE_ARTIFACT_CHANGED')
            bindings[value['path']] = value['sha256']
        artifacts = {**common, 'programs': programs}
        write(root/'artifacts.json', artifacts)
        ledger = manifest(root.name, pool)
        require(ledger['total_target_calls'] == 240 and len(ledger['sessions']) == 40 and
                ledger['formal_samples'] == 72, 'FIXED_A_C_COMPARISON_BUDGET')
        write(root/'call_manifest.json', ledger)
        write(root/'transport_contract.json', transport_contract(ledger))
        write(root/'comparison_semantics.json', {'internal_slot_mapping': SLOTS,
              'no_TopHub_program_in_this_comparison': True, 'network': 'YOLOv3-tiny-224',
              'nodes': [34, 46], 'A': 'sealed oc2/h1 TPS plus cache-identity reuse',
              'C': 'preselected broader-thread-domain TPS, with no identity optimization',
              'A_over_C_is_competitiveness_not_isolated_identity_effect': True,
              'historical_B_A_times_excluded_from_all_new_statistics': True,
              'six_rounds_three_seeds_once_per_round': True,
              'stop_on_error_no_retry_no_replacement': True,
              'no_SD_write_reboot_or_external_service_termination': True})
        for p in [*HERE.glob('c3_*.py'), HERE/'test_c3_tps_broader_baseline_compare.py', Path(known_hosts)]:
            bindings[str(p.resolve())] = sha(p)
        for p in root.rglob('*'):
            if p.is_file(): bindings[str(p.resolve())] = sha(p)
        verify(bindings)
        write(root/'freeze.json', {'status': 'TH_P_OPERATORS_AND_GRAPH_READY_FPGA_PENDING',
              'actual_comparison': 'C_vs_A_NO_TOPHUB', 'bindings': bindings, 'host': host,
              'known_hosts': str(Path(known_hosts).resolve()), 'FPGA_calls': 0,
              'target_calls': 240, 'formal_samples': 72, 'sessions': 40})
        print(json.dumps({'status': 'FROZEN_C_A_COMPARISON_READY', 'FPGA_calls': 0, 'planned_calls': 240}), flush=True)
        return 0
    except Exception as error:
        write(root/'prepare_failure.json', {'type': type(error).__name__, 'message': str(error), 'FPGA_calls': 0})
        traceback.print_exc()
        return 1


def report(root):
    from c3_yolo416_board_execute import accounting
    require(not (root/'completion_manifest.json').exists(), 'COMPARISON_ALREADY_SEALED')
    summary = read(root/'board/summary.json')
    success = summary['status'] == 'COMPLETE_TH_P_OPERATORS_AND_GRAPH_SINGLE_BOOT_ATTEMPT'
    account = accounting(root/'board')
    write(root/'call_accounting.json', account)
    result = {'status': 'C_A_COMPARISON_COMPLETE' if success else 'C_A_COMPARISON_STOPPED',
              'success': success, 'actual_variants': SLOTS, 'no_TopHub_program_in_this_comparison': True,
              'accounting': account, 'historical_performance_samples_reused': 0}
    if success:
        verify(read(root/'freeze.json')['bindings'])
        rows = [json.loads(line) for line in (root/'board/results.jsonl').read_text().splitlines()]
        stats = analyse(read(root/'call_manifest.json'), rows)
        result['statistics_legacy_slot_names'] = stats
        result['comparisons'] = []
        for item in stats['objects']:
            result['comparisons'].append({'object_id': item['object_id'],
                'T_ms': {SLOTS[k]: v for k, v in item['T_ms'].items()},
                'round_medians_ms': {SLOTS[k]: v for k, v in item['round_medians_ms'].items()},
                'A_over_C_gain_percent': item['Gain_P_over_TH_percent'],
                'A_over_C_round_gains_percent': item['round_gains_percent'],
                'A_strict_wins_over_C': item['strict_wins'],
                'A_saves_over_C_median_ms': item['median_absolute_ms'],
                'gain_MAD_percent': item['gain_MAD_percent']})
        require(account['certified'] == 240 and not account['execution_unresolved_units'] and not account['invalid_records'],
                'COMPARISON_ACCOUNTING_INCOMPLETE')
    write(root/'result.json', result)
    write(root/'completion_manifest.json', {'status': result['status'], 'success': success, 'paused': True,
          'artifact_sha256': {str(p.relative_to(root)): sha(p) for p in root.rglob('*') if p.is_file()}})
    print(json.dumps({k: v for k, v in result.items() if k in ('status', 'comparisons')}, ensure_ascii=False), flush=True)
    return 0 if success else 1


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=['prepare', 'board', 'report'])
    p.add_argument('output', type=Path)
    p.add_argument('--host', default='192.168.1.247')
    p.add_argument('--known-hosts', default='/home/orange/.ssh/known_hosts')
    args = p.parse_args()
    if args.phase == 'prepare':
        sys.exit(prepare(args.output.resolve(), args.host, args.known_hosts))
    if args.phase == 'board':
        from c3_yolo416_board_execute import execute
        sys.exit(execute(args.output.resolve()))
    sys.exit(report(args.output.resolve()))
