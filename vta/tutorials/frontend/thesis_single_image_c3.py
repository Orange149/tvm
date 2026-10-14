#!/usr/bin/env python3
"""One bounded current-graph residency proposal; no unseen-online claim."""
import argparse
import json
import math
import statistics
from pathlib import Path
import tarfile
import tvm
from tvm import autotvm, relay
from tvm.autotvm.task.topi_integration import TaskExtractEnv
from tvm.autotvm.task.relay_integration import _lower
import vta
from vta.top.residency_dispatch import ExplicitResidencyDispatch, _json_value
from run_thesis_single_image_smoke import SSH, checked
from thesis_supplement_run import RunStore, sha256, write_new_json
from thesis_single_image_integrated import make_package, make_anchor, deploy_controls, make_q1, qualify, boot_identity
from thesis_single_image_native import native
from build_vta_resnet50_generic_residency_tir_audit import FullTIRArchive
from build_vta_resnet50_fused_program_pool import save_snapshots
from deploy_classification_stage_pipeline_native import export_stage_lib


def collect_packed_workloads(mod, target, history):
    # Capture registered calls during actual fused lowering. Standalone Task
    # construction in this tree splits the bare conv output twice; it is not
    # necessary to instantiate a new space when binding complete cached entities.
    tracing = TaskExtractEnv.get()
    tracing.reset()
    relay.backend.te_compiler.get().clear()
    with autotvm.apply_history_best([str(history)]), tracing:
        _lower(mod, target, {})
    return [(name,) + args for name, args in tracing.get_tasks() if name == 'conv2d_packed.vta']


def logical_flops(workload):
    data, weight = workload[1][1], workload[2][1]
    stride, padding, dilation = workload[3], workload[4], workload[5]
    if len(padding) != 4 or list(dilation) != [1, 1]:
        raise ValueError('Unsupported static convolution geometry')
    h = (data[2] + padding[0] + padding[2] - weight[2]) // stride[0] + 1
    w = (data[3] + padding[1] + padding[3] - weight[3]) // stride[1] + 1
    return 2 * math.prod([data[0], data[4], h, w, weight[0], weight[4], data[1], data[5], weight[2], weight[3]])


def build(store):
    r = store.directory
    selected = json.loads((r / 'analysis/B_selection.json').read_text())
    stage = next(stage for stage in selected['stages'] if stage['device'] == 'vta')
    source = r / 'raw' / stage['directory'] / 'lowering_qir.json'
    if sha256(source) != stage['files']['lowering_qir.json']['sha256']:
        raise ValueError('B lowering program drifted')
    history = r / 'raw/stock_compatible_history.log'
    if sha256(history) != store.contract['baseline_history_sha256']:
        raise ValueError('Baseline Tile configurations drifted')
    mod = tvm.ir.load_json(source.read_text())
    env = vta.get_env()
    target = tvm.target.Target(env.target, host=env.target_host)
    relay.backend.te_compiler.get().clear()
    workloads = sorted(collect_packed_workloads(mod, target, history),
                       key=lambda workload: (-logical_flops(workload), str(workload)))
    routes = []
    excluded = []
    with autotvm.apply_history_best([str(history)]) as context:
        for index, workload in enumerate(workloads):
            cfg = context.query(target, workload)
            if cfg.is_fallback or not hasattr(cfg, 'to_json_dict'):
                excluded.append({'workload': _json_value(workload),
                    'reason': 'no complete stock ConfigEntity; no invented configuration'})
                continue
            config = _json_value(cfg.to_json_dict())
            config.pop('index', None)
            if cfg['h_nthread'].val != 1:
                excluded.append({'workload': _json_value(workload),
                    'reason': 'current production input-stationary schedule requires h_nthread=1; Tile configuration fixed'})
                continue
            if len(routes) >= 3:
                continue
            routes.append({'candidate_id': 'current_input_stationary_' + str(index), 'identity': {
                'workload': _json_value(workload), 'complete_config_entity': config,
                'public_mode': 'input_stationary', 'implementation_mode': 1}})
    write_new_json(r / 'raw/c3_route_contract.json', {
        'status': 'frozen_before_residency_build_or_new_residency_performance_labels',
        'scope': 'one joint input-stationary proposal on up to three current B conv workloads; engineering smoke',
        'candidate_program_budget': 1, 'routes': routes, 'excluded_workloads': excluded,
        'task_selection': 'static input-mode legality first, then descending logical FLOPs, tie workload; at most three unique tasks',
        'baseline_history_sha256': sha256(history), 'B_lowering_sha256': sha256(source),
        'canonical_qir_sha256': store.contract['canonical_qir_sha256'], 'extra_quantize_calls': 0,
        'unseen_online_evidence': False, 'historical_model_outputs_used': False,
        'weight_and_joint_modes': 'outside this minimal proposal space; no full C3 or mode5 benefit claim',
        'deployment_gate': 'seven paired full-model serial observations; D first in rounds0..3, proposal first in4..6; five warmup per fresh context; seven of seven faster and strict median improvement',
        'fallback_rule': 'static build/ineligible rejection => E=D with costs; board timeout/numeric fault stops this run',
        'dispatch_source_sha256': sha256(Path(vta.__file__).parent / 'top/residency_dispatch.py')})
    if not routes:
        write_new_json(r / 'analysis/c3_route_rejection.json', {'reason': 'no current compatible complete stock routes',
            'board_candidate_calls': 0, 'all_search_costs_retained': True})
        return
    local = r / 'raw/c3_program'
    local.mkdir()
    (local / 'lowering_qir.json').write_bytes(source.read_bytes())
    relay.backend.te_compiler.get().clear()
    archive = FullTIRArchive(env)
    try:
        with autotvm.apply_history_best([str(history)]), ExplicitResidencyDispatch(routes) as dispatch:
            with vta.build_config(opt_level=3, disabled_pass={'AlterOpLayout', 'tir.CommonSubexprElimTIR'},
                                  instruments=[archive]):
                factory = relay.build(mod, target=target)
        audit = dispatch.summary()
        if not audit['all_routes_scheduled'] or not archive.snapshots:
            raise RuntimeError('Exact routes did not all reach final residency scheduling')
        (local / 'graph.json').write_text(factory.get_graph_json())
        (local / 'params.bin').write_bytes(relay.save_param_dict(factory.get_params()))
        export_stage_lib(factory.get_lib(), local / 'module.so')
    except Exception as error:
        write_new_json(r / 'analysis/c3_route_rejection.json', {'reason': str(error),
            'phase': 'static_build', 'board_candidate_calls': 0, 'all_search_costs_retained': True})
        print('C3 proposal static rejection retained; eligible fallback E=D', flush=True)
        return
    result = dict(stage, name='c3_' + stage['name'], directory='c3_program',
        files={filename: {'sha256': sha256(local / filename), 'bytes': (local / filename).stat().st_size}
               for filename in ('graph.json', 'params.bin', 'module.so', 'lowering_qir.json')})
    write_new_json(local / 'program_manifest.json', {'program': result, 'dispatch_audit': audit,
        'snapshots': save_snapshots(local, archive), 'extra_quantize_calls': 0})
    print('Current B residency proposal built, exact routes scheduled, new quantization=0', flush=True)


def integrate(store):
    r = store.directory
    base = json.loads((r / 'raw/ABCCaD_conditions.json').read_text())
    if (r / 'analysis/c3_route_rejection.json').exists():
        condition = dict(base['D'], c3_status='E_equals_D_static_proposal_rejected',
            c3_search_contract_sha256=sha256(r / 'raw/c3_route_contract.json'))
        qualify(store, 'E', condition)
        write_new_json(r / 'raw/frozen_six_conditions.json', {**base, 'E': condition})
        return
    record = json.loads((r / 'raw/c3_program/program_manifest.json').read_text())
    stage = record['program']
    archive = r / 'raw/c3_program_bundle.tgz'
    with tarfile.open(archive, 'w:gz') as tar:
        tar.add(r / 'raw/c3_program', arcname='c3_program')
    board = json.loads((r / 'analysis/board_directory.json').read_text())['path']
    checked(store, SSH + ['cat > ' + board + '/' + archive.name], 'c3_program_transfer', 'one_time_preparation', 60, archive)
    script = r / 'raw/c3_program_verify.sh'
    lines = ['set -eu', 'cd ' + board, 'tar -xzf ' + archive.name]
    for name, value in stage['files'].items():
        lines.append('test "$(sha256sum c3_program/' + name + ' | cut -d \' \' -f 1)" = ' + value['sha256'])
    script.write_text('\n'.join(lines) + '\n')
    checked(store, SSH + ['sh -s'], 'c3_program_unpack_verify', 'one_time_preparation', 60, script)
    stages = [stage if old['device'] == 'vta' else old for old in base['D']['stages']]
    # First qualify the new program before any performance/profile measurements.
    qualify(store, 'E_proposal_copy', {'stages': stages, 'threads': base['D']['threads'], 'managed': False})
    package = make_package(store, 'E', stages, 'c3_current_input_stationary')
    anchor = make_anchor(store, 'E', package)
    deploy_controls(store, 'E')
    provisional = {'stages': stages, 'threads': base['D']['threads'], 'managed': True, 'anchor': anchor}
    qualify(store, 'E_proposal_anchor', provisional)
    capacity = make_q1(store, 'E', package, stages, provisional['threads'], anchor)
    candidate = {**provisional, 'capacity': capacity,
        'c3_status': 'current_graph_input_stationary_qualified_smoke_only',
        'c3_search_contract_sha256': sha256(r / 'raw/c3_route_contract.json'),
        'deployment_policy': 'seven of seven full-model pairs faster and strict median improvement; engineering guard, not significance'}
    qualify(store, 'E_candidate', candidate)
    boot, index = boot_identity(r)
    pairs = []
    for round_index in range(7):
        order = ['D', 'candidate'] if round_index < 4 else ['candidate', 'D']
        observed = {}
        for label in order:
            variant = base['D'] if label == 'D' else candidate
            identity = f'b{index}_C3_gate_r{round_index}_{label}'
            value = native(store, identity, variant['stages'], variant['threads'], index, boot,
                'gate_serial', 6, managed=True, anchor_plan=variant['anchor'], capacity=variant['capacity'])
            observed[label] = value['gate_latency_ms']
        pairs.append({'round': round_index, 'order': order, 'latency_ms': observed,
            'ratio_candidate_over_D': observed['candidate'] / observed['D'],
            'candidate_faster': observed['candidate'] < observed['D']})
    passed = all(pair['candidate_faster'] for pair in pairs) and statistics.median(
        pair['latency_ms']['candidate'] for pair in pairs) < statistics.median(pair['latency_ms']['D'] for pair in pairs)
    write_new_json(r / 'analysis/c3_current_graph_deployment_gate.json', {'pairs': pairs, 'passed': passed,
        'scope': 'conservative engineering admission, one image; not a significance test',
        'all_search_deployment_costs_retained': True})
    condition = candidate if passed else dict(base['D'],
        c3_status='E_equals_D_deployment_gate_rejected',
        c3_search_contract_sha256=sha256(r / 'raw/c3_route_contract.json'),
        rejected_candidate='raw/condition_controls/E', qualification='full numeric pass, seven-pair performance guard not passed')
    qualify(store, 'E', condition)
    write_new_json(r / 'raw/frozen_six_conditions.json', {**base, 'E': condition})
    print('E: ' + condition['c3_status'] + '; serial/pipeline output qualification PASS', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('build', 'integrate'))
    parser.add_argument('run_dir', type=Path)
    args = parser.parse_args()
    store = RunStore(args.run_dir)
    if (store.directory / 'invalid_session.json').exists():
        raise ValueError('Invalid run')
    try:
        build(store) if args.operation == 'build' else integrate(store)
    except BaseException as error:
        write_new_json(store.directory / 'invalid_session.json', {'operation': 'c3_' + args.operation,
            'reason': str(error), 'all_paid_costs_retained': True, 'stop_later_conditions': True})
        store.event('single_image_run_invalid', operation='c3_' + args.operation, reason=str(error))
        raise
