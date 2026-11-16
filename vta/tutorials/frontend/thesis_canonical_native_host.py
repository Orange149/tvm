#!/usr/bin/env python3
"""Native CPU GraphExecutor qualification of once-quantized canonical stages."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time
import uuid

import numpy as np

from thesis_supplement_run import RunStore, sha256, write_new_json
from thesis_supplement_audit import reconcile, seal

FRONTEND = Path(__file__).resolve().parent
ROOT = FRONTEND.parents[2]
BASE = FRONTEND / 'report_out/stage_tile_cotuning/thesis_supplement_experiments'


def qualify(preparation, retry_of=None):
    qualification = preparation / 'raw/canonical_host_qualification_01'
    interfaces_path = preparation / 'raw/canonical_interface_check_01/native_transport_interface.json'
    interfaces = json.loads(interfaces_path.read_text())['stage_interfaces']
    lookup = {(s['begin'], s['end_exclusive']): s for s in interfaces}
    manifest = json.loads((qualification / 'artifact_hashes.json').read_text())
    for relative, expected in manifest.items():
        path = qualification / relative
        if sha256(path) != expected['sha256']:
            raise ValueError('Changed host-qualified artifact: ' + str(path))
    source_paths = [
        ROOT / 'vta/apps/native_deploy/vta_stage_pipeline_runner.cc',
        ROOT / 'vta/apps/native_deploy/vta_shared_buffer_plan.h',
        ROOT / 'vta/apps/native_deploy/vta_managed_lifecycle_control.h',
        ROOT / '3rdparty/picojson/picojson.h',
        ROOT / '3rdparty/vta-hw/include/vta/driver.h',
        FRONTEND / 'thesis_native_host_driver_guard.cc',
        FRONTEND / 'thesis_native_carry_probe.cc',
    ]
    ranges = {'full': [(0, 21)], 'integrated': [(0, 3), (3, 18), (18, 21)],
              'units': [(i, i + 1) for i in range(21)]}
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_canonical_native_host_' + uuid.uuid4().hex[:8]
    directory = BASE / 'results/02' / run_id
    store = RunStore.create(directory, {
        'protocol_version': 'thesis_supplement_v1_20260916_independent_native_cpu',
        'campaign_id': json.loads((BASE / 'results/ACTIVE_CAMPAIGN.json').read_text())['campaign_id'],
        'stage': '02', 'run_id': run_id, 'retry_of': retry_of,
        'purpose': 'native完整单图/3段/21单元合成语义预资格',
        'scope': 'host_native_cpu_prequalification_only', 'formal_phase02_completed': False,
        'phase01_certificate': None, 'real_dataset': None, 'seed': [59001, 59002],
        'inputs': 'reuse exact frozen zeros/uniform/normal float32 arrays; do not redraw RNG',
        'model': 'pretrained ResNet18_v1@224 from common once-quantized qIR',
        'quantize_calls_this_run': 0,
        'host_qualification_manifest_sha256': sha256(qualification / 'artifact_hashes.json'),
        'native_interface_sha256': sha256(interfaces_path),
        'native_sources': {str(p.relative_to(ROOT)): sha256(p) for p in source_paths},
        'controller_sha256': sha256(Path(__file__)),
        'host_runtime_sha256': sha256(ROOT / 'build/libtvm_runtime.so'),
        'protocol_sha256': sha256(BASE / '00_GLOBAL_PROTOCOL.md'),
        'condition_order': ['full-copy', 'full-input-zero-copy', 'integrated-copy',
                            'integrated-input-zero-copy', 'units-copy', 'units-input-zero-copy'],
        'full_native_calls': 18, 'carry_probe_calls': 12,
        'carry_probes': 'units03/06 x3 frozen inputs x naive/routed; original input pools complemented',
        'driver_guard': 'host-only link symbols exit86 on any VTA memory/address call; no simulator',
        'expected_naive_wrong_cases': 6, 'expected_routed_correct_cases': 6,
        'numeric': {'integer': 'byte exact', 'float_atol': 1e-5, 'float_rtol': 1e-4,
                    'copy_vs_input_zero_copy': 'full byte exact'},
        'warmup': 0, 'repeats': 1, 'CPU_threads': 1, 'hardware_calls': 0,
        'compile_timeout_seconds': 60, 'native_action_timeout_seconds': 60,
        'session_budget_seconds': 300,
        'timing': 'functional action walls only; no inference performance labels or online T0',
        'output_ownership': 'retain actual incoming buffer for live-input heads; snapshot all computed outputs',
        'stop_rule': 'unexpected action/bytes/shape error invalidates run; preserve all paid costs; retry new run',
        'source_costs': 'previous qIR/LLVM build costs remain charged to prior run, not claimed free cold preparation',
        'non_applicable': ['board DMA/FINISH/coherence', 'managed slot qualification',
                           'real-image accuracy', 'five boot performance', 'C3 unseen targets'],
    })
    print(json.dumps({'run_started': str(directory)}), flush=True)
    run_begin = time.monotonic()
    def execute(argv, action, cost_class, timeout):
        remaining = 300 - (time.monotonic() - run_begin)
        if remaining <= 0:
            raise TimeoutError('Native host session budget exceeded')
        return store.execute(argv, action, cost_class, min(timeout, remaining))
    try:
        preparation_begin = time.monotonic_ns()
        store.event('native_materials_prepare_start', cost_class='functional_qualification')
        native_dir = directory / 'raw/source/vta/apps/native_deploy'
        native_dir.mkdir(parents=True)
        for path in source_paths:
            relative = path.relative_to(ROOT)
            target = directory / 'raw/source' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            if sha256(target) != store.contract['native_sources'][str(relative)]:
                raise ValueError('Source changed while being frozen')
        shutil.copyfile(FRONTEND / 'thesis_native_carry_probe.cc', native_dir / 'carry_probe.cc')
        shutil.copyfile(Path(__file__), directory / 'raw/controller.py')
        write_new_json(directory / 'raw/native_transport_interface.json', json.loads(interfaces_path.read_text()))
        compiled = directory / 'raw/compiled'
        compiled.mkdir()
        names = ['canonical_qir'] + [f'integrated_{a:02}_{b-1:02}' for a,b in ranges['integrated']] + [f'units_{a:02}_{a:02}' for a,b in ranges['units']]
        for name in names:
            shutil.copytree(qualification / 'compiled' / name, compiled / name)
        inputs_dir = directory / 'raw/inputs'
        inputs_dir.mkdir()
        for case in ('zeros', 'uniform', 'normal'):
            target = inputs_dir / case
            target.mkdir()
            with np.load(qualification / case / 'canonical_boundaries.npz', allow_pickle=False) as saved:
                for node, value in saved.items():
                    (target / (node + '.bin')).write_bytes(value.tobytes())
            values = np.load(qualification / case / 'input.npy', allow_pickle=False)
            (target / 'input.bin').write_bytes(values.tobytes())
            shutil.copyfile(qualification / case / 'input.npy', target / 'input.npy')
            shutil.copyfile(qualification / case / 'canonical_logits.npy', target / 'reference_logits.npy')
        preparation_seconds = (time.monotonic_ns() - preparation_begin) / 1e9
        write_new_json(directory/'analysis/material_preparation_cost.json', {
            'wall_seconds':preparation_seconds, 'cost_class':'functional_qualification',
            'scope':'source/library copies and frozen input/byte materialization',
            'charged_to_action_sum':False,
        })
        store.event('native_materials_prepare_end',wall_seconds=preparation_seconds,
                    cost_class='functional_qualification')
        include = ['-I', str(ROOT / 'include'), '-I', str(ROOT / '3rdparty/dlpack/include'),
                   '-I', str(ROOT / '3rdparty/dmlc-core/include'),
                   '-I', str(directory / 'raw/source/3rdparty/vta-hw/include')]
        guard = directory / 'raw/source/vta/tutorials/frontend/thesis_native_host_driver_guard.cc'
        common = ['g++', '-std=c++17', '-O1', '-g', *include]
        link = [str(guard), '-L', str(ROOT / 'build'), '-ltvm_runtime', '-ldl', '-pthread',
                '-Wl,-rpath,' + str(ROOT / 'build')]
        for source, name in [(native_dir / 'vta_stage_pipeline_runner.cc', 'native_runner'),
                             (native_dir / 'carry_probe.cc', 'carry_probe')]:
            result = execute([*common, str(source), *link, '-o', str(directory / 'raw' / name)],
                                   'compile_' + name, 'functional_qualification', 60)
            if result['status'] != 'ok':
                raise ValueError('Host native build failed; see full raw compiler diagnostics')
        checks, actual_calls = [], 0
        for kind, stage_ranges in ranges.items():
            for transport in ('copy', 'input_zero_copy'):
                output = directory / 'raw' / (kind + '_' + transport)
                output.mkdir()
                input_list = output / 'inputs.txt'
                input_list.write_text(''.join(str(inputs_dir / case / 'input.bin') + '\n'
                                             for case in ('zeros', 'uniform', 'normal')))
                argv = ['env', 'TVM_NUM_THREADS=1', str(directory / 'raw/native_runner'),
                        '--serial', '--runs', '3', '--warmup-runs', '0', '--runtime-num-threads', '1',
                        '--input-list', str(input_list), '--output-jsonl', str(output / 'result.jsonl'),
                        '--output-mode', 'raw_all_stages',
                        '--output-dump-dir', str(output / 'outputs')]
                if transport == 'input_zero_copy': argv.append('--cpu-input-zero-copy')
                for index, (begin, end) in enumerate(stage_ranges):
                    name = 'canonical_qir' if kind == 'full' else f'{kind}_{begin:02}_{end-1:02}'
                    folder = compiled / name
                    arity = 1 if kind == 'full' else len(lookup[(begin,end)]['input_node_ids'])
                    parameters = 'data' if kind == 'full' else ','.join(f'edge{begin}_{i}' for i in range(arity))
                    prefix = f'--stage{index}-'
                    argv += [prefix+'graph', str(folder/'graph.json'), prefix+'lib', str(folder/'module.so'),
                             prefix+'params', str(folder/'params.bin'), prefix+'device', 'cpu',
                             prefix+'input-names', parameters, prefix+'runtime-num-threads', '1']
                result = execute(argv, kind + '_' + transport, 'functional_qualification', 60)
                if result['status'] != 'ok': raise ValueError('Native execution failed')
                actual_calls += 3
                records = [json.loads(line) for line in (output/'result.jsonl').read_text().splitlines()]
                if len(records) != 3: raise ValueError('Wrong actual full-call count')
                for record, case in zip(records, ('zeros','uniform','normal')):
                    # Dump filenames are bound to completion index; determine final output from metadata.
                    final = record['raw_output_files'][0]
                    raw_bytes = Path(final['path']).read_bytes()
                    reference = np.load(inputs_dir / case / 'reference_logits.npy', allow_pickle=False)
                    if len(raw_bytes) != reference.nbytes: raise ValueError('Final output byte size changed')
                    actual = np.frombuffer(raw_bytes, dtype=reference.dtype).reshape(reference.shape)
                    if not np.allclose(actual, reference, atol=1e-5, rtol=1e-4):
                        raise ValueError('Native final output differs from common reference')
                    check = {'program': kind, 'transport': transport, 'input': case,
                             'final_bytes_exact': raw_bytes == reference.tobytes(),
                             'max_abs_error': float(np.max(np.abs(actual-reference))), 'boundary_checks': []}
                    if kind != 'full':
                        with np.load(qualification/case/'canonical_boundaries.npz', allow_pickle=False) as refs:
                            for index, (begin,end) in enumerate(stage_ranges):
                                stage_record = record['stage_raw_outputs'][index]['output_files']
                                if len(stage_record)!=len(lookup[(begin,end)]['output_sources']):
                                    raise ValueError('Native output bundle arity changed')
                                for item, source in zip(stage_record, lookup[(begin,end)]['output_sources']):
                                    expected = refs[f'node_{source["node_id"]}']
                                    emitted = Path(item['path']).read_bytes()
                                    if expected.dtype.kind in 'iu' and emitted != expected.tobytes():
                                        raise ValueError('Native integer live boundary mismatch')
                                    check['boundary_checks'].append({'stage':index,'node_id':source['node_id'],
                                                                     'integer':expected.dtype.kind in 'iu',
                                                                     'bytes_exact': emitted==expected.tobytes()})
                    checks.append(check)
        negative, routed = [], []
        for begin in (3,6):
            interface = lookup[(begin,begin+1)]
            folder = compiled/f'units_{begin:02}_{begin:02}'
            parameters = ','.join(f'edge{begin}_{i}' for i in range(len(interface['input_node_ids'])))
            for case in ('zeros','uniform','normal'):
                paths = ','.join(str(inputs_dir/case/f'node_{node}.bin') for node in interface['input_node_ids'])
                with np.load(qualification/case/'canonical_boundaries.npz', allow_pickle=False) as refs:
                    for mode in ('naive','routed'):
                        out = directory/'raw'/f'carry_{begin:02}_{case}_{mode}'
                        result = execute(['env','TVM_NUM_THREADS=1',str(directory/'raw/carry_probe'),
                                                str(folder/'graph.json'),str(folder/'module.so'),str(folder/'params.bin'),
                                                paths,str(out),mode,parameters], f'carry_{begin}_{case}_{mode}',
                                               'functional_qualification',60)
                        if result['status'] != 'ok': raise ValueError('Native carry probe failed')
                        wrong = []
                        for i, source in enumerate(interface['output_sources']):
                            expected = refs[f'node_{source["node_id"]}']
                            value = np.frombuffer((out/f'output{i}.bin').read_bytes(), dtype=expected.dtype).reshape(expected.shape)
                            count = int(np.count_nonzero(value != expected))
                            if mode == 'routed' and count: raise ValueError('Native live-input route wrong')
                            if source['kind']=='live_input': wrong.append(count)
                            elif count: raise ValueError('Computed output affected by pool poison')
                        entry = {'unit':begin,'input':case,'carry_mismatching_elements':wrong}
                        (negative if mode=='naive' else routed).append(entry)
                        if mode=='naive' and not all(wrong): raise ValueError('Naive negative control did not fail')
        assert actual_calls == 18 and len(negative)==6 and len(routed)==6
        for kind in ranges:
            for case in ('zeros','uniform','normal'):
                rows=[c for c in checks if c['program']==kind and c['input']==case]
                if not all(c['final_bytes_exact'] for c in rows):
                    raise ValueError('CPU final copy/input-zero-copy not byte exact with reference')
        write_new_json(directory/'analysis/native_checks.json',checks)
        write_new_json(directory/'analysis/carry_controls.json',{'naive':negative,'routed':routed})
        summary={'status':'NATIVE_CPU_PREQUALIFICATION_PASS','formal_phase02_completed':False,
                 'actual_full_native_calls':actual_calls,'actual_carry_probe_calls':12,
                 'actual_hardware_calls':0,'real_images':0,'accuracy_measured':False,
                 'boundary_output_checks':sum(len(c['boundary_checks']) for c in checks),
                 'integer_boundary_checks':sum(b['integer'] for c in checks for b in c['boundary_checks']),
                 'naive_wrong_cases':len(negative),'routed_correct_cases':len(routed)}
        write_new_json(directory/'summary.json',summary)
        store.event('native_cpu_prequalification_verified',**summary)
    except Exception as error:
        write_new_json(directory/'invalid_session.json',{'reason':str(error),'all_costs_preserved':True})
        completed_rows = sum(len(p.read_text().splitlines())
                             for p in directory.glob('raw/*/result.jsonl'))
        started_probes = len(list(directory.glob('raw/carry_*_*.json')))
        write_new_json(directory/'summary.json',{'status':'FAILED_NO_CERTIFICATE','reason':str(error),
                                                'formal_phase02_completed':False,'actual_hardware_calls':0,
                                                'completed_full_native_result_rows':completed_rows,
                                                'recorded_carry_probe_actions':started_probes})
        store.event('native_cpu_prequalification_failed',reason=str(error))
        success=False
    else:
        success=True
    cost=reconcile(directory)
    write_new_json(directory/'analysis/cost_audit.json',cost)
    (directory/'REPORT.md').write_text(
        '# Native canonical CPU 本机预资格\n\n'
        + ('18次完整native输出及12次carry对照通过。' if success else '运行失败，原因见summary及invalid_session。')
        + '\n\n全图、三段及21单元使用同一已冻结一次量化qIR的CPU LLVM库；不重新量化或调优。'
        '真实native入口分别执行copy和CPU输入零拷贝，计算输出始终快照，live-input输出保留真正incoming buffer。'
        'carry探针直接调用生产LoadStage/RunStage，原输入池逐字节取反，naive对照只清除推断的live-input映射，'
        '不替换计算图或算子。原始bin与NPY参考完整保留，可复算整数与最终bytes。\n\n'
        'host link guard在任何VTA内存/地址调用时exit86，提供不了设备或模拟器；全部路线CPU-only。'
        '实际DMA、FINISH、managed slot复制、两boot及1000图任务准确率未验证。'
        'native managed实现新增live-input→outgoing slot复制后统一Publish，复制字节计入trace，'
        '但本轮CPU普通执行不认证该板端路径。正式02仍未完成。\n\n'
        f'实测动作成本{cost["measured_action_total_seconds"]:.6f}s；前期qIR/LLVM生成成本仍归原run，'
        'material_preparation_cost单列已测准备成本，不重复计入action_sum；'
        '开发成本unknown/null，不产生正式性能标签。合同、源码、输入、完整输出与所有失败均保留。\n')
    seal(directory)
    print(json.dumps({'run_dir':str(directory),'status':'PASS' if success else 'FAILED'},ensure_ascii=False))
    return 0 if success else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation-run',type=Path,required=True)
    parser.add_argument('--retry-of')
    args=parser.parse_args()
    sys.exit(qualify(args.preparation_run.resolve(),args.retry_of))
