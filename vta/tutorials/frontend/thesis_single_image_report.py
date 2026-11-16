#!/usr/bin/env python3
"""Recompute smoke evidence from raw files; seal only after the six-condition gate."""
import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import numpy as np
from thesis_supplement_run import RunStore, sha256, write_new_json
from thesis_supplement_audit import reconcile, seal
from thesis_single_image_board_check import numeric
from thesis_single_image_integrated import boot_identity
from thesis_e2e_ablation_protocol import window_metrics

NAMES = ['A', 'B', 'C', 'Ca', 'D', 'E']


def checkpoint(store):
    r = store.directory
    counts = {str(index): len(list((r / 'raw/native_sessions').glob(f'b{index}_C1_*_window/metrics.json')))
              for index in range(1, 4)}
    invalid = (r / 'invalid_session.json').exists()
    lines = ['# 单张图片语义一致性与递进消融试运行', '',
        '当前状态：' + ('停止，未通过本轮验收。' if invalid else '执行中，尚未签发单图试运行通过结论。'), '',
        '更新时间：' + datetime.now(timezone.utc).isoformat() + '。', '',
        '固定本地猫图片、预训练 ResNet18_v1 和共同 once-qIR，新增量化 0 次。图片没有真实标签，本轮不计算任务准确率，也不签发正式阶段02或06证书。', '',
        '合同及时间线：[contract.json](contract.json)、[timeline.jsonl](timeline.jsonl)。', '',
        '| 检查 | 当前证据 |', '|---|---|']
    for filename, label in [('host_reference_checks.json', '共同LLVM参考'),
            ('Qboard_numeric_checks.json', '同图板端单图路线'),
            ('boot1_trace_audit.json', '启动1完成握手'), ('boot1_dependency_audit.json', '启动1 STORE依赖')]:
        exists = (r / 'analysis' / filename).exists()
        lines.append(f'| {label} | ' + (f'[已保存](analysis/{filename})' if exists else '待采集') + ' |')
    a = r / 'raw/native_sessions/b1_A_stock_correctness/metrics.json'
    lines += ['| 合理A [4,1,4] | ' + ('10次全模型、40数值比较及30实际存储10ms稳定性检查通过' if a.exists() else '待采集') + ' |',
        f'| C1冻结支持集 | 8拓扑×2完整线程身份；窗口进度：boot1 {counts["1"]}/16、boot2 {counts["2"]}/16、boot3 {counts["3"]}/16 |', '',
        '| 条件 | 当前状态 |', '|---|---|']
    _, index = boot_identity(r)
    for name in NAMES:
        path = r / 'raw/native_sessions' / f'b{index}_{name}_integrated_correctness/metrics.json'
        pipe = path.parent.parent / (path.parent.name + '_pipeline') / 'metrics.json'
        state = '串行及流水完整输出通过，待性能汇总' if path.exists() and pipe.exists() else '待本轮集成资格核验'
        lines.append(f'| {name} | {state} |')
    lines += ['', 'A使用兼容stock调度配置在当前AXU板端重测；不复用历史性能标签。前一轮fallback及单线程诊断run在读取C1候选窗口之前封存，其全部成本仍保留。', '',
        'C1仅含单VTA island；多worker TLS队列交接尚未认证，在读取新标签之前排除。DMA静态字段未知时按协议回退M1，不能用零填补未知。', '',
        'C3限定当前图的输入驻留集成候选；其结果不构成未见任务在线调优或输入＋权重联合驻留效果证据。', '',
        '本目录未完成时不封存根级artifact账本，避免把活动时间线误当作不可变结果。']
    if invalid:
        lines += ['', '停止原因见：[invalid_session.json](invalid_session.json)。不得继续本run，重试需建立新目录。']
    if (r / 'analysis/imported_evidence.json').exists():
        imported = json.loads((r / 'analysis/imported_evidence.json').read_text())
        lines += ['', '相同图首boot证据经封存哈希核验后显式导入：[来源及哈希](analysis/imported_evidence.json)。原run为`' + Path(imported['source_run']).name + '`，原boot为`' + imported['original_boot_id'] + '`；不能计为当前run新测量，其成本保留在原run。所有新boot完成和输出检查重新采集。']
    (r / 'REPORT.md').write_text('\n'.join(lines) + '\n')


def final(store):
    r = store.directory
    if (r / 'invalid_session.json').exists():
        raise ValueError('Invalid run cannot receive smoke-pass summary')
    contract = store.contract
    if sha256(r / 'raw/input_image.png') != contract['image_sha256']:
        raise ValueError('Image contract drift')
    input_value = np.load(r / 'raw/inputs/input.npy', allow_pickle=False)
    if input_value.tobytes() != (r / 'raw/inputs/input.bin').read_bytes() or list(input_value.shape) != contract['input_shape']:
        raise ValueError('Actual input bytes/shape drift')
    if sha256(Path(contract['canonical_source'])) != contract['canonical_qir_sha256']:
        raise ValueError('Canonical quantized graph drift')
    conditions = json.loads((r / 'raw/frozen_six_conditions.json').read_text())
    imported_path = r / 'analysis/imported_evidence.json'
    imported = json.loads(imported_path.read_text()) if imported_path.exists() else None
    if imported:
        source = Path(imported['source_run'])
        if sha256(source / 'artifact_hashes.json') != imported['source_ledger_sha256']:
            raise ValueError('Imported evidence ledger drifted')
        for item in imported['imported_files']:
            if sha256(r / item['path']) != item['sha256']:
                raise ValueError('Imported evidence drifted: ' + item['path'])
        if contract.get('audit_only_recovery'):
            original = Path(imported['original_control_source_run'])
            if sha256(original / 'artifact_hashes.json') != imported['original_control_source_ledger_sha256']:
                raise ValueError('Original control version ledger drifted')
            source_contract = json.loads((source / 'contract.json').read_text())
            for key in ('canonical_qir_sha256', 'image_sha256', 'weights_sha256', 'baseline_history_sha256',
                        'A_thread_vector', 'numeric', 'performance', 'udmabuf_bytes', 'bitstream_sha256', 'protocol_version'):
                if source_contract[key] != contract[key]:
                    raise ValueError('Audit-only source execution contract differs: ' + key)
    boot, index = boot_identity(r)
    references = np.load(r / 'raw/host_reference/reference_boundaries.npz', allow_pickle=False)
    expected = np.load(r / 'raw/host_reference/reference_logits.npy', allow_pickle=False)
    correct = []
    final_output_hashes = {}
    memory = {}
    q1_runtime = []
    board = json.loads((r / 'analysis/board_directory.json').read_text())['path']
    for name in NAMES:
        condition = conditions[name]
        for stage in condition['stages']:
            for filename, binding in stage['files'].items():
                if sha256(r / 'raw' / stage['directory'] / filename) != binding['sha256']:
                    raise ValueError('Final stage artifact differs from its frozen identity')
        for suffix in ('', '_pipeline'):
            identity = f'b{index}_{name}_integrated_correctness' + suffix
            directory = r / 'raw/native_sessions' / identity
            rows = [json.loads(line) for line in (directory / 'result.jsonl').read_text().splitlines()]
            if len(rows) != 10 or [row['frame_id'] for row in rows] != list(range(10)):
                raise ValueError('Missing per-condition complete outputs')
            checks = []
            hashes = []
            for row in rows:
                item = row['raw_output_files'][0]
                local = directory / Path(item['path']).relative_to(Path(board) / 'native_sessions' / identity)
                actual = np.frombuffer(local.read_bytes(), dtype=item['dtype']).reshape(item['shape'])
                checks.append(numeric(actual, expected, contract['numeric']))
                hashes.append(sha256(local))
            final_output_hashes[(name, suffix)] = hashes
            metric = json.loads((directory / 'metrics.json').read_text())
            if condition.get('capacity'):
                certificate_path = r / 'raw' / condition['capacity']
                certificate = json.loads(certificate_path.read_text())
                stderr = (directory / 'native.stderr').read_text()
                queues = [json.loads(line.split('[VTA_QUEUE] ', 1)[1])
                          for line in stderr.splitlines() if '[VTA_QUEUE] ' in line]
                insn = certificate['derivation']['insn_capacity_bytes']
                uop = certificate['derivation']['uop_capacity_bytes']
                if not queues or any(row['insn_capacity'] != insn or row['uop_capacity'] != uop
                                     or row['insn_peak'] > insn or row['uop_peak'] > uop
                                     or row['timeout'] != 0 or row['submit_threshold_configured']
                                     for row in queues):
                    raise ValueError('Actual Q1 queue capacity/peak/timeout differs from certificate')
                marker = '[COMMAND-CAPACITY] certificate=' + sha256(certificate_path)
                if marker not in (directory / 'native.stdout').read_text():
                    raise ValueError('Actual Q1 preflight has no certificate SHA evidence')
                q1_runtime.append({'condition': name, 'session': identity,
                    'certificate_sha256': sha256(certificate_path), 'queue_records': len(queues),
                    'insn_capacity_bytes': insn, 'uop_capacity_bytes': uop,
                    'observed_insn_peak_bytes': max(row['insn_peak'] for row in queues),
                    'observed_uop_peak_bytes': max(row['uop_peak'] for row in queues),
                    'raw_stderr_sha256': sha256(directory / 'native.stderr'), 'passed': True})
            if not suffix:
                expected_count = 10 * sum(len(stage['output_node_ids']) for stage in condition['stages'])
                if len(metric['stability_observations']) != expected_count:
                    raise ValueError('Incomplete actual-storage stability proof')
                by_name = {stage['name']: stage for stage in condition['stages']}
                for item in metric['stability_observations']:
                    node = by_name[item['stage']]['output_node_ids'][item['output_index']]
                    pair = []
                    for tag in ('immediate', 'delayed'):
                        local = directory / 'stability' / (item['stem'] + '_' + tag + '.bin')
                        value = np.frombuffer(local.read_bytes(), dtype=item['dtype']).reshape(item['shape'])
                        numeric(value, references[f'node_{node}'], contract['numeric'])
                        pair.append(value.tobytes())
                    if pair[0] != pair[1] or item['delayed_ms'] - item['immediate_ms'] < 9.999:
                        raise ValueError('Actual frontier instability')
            else:
                snapshots = [json.loads(line) for line in (directory / 'allocations.jsonl').read_text().splitlines()]
                memory[name] = next(value for value in snapshots if value['phase'] == 'after_steady_state')
            correct.append({'condition': name, 'mode': 'pipeline' if suffix else 'serial',
                'complete_outputs': 10, 'all_finite': True, 'max_abs_error': max(value['max_abs_error'] for value in checks),
                'all_byte_exact_to_reference': all(value['exact'] for value in checks)})
    byte_proof = json.loads((r / 'analysis/B_C_Ca_D_byte_exact.json').read_text())
    if not byte_proof['passed']:
        raise ValueError('Copy/shared byte-exact gate failed')
    base_hashes = final_output_hashes[('B', '')]
    same_program_names = ['B', 'C', 'Ca', 'D']
    if conditions['E']['c3_status'].startswith('E_equals_D'):
        same_program_names.append('E')
    for name in same_program_names:
        for suffix in ('', '_pipeline'):
            if final_output_hashes[(name, suffix)] != base_hashes:
                raise ValueError('Independent raw copy/shared byte-exact recomputation failed')
    c1_boots = [json.loads((r / f'analysis/c1_boot{i}_windows.json').read_text()) for i in range(1, 4)]
    if len({value['boot_id'] for value in c1_boots}) != 3 or any(len(value['results']) != 16 for value in c1_boots):
        raise ValueError('Incomplete three-boot C1 support measurements')
    frozen = json.loads((r / 'raw/c1_frozen/fair_support_contract.json').read_text())
    c1_values = {item['candidate_id']: [] for item in frozen['support']}
    c1_raw = []
    for boot_index, value in enumerate(c1_boots, 1):
        if {point['candidate_id'] for point in value['results']} != set(c1_values):
            raise ValueError('C1 raw support coverage differs')
        for point in value['results']:
            identity = f'b{boot_index}_C1_' + point['candidate_id'] + '_window'
            directory = r / 'raw/native_sessions' / identity
            rows = [json.loads(line) for line in (directory / 'result.jsonl').read_text().splitlines()]
            session = json.loads((directory / 'session_contract.json').read_text())
            if len(rows) != 307 or session['boot_id'] != value['boot_id'] or [row['frame_id'] for row in rows] != list(range(307)):
                raise ValueError('Incomplete or mismatched C1 raw window identity')
            ii = window_metrics([row['completion_ms'] / 1000 for row in rows[5:]])['ii_seconds']
            endpoint = (rows[-1]['completion_ms'] - rows[7]['completion_ms']) / 299 / 1000
            if not math.isclose(ii, point['ii_seconds'], rel_tol=1e-10, abs_tol=1e-12) or not math.isclose(ii, endpoint, rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError('C1 raw II recomputation differs')
            c1_values[point['candidate_id']].append(ii)
            c1_raw.append({'candidate_id': point['candidate_id'], 'boot_id': value['boot_id'],
                          'ii_seconds': ii, 'raw_rows_sha256': sha256(directory / 'result.jsonl')})
    c1_medians = {key: statistics.median(values) for key, values in c1_values.items()}
    selection = json.loads((r / 'analysis/B_selection.json').read_text())
    selected = min(frozen['orders']['O2_semantic_band'][:5], key=lambda key: (c1_medians[key], key))
    if selection['candidate_id'] != selected:
        raise ValueError('B selection differs from independently recomputed frozen-prefix result')
    with (r / 'analysis/c1_three_boot_selection.csv').open('x', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['candidate_id', 'M1_ms', 'boot1_II_ms', 'boot2_II_ms', 'boot3_II_ms', 'median_II_ms', 'in_frozen_O2_first5', 'selected_B'])
        for item in frozen['support']:
            key = item['candidate_id']
            writer.writerow([key, item['m1_score_ms'], *[value * 1000 for value in c1_values[key]],
                             c1_medians[key] * 1000, key in frozen['orders']['O2_semantic_band'][:5], key == selected])
    performance = []
    for block in range(2):
        for name in NAMES:
            for mode, count in [('serial', 20), ('window', 307)]:
                identity = f'b{index}_block{block}_{name}_{mode}'
                directory = r / 'raw/native_sessions' / identity
                rows = [json.loads(line) for line in (directory / 'result.jsonl').read_text().splitlines()]
                if len(rows) != count:
                    raise ValueError('Performance raw call count mismatch')
                value = {'condition': name, 'block': block, 'mode': mode, 'complete_calls': count,
                    'raw_rows_sha256': sha256(directory / 'result.jsonl')}
                if mode == 'window':
                    score = window_metrics([row['completion_ms'] / 1000 for row in rows[5:]])
                    independent = (rows[-1]['completion_ms'] - rows[7]['completion_ms']) / 299 / 1000
                    if not math.isclose(score['ii_seconds'], independent, rel_tol=1e-10, abs_tol=1e-12):
                        raise ValueError('Independent II endpoint recomputation differs')
                    value['window'] = score
                else:
                    value['serial_latency_ms'] = [row['total_latency_ms'] for row in rows[5:]]
                performance.append(value)
    table = []
    for name in NAMES:
        serial = [point for value in performance if value['condition'] == name and value['mode'] == 'serial'
                  for point in value['serial_latency_ms']]
        ii = [value['window']['ii_seconds'] for value in performance if value['condition'] == name and value['mode'] == 'window']
        a_ii = [value['window']['ii_seconds'] for value in performance if value['condition'] == 'A' and value['mode'] == 'window']
        table.append({'condition': name, 'serial_calls_scored': 30, 'serial_median_ms': statistics.median(serial),
            'serial_p95_ms_descriptive': float(np.percentile(serial, 95)), 'ii_seconds_blocks': ii,
            'ii_median_ms': statistics.median(ii) * 1000, 'fps_from_median_ii': 1 / statistics.median(ii),
            'paired_speedup_over_A': math.sqrt(math.prod(a / b for a, b in zip(a_ii, ii))),
            'udmabuf_allocator_high_water_mib': memory[name]['high_water_bytes'] / (1 << 20)})
    result = {'status': 'SINGLE_IMAGE_SMOKE_PASS', 'run_id': contract['run_id'],
        'formal_stage_completed': False, 'accuracy_evidence': False, 'input_images': 1,
        'extra_quantize_calls': 0, 'C1_boot_ids': [value['boot_id'] for value in c1_boots],
        'final_ablation_boot_ids': [boot], 'correctness': correct, 'performance': table,
        'C3_scope': json.loads((r / 'raw/c3_route_contract.json').read_text())['scope'],
        'E_status': conditions['E']['c3_status'], 'inferential_confidence_interval': None,
        'claim_boundary': 'one real image, one final-system boot, two order blocks; no ImageNet accuracy or five-boot conclusion'}
    result['imported_evidence'] = None if not imported else {
        'source_run': imported['source_run'], 'original_boot_id': imported['original_boot_id'],
        'manifest': 'analysis/imported_evidence.json', 'not_new_measurements': True}
    result['A_B_same_program_identity'] = (
        conditions['A']['stages'] == conditions['B']['stages']
        and conditions['A']['threads'] == conditions['B']['threads']
        and conditions['A']['final_runner_sha256'] == conditions['B']['final_runner_sha256'])
    result['selection_deployment_lineage'] = json.loads((r / 'analysis/selection_deployment_lineage.json').read_text())
    result['audit_only_recovery'] = bool(contract.get('audit_only_recovery'))
    if result['audit_only_recovery']:
        result['source_execution_run_id'] = contract['source_execution_run_id']
        result['new_board_calls_in_this_directory'] = 0
    write_new_json(r / 'analysis/independent_raw_recomputation.json', {'correctness': correct,
        'performance_sessions': performance, 'memory_snapshots': memory,
        'C1_raw_windows': c1_raw, 'B_selection_independently_recomputed': selected,
        'Q1_actual_runtime_capacity_checks': q1_runtime,
        'copy_shared_raw_output_hashes_equal': True})
    store.event('single_image_smoke_gates_passed', formal_stage_completed=False, accuracy_evidence=False)
    costs = reconcile(r)
    if costs['unfinished_action_ids']:
        raise ValueError('Cannot seal with unfinished actions')
    write_new_json(r / 'analysis/cost_audit.json', costs)
    retry = []
    previous = contract.get('retry_of')
    while previous:
        prior = r.parent / previous
        prior_cost = reconcile(prior)
        retry.append({'run_id': previous, 'report': '../' + previous + '/REPORT.md', **prior_cost})
        previous = json.loads((prior / 'contract.json').read_text()).get('retry_of')
    write_new_json(r / 'analysis/retry_chain_cost_audit.json', {'current': costs, 'predecessors': retry,
        'interactive_cost_seconds': None, 'common_preparation_run': contract['preparation_run'],
        'common_preparation_costs': reconcile(Path(contract['preparation_run'])),
        'historical_cache_acquisition_cost_seconds': None,
        'note': 'sum of primitive actions, not online wall time; unknown costs are not zero'})
    figures(r, table)
    write_new_json(r / 'summary.json', result)
    lines = ['# 单张图片语义一致性与递进消融试运行结果', '',
        '**单图试运行通过；正式阶段未完成。** 固定预训练 ResNet18_v1、共同量化图及本地猫图片，新增量化0次。', '',
        '本轮10次串行及10次流水完整输出／条件通过共同参考比较。串行会话逐项重读实际executor或slot存储，立即与延迟10ms输出稳定；整数切分前沿精确一致，B/C/Ca/D最终输出在两种执行模式下均字节一致。', '',
        '共同参考及同图板端单图路线另各验证10次。图片没有真实标签，因此不报告Top-1/Top-5准确率。', '',
        '| 条件 | 串行延迟中位数/ms | 流水II/ms | FPS | 相对A配对速度比 | u-dma-buf申请高水位/MiB |',
        '|---|---:|---:|---:|---:|---:|']
    for value in table:
        lines.append('| ' + value['condition'] + ' | ' + ' | '.join(f'{value[key]:.3f}' for key in
            ('serial_median_ms', 'ii_median_ms', 'fps_from_median_ii', 'paired_speedup_over_A', 'udmabuf_allocator_high_water_mib')) + ' |')
    lines += ['', '串行延迟从帧enqueue计至最终输出完成，含set/run/get及复制；每条件每块5次预热＋15次计分，共30次计分。流水每块5次预热后运行302帧，丢弃前两帧，用保留300帧的299个间隔计分。正序、逆序各一块；表中II为两块中位数，速度比为同块A/条件II的几何平均。启动、构建、上传、恢复等成本另记账，不混入帧延迟。', '',
        '内存列是诊断driver记录的u-dma-buf分配高水位，来自各条件独立流水正确性会话；不是全平台RSS、FPGA片上SRAM或物理保留内存。u-dma-buf静态保留192MiB保持不变。', '',
        '流水managed条件统一使用K=2。串行C按现有CLI约束使用1个slot，带anchor计划的Ca/D/E使用2个；因此串行C→Ca同时包含slot配置差异，不能单独归因于anchor。anchor内存比较使用slot数相同的流水资格会话。', '',
        'C1使用当前共同图冻结的8拓扑×2线程支持集，三次不同kernel boot共48个窗口；B只从冻结O2前五点按三boot中位II选择。未知内部DMA字段触发M1回退，不能将此数据作为语义排序优越性的证据；多VTA island尚未认证。', '',
        'E状态：`' + conditions['E']['c3_status'] + '`。' + result['C3_scope'] + '。该有限候选的构建及搜索成本全部保留，不作为未见任务在线调优或完整C3创新的验证。' + ('合格新程序的anchor和Q1均重新生成。' if not conditions['E']['c3_status'].startswith('E_equals_D') else '本轮E实际使用D程序及其已认证anchor/Q1；被拒候选的artifact、资格与成本单独保留，不能称C3带来收益。'), '',
        '本轮最终六条件性能仅来自同一kernel boot，不计算推断性置信区间，也不签发完整语义验证、任务准确率或五次独立启动证书。任何下降或零收益均按实记录，不能称每一步都加速。', '',
        '原始及复算入口：[合同](contract.json)、[六条件身份](raw/frozen_six_conditions.json)、[原始数据复算](analysis/independent_raw_recomputation.json)、[Q1全部worker峰值](analysis/D_all_worker_queue_status.json)、[成本账本](analysis/cost_audit.json)、[重试链成本](analysis/retry_chain_cost_audit.json)、[时间线](timeline.jsonl)。', '',
        '图表：[II及延迟](figures/ablation_performance.png)、[分配高水位](figures/ablation_allocator.png)。可发表版本另存同名PDF。']
    if imported:
        lines += ['', '首boot及此前同图输出证据来自已封存run`' + Path(imported['source_run']).name + '`，经逐文件SHA核验显式导入：[来源清单](analysis/imported_evidence.json)。保留原boot身份及原始成本，不宣称是本run新增测量；新boot资格重新采集。']
    lines += ['', '最终A使用冻结支持集事先指定的合理基线及固定线程向量，与B采用同一构建/参数序列化策略：[A绑定](analysis/A_fair_program_binding.json)。原A诊断参数文件虽然30个参数张量逐字节一致，但序列化顺序不同，其旧诊断样本不混入最终性能。', '',
        '原C1窗口保持原runner和原候选身份，只用于冻结划分及线程选择。修复仅涉及私有读回诊断；最终六条件统一使用修复后的runner重新验证、重新计时。原窗口不能描述成新runner测量：[选择与部署身份](analysis/selection_deployment_lineage.json)。']
    if result['A_B_same_program_identity']:
        lines += ['', '本轮B与A选择相同完整程序及线程身份。A→B的计时差异属于同一程序重复测量，不能归因于C1划分选择带来的加速。']
    if result['audit_only_recovery']:
        lines += ['', '本目录仅审计已完成执行run`' + contract['source_execution_run_id'] + '`，新增板端调用0。源run全部六条件资格及24性能会话已完成，最终来源检查因两份恢复控制脚本同名覆盖停止。仅这两份控制存在来源差异；当前版逐一匹配源run实际成功执行action输入SHA，原版另名归档且原封存账本再次核验。其余导入文件、计算artifact、输出和时间戳均精确一致。本目录重新从raw复算后签发单图试运行结论，原执行run、kernel boot及全部成本保持来源身份：[来源核对](analysis/provenance_reconciliation.json)、[导入及两版控制清单](analysis/imported_evidence.json)。']
    (r / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    seal(r)
    update_status(r, result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def figures(r, table):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x = np.arange(len(table))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), constrained_layout=True)
    for axis, key, ylabel in [(axes[0], 'serial_median_ms', 'Serial latency (ms)'),
                              (axes[1], 'ii_median_ms', 'Pipeline II (ms)')]:
        axis.bar(x, [value[key] for value in table], color='#326b9b')
        axis.set_xticks(x, NAMES)
        axis.set_ylabel(ylabel)
        axis.grid(axis='y', alpha=.2)
    for i, value in enumerate(table):
        axes[1].scatter([i, i], np.array(value['ii_seconds_blocks']) * 1000, color='#b34431', s=20, zorder=3)
    fig.suptitle('Single-image smoke: one boot, forward/reverse blocks')
    for suffix in ('png', 'pdf'):
        fig.savefig(r / 'figures' / ('ablation_performance.' + suffix), dpi=220)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(6, 3.5), constrained_layout=True)
    axis.bar(NAMES, [value['udmabuf_allocator_high_water_mib'] for value in table], color='#3b8272')
    axis.set_ylabel('u-dma-buf allocation high water (MiB)')
    axis.set_title('10-frame pipeline qualification; 192 MiB reservation unchanged')
    axis.grid(axis='y', alpha=.2)
    for suffix in ('png', 'pdf'):
        fig.savefig(r / 'figures' / ('ablation_allocator.' + suffix), dpi=220)
    plt.close(fig)


def update_status(r, result):
    path = r.parents[2] / 'EXECUTION_STATUS.md'
    text = path.read_text()
    marker = '当前run：`' + result['run_id'] + '`'
    start = text.index(marker)
    end = text.index('\n\n', start)
    text = text[:start] + marker + '，结果入口：[报告](results/06/' + result['run_id'] + '/REPORT.md)。状态：完成（仅单图试运行通过，正式06仍为准备中）。' + text[end:]
    text += '\n- 2026-09-16：单图试运行 `' + result['run_id'] + '`通过六条件完整输出、实际存储10ms稳定性和正逆序性能复算，原始文件及直接重试链成本已封存。C1三个boot不代替最终系统五boot；准确率证据为0，正式02/06状态不自动完成。\n'
    path.write_text(text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('checkpoint', 'final'))
    parser.add_argument('run_dir', type=Path)
    args = parser.parse_args()
    store = RunStore(args.run_dir)
    checkpoint(store) if args.operation == 'checkpoint' else final(store)
