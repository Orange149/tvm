"""Read measured FPS only after exact ranking has been sealed and completed."""
import csv
import json
import math
from pathlib import Path
import statistics

from cpu_vta_partitioning.common import digest, read, sha256, verify_inventory, write
from .common import LABELS, METHODS
from .runtime import summary, task


def receipt(path, document=None, boot=None, *, bundle=None):
    """Reject a receipt pointing to another raw task or boot, before using it."""
    path = Path(path)
    item = read(path)
    document = item['task'] if document is None else document
    boot = item['boot_id'] if boot is None else boot
    if item['task'] != document or item['boot_id'] != boot:
        raise ValueError('analysis receipt task or boot identity differs')
    verify_inventory(item['raw'], item['files'])
    raw = Path(item['raw'])
    if ((raw / 'boot_id').read_text().strip() != boot
            or (raw / 'task_hash').read_text().strip() != digest(document)):
        raise ValueError('analysis raw task or boot identity differs')
    if bundle is not None and summary(raw, document, bundle) != item['summary']:
        raise ValueError('analysis result does not replay')
    return item


def _divide(value, denominator):
    return value / denominator if denominator else None


def thread_curve_rows(records):
    """Keep every measured point, including non-monotonic curves and fallbacks."""
    groups = {}
    for key, record in records.items():
        cfg = record['configuration']; stage = cfg['stages'][0]
        if stage['device'] != 'cpu':
            continue
        name, threads = stage['program'], stage['threads']
        if threads in groups.setdefault(name, {}):
            raise ValueError('duplicate measured CPU thread signature')
        groups[name][threads] = (key, record, stage)
    rows = []
    for name, points in sorted(groups.items()):
        if set(points) != {1, 2, 3, 4}:
            raise ValueError('CPU thread curve must include all four measured signatures')
        baseline = points[1][1]['cost']
        for threads in range(1, 5):
            key, record, stage = points[threads]; value = record['cost']
            prior = points[threads - 1][1]['cost'] if threads > 1 else None
            source = record.get('source', {})
            shares = record.get('session_thread_weights')
            rows.append(dict(program=name, begin=stage['begin'], end=stage['end'], threads=threads,
                measured_core_set=stage['mask'], worker_core_map=stage['worker_core_map'],
                service_ms=value['service_ms'], input_ms=value.get('set_ms'),
                compute_ms=value.get('run_ms'), output_ms=value.get('get_ms'),
                compute_core_ms=value['run_core_ms'], input_core_ms=value.get('set_core_ms'),
                output_core_ms=value.get('get_core_ms'), boundary_core_ms=value['boundary_core_ms'],
                total_core_ms=value['run_core_ms'] + value['boundary_core_ms'],
                speedup_over_one_thread=_divide(baseline['service_ms'], value['service_ms']),
                compute_core_ratio_over_one_thread=_divide(value['run_core_ms'], baseline['run_core_ms']),
                total_core_ratio_over_one_thread=_divide(value['run_core_ms'] + value['boundary_core_ms'],
                    baseline['run_core_ms'] + baseline['boundary_core_ms']),
                marginal_service_reduction_ms=prior['service_ms'] - value['service_ms'] if prior else None,
                marginal_service_reduction_percent=100 * (1 - value['service_ms'] / prior['service_ms'])
                    if prior and prior['service_ms'] else None,
                slower_than_previous_threads=value['service_ms'] > prior['service_ms'] if prior else False,
                session_thread_weights=shares,
                thread_share_source='whole_native_session_including_warmup_and_startup_probe'
                    if shares is not None else 'uniform_fallback_unobservable_session_increments',
                compute_participating_threads=None,
                compute_participation_note='configured slots and session shares do not prove run-only participation',
                other_masks='homogeneous_core_permutation_approximation_not_per_mask_measurement',
                component_key=key, boot_id=source.get('boot_id'), task_id=source.get('task_id'),
                source_receipt=source.get('receipt'), source_receipt_sha256=source.get('receipt_sha256')))
    return rows


def _csv(path, rows):
    if not rows:
        raise ValueError('cannot silently produce an empty required result table')
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list, tuple)) else v
                         for k, v in row.items()} for row in rows)


def score_rows(lists, pool, results, records, programs, ddr, frozen_scores):
    """Explain existing frozen scores, without changing or using FPS to score."""
    from cpu_policies.open_resources.costs import Costs
    from resource_envelope.model import score as envelope_score
    from .ranking import scalar_score
    configurations, constraints = [], []
    for method in METHODS:
        costs = Costs(records, programs=programs, ddr=ddr,
                      distribution='balanced' if method == 'tarnawski' else 'session_weights')
        for rank, row in enumerate(lists[method], 1):
            name = row['id']; cfg = pool[name]
            value = scalar_score(cfg, costs, method)
            if value != frozen_scores[method][name] or value['score_ns'] != row['score_ns']:
                raise ValueError('analysis scalar score differs from the frozen selection')
            if method == 'tarnawski':
                capacity = value['capacity_terms_ns']
            else:
                original = envelope_score(cfg, costs)['original_components']
                capacity = dict(CPU_stage=original['cpu_service_max_ns'],
                    VTA_sum=original['vta_service_sum_ns'], DDR_sum=original['ddr_service_ns'],
                    **{'CPU_subset_' + str(i + 1): v for i, v in enumerate(original['subset_capacity_ns'])},
                    **{'serial_group_' + g: v for g, v in original['group_services_ns'].items()})
            configurations.append(dict(method=method, static_rank=rank, configuration=name,
                context=cfg['context_id'], gate=cfg['execution_mode'], stage_count=len(cfg['stages']),
                VTA_islands=sum(s['device'] == 'vta' for s in cfg['stages']),
                CPU_threads=[s['threads'] for s in cfg['stages'] if s['device'] == 'cpu'],
                CPU_core_sets=[s['mask'] for s in cfg['stages'] if s['device'] == 'cpu'],
                host_mask=cfg['host_mask'], control_mask=cfg['control_mask'],
                predicted_II_ns=row['score_ns'], actual_II_ms=results[name]['II_ms'],
                FPS=results[name]['FPS'], binding_terms=value['binding_terms'],
                capacity_binding_terms=[k for k, v in capacity.items() if v == value['terms_ns']['capacity']],
                equal_score_count=sum(r['score_ns'] == row['score_ns'] for r in lists[method])))
            for scope, terms in (('score', value['terms_ns']), ('capacity', capacity)):
                for term, demand in terms.items():
                    constraints.append(dict(method=method, static_rank=rank, configuration=name,
                        scope=scope, term=term, demand_ns=demand,
                        equals_final_score=demand == row['score_ns'],
                        equals_capacity=demand == value['terms_ns']['capacity']))
    return configurations, constraints


def stage_rows(pool, results, records, programs, ddr):
    from cpu_policies.open_resources.costs import Costs
    costs = Costs(records, programs=programs, ddr=ddr, distribution='session_weights')
    rows = []
    for name, cfg in sorted(pool.items()):
        services = results[name]['stage_service_ms']
        if len(services) != len(cfg['stages']):
            raise ValueError('pipeline stage service coverage differs')
        for index, stage in enumerate(cfg['stages']):
            value = costs.get(stage, cfg)
            source = records[value['source_key']].get('source', {})
            rows.append(dict(configuration=name, stage_index=index, program=stage['program'],
                device=stage['device'], begin=stage['begin'], end=stage['end'], threads=stage['threads'],
                core_set=stage['mask'], worker_core_map=stage['worker_core_map'],
                caller_core=stage['caller_core'], context=cfg['context_id'], gate=cfg['execution_mode'],
                isolated_service_ms=value['service_ms'], pipeline_service_ms=services[index],
                pipeline_minus_isolated_ms=services[index] - value['service_ms'],
                pipeline_over_isolated=_divide(services[index], value['service_ms']),
                isolated_input_ms=value.get('set_ms'), isolated_compute_ms=value.get('run_ms'),
                isolated_output_ms=value.get('get_ms'), isolated_compute_core_ms=value['run_core_ms'],
                isolated_boundary_core_ms=value['boundary_core_ms'],
                VTA_host_core_ms=value['run_core_ms'] + value['boundary_core_ms'] if stage['device'] == 'vta' else None,
                CPU_thread_weights=value['cpu_core_weights'] if stage['device'] == 'cpu' else None,
                assumptions=value['assumptions'], component_source_key=value['source_key'],
                component_source_receipt=source.get('receipt'), component_source_boot=source.get('boot_id'),
                interpretation='stage wall service comparison; concurrent stages must not be summed'))
    return rows


def measurement_cost(items):
    """Count accepted work once; startup/empty-callback probes remain separate."""
    tasks = [item['task'] for item in items]
    if len({doc['task_id'] for doc in tasks}) != len(tasks):
        raise ValueError('duplicate receipts would double count measured work')
    kinds = {doc['kind'] for doc in tasks}
    if not (kinds <= {'calibration'} or kinds <= {'traffic'} or kinds <= {'qualify_pipeline', 'performance'}):
        raise ValueError('component, DMA and full-network calls must be counted separately')
    durations = [item.get('measurement_and_collection_seconds') for item in items]
    if any(v is not None and (type(v) not in (int, float) or not math.isfinite(v) or v < 0)
           for v in durations):
        raise ValueError('invalid measured collection duration')
    stage_calls = sum(sum(doc[k] for k in ('warmup', 'scored', 'tail')) * len(doc['stages']) for doc in tasks)
    network = kinds <= {'qualify_pipeline', 'performance'} and bool(tasks)
    return dict(accepted_native_sessions=len(items),
        scheduled_stage_calls_including_warmup=stage_calls,
        complete_network_inferences=sum(sum(doc[k] for k in ('warmup', 'scored', 'tail')) for doc in tasks)
            if network else None,
        measurement_and_collection_seconds=sum(durations) if all(v is not None for v in durations) else None,
        duration_scope='sum of accepted per-session measure+collect+initial-validation elapsed; excludes deployment and failed attempts',
        recorded_duration_sessions=sum(v is not None for v in durations),
        initialization_and_empty_callback_probe_calls=None,
        initialization_note='not counted as stage or full-network calls; no reliable separate call counter in receipts')


def cost_report(output, run_dir, component_items, dma_items, pipeline_items):
    output, run_dir = Path(output), Path(run_dir)
    build_paths = sorted((output / 'build/build_reports').glob('*.json'))
    builds = [read(path) for path in build_paths]
    compiles = sorted((output / 'build/programs').glob('*/record.json'))
    compile_records = [read(path) for path in compiles]
    search_path = output / 'ranking/SEARCH_STATUS.json'
    search = read(search_path) if search_path.exists() else None
    reference_path = output / 'references/REFERENCE_PROOF.json'
    reference = read(reference_path) if reference_path.exists() else None
    local_audit_path = output / 'LOCAL_REFERENCE_AUDIT.json'
    local_audit = read(local_audit_path) if local_audit_path.exists() else None
    archives = sorted(run_dir.glob('*/deployment.tar.gz'))
    deployment_paths = sorted(run_dir.glob('*/deployments/*.json'))
    deployments = [read(path) for path in deployment_paths]
    import_path = run_dir / 'components/imported_components.json'
    imported = read(import_path) if import_path.exists() else None
    if imported is not None:
        current_id = read(output / 'profile_capsule/manifest.json')['bundle_id']
        new_components = [item for item in component_items if item['task']['bundle_id'] == current_id]
        reused_components = [item for item in component_items if item['task']['bundle_id'] != current_id]
        archived = imported['archive_audit']
        archived_items = [receipt(row['receipt']) for row in archived['records']]
    else:
        new_components, reused_components, archived_items, archived = component_items, [], [], None
    actual_component_work = {}
    for item in new_components + archived_items:
        identity = item['boot_id'], item['task']['task_id']
        if identity in actual_component_work and actual_component_work[identity] != item:
            raise ValueError('one historical task has conflicting receipts in the cost ledger')
        actual_component_work[identity] = item
    return dict(components=measurement_cost(component_items), DMA=measurement_cost(dma_items),
        pipeline=measurement_cost(pipeline_items),
        component_source_scope='cost-table evidence includes exact imported profiles; it is not all newly executed work',
        new_component_measurements=measurement_cost(new_components),
        reused_component_sources=measurement_cost(reused_components),
        cumulative_unique_component_work=measurement_cost(list(actual_component_work.values())),
        archived_compiler_regression=dict(
            archive_evidence=archived,
            archived_measurements=measurement_cost(archived_items) if archived is not None else None,
            already_counted_as_reused_sources=len(reused_components),
            additional_stopped_sessions_not_used_for_costs=len(archived_items) - len(reused_components)
                if archived is not None else 0,
            interpretation='stopped preparation and repaired preparation are reported separately; reused receipts enter cumulative work once'),
        compilation=dict(successful_program_records=len(compile_records),
            successful_program_worker_seconds=sum(r['compile_seconds'] for r in compile_records)
                if compile_records and all('compile_seconds' in r for r in compile_records) else None,
            preparation_invocation_wall_seconds=sum(b['wall_seconds'] for b in builds) if builds else None,
            invocation_reports=[dict(path=str(p), sha256=sha256(p)) for p in build_paths],
            wall_scope='sum of recorded preparation invocations; may overlap with other host work',
            failed_compile_worker_seconds=None,
            unknown='failed compiler subprocess durations are not recorded; successful records are counted once'),
        search=dict(last_invocation_elapsed_seconds=search.get('elapsed_seconds') if search else None,
            all_invocations_elapsed_seconds=None,
            checkpoint_statistics=search.get('statistics') if search else None,
            source=str(search_path) if search else None,
            unknown='SEARCH_STATUS stores the latest invocation; cumulative resumptions are not inferred'),
        deployment=dict(elapsed_seconds=sum(d['elapsed_seconds'] for d in deployments) if deployments else None,
            timed_attempts=len(deployments),
            failed_timed_attempts=sum(d['status'] == 'FAILED' for d in deployments),
            timing_scope='deployment transfer or cached-payload verification, as actually executed; includes failed attempts',
            timing_sources=[dict(path=str(p), sha256=sha256(p)) for p in deployment_paths],
            last_local_archive_bytes={str(p): p.stat().st_size for p in archives},
            profile_payload_unpacked_bytes=read(output / 'profile_capsule/plans/board_payload.json')['unpacked_bytes'],
            pipeline_payload_unpacked_bytes=read(output / 'pipeline_bundle/plans/board_payload.json')['unpacked_bytes'],
            unknown='transfer-only time and cumulative transmitted bytes are not separated from verification'),
        local_reference=dict(complete_graph_calls=reference.get('local_complete_graph_calls') if reference else None,
            calls_scope='accepted full-graph or all-frontier QEMU calls; failed preflight is separate',
            failed_preflight_calls=local_audit.get('failed_preflight_full_calls') if local_audit else None,
            known_total_calls=(local_audit['accepted_full_or_probe_calls'] + local_audit['failed_preflight_full_calls'])
                if local_audit else None,
            local_call_audit_sha256=sha256(local_audit_path) if local_audit else None,
            elapsed_seconds=reference.get('elapsed_seconds') if reference else None, board_inferences=0,
            target='local AArch64 CPU under QEMU; excluded from board DNN counts'),
        unknown_counts_are_not_zero=True)


def plot_thread_curves(rows, selected, target):
    import matplotlib.pyplot as plt
    grouped = {}
    for row in rows:
        grouped.setdefault(row['program'], []).append(row)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for points in grouped.values():
        for ax, field in zip(axes, ('speedup_over_one_thread', 'total_core_ratio_over_one_thread')):
            ax.plot([p['threads'] for p in points], [p[field] for p in points], color='0.5', alpha=.12, linewidth=.7)
    for ax, label in zip(axes, ('Isolated service speedup', 'Total CPU core-time / one thread')):
        ax.set(xlabel='Configured threads', ylabel=label, xticks=[1, 2, 3, 4]); ax.grid(alpha=.2)
    fig.suptitle('Every measured CPU subgraph; non-monotonic points retained')
    fig.tight_layout()
    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(target / ('all_thread_curves.' + ext), dpi=240)
    plt.close(fig)
    names = sorted(set(selected) & set(grouped))
    for offset in range(0, len(names), 8):
        current = names[offset:offset + 8]
        fig, axes = plt.subplots(len(current), 2, figsize=(8.5, max(2.3, 1.8 * len(current))), squeeze=False)
        for line, name in enumerate(current):
            points = grouped[name]
            for column, fields in enumerate((('service_ms',), ('compute_core_ms', 'boundary_core_ms'))):
                for field in fields:
                    axes[line, column].plot([p['threads'] for p in points], [p[field] for p in points],
                                            marker='o', label=field.replace('_', ' '))
                axes[line, column].set(title=name, xticks=[1, 2, 3, 4], ylabel='ms' if column == 0 else 'core-ms')
                axes[line, column].grid(alpha=.2); axes[line, column].legend(fontsize=7)
        axes[-1, 0].set_xlabel('Configured threads'); axes[-1, 1].set_xlabel('Configured threads')
        fig.tight_layout()
        for ext in ('pdf', 'svg', 'png'):
            fig.savefig(target / ('selected_thread_curves_' + str(offset // 8 + 1) + '.' + ext), dpi=240)
        plt.close(fig)


def budgets(lists, results):
    rows = []
    if set(lists) != set(METHODS):
        raise ValueError('two frozen methods required')
    for method in METHODS:
        selected = lists[method]
        if len(selected) != 20 or any(r['id'] not in results for r in selected):
            raise ValueError('all selected configurations must have new accepted measurements')
        for budget in (1, 5, 20):
            rank = min(range(budget), key=lambda i: (results[selected[i]['id']]['II_ms'], i))
            row = results[selected[rank]['id']]
            rows.append(dict(method=method, budget=budget, static_rank=rank + 1,
                configuration=selected[rank]['id'], II_ms=row['II_ms'], FPS=row['FPS']))
    return rows


def analyze(output, run_dir):
    from .validation import validate_bundle
    output, run_dir = Path(output), Path(run_dir)
    bundle = output / 'pipeline_bundle'
    validate_bundle(bundle)
    state = read(run_dir / 'pipeline/state.json')
    if state['status'] != 'COMPLETE':
        raise ValueError('the complete frozen union must be measured before analysis')
    boot = state['boot_id']
    pool = read(bundle / 'plans/pool.json')
    lists = read(bundle / 'plans/rankings.json')['lists']
    results, sources, pipeline_items = {}, {}, []
    for name, cfg in pool.items():
        qualified = None
        for phase in ('qualify', 'short'):
            doc = task(bundle, cfg, phase)
            path = run_dir / 'pipeline/accepted' / boot / (doc['task_id'] + '.json')
            item = receipt(path, doc, boot, bundle=bundle)
            value = item['summary']; pipeline_items.append(item)
            sources[str(path)] = sha256(path)
            if phase == 'short':
                if value['output_fingerprints'] != qualified:
                    raise ValueError('analysis performance output differs from qualification')
                results[name] = value
            else:
                qualified = value['output_fingerprints']
    # The whole profile table is revalidated independently of measured pipeline FPS.
    from .components import replay as replay_components
    from yolo_transfer.ranking import replay_traffic
    records = replay_components(output, run_dir / 'components')
    projection = bundle / 'provenance/static_inputs'
    if records != read(projection / 'isolated_costs.json')['costs']:
        raise ValueError('accepted component records differ from frozen ranking inputs')
    dma = replay_traffic(output / 'profile_capsule', run_dir / 'traffic')
    if dma != read(projection / 'traffic.json')['records']:
        raise ValueError('accepted DMA records differ from frozen ranking inputs')
    programs, ddr = read(projection / 'programs.json'), read(projection / 'ddr.json')
    component_items, dma_items = [], []
    for record in records.values():
        source = record['source']
        if ((source['boot_id'] != boot and not source.get('archived_output'))
                or sha256(source['receipt']) != source['receipt_sha256']):
            raise ValueError('component source is outside the pipeline cohort')
        component_items.append(receipt(source['receipt'], boot=boot))
        sources[source['receipt']] = source['receipt_sha256']
    for record in dma.values():
        source = record['source']
        path = run_dir / 'traffic/accepted' / source['boot_id'] / (source['task_id'] + '.json')
        if source['boot_id'] != boot or sha256(path) != source['receipt_sha256']:
            raise ValueError('DMA source is outside the pipeline cohort')
        dma_items.append(receipt(path, boot=boot)); sources[str(path)] = sha256(path)
    rows = budgets(lists, results)
    target = output / 'analysis'
    target.mkdir(parents=True, exist_ok=True)
    _csv(target / 'budgets.csv', rows)
    predictions = []
    for method, selected in lists.items():
        for rank, row in enumerate(selected, 1):
            observed = results[row['id']]
            predictions.append(dict(method=method, static_rank=rank, configuration=row['id'],
                predicted_II_ms=row['score_ns'] / 1e6, actual_II_ms=observed['II_ms'], FPS=observed['FPS'],
                II_error_percent=100 * (row['score_ns'] / 1e6 / observed['II_ms'] - 1),
                context=pool[row['id']]['context_id'], gate=pool[row['id']]['execution_mode']))
    _csv(target / 'predictions.csv', predictions)
    curves = thread_curve_rows(records)
    _csv(target / 'CPU_thread_curves.csv', curves)
    selected_scores, constraints = score_rows(lists, pool, results, records, programs, ddr,
                                              read(bundle / 'plans/scores.json'))
    _csv(target / 'configuration_scores.csv', selected_scores)
    _csv(target / 'constraint_scores.csv', constraints)
    _csv(target / 'stage_isolated_pipeline.csv', stage_rows(pool, results, records, programs, ddr))
    costs = cost_report(output, run_dir, component_items, dma_items, pipeline_items)
    write(target / 'COSTS.json', costs, replace=True)
    gains = {str(b): 100 * (next(r['FPS'] for r in rows if r['method'] == 'complete' and r['budget'] == b)
                           / next(r['FPS'] for r in rows if r['method'] == 'tarnawski' and r['budget'] == b) - 1)
             for b in (1, 5, 20)}
    report = dict(status='COMPLETE', budgets=rows, FPS_gain_percent=gains,
        input=0, boot_id=boot, accepted_receipt_hashes=sources, unique_configurations=len(pool),
        native_pipeline_sessions=2 * len(pool), complete_board_DNN_inferences=49 * len(pool),
        single_input_single_boot_observation=True, significance_claimed=False,
        predicted_best_is_not_global_measured_optimum=True,
        independent_ResNet50_global_enumeration=False,
        numerical_error_is_not_selection_gain=True, cost_report=str(target / 'COSTS.json'),
        component_source_boot_ids=sorted({r['source']['boot_id'] for r in records.values()}),
        imported_component_sources=sum(bool(r['source'].get('archived_output')) for r in records.values()),
        thread_curves=dict(CPU_programs=len({r['program'] for r in curves}), measured_signatures=len(curves),
            non_monotonic_increases=sum(r['slower_than_previous_threads'] for r in curves),
            per_mask_measurement=False, run_only_worker_participation='unknown'),
        distinct_score_counts={m: len({r['score_ns'] for r in lists[m]}) for m in METHODS},
        prediction_summary={m: dict(
            mean_absolute_II_error_percent=statistics.mean(abs(r['II_error_percent']) for r in predictions if r['method'] == m),
            static_first_tie_count=sum(r['score_ns'] == lists[m][0]['score_ns'] for r in lists[m]),
            static_first_selection_loss_percent=100 * (results[lists[m][0]['id']]['II_ms']
                / min(results[r['id']]['II_ms'] for r in lists[m]) - 1),
            scope='own measured Top-20 only, not the full legal-domain optimum') for m in METHODS},
        required_tables=['CPU_thread_curves.csv', 'stage_isolated_pipeline.csv',
                         'configuration_scores.csv', 'constraint_scores.csv', 'COSTS.json'])
    write(target / 'RESULTS.json', report, replace=True)
    text = ['ResNet50-v2：冻结方法迁移验证', '',
            '同一启动、输入0、8预热＋16完成时间戳＋24尾帧；16时间戳对应15间隔。',
            '两种方法各测严格Top-20，重复执行配置共用本批测量。', '']
    for row in rows:
        text.append(f"{LABELS[row['method']]}／预算{row['budget']}：{row['FPS']:.6f} FPS，"
                    f"{row['II_ms']:.6f} ms，静态名次{row['static_rank']}。")
    text += ['', '完整模型相对Tarnawski平台适配的本批FPS变化：' + str(gains),
             '提升包含切图、线程／核心配置与评分共同作用，不全部归给某个评分项。',
             '本轮不证明跨启动稳定性、整个ResNet家族有效或候选域内实测全局最优。',
             'ResNet50仅由精确DP生成全域排名；独立核验范围另列，不冒称全域逐项二次枚举。',
             'CPU_thread_curves.csv保留每个CPU子图1—4线程真实曲线，包括非单调点；其他同核数集合采用同构核近似。',
             '线程份额覆盖完整隔离会话，不等于计入窗口计算期的逐线程精确份额；实际参与计算线程数未知。',
             'stage_isolated_pipeline.csv逐阶段比较服务墙钟，不把并发阶段差值相加解释整体间隔。',
             'COSTS.json将组件、DMA、完整网络和本地参考分别计数；未独立记录的初始化探针保持未知。']
    (target / 'FINDINGS_zh.txt').write_text('\n'.join(text) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected_programs = {s['program'] for method in METHODS for s in lists[method][0]['stages']
                         if s['device'] == 'cpu'}
    plot_thread_curves(curves, selected_programs, target)
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    for index, method in enumerate(METHODS):
        ax.bar([j + (index-.5)*.35 for j in range(3)],
               [r['FPS'] for r in rows if r['method'] == method], width=.35, label=LABELS[method])
    ax.set_xticks(range(3), ['Top-1', 'Best of top-5', 'Best of top-20'])
    ax.set_ylabel('Throughput (FPS)'); ax.legend(); fig.tight_layout()
    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(target / ('budget_comparison.' + ext), dpi=300)
    plt.close(fig)
    return dict(status='COMPLETE', results=str(target / 'RESULTS.json'), FPS_gain_percent=gains)
