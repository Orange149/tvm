"""Locate residuals of the strongest static score in an already measured pool.

This post-hoc diagnostic audit produces no candidate ranking or new Top20.
All score inputs/formulas stay frozen; measured times select diagnostic cases
only after every scalar score has been sealed.
"""
from collections import defaultdict
from pathlib import Path
import json

import run_scope_rank as rank
from cpu_vta_partitioning.common import read, write, sha256, verify_inventory
from model_ablation.model import score as complete_score
from resource_envelope.model import score as details_score
from common_top20.runtime import summary as native_summary
from scheduler_gap_audit.curves import csv_write

ROOT = rank.ROOT
OUT = ROOT / 'artifacts/formal_gap_coverage_01'
REPORT = ROOT.parents[4] / 'paper/my_paper/analysis/20261111_formal_gap_coverage'
PARENT = ROOT / 'artifacts/cross_network_model_01/common_top20'
NETWORKS = ('resnet18', 'yolov3_tiny416')


def run():
    if OUT.exists():
        raise FileExistsError('preserve the frozen gap coverage audit')
    plan = rank.assert_plan()
    parent = read(PARENT / 'SHORT_RESULTS.json')
    source_hashes = {str(PARENT / 'SHORT_RESULTS.json'): sha256(PARENT / 'SHORT_RESULTS.json'),
                     str(rank.OUT / 'PLAN.json'): sha256(rank.OUT / 'PLAN.json')}
    costs, programs, configs, receipts, scores = {}, {}, {}, {}, {}
    inventories = {}
    for network in NETWORKS:
        programs[network], costs[network], _ = rank.inputs_for(plan, network, 'original')
        if parent['networks'][network]['component_hashes']['isolated_costs.json'] != plan['networks'][network]['cost_sha256']:
            raise ValueError('historical native pool used different component costs')
        inventories[network] = read(PARENT / network / 'capsule/plans/programs.json')
    # Only execution metadata are projected here. Performance is read below,
    # after all exact, unchanged scalar scores are written to SCORE_FREEZE.
    for label, pair in parent['source_receipts'].items():
        path = pair['short']['path']
        if sha256(path) != pair['short']['sha256']:
            raise ValueError('frozen native receipt changed')
        receipt = read(path)
        task = receipt['task']
        network = task['network']
        cfg = task['configuration']
        if label != network + '/' + cfg['id'] or (task['warmup'], task['scored'], task['tail']) != (8, 16, 4):
            raise ValueError('historical pool identity/window differs')
        if task['detailed'] or task['kind'] != 'performance':
            raise ValueError('pool includes an instrumented or nonperformance session')
        for stage in cfg['stages']:
            a, b = programs[network][stage['program']], inventories[network][stage['program']]
            for name in ('graph.json', 'module.so', 'params.bin'):
                if a['files'][name]['sha256'] != b['files'][name]['sha256']:
                    raise ValueError('native and isolated-cost modules differ')
        details = details_score(cfg, costs[network])
        exact = complete_score(cfg, costs[network], 'complete')
        if exact['score_ns'] != max(details['original'], details['host_concentration_guard_ns'], details['CPU_team_guard_ns']):
            raise ValueError('strongest complete formula not reproduced')
        configs[label] = cfg
        scores[label] = dict(score=exact, details=details,
            isolated_stage_service_ms=[costs[network].get(s, cfg)['service_ms'] for s in cfg['stages']])
        receipts[label] = pair
    OUT.mkdir(parents=True)
    write(OUT / 'SCORE_FREEZE.json', dict(scores=scores, configurations=configs,
        source_hashes=source_hashes, original_component_inputs_unchanged=True,
        full_original_formula=True, no_new_model_or_Top20=True,
        diagnostic_case_rule='For each network, largest positive residual among multi-thread configurations and largest negative residual overall; ties by full id.',
        source_sha256=sha256(__file__)))
    rows = []
    for label, pair in receipts.items():
        receipt = read(pair['short']['path'])
        network, cid = label.split('/')
        cfg, values = configs[label], scores[label]
        actual = receipt['summary']['II_ms']
        predicted = values['score']['score_ns'] / 1e6
        cpu = [s for s in cfg['stages'] if s['device'] == 'cpu']
        row = dict(network=network, configuration_id=cid, actual_II_ms=actual,
            complete_predicted_II_ms=predicted, actual_minus_prediction_ms=actual - predicted,
            relative_residual_percent=100 * (actual / predicted - 1),
            has_multithread=any(s['threads'] > 1 for s in cpu),
            binding_terms=values['score']['binding_terms'], terms_ns=values['score']['terms_ns'],
            context=cfg['context_id'], execution_mode=cfg['execution_mode'],
            CPU_stages=[{k: s[k] for k in ('program', 'threads', 'mask')} for s in cpu],
            isolated_stage_service_ms=values['isolated_stage_service_ms'],
            actual_stage_service_ms=receipt['summary']['stage_service_ms'],
            boot_id=receipt['boot_id'], executor_sha256=receipt['task']['executor_sha256'])
        old = next(r for r in parent['rows'] if r['network'] == network and r['configuration_id'] == cid)
        if old['II_ms'] != actual:
            raise ValueError('parent performance table and receipt differ')
        rows.append(row)
    focus = []
    for network in NETWORKS:
        selected = [r for r in rows if r['network'] == network]
        multi = [r for r in selected if r['has_multithread']]
        for kind, row in [('remaining_multithread_underestimate', min(multi, key=lambda r: (-r['actual_minus_prediction_ms'], r['configuration_id']))),
                          ('conservative_overestimate', min(selected, key=lambda r: (r['actual_minus_prediction_ms'], r['configuration_id'])))]:
            label = network + '/' + row['configuration_id']
            proof = []
            fingerprints = None
            for phase in ('qualify', 'short'):
                ref = receipts[label][phase]
                if sha256(ref['path']) != ref['sha256']:
                    raise ValueError('focused raw receipt changed')
                rec = read(ref['path'])
                verify_inventory(rec['raw'], rec['files'])
                calc = native_summary(rec['raw'], rec['task'], PARENT / network / 'capsule')
                if calc != rec['summary']:
                    raise ValueError('focused native raw no longer replays')
                if fingerprints is not None and calc['output_fingerprints'] != fingerprints:
                    raise ValueError('focused qualification and performance outputs differ')
                fingerprints = calc['output_fingerprints']
                proof.append(dict(phase=phase, **ref, raw_files_rehashed=True, native_summary_exact=True))
            focus.append(dict(kind=kind, **row, raw_proof=proof,
                              next_use='stage-gap diagnosis only; never injected into Top20'))
    # Existing detailed-trace layout: compare its formal II score separately
    # from the already-frozen experimental stage-service replay.
    current_path = ROOT / 'artifacts/current_model_gap_01/INPUTS.json'
    current = read(current_path)
    current_cfg = current['case']['configuration']
    current_static = complete_score(current_cfg, costs['yolov3_tiny416'], 'complete')
    native_path = ROOT / 'artifacts/yolo_switch_without_queries_01/analysis/RESULTS.json'
    native_manifest = read(native_path.parent / 'MANIFEST.json')
    if sha256(native_path) != native_manifest['files'][str(native_path)]:
        raise ValueError('latest native diagnostic result changed')
    native = read(native_path)
    current_case = dict(configuration_id=current_cfg['id'], formal_score_ns=current_static,
        clean_II_ms=native['clean']['II_ms'], clean_front_service_ms=native['clean']['stage_service_ms'][0],
        diagnostic_replay_front_run_ms=192.52706458333336,
        diagnostic_replay_is_not_formal_candidate_II=True,
        native_executor_is_optional_diagnostic_runner_not_legacy_formal_executor=True,
        no_exact_cross_executor_accuracy_claim=True,
        sources={str(current_path): sha256(current_path), str(native_path): sha256(native_path)})
    summaries = []
    for network in NETWORKS:
        selected = [r for r in rows if r['network'] == network]
        summaries.append(dict(network=network, configurations=len(selected),
            MAE_ms=sum(abs(r['actual_minus_prediction_ms']) for r in selected) / len(selected),
            under_by_more_than_ten_percent=sum(r['relative_residual_percent'] > 10 for r in selected),
            multithread_cases=sum(r['has_multithread'] for r in selected),
            scope='old measured pool only, not full-domain validation or a newly selected Top20'))
    result = dict(summary=summaries, focused_cases=focus, detailed_trace_case=current_case,
        source_hashes=source_hashes, score_freeze_sha256=sha256(OUT / 'SCORE_FREEZE.json'),
        source_sha256=sha256(__file__), no_new_search_or_ranking=True,
        no_new_board_or_model_execution=True, no_new_performance_or_improvement_claim=True,
        receipt_hashes_checked_for_all_cases=True, full_raw_checked_only_for_focused_cases=True)
    write(OUT / 'RESULTS.json', result)
    csv_write(OUT / 'measured_pool_residuals.csv', rows)
    REPORT.mkdir(parents=True, exist_ok=True)
    text = ['阶段误差应对应到正式静态评分的剩余问题', '',
        '本轮没有板测、DP搜索、改公式或生成新Top20。先封存全部原公式标量分数，再用旧实测定位诊断对象。',
        '复用同一组件成本；全部158项凭据哈希核验，选出的4个诊断案例另对资格与性能原始文件全量哈希并重放验收。', '',
        '先前详细追踪的YOLO五阶段布局：基础容量约164.490ms；完整模型主机集中约束228.767ms。',
        f'本批无探针完成间隔为{native["clean"]["II_ms"]:.3f}ms。它是诊断执行器，并非原正式执行器的严格同身份复测。',
        '因此诊断回放前段192.527ms低于流水前段，并不等于正式候选评分也只给192.527ms。两类输出必须分开。', '',
        '旧实测池中，最强完整模型仍低估的多线程案例 / 预测II / 实测II / 差ms']
    for row in focus:
        if row['kind'] == 'remaining_multithread_underestimate':
            text.append(f'{row["network"]} {row["configuration_id"]} / {row["complete_predicted_II_ms"]:.3f} / {row["actual_II_ms"]:.3f} / {row["actual_minus_prediction_ms"]:.3f}')
            text.append('CPU布局：' + json.dumps(row['CPU_stages'], ensure_ascii=False))
    text += ['', '同时保留过度悲观的反例，避免只会加惩罚：']
    for row in focus:
        if row['kind'] == 'conservative_overestimate':
            text.append(f'{row["network"]} {row["configuration_id"]}：预测{row["complete_predicted_II_ms"]:.3f}ms，实测{row["actual_II_ms"]:.3f}ms，主导{row["binding_terms"]}。')
    text += ['',
        '结论：剩余问题同时包含某些开放共享布局仍偏乐观、某些固定分组/共享主机布局被主机集中项过度惩罚。不能把所有配置再统一增加主机或线程等待成本。',
        '新诊断优先选YOLO前段仍为CPU0–2四线程、但中段变为CPU4三线程的未覆盖布局；原详细案例只作机制参照，不再把其35ms回放差当作正式模型漏计量。',
        '这些案例按旧误差挑选，属于开发诊断，不是未见配置验证；不声称新排序、全域预测改善或最终选点收益。',
        '下一步先检查真实资源请求/获得次序和同核竞争，尤其区分CPU完成拖尾与单VTA重入、队列反压；只有可事前取得的输入和统一方法才能用于修正。',
        '完整原合法域、最强ResNet约13FPS结果、旧名单及原始数据均保持不变。']
    report = REPORT / 'FINDINGS_zh.txt'
    report.write_text('\n'.join(text) + '\n')
    paths = [OUT / 'SCORE_FREEZE.json', OUT / 'RESULTS.json', OUT / 'measured_pool_residuals.csv', report]
    write(OUT / 'MANIFEST.json', dict(files={str(p): sha256(p) for p in paths}))
    print(json.dumps(dict(summary=summaries, focused=[{k: r[k] for k in ('network', 'kind', 'configuration_id',
        'complete_predicted_II_ms', 'actual_II_ms')} for r in focus], report=str(report)), ensure_ascii=False))
    return result


if __name__ == '__main__':
    run()
