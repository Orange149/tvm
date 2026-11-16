"""User-selected YOLO416 nodes 34/46, using frozen TH/P operators unchanged.

I is an explicitly proved alias of P when the content-equivalence transform
returns the same function. No synthetic third sample or new schedule is made.
The existing executor owns the exact-call, correctness and stop contracts.
"""
import argparse
import csv
from pathlib import Path
import shutil

from c3_tps_sync_release_local import HERE, HOLD, read, write, sha, verify, require
from c3_yolo416_comparison_protocol import manifest, canonical, analyse
from c3_yolo416_board_transport import transport_contract

OBJECTS = ('yolov3_tiny_416_node034', 'yolov3_tiny_416_node046')
SOURCE = HOLD/'20261017_c3_yolo416_TH_P_board_run02'
COMPILED = HOLD/'20261017_c3_yolo416_tps_prepare_run02'


def select_pool(ledger):
    require(canonical(ledger) == canonical(manifest(ledger['attempt'], ledger['pool'])),
            'SOURCE_LEDGER_CHANGED')
    pool = [g for g in ledger['pool'] if g['object_id'] in OBJECTS]
    require(tuple(g['object_id'] for g in pool) == OBJECTS and
            all(g['kind'] == 'operator' for g in pool), 'EXACT_USER_SELECTED_NODES_REQUIRED')
    require(all(g['representatives'] == {'TH': 'TH', 'P': 'P'} for g in pool),
            'TH_P_PAIR_UNEXPECTEDLY_IDENTICAL')
    return pool


def source_bindings(frozen):
    """Only the explicitly external SSH trust file may be rebound.

    Every compiler, program, proof, helper and runtime byte must still match.
    StrictHostKeyChecking remains enabled in the inherited transport.
    """
    bindings = dict(frozen['bindings'])
    path = str(Path(frozen['known_hosts']).resolve())
    previous = bindings.pop(path, None)
    verify(bindings)
    current = sha(path)
    bindings[path] = current
    return bindings, {'path': path, 'historical_sha256': previous,
                      'current_sha256': current, 'changed': previous != current,
                      'program_or_runtime_changed': False,
                      'StrictHostKeyChecking': True}


def identity_alias(descriptor):
    import tvm
    import vta  # Register the VTA TIR operators before loading archived IR.
    from c3_tps_sync_release_prepare import complete_function
    from c3_uop_identity_method import transform
    parent = tvm.ir.load_json(Path(descriptor['parent_module']).read_text())
    _, fn = complete_function(parent, descriptor['symbol'])
    candidate, proof = transform(fn)
    require(not proof['changes'] and tvm.ir.structural_equal(fn, candidate),
            'I_IS_NOT_P_REQUIRES_SEPARATE_QUALIFICATION')
    require(proof['census']['status'] == 'FINITE_DOMAIN_CONTENT_SCREEN_COMPLETE' and
            not proof['census']['unresolved'] and proof['census']['MAC_redundant_identities'] == 0,
            'IDENTITY_SUPPORT_OR_REDUNDANCY_CHANGED')
    return {'status': 'I_EQUALS_P_EXACT_FUNCTION_NO_OP', 'method_proof': proof,
            'parent_sha256': sha(descriptor['parent_module']),
            'execution_identity': descriptor['execution_identity'],
            'arm_module_sha256': sha(descriptor['arm_module']),
            'measurement_representative': 'P', 'independent_I_measurement': False,
            'new_target_calls': 0}


def prepare(root, source=SOURCE):
    require(not root.exists(), 'OUTPUT_ALREADY_EXISTS')
    require(not (source/'board').exists(), 'SOURCE_OPERATOR_POOL_ALREADY_EXECUTED')
    frozen = read(source/'freeze.json')
    require(frozen['status'] == 'TH_P_OPERATORS_AND_GRAPH_READY_FPGA_PENDING', 'SOURCE_NOT_READY')
    bindings, trust = source_bindings(frozen)
    ledger = manifest(root.name, select_pool(read(source/'call_manifest.json')))
    require((ledger['unique_programs'], ledger['total_target_calls'],
             len(ledger['sessions']), ledger['formal_samples']) == (4, 240, 40, 72),
            'TWO_NODE_CALL_BUDGET_CHANGED')
    artifacts = read(source/'artifacts.json')
    artifacts['programs'] = {key: value for key, value in artifacts['programs'].items()
                             if key.split('/')[0] in OBJECTS}
    require(len(artifacts['programs']) == 4, 'OPERATOR_DESCRIPTOR_COVERAGE_CHANGED')
    aliases = {oid: identity_alias(artifacts['programs'][oid+'/P']) for oid in OBJECTS}
    rows = {r['object_id']: r for r in read(COMPILED/'selection.json')}
    selections = [rows[oid] for oid in OBJECTS]
    for oid in OBJECTS:
        th, p = (artifacts['programs'][oid+'/'+v] for v in ('TH', 'P'))
        require(th['abi'] == p['abi'] and th['model_cases'] == p['model_cases'],
                'FUSION_ABI_OR_INPUT_CASE_CHANGED')
    root.mkdir()
    write(root/'artifacts.json', artifacts)
    write(root/'call_manifest.json', ledger)
    write(root/'transport_contract.json', transport_contract(ledger))
    write(root/'identity_aliases.json', aliases)
    write(root/'selection.json', selections)
    write(root/'source_adoption.json', {'source': str(source), 'selection_basis':
          'user explicitly requested YOLO416 nodes 34 and 46; no per-operator timing selection',
          'source_measurements_adopted': 0, 'new_local_target_calls': 0,
          'local_qualification': 'existing 14 calls per TH/P function; frozen bindings verified',
          'SSH_trust_rebinding': trust, 'Tile_threads_tensor_requests_unchanged': True,
          'I_is_P_alias': True})
    bindings[str(source/'freeze.json')] = sha(source/'freeze.json')
    for name in ('c3_yolo416_two_node_comparison.py', 'test_c3_yolo416_two_node_comparison.py',
                 'c3_uop_identity_method.py', 'c3_uop_identity_applicability.py'):
        path = HERE/name
        bindings[str(path)] = sha(path)
    for path in root.iterdir():
        bindings[str(path.resolve())] = sha(path)
    verify(bindings)
    write(root/'freeze.json', {**frozen, 'bindings': bindings,
          'target_calls': 240, 'sessions': 40, 'formal_samples': 72,
          'comparison_stratum': 'user_selected_two_operators',
          'source_pool': str(source), 'I_measurement_alias': 'P'})
    return {'unique_programs': 4, 'target_calls': 240, 'formal_samples': 72,
            'I_equals_P': True, 'new_local_target_calls': 0}


def report(root):
    require(not (root/'completion_manifest.json').exists(), 'ATTEMPT_ALREADY_SEALED')
    summary = read(root/'board/summary.json')
    success = summary['status'] == 'COMPLETE_TH_P_OPERATORS_AND_GRAPH_SINGLE_BOOT_ATTEMPT'
    text = ['# YOLO416节点34和46的TopHub与TPS性能对照', '']
    if success:
        frozen = read(root/'freeze.json'); verify(frozen['bindings'])
        ledger = read(root/'call_manifest.json'); rows = read(root/'board/results.json')
        account = read(root/'board/call_accounting.json')
        require(account['registered_attempts'] == account['helper_started_observed_lower_bound'] ==
                account['helper_completed_observed_lower_bound'] == account['certified'] == 240 and
                not account['execution_unresolved_units'] and not account['invalid_records'],
                'CALL_ACCOUNTING_NOT_CLOSED')
        require(all(r['output_checks']['correct'] for r in rows), 'OUTPUT_CHECK_FAILED')
        stats = analyse(ledger, rows); write(root/'statistics.json', stats)
        aliases = read(root/'identity_aliases.json')
        exported = []
        text += ['仅测YOLOv3-tiny-416的两个完整融合算子。TopHub采用精确缓存配置在当前板上重实例化；'
                 'TPS沿用已经资格化的本地适配程序。三种方案保持实际输入、训练权重及融合语义一致。', '',
                 '|节点|TopHub ms|TPS ms|TPS加身份复用 ms|TPS相对TopHub配对改善|胜场|',
                 '|---|---:|---:|---:|---:|---:|']
        for item in stats['objects']:
            oid = item['object_id']; require(item['status'] == 'MEASURED_FULL_CALL', 'PAIR_NOT_MEASURED')
            require(aliases[oid]['status'] == 'I_EQUALS_P_EXACT_FUNCTION_NO_OP', 'ALIAS_PROOF_MISSING')
            th, p = item['T_ms']['TH'], item['T_ms']['P']
            text.append('|%s|%.6f|%.6f|%.6f（同TPS）|%+.4f%%|%d/6|' %
                        (oid[-3:], th, p, p, item['Gain_P_over_TH_percent'], item['strict_wins']))
            for label, value, rep in (('TopHub', th, 'TH'), ('TPS', p, 'P'), ('TPS+identity', p, 'P')):
                exported.append({'object_id': oid, 'variant': label, 'T_ms': value,
                    'operator_calls_per_second': 1000/value, 'measurement_representative': rep,
                    'independent_measurement': label != 'TPS+identity'})
        with (root/'three_way_performance.csv').open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(exported[0])); writer.writeheader(); writer.writerows(exported)
        write(root/'three_way_performance.json', exported)
        organization = []
        for i, unit in enumerate(ledger['sessions'], 1):
            if unit['phase'] != 'diagnostic': continue
            for position in range(1, 7):
                path = root/'board'/('session_%03d'%i)/('organization_%02d.json'%position)
                organization.append({'object_id': unit['object_id'], 'variant': unit['variant'],
                    'rotation': unit['rotation'], 'position': position, 'organization': read(path)})
        write(root/'organization.json', organization)
        write(root/'qualification_first_calls.json', [r for r in rows if r['phase'] == 'qualification' and r['position'] == 1])
        text += ['', '## 能不能复用', '',
                 '当前内容等价判定器在两份TPS上均返回零改写，函数结构完全相同；每份TPS的MAC身份数与内容类数均为1。'
                 '因此TPS加身份复用沿用同一ARM模块和同一组TPS样本，没有第三组独立实测，也没有虚构0%差异样本。', '',
                 '## 能复用多少', '',
                 '这两个固定oc1/h1表示没有同一静态kernel内可归并的冗余缓存身份。'
                 '未更换Tile、线程或数据安排来制造机会。编译、完整融合FSim及ARM initializer绑定采纳原封存资格。', '',
                 '## 完整调用值不值', '',
                 '240次完整算子调用全部通过，40个fresh RPC单元，72个正式样本。三seed、同实际权重存储W0→W1→W0、'
                 '输出哨兵及只读参数均检查；72次详细诊断认证实际指令、μop、缓存和有限FIFO进度。',
                 '每轮各seed正式调用一次，轮内取中位数m，T为六个m的中位数；改善为六个100×(1−m[TPS]/m[TH])的中位数。'
                 '全部六轮、绝对节省、范围和MAD见statistics.json。',
                 '正式采用最小采集模式，单次helper计入完整融合函数的命令组织、提交及等待。'
                 'RPC、参数准备、权重更新、初始化及输出校验在计时外；首调用单列。',
                 'CSV中的1000/T为单算子调用率，不是DNN整图FPS。三条方案线中TPS和TPS加优化重合，标注共享测量。',
                 '本次仅支持这两个固定程序的同启动受控比较；TopHub原搜索预算未知，TPS解析流量目标不保证本板最低时延。'
                 'TPS相对TopHub的变化属于底座对照；这两个表示未产生身份复用的增量收益。',
                 '逐调用原始证据位于board目录；历史YOLO224正例和YOLO416整图结果均未混入统计。完成后暂停。']
    else:
        failure = read(root/'board/failure.json')
        text += ['本attempt停止，不能形成性能结论。', '', str(failure),
                 '全部失败证据保留，不重试、不补样、不拼接。']
    (root/'REPORT.md').open('x').write('\n'.join(text)+'\n')
    snapshots = root/'source_snapshots'; snapshots.mkdir()
    for file, digest in read(root/'freeze.json')['bindings'].items():
        path = Path(file)
        if path.parent == HERE and path.suffix in ('.py', '.cc'):
            require(sha(path) == digest, 'CLOSEOUT_SOURCE_CHANGED')
            shutil.copyfile(path, snapshots/path.name)
    write(root/'completion_manifest.json', {'status': 'SEALED_COMPLETE_TWO_NODE_COMPARISON' if success else
          'SEALED_STOPPED_TWO_NODE_COMPARISON', 'performance_conclusion_allowed': success,
          'artifact_sha256': {str(p.relative_to(root)): sha(p) for p in root.rglob('*') if p.is_file()}})
    return {'success': success}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=('prepare', 'board', 'report'))
    parser.add_argument('output', type=Path); args = parser.parse_args(); root = args.output.resolve()
    if args.phase == 'prepare': print(prepare(root))
    elif args.phase == 'report': print(report(root))
    else:
        from c3_yolo416_board_execute import execute
        raise SystemExit(execute(root))
