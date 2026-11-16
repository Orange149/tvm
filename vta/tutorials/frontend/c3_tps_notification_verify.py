#!/usr/bin/env python3
"""Prepare, execute and seal the fixed B/Nprime/Gprime carrier comparison.

No schedule/module rebuild, new candidate, retry or historical timing splice.
The ARM runtime uses the byte-identical generated C++ qualified in local run02.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess

from c3_tps_aggregation_board import sha, write, verify_bindings
from c3_tps_aggregation_prepare import HERE, HOLD, OFFICIAL
from c3_tps_aggregation_protocol import manifest, analyse, _check_manifest
from c3_tps_aggregation_transport_v2 import transport_contract
from c3_tps_notification_probe import require

LOCAL = HOLD / '20261015_c3_tps_notification_local_run02'
PREVIOUS = HOLD / '20261015_c3_tps_aggregation_run02'
DISPLAY = {'B': 'B', 'N': 'N′', 'G': 'G′'}


def read(path):
    return json.loads(Path(path).read_text())


def sealed(directory):
    seal = read(directory / 'completion_manifest.json')
    bindings = {str(directory / p): value for p, value in seal['artifact_sha256'].items()}
    verify_bindings(bindings)
    bindings[str(directory / 'completion_manifest.json')] = sha(directory / 'completion_manifest.json')
    return bindings


def adoption(out):
    bindings = sealed(LOCAL)
    bindings.update(sealed(PREVIOUS))
    verify_bindings(read(LOCAL / 'freeze.json')['bindings'])
    old_freeze = read(PREVIOUS / 'freeze.json')['source_and_artifact_bindings']
    verify_bindings(old_freeze)
    bindings.update(old_freeze)
    bindings.update(read(LOCAL / 'freeze.json')['bindings'])
    artifacts = deepcopy(read(PREVIOUS / 'artifacts/prepared.json'))
    require(read(LOCAL / 'stage_result.json')['status'] == 'LOCAL_CARRIER_PREEXPERIMENT_COMPLETE',
            'LOCAL_CARRIER_ATTEMPT_NOT_COMPLETE')
    workers = []
    for variant in 'BNG':
        descriptor = artifacts['programs'][variant]
        verify_bindings({descriptor[k]: v for k, v in descriptor['sha256'].items()})
        from c3_tps_aggregation_prepare import compare_initializer_bindings
        arm = read(descriptor['arm_contract'])
        comparison = compare_initializer_bindings(Path(descriptor['fsim_llvm']).read_text(),
            Path(descriptor['arm_llvm']).read_text(), arm['fsim_cache_contract'], arm['arm_cache_contract'])
        require(comparison == arm['initializer_comparison'], 'ARM_INITIALIZER_ADOPTION_CHANGED')
        for field in ('compiler', 'sdk'):
            verify_bindings({arm[field]['path']: arm[field]['sha256']})
            bindings[arm[field]['path']] = arm[field]['sha256']
        for mode in ('detailed', 'minimal'):
            folder = LOCAL / (variant + '_' + mode)
            summary = read(folder / 'summary.json')
            require(summary['status'] == 'LOCAL_CARRIER_CERTIFICATES_AND_FSIM_PASSED'
                    and summary['target_calls'] == 7 and len(summary['calls']) == 7,
                    'CARRIER_WORKER_NOT_QUALIFIED')
            require(summary['variant'] == variant and summary['mode'] == mode
                    and summary['runtime_sha256'] == sha(LOCAL / 'runtime/libvta_fsim.so')
                    and summary['unchanged_module_sha256'] == descriptor['sha256']['fsim_module'],
                    'CARRIER_WORKER_IDENTITY_CHANGED')
            for i, call in enumerate(summary['calls'], 1):
                require(call['correct'] is True and call['output_sha256'] == call['reference_sha256']
                        and call['helper']['started_calls'] == call['helper']['completed_calls'] == i,
                        'CARRIER_CALL_NOT_QUALIFIED')
            workers.append({'variant': variant, 'mode': mode, 'path': str(folder),
                            'calls': 7, 'summary_sha256': sha(folder / 'summary.json')})
        descriptor['carrier_qualification'] = str(LOCAL / (variant + '_detailed/summary.json'))
        descriptor['display_name'] = DISPLAY[variant]
    write(out / 'adoption.json', {'status': 'TPS_CARRIER_42_LOCAL_CALLS_ADOPTED',
        'workers': workers, 'adopted_calls': 42, 'new_FSim_calls': 0,
        'historical_board_calls_reused': 0, 'previous_attempt': str(PREVIOUS),
        'previous_sealed_artifacts_verified': len(read(PREVIOUS / 'completion_manifest.json')['artifact_sha256']),
        'local_sealed_artifacts_verified': len(read(LOCAL / 'completion_manifest.json')['artifact_sha256']),
        'ARM_modules': 'exact previously certified binaries, initializer contracts and source bindings retained',
        'old_attempts_unchanged': True})
    return artifacts, bindings


def build_arm(out):
    from c3_official_window_runtime import run_compiler
    out.mkdir()
    base = OFFICIAL / 'runtime_arm'
    source_name = 'c3_official_window_runtime_generated.cc'
    local_build = read(LOCAL / 'runtime/build.json')
    require(sha(base / source_name) == local_build['parent_generated_sha256'],
            'ARM_BASE_NOT_LOCAL_QUALIFIED_PARENT')
    for name in (source_name, 'c3_official_window_runtime_collection.h',
                 'c3_official_window_runtime_uop_queue.h'):
        if name != source_name:
            require(sha(base / name) == sha(LOCAL / 'runtime' / name), 'ARM_LOCAL_HEADER_DIFFERENCE')
        shutil.copy2(LOCAL / 'runtime' / name, out / name)
    driver_name = 'c3_official_window_runtime_driver_generated.cc'
    shutil.copy2(base / driver_name, out / driver_name)
    command = [v.replace(str(base), str(out)) for v in read(base / 'build.json')['command']]
    result = run_compiler(command, 'arm')
    (out / 'build.log').write_text(result.stdout + result.stderr)
    write(out / 'build.json', {'returncode': result.returncode, 'command': command,
        'parent': str(base), 'runtime_source_sha256': sha(out / source_name),
        'qualified_local_source_sha256': sha(LOCAL / 'runtime' / source_name),
        'driver_source_sha256': sha(out / driver_name), 'local_generated_source_identical': True,
        'driver_unchanged': True, 'scan_and_erase_inside_timing': True,
        'new_FSim_calls': 0, 'target': 'AArch64 AXU5EVB',
        'binary_sha256': sha(out / 'libvta.so') if result.returncode == 0 else None})
    require(result.returncode == 0, 'ARM_CARRIER_BUILD_FAILED')
    header = subprocess.run(['readelf', '-h', str(out / 'libvta.so')],
                            check=True, capture_output=True, text=True).stdout
    (out / 'elf_header.txt').write_text(header)
    require('AArch64' in header, 'RUNTIME_NOT_AARCH64')
    return {'path': str(out / 'libvta.so'), 'sha256': sha(out / 'libvta.so')}


def prepare(args):
    out = args.output_dir
    test_log = Path(args.test_log).read_text()
    require(test_log.rstrip().endswith('OK') and 'Ran ' in test_log, 'TOOLS_NOT_TESTED')
    out.mkdir(exist_ok=False)
    (out / 'tests.log').write_text(test_log)
    artifact_dir = out / 'artifacts'
    artifact_dir.mkdir()
    artifacts, bindings = adoption(artifact_dir)
    artifacts['runtime'] = build_arm(out / 'runtime_arm')
    artifacts['adoption'] = str(artifact_dir / 'adoption.json')
    artifacts['status'] = 'TPS_CARRIER_ARM_PREPARED_FPGA_PENDING'
    reference = read(artifacts['programs']['B']['reference_contract'])
    require(all(read(d['reference_contract']) == reference for d in artifacts['programs'].values()),
            'REFERENCE_CONTRACTS_DIFFER')
    write(artifact_dir / 'prepared.json', artifacts)
    identities = {p: {'variant': p, 'display_name': DISPLAY[p], 'symbol': d['symbol'],
        'sha256': d['sha256'], 'runtime_sha256': artifacts['runtime']['sha256'],
        'helper_sha256': artifacts['helper']['sha256'], 'bitstream_sha256': artifacts['bitstream']['sha256'],
        'carrier_qualification_sha256': sha(d['carrier_qualification'])}
        for p, d in artifacts['programs'].items()}
    ledger = manifest(out.name, identities)
    write(out / 'call_manifest.json', ledger)
    write(out / 'transport_contract.json', transport_contract(ledger))
    lifetime = read(PREVIOUS / 'storage_lifetime.json')
    lifetime['carrier_runtime'] = artifacts['runtime']
    lifetime['allocation_change'] = False
    write(out / 'storage_lifetime.json', lifetime)
    restorations = []
    for dirname in args.restoration:
        dirname = Path(dirname).resolve()
        record = read(dirname / 'summary.json')
        require(record['status'] in ('VOLATILE_TASKSET_RESTORED', 'VOLATILE_PREREQUISITES_RESTORED')
                and record['SD_writes'] is False, 'RESTORATION_NOT_VERIFIED')
        restorations.append({'path': str(dirname), 'summary': record})
        bindings.update({str(p): sha(p) for p in dirname.iterdir() if p.is_file()})
    write(out / 'environment_restoration.json', restorations)
    # Preserve every certification/protocol source, plus new adapter and runner.
    sources = sorted(set([*HERE.glob('c3_tps_notification*'), *HERE.glob('test_c3_tps_notification*'),
                          *HERE.glob('c3_tps_aggregation_*.py')]))
    snapshot = out / 'tool_sources'
    snapshot.mkdir()
    for source in sources:
        if source.is_file():
            shutil.copy2(source, snapshot / source.name)
            bindings[str(source)] = sha(source)
    for key in ('runtime', 'helper', 'storage_helper', 'runtime_tvm', 'rpc_server', 'bitstream', 'hardware_archive'):
        bindings[artifacts[key]['path']] = artifacts[key]['sha256']
    for program in artifacts['programs'].values():
        bindings.update({program[k]: v for k, v in program['sha256'].items()})
    for path in out.rglob('*'):
        if path.is_file():
            bindings[str(path)] = sha(path)
    verify_bindings(bindings)
    write(out / 'freeze.json', {'status': 'READY_FOR_FPGA_GATES', 'source_and_artifact_bindings': bindings,
        'adopted_FSim_calls': 42, 'new_FSim_calls': 0, 'planned_FPGA_calls': 180,
        'planned_sessions': 30, 'planned_formal_samples': 54,
        'scope': 'B/Nprime/Gprime; same new runtime; fixed six carrier NOP deletions in N/G only',
        'timing': 'complete fused call including queue rewrite; minimal counters; no per-command dump',
        'primary': 'Gprime/B', 'stopping': 'one attempt; no retry, sample replacement or expanded pool'})
    return {'status': 'READY_FOR_FPGA_GATES', 'directory': str(out), 'calls': 180}


def board(args):
    from c3_tps_notification_board import execute
    out = args.output_dir
    frozen = read(out / 'freeze.json')
    verify_bindings(frozen['source_and_artifact_bindings'])
    ledger = read(out / 'call_manifest.json')
    _check_manifest(ledger)
    args.transport_contract = read(out / 'transport_contract.json')
    require(args.transport_contract == transport_contract(ledger), 'TRANSPORT_PLAN_CHANGED')
    bindings = {**frozen['source_and_artifact_bindings'], str(out / 'freeze.json'): sha(out / 'freeze.json')}
    return execute(args, read(out / 'artifacts/prepared.json'), ledger, out / 'board', bindings)


def report(args):
    out = args.output_dir
    summary = read(out / 'board/summary.json')
    complete = summary['status'] == 'COMPLETE_SUCCESS'
    ledger = read(out / 'call_manifest.json')
    result_file = out / 'board/results.jsonl'
    results = [json.loads(line) for line in result_file.read_text().splitlines()] if result_file.exists() else []
    verify_bindings(read(out / 'freeze.json')['source_and_artifact_bindings'])
    organizations = []
    for ordinal, session in enumerate(ledger['sessions'], 1):
        for position in range(1, 7):
            path = out / 'board' / ('session_%02d' % ordinal) / ('native_%d.json' % position)
            if path.exists():
                cert = read(path)
                organizations.append({'session_id': session['session_id'], 'program': session['program'],
                    'rotation': session['rotation'], 'position': position,
                    'organization': cert['organization'], 'removed_NOPs': len(cert['rewrite']['edits'])})
    write(out / 'organization.json', organizations)
    text = ['# TPS通知承载与指令聚合板测', '',
        '固定对象：YOLOv3-tiny-224节点24，128→256、14×14、3×3；Tile b1h14w14ci4co4，oc2/h1。', '',
        'B为原TPS；N′为调整bank通知后的程序，使用已认证的通知承载规则；G′在同一通知安排下连接两份μop清单。三版共同使用新的ARM runtime。', '',
        '本轮只采纳本地run02的42次FSim资格，没有重新执行FSim；ARM算子模块与initializer绑定合同完全复用。新的ARM runtime与已资格FSim使用相同生成C++源码，板驱动及执行配置延续冻结底座。', '',
        '## 能不能合？', '']
    if complete:
        stats = analyse(ledger, results)
        write(out / 'statistics.json', stats)
        require(len(organizations) == 54, 'DIAGNOSTIC_CENSUS_NOT_54')
        text += ['180次完整FPGA调用全部通过，30个fresh RPC单元，正式样本54个。三seed资格、同一权重存储W0→W1→W0、逐元素输出、只读参数和54次详细诊断均通过。', '',
            '每次详细诊断核对实际改写、非NOP载荷顺序、所有μop逐引用内容、通知配对及有限FIFO进度。改写前队列只被捕获而未执行；实际driver提交与完成记录属于改写后队列。正式最小模式按自身暖诊断核对提交路径。', '']
    else:
        text += [f"**本attempt已停止。** 开始{summary['begun_calls']}次，完整通过{summary['passed_calls']}次。原因：`{summary['error']}`。保留失败，不重试、不补样、不生成性能结论。", '']
    text += ['## 能合多少？', '', '每段仍只合两个bank、4704条基址μop，UOP SRAM容量8192条。本轮未改聚合量、LOAD合并、缓存分配、Tile或张量请求。', '',
        '|程序|乘加GEMM|RESET|ALU|总指令|compute NOP|UOP LOAD|LOAD字节|回绕|提交|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for p in 'BNG':
        rows = [x['organization'] for x in organizations if x['program'] == p]
        if rows:
            keys = ('MAC_GEMM', 'RESET_GEMM', 'ALU', 'all_instructions', 'compute_NOP', 'UOP_LOAD',
                    'UOP_LOAD_bytes', 'UOP_SRAM_wraps', 'submissions')
            require(len({tuple(r[k] for k in keys) for r in rows}) == 1, 'ORGANIZATION_UNEXPLAINED_VARIATION')
            text.append('|' + DISPLAY[p] + '|' + '|'.join(str(rows[0][k]) for k in keys) + '|')
    text += ['', '三版每提交序列化37760字节。实际回绕、首次装载、容量驱逐重装和提交重置重装逐调用见[容量账](organization.json)。NOP减少并不意味着通知依赖减少；两条规则保持原N/G的实质依赖关系。', '',
        '## 合了值不值？', '']
    if complete:
        text += ['|程序|六轮中位时延T（ms）|轮中位数范围（ms）|', '|---|---:|---|']
        for p in 'BNG':
            values = stats['round_medians'][p]
            text.append(f'|{DISPLAY[p]}|{stats["T"][p]:.6f}|{min(values):.6f}–{max(values):.6f}|')
        text += ['', '|对照|逐轮改善中位数|严格胜场|绝对节省中位数ms|', '|---|---:|---:|---:|']
        for pair in ('N/B', 'G/N', 'G/B'):
            value = stats['pairwise'][pair]
            display = '/'.join(DISPLAY[p] for p in pair.split('/'))
            text.append(f'|{display}|{value["Gain_percent"]:.4f}%|{value["wins"]}/6|{value["absolute_ms"]:.6f}|')
        text += ['', '**主要结果G′/B：' + ('达到' if stats['engineering_goal_met'] else '未达到') +
            '冻结工程门槛（Gain≥2%，严格胜场≥5/6）。**', '',
            '|轮次|B ms|N′ ms|G′ ms|N′/B %|G′/N′ %|G′/B %|', '|---|---:|---:|---:|---:|---:|---:|']
        for i in range(6):
            vals = [stats['round_medians'][p][i] for p in 'BNG'] + [
                stats['pairwise'][p]['round_gains_percent'][i] for p in ('N/B', 'G/N', 'G/B')]
            text.append('|' + str(i+1) + '|' + '|'.join('%.6f' % v for v in vals) + '|')
        text += ['', 'Gain为六个逐轮比值的中位数；不能用两个T直接相除替代。完整范围、MAD及标准差见[统计数据](statistics.json)。', '',
            'N′/B反映通知调整、编码和必要缓存变化的整体影响；G′/N′反映相同通知安排下聚合增量；G′/B是整套方案净收益。新runtime的扫描和删除指令开销计入完整调用，未增加开关消融。', '',
            '资格首调用使用详细模式，其成本单列在[首调用记录](qualification_first_calls.json)，不与最小模式正式时延相减解释缓存开销。旧板测180次结果保持独立，不拼接或通过跨日差值归因NOP效果。']
        write(out / 'qualification_first_calls.json', [r for r in results if r['position'] == 1 and
                                                     r['phase'] == 'qualification'])
    else:
        text += ['没有完整成功的账目，不对已返回的时延做性能统计。']
    text += ['', '结论限于当前TPS双bank表示，不称通用聚合已完成、统计显著性、超过TopHub或整网加速。', '',
        '## 证据与暂停', '',
        '- [资格采纳](artifacts/adoption.json)、[三版ARM身份](artifacts/prepared.json)、[runtime构建](runtime_arm/build.json)',
        '- [冻结身份](freeze.json)、[预登记180次调用清单](call_manifest.json)、[权重生命周期](storage_lifetime.json)',
        '- [板测状态](board/summary.json)、[逐调用账目](board/results.jsonl)、board目录内的原始诊断', '',
        '**本轮结束后暂停，不追加候选或性能选优。**', '']
    (out / 'REPORT.md').write_text('\n'.join(text))
    write(out / 'completion_manifest.json', {'status': summary['status'], 'paused': True,
        'artifact_sha256': {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*')) if p.is_file()}})
    return {'status': summary['status'], 'report': str(out / 'REPORT.md'), 'paused': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'board', 'report'))
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--restoration', type=Path, action='append', default=[])
    parser.add_argument('--test-log', type=Path)
    parser.add_argument('--host', default='192.168.1.247')
    parser.add_argument('--known-hosts', default='/home/orange/.ssh/known_hosts')
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    print(json.dumps(globals()[args.phase](args)), flush=True)


if __name__ == '__main__':
    main()
