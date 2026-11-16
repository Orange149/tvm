#!/usr/bin/env python3
"""Independent prepare/board/report entry for the fixed TPS B/N/G experiment."""
import argparse
import json
from pathlib import Path
import re
import shutil
import sys

from c3_tps_aggregation_board import sha, write, verify_bindings
from c3_tps_aggregation_protocol import manifest, analyse, _check_manifest

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def freeze(out):
    from c3_tps_aggregation_prepare import prepare_artifacts
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'freeze.json').exists() or (out / 'board').exists():
        raise ValueError('ATTEMPT_ALREADY_FROZEN_OR_STARTED')
    artifact_path = out / 'artifacts/prepared.json'
    artifacts = read(artifact_path) if artifact_path.exists() else prepare_artifacts(out / 'artifacts')
    if artifacts['status'] != 'TPS_ARM_ARTIFACTS_PREPARED_FPGA_PENDING':
        raise ValueError('PREPARATION_NOT_QUALIFIED')
    verify_bindings(artifacts['source_bindings'])
    for program in artifacts['programs'].values():
        verify_bindings({program[k]: v for k, v in program['sha256'].items()})
    reference = read(artifacts['programs']['B']['reference_contract'])
    if any(read(p['reference_contract']) != reference for p in artifacts['programs'].values()):
        raise ValueError('THREE_PROGRAM_REFERENCES_DIFFER')
    identities = {p: {'variant': p, 'symbol': d['symbol'], 'sha256': d['sha256'],
                      'runtime_sha256': artifacts['runtime']['sha256'],
                      'helper_sha256': artifacts['helper']['sha256'],
                      'bitstream_sha256': artifacts['bitstream']['sha256']}
                  for p, d in artifacts['programs'].items()}
    ledger = manifest(out.name, identities)
    write(out / 'call_manifest.json', ledger)
    # This runtime's ext_dev data pointer is an immutable DataBuffer identity,
    # not a physical address. Its single allocation owns a stable backing store.
    runtime_source = HERE.parents[2] / 'vta/runtime/runtime.cc'
    runtime_text = runtime_source.read_text()
    if len(re.findall(r'buffer->data_\s*=\s*data;', runtime_text)) != 1 or \
            'VTAMemFree(buffer->data_);' not in runtime_text:
        raise ValueError('STORAGE_LIFETIME_SOURCE_REQUIRES_REVIEW')
    write(out / 'storage_lifetime.json', {
        'source': str(runtime_source), 'source_sha256': sha(runtime_source),
        'observed_identity': 'ext_dev DLTensor.data is the opaque live DataBuffer pointer; not physical DRAM',
        'backing_lifetime': 'DataBuffer::Alloc sets data_/phy_addr_ once; DataBuffer::Free releases it',
        'scope': 'four NDArrays and module retained for all six calls in each fresh RPC',
        'checks': 'same live NDArray handles, DataBuffer identities, byte offsets, output and read-only parameters every call',
        'weight_check': 'fixed seed0 input, W0/W1 independently produce different final outputs; same weight allocation'})
    sources = sorted([*HERE.glob('c3_tps_aggregation_*.py'), *HERE.glob('test_c3_tps_aggregation_*.py')])
    snapshot = out / 'tool_sources'
    snapshot.mkdir()
    for source in sources:
        shutil.copy2(source, snapshot / source.name)
    bindings = dict(artifacts['source_bindings'])
    bindings.update({str(p): sha(p) for p in sources})
    adoption = read(artifacts['adoption'])
    for program in adoption['programs'].values():
        for mode in ('detailed', 'minimal'):
            for evidence in sorted(Path(program[mode]).rglob('*')):
                if evidence.is_file():
                    bindings[str(evidence)] = sha(evidence)
    for restoration in ('20261015_c3_tps_aggregation_volatile_restore01',
                        '20261015_c3_tps_aggregation_taskset_restore01'):
        directory = out.parent / restoration
        if directory.exists():
            for path in directory.iterdir():
                if path.is_file():
                    bindings[str(path)] = sha(path)
    for item in ('runtime', 'helper', 'storage_helper', 'runtime_tvm', 'rpc_server', 'bitstream', 'hardware_archive'):
        bindings[artifacts[item]['path']] = artifacts[item]['sha256']
    for path in sorted(out.rglob('*')):
        if path.is_file():
            bindings[str(path)] = sha(path)
    verify_bindings(bindings)
    write(out / 'freeze.json', {'schema': 'c3_tps_aggregation_execution_freeze_v1',
          'status': 'READY_FOR_FPGA_GATES', 'source_and_artifact_bindings': bindings,
          'adopted_FSim_calls': 42, 'new_FSim_calls': 0, 'planned_FPGA_calls': 180,
          'planned_sessions': 30, 'planned_formal_samples': 54,
          'scope': 'TPS224 B/N/G only; original representations and common runtime unchanged'})
    return {'status': 'READY_FOR_FPGA_GATES', 'directory': str(out), 'calls': 180}


def run_board(args, out):
    from c3_tps_aggregation_board import execute
    frozen = read(out / 'freeze.json')
    verify_bindings(frozen['source_and_artifact_bindings'])
    artifacts, ledger = read(out / 'artifacts/prepared.json'), read(out / 'call_manifest.json')
    _check_manifest(ledger)
    bindings = {**frozen['source_and_artifact_bindings'], str(out / 'freeze.json'): sha(out / 'freeze.json')}
    return execute(args, artifacts, ledger, out / 'board', bindings)


def report(out):
    from c3_tps_aggregation_capacity import organization
    artifacts = read(out / 'artifacts/prepared.json')
    ledger = read(out / 'call_manifest.json')
    summary = read(out / 'board/summary.json')
    path = out / 'board/results.jsonl'
    results = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    complete = summary['status'] == 'COMPLETE_SUCCESS'
    organizations = []
    for ordinal, session in enumerate(ledger['sessions'], 1):
        folder = out / 'board' / ('session_%02d' % ordinal)
        for position in range(1, 7):
            native_path = folder / ('native_%d.json' % position)
            if native_path.exists():
                organizations.append({'session_id': session['session_id'], 'program': session['program'],
                    'rotation': session['rotation'], 'position': position,
                    'organization': organization((folder / ('native_%d.txt' % position)).read_text(), read(native_path))})
    write(out / 'organization.json', organizations)
    text = ['# TPS原有指令聚合验证结果', '',
            '对象：YOLOv3-tiny-224节点24，128→256、14×14、3×3，Tile b1h14w14ci4co4、oc2/h1。', '',
            'B为原TPS，N仅调整已有通知位置，G在N上连接两个bank清单。三版保留原有NOP、LOAD和回绕安排。', '',
            '## 能不能合？', '',
            '已采纳同一旧attempt的TPS六worker、42次完整融合FSim；本轮新增FSim调用为0。原完整attempt因后续官方G失败而停止的状态保持不变。', '',
            '三版从已认证最终IR构建AArch64模块，逐TIR作用域的缓存handle、initializer、签名、规范化函数体及共享关系均通过对照。', '']
    if complete:
        verify_bindings(read(out / 'freeze.json')['source_and_artifact_bindings'])
        stats = analyse(ledger, results)
        write(out / 'statistics.json', stats)
        text += ['FPGA完成180次完整调用，30个fresh RPC单元。三seed资格、同存储W0→W1→W0、逐元素输出、只读参数和54次完整诊断均通过。正式样本为54个；每次只有一次完整目标调用。', '']
    else:
        text += ['**本attempt已停止，不能形成性能结论。**', '',
                 f"已开始{summary['begun_calls']}次目标调用，{summary['passed_calls']}次通过全部调用后检查。停止原因：`{summary['error']}`。不自动重试、不补样、不拼接。", '']
    text += ['## 能合多少？', '',
             '当前段固定连接两个bank，每段4704条基址μop，容量为8192条；没有扩大聚合量。已有本地证书的完整调用结构如下，FPGA实测诊断另见逐调用证书。', '',
             '|程序|乘加GEMM|RESET|ALU|全部指令|compute NOP|真实UOP LOAD|载荷字节|回绕|提交|',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
             '|B|8|4|16|76|3|4|75520|2|1|',
             '|N|8|4|16|89|11|9|75520|2|1|',
             '|G|4|4|16|83|11|7|75504|3|1|', '',
             '三版每提交序列化37760字节；host清单缓存、提交DRAM源及UOP SRAM驻留各按冻结runtime核验。回绕从SRAM起点重新分配合法kernel，并不拆开超大kernel。', '',
             '实际诊断的逐调用组织、容量与重装原因投影见[organization.json](organization.json)。kernel装入事件与greedy合并后的真实LOAD指令分账；原因来自冻结缓存规则及逐引用地址核对，不是周期成本测量。', '']
    if complete:
        text += ['|FPGA暖诊断|GEMM|RESET|ALU|NOP|UOP LOAD|LOAD字节|回绕|容量驱逐重装kernel|提交重置重装kernel|',
                 '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for p in ('B','N','G'):
            obs = next(x['organization'] for x in organizations if x['program'] == p and x['rotation'] == 2 and x['position'] == 6)
            reasons = obs['allocation_projection']['reason_counts']
            text.append('|%s|%d|%d|%d|%d|%d|%d|%d|%d|%d|' %
                        (p, obs['MAC_GEMM'], obs['RESET_GEMM'], obs['ALU'], obs['compute_NOP'], obs['UOP_LOAD'],
                         obs['UOP_LOAD_bytes'], obs['UOP_SRAM_wraps'], reasons.get('capacity_eviction_reload',0),
                         reasons.get('submission_reset_reload',0)))
        text += ['']
    text += ['## 合了值不值？', '']
    if complete:
        text += ['|程序|六轮中位时延 T（ms）|', '|---|---:|']
        text += [f"|{p}|{stats['T'][p]:.6f}|" for p in ('B','N','G')]
        text += ['', '|对照|逐轮改善中位数|严格胜场|绝对节省中位数 ms|', '|---|---:|---:|---:|']
        for pair in ('N/B','G/N','G/B'):
            value = stats['pairwise'][pair]
            text.append(f"|{pair}|{value['Gain_percent']:.4f}%|{value['wins']}/6|{value['absolute_ms']:.6f}|")
        text += ['', '**主要结果G/B：' + ('达到' if stats['engineering_goal_met'] else '未达到') +
                 '冻结工程门槛（改善≥2%且严格胜场≥5/6）。**', '',
                 '|轮次|B ms|N ms|G ms|G/B改善 %|', '|---|---:|---:|---:|---:|']
        for i in range(6):
            text.append('|%d|%.6f|%.6f|%.6f|%.4f|' % (i+1, *(stats['round_medians'][p][i] for p in ('B','N','G')),
                         stats['pairwise']['G/B']['round_gains_percent'][i]))
        text += ['', '完整六轮比较、范围与离散程度见[统计数据](statistics.json)。Gain使用六个逐轮比值的中位数，不以两个T相除替代。', '',
                 '正式最小计数模式计入命令组织、全部提交及必要等待；保留固定模式检查和每提交记录成本。详细资格首调用成本单列在调用账中，不能直接与最小模式正式时延相减解释缓存成本。', '',
                 'N/B体现通知调整及编码变化整体影响，G/N体现同通知安排下聚合及必要缓存变化增量；只有G/B说明整套方案净收益。无法分解的成本不强行归因。']
    else:
        text += ['没有完整成功的180次账目，正式性能统计不生成。已返回的时间只保留为失败attempt诊断，不能用来判断加速。']
    text += ['', '本轮仅覆盖当前受限双bank表示；不宣称通用算法完成、统计显著性、超过TopHub或整网加速。', '',
             '## 证据与暂停', '',
             '- [资格采纳](artifacts/adoption.json)与[三版ARM合同](artifacts/prepared.json)',
             '- [冻结身份](freeze.json)、[180次预登记清单](call_manifest.json)、[权重存储生命周期](storage_lifetime.json)',
             '- [板测状态](board/summary.json)与board目录逐调用原始记录', '',
             '**本轮已暂停。** 后续通知承载、ALU或缓存方向的本地预实验另行设计，不在本轮追加候选。', '']
    with (out / 'REPORT.md').open('x') as stream:
        stream.write('\n'.join(text))
    write(out / 'completion_manifest.json', {'status': summary['status'], 'paused': True,
          'artifact_sha256': {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*')) if p.is_file()}})
    return {'status': summary['status'], 'report': str(out / 'REPORT.md'), 'paused': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare','board','report'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--host', default='192.168.1.247')
    parser.add_argument('--known-hosts', default='/home/orange/.ssh/known_hosts')
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if args.phase == 'prepare':
        result = freeze(out)
    elif args.phase == 'board':
        result = run_board(args, out)
    else:
        result = report(out)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
