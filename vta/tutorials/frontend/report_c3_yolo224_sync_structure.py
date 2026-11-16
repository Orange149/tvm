"""Verify and summarize the sealed, read-only network synchronization census."""
import csv
import json
import re
from pathlib import Path

from c3_yolo224_sync_structure import HOLD, SOURCE, read, require, sha, write


def main():
    audit = HOLD / '20261016_c3_yolo224_sync_structure_readonly03'
    manifest = read(audit / 'completion_manifest.json')
    for name, digest in manifest['artifact_sha256'].items():
        require(sha(audit / name) == digest, 'AUDIT_CHANGED:' + name)
    for name, digest in read(SOURCE / 'artifact_hashes.json')['artifacts'].items():
        require(sha(SOURCE / name) == digest, 'SOURCE_CHANGED:' + name)
    summary = read(audit / 'summary.json')
    mapping = read(audit / 'graph_mapping.json')
    coverage = {}
    for variant in ('G0', 'G1', 'G2'):
        nodes = read(SOURCE / variant / 'graph.json')['nodes']
        other = []
        for index, node in enumerate(nodes):
            if node['op'] != 'tvm_op':
                continue
            symbol = node['attrs']['func_name']
            if 'conv2d' in symbol:
                continue
            path = SOURCE / variant / (symbol + '.tir')
            text = path.read_text()
            require(not re.search(r'vta\.(uop_push|coproc_dep_push|coproc_dep_pop)|VTALoadBuffer2D|VTAStoreBuffer2D', text),
                    'UNENUMERATED_DEVICE_COMMAND:' + symbol)
            other.append({'node': index, 'symbol': symbol, 'tir_sha256': sha(path),
                          'device_compute_or_load_store_notifications': False})
        rows = [r for r in mapping if r['variant'] == variant]
        require(len(rows) == 13 and sum(r['device'] == 'VTA' for r in rows) == 12,
                'GRAPH_COVERAGE_CHANGED')
        require(len(other) == 20, 'NONCONV_COVERAGE_CHANGED')
        require(summary[variant]['unique_functions'] == 11, 'FUNCTION_COVERAGE_CHANGED')
        coverage[variant] = {'graph_tvm_op_calls': 33, 'CPU_first_convolution_node': 3,
                             'other_calls': other, 'VTA_convolution_calls': 12}

    out = HOLD / '20261016_c3_yolo224_sync_structure_report01'
    out.mkdir(exist_ok=False)
    write(out / 'coverage.json', coverage)
    write(out / 'evidence.json', {
        'audit_manifest': str(audit / 'completion_manifest.json'),
        'audit_manifest_sha256': sha(audit / 'completion_manifest.json'),
        'source_archive_manifest_sha256': sha(SOURCE / 'artifact_hashes.json'),
        'report_generator_sha256': sha(__file__),
        'target_calls': 0, 'compilations': 0, 'board_operations': 0,
        'full_network_TPS_port': False,
        'prior_tool_failure_correction': 'readonly01 and readonly02 failed because VTA intrinsics were not registered before JSON deserialization; neither is an original-program failure; zero target calls.'})
    keys = ['variant', 'node', 'symbol', 'force_serial_requested', 'g2l_count',
            'MAC_input_overwrite', 'MAC_disjoint_candidate', 'RESET_physical_unresolved',
            'RESET_disjoint_candidate', 'status']
    table = []
    for variant, group in summary.items():
        for e in group['entries']:
            mac, reset = e['MAC_categories'] or {}, e['RESET_categories'] or {}
            table.append({k: e[k] for k in keys if k in e} | {
                'MAC_input_overwrite': mac.get('MAC_INPUT_OVERWRITE_PROTECTION', 0),
                'MAC_disjoint_candidate': mac.get('ADDRESS_DISJOINT_PREFIX_NEEDS_FULL_CERTIFICATE', 0),
                'RESET_physical_unresolved': reset.get('RESET_PHYSICAL_READ_UNRESOLVED', 0),
                'RESET_disjoint_candidate': reset.get('ADDRESS_DISJOINT_PREFIX_NEEDS_FULL_CERTIFICATE', 0)})
            if e['status'] != 'TIR_STRUCTURE_AND_DIRECT_OVERWRITE_SCREEN_ONLY':
                for key in keys[5:9]:
                    table[-1][key] = None
    with (out / 'per_node.csv').open('x') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(table)
    lines = [
        '# YOLO224 全网同步现象结构核验', '',
        '本次核验现有 YOLOv3-tiny-224 的 G0、G1、G2 三个封存完整图，以 G2 为主要对象。'
        '每图覆盖 12 次 VTA 卷积调用、11 个不同编译函数；节点29和57共用函数但在全网计数中分别计入。'
        'CPU首卷积和20个非卷积调用不含本次关注的VTA计算及装载通知。三个版本不是三个网络，也不是全网TPS移植。', '',
        '统计将每个卷积完整调用的有限循环展开，配对compute→load通知，检查生产者MAC/RESET的有效INP读地址与随后INP装载的写地址（含padding）。'
        '仅分析把等待移过连续INP装载、停在下一条WGT装载之前的原有规则；遇到其他push/pop或同步边界即停止。', '',
        '| 版本 | 完成结构分析的VTA调用 | MAC同址覆盖保护 | MAC异址前缀候选 | RESET异址前缀候选 | 请求强制串行的调用 |',
        '|---|---:|---:|---:|---:|---:|']
    for variant, group in summary.items():
        mac, reset = group['MAC_categories'], group['RESET_categories']
        lines.append(f"| {variant} | {group['analyzed_call_sites']}/12 | {mac.get('MAC_INPUT_OVERWRITE_PROTECTION', 0)} | {mac.get('ADDRESS_DISJOINT_PREFIX_NEEDS_FULL_CERTIFICATE', 0)} | {reset.get('ADDRESS_DISJOINT_PREFIX_NEEDS_FULL_CERTIFICATE', 0)} | {group['force_serial_call_sites']} |")
    lines += ['', '表内次数是一次完整图调用中已成功分析部分的TIR动态通知计数，不是板端实测停顿次数、周期或耗时。未决对象不计为零机会。', '']
    g2 = summary['G2']
    if g2['analyzed_call_sites'] == 12 and g2['MAC_categories'].get('ADDRESS_DISJOINT_PREFIX_NEEDS_FULL_CERTIFICATE', 0) == 0:
        lines += ['**主要结论：G2的12次VTA卷积调用均完成本次结构分析，未发现可按原TPS规则后移的MAC异址装载前缀。'
                  '等待计算后再装载的现象普遍，但“等待可后移”尚未在该现有全网程序中得到推广证据。**', '']
    for variant, group in summary.items():
        for e in group['entries']:
            if e['status'] != 'TIR_STRUCTURE_AND_DIRECT_OVERWRITE_SCREEN_ONLY':
                lines.append(f"- {variant} 节点{e['node']}：分析未决，原因 `{e.get('error')}`。TIR同步处存在尚未配对通知，当前投影不能解释runtime提交和收尾；不据此判定原程序错误或不存在机会。")
    lines += ['',
              '**能不能撤除等待边界？** 等待本身重复出现，但需区分保护同址覆盖的等待与TPS式异址装载机会。'
              '同址覆盖见证说明单独后移该等待会撤除必要的先读后写关系。异址前缀只通过直接生产者地址筛选，仍需全局物理访问、runtime编码和有限FIFO认证。', '',
              '| G2图节点 | MAC同址覆盖保护 | MAC异址候选 | RESET异址候选 | 强制串行请求 |',
              '|---|---:|---:|---:|---|']
    for e in [r for r in table if r['variant'] == 'G2']:
        values = [('未决' if e[k] is None else str(e[k])) for k in ['MAC_input_overwrite', 'MAC_disjoint_candidate', 'RESET_disjoint_candidate']]
        serial = '未决' if e['force_serial_requested'] is None else '是' if e['force_serial_requested'] else '否'
        lines.append(f"| {e['node']} | {' | '.join(values)} | {serial} |")
    lines += ['', '**能提前到哪里？** 封存TPS R的两处候选来自下一批INP写入另一片区域，而WGT继续复用原区域：可先装INP，但覆盖WGT前仍要等旧MAC。'
              '普通程序中的同址覆盖保护不能套用这个改动。真实节点24回归测试用同一分析器对比普通程序与TPS R，分别识别出0处和2处MAC异址前缀候选。', '',
              'RESET异址候选不等于MAC机会。RESET在物理实现中仍可能读取INP/WGT、写OUT；本次仅检查其直接INP关系，不能据此把它们称为冗余等待。'
              '其他既有读取、STORE观察、通知承载和有限FIFO关系仍需要认证。', '',
              '**值得进入性能验证吗？** 本次没有执行目标函数或测量设备停顿，不能从通知次数判断收益。'
              '先依据MAC/RESET分类判断是否存在同类机会，不能将节点24 TPS的两个改动复制到全网，或乘以算子数估算加速。', '',
              'G2部分驻留程序显式设置 `VTASetDebugMode(...,32)`，结束时恢复0。当前本地runtime将32定义为 `VTA_DEBUG_FORCE_SERIAL`，并在提交时调用 `RewriteForceSerial`。'
              '本次记录这项请求；为了展开TIR，仅在内存中的分析投影里移除模式设置语句，不改变、编译或执行任何程序。'
              '因此这些函数的TIR通知图不能替代runtime改写后的实际指令依赖图，也不能据此直接关闭强制串行。', '',
              '7项工具测试通过，包括同址拒绝、RESET物理读取、padding写入、内部通知边界和真实普通节点24/TPS对照。'
              '逐函数结果保留未决状态；成功结构分析不等于完整数值或硬件资格。每worker限制8 GiB、10分钟，展开步数沿用500万上限。', '',
              f'证据：[逐节点表](per_node.csv)、[覆盖核验](coverage.json)、[冻结审计汇总]({audit}/summary.json)、'
              f'[审计完成manifest]({audit}/completion_manifest.json)。'
              '逐等待见证保存在审计目录objects中，包含生产者、消费者、INP装载事件、WGT目标和同址冲突地址。', '',
              '旧只读工具尝试readonly01、readonly02的反序列化失败原因统一更正为缺少VTA指令注册。'
              '它们均未执行目标函数，不是基线程序失败；历史目录保持封存。', '']
    (out / 'REPORT.md').write_text('\n'.join(lines))
    write(out / 'completion_manifest.json', {'status': 'READONLY_REPORT_COMPLETE',
          'artifact_sha256': {p.name: sha(p) for p in out.iterdir() if p.is_file()}})
    print(out)


if __name__ == '__main__':
    main()
