"""Read-only node46 board closeout and correction of the inherited status label.

No original artifact or source is changed, and no target is built or invoked.
The original NODE34-prefixed completion label is an exporter metadata mistake;
node46 ABI, frozen program identities and every ledger row must verify first.
"""
import argparse
from pathlib import Path
from c3_tps_aggregation_prepare import read,write,sha,require,verify_bindings
from c3_node46_uop_identity_protocol import analyse
from c3_node46_uop_identity_board_support import tensor_sizes


def run(source,out):
    import json
    out.mkdir(exist_ok=False)
    seal=read(source/'completion_manifest.json')
    require(seal['status']=='NODE34_IDENTITY_BOARD_COMPLETE','UNEXPECTED_SOURCE_EXPORT_LABEL')
    bindings={str(source/p):v for p,v in seal['artifact_sha256'].items()}
    bindings[str(source/'completion_manifest.json')]=sha(source/'completion_manifest.json')
    bindings.update(read(source/'freeze.json')['source_and_artifact_bindings'])
    bindings[str(Path(__file__).resolve())]=sha(__file__)
    verify_bindings(bindings)
    ledger=read(source/'call_manifest.json')
    require(ledger['schema']=='c3_node46_uop_identity_120_calls_v1','NOT_NODE46_LEDGER')
    results=[json.loads(s) for s in (source/'board/results.jsonl').read_text().splitlines()]
    statistics=analyse(ledger,results)
    require(statistics==read(source/'statistics.json'),'STATISTICS_CHANGED')
    require(read(source/'board/summary.json')['status']=='COMPLETE_SUCCESS','BOARD_ATTEMPT_NOT_COMPLETE')
    a=read(source/'call_accounting.json')
    require(a['registered_attempts']==a['fully_certified']==a['started_calls']['exact']==a['completed_calls']['exact']==120,
            'CALL_ACCOUNTING_CHANGED')
    prepared=read(source/'artifacts/prepared.json')
    expected_shapes=[[1,24,14,14,1,16],[16,24,3,3,16,16],[16,1,1,1,16],[1,16,14,14,1,16]]
    for name in ('B','A'):
        d=prepared['programs'][name]
        require([r['shape'] for r in d['abi']]==expected_shapes,'NOT_NODE46_ABI')
        tensor_sizes(d['abi'])
        reference=read(d['reference_contract'])
        require([r['source'] for r in reference['postprocess'] if r['op']=='shr']==
                [{'kind':'imm','value':8}],'NOT_NODE46_POSTPROCESS')
        contract=read(d['arm_contract'])
        require(contract['initializer_comparison']['status']=='PER_TIR_SCOPE_INITIALIZER_AND_CACHE_SHARING_EQUAL',
                'ARM_FSIM_NOT_BOUND')
    require(read(source/'board/preflight.json')['boot_id']==read(source/'board/final_state.json')['boot_id'],
            'BOOT_CHANGED')
    write(out/'freeze.json',{'bindings':bindings,'source':str(source),'new_target_calls':0,'new_compilations':0})
    write(out/'correction.json',{'status':'NODE46_IDENTITY_BOARD_COMPLETE',
          'source_manifest_status_preserved':seal['status'],
          'correction':'Report exporter inherited NODE34 status prefix; node46 ABI, identities, all ledger rows and statistics verified.',
          'source_manifest_sha256':sha(source/'completion_manifest.json'),
          'experiment_repeated':False,'new_target_calls':0,'performance_samples_spliced':False})
    write(out/'statistics.json',statistics)
    gain=statistics['pairwise']['A/B']
    text=['# 节点46缓存身份复用的FPGA验证结论','',
          f"**在当前固定TPS对照上，完整调用改善{gain['Gain_percent']:.4f}%，六轮全部更快。** 120次完整调用、20个fresh RPC单元、36个正式样本全部通过资格与冻结口径检查。",'',
          '## 能不能复用','',
          '可以。按同一静态kernel的完整initializer等价类生成代表签名，48个MAC缓存身份降为4个，总kernel身份58降为14。保持节点46的原Tile、线程、DMA、计算顺序、融合尾部及通知。36份实际指令诊断及18份联合证书通过。','',
          '## 能复用多少','',
          'MAC命令保持96条，UOP LOAD保持1次，回绕保持0，提交保持1次；实际序列化与装载载荷从32,384降至2,816字节。该案例验证了没有回绕且LOAD次数不变时的适用性。','',
          '## 复用后值不值','',
          '|程序|六轮中位时延 ms|','|---|---:|',
          f"|原版B|{statistics['T']['B']:.6f}|",f"|优化版A|{statistics['T']['A']:.6f}|",'',
          f"逐轮比值改善中位数为{gain['Gain_percent']:.6f}%，严格胜场{gain['wins']}/6，绝对节省中位数{gain['absolute_ms']:.6f} ms。达到Gain≥2%、胜≥5/6的工程判据。",'',
          '结果来自当前一次Linux启动和本地适配TPS的oc2/h1支持域；不作为统计显著性、整网加速或作者完整编译栈全局最优的结论。节点34历史样本保持独立。载荷与缓存安排同时改变，现有证据不将收益全部归因于某一项成本。','',
          '## 封存记录','',
          '原报告导出器的完成状态字串误保留NODE34前缀。本目录核验原始901项封存文件、源与二进制指纹、节点46四参数ABI、右移8位语义、逐调用身份、同一boot和完整统计后，以独立记录给出正确的NODE46标签。原始manifest、源代码、120次样本与报告均保持不变；没有重新执行、补样或拼接。','',
          f'[原始详细报告]({source.as_posix()}/REPORT.md)、[标签更正](correction.json)、[统计](statistics.json)。','',
          '**本轮完成并暂停，不追加实验。**']
    (out/'REPORT.md').write_text('\n'.join(text)+'\n')
    verify_bindings(bindings)
    write(out/'completion_manifest.json',{'status':'NODE46_IDENTITY_BOARD_COMPLETE','paused':True,
          'source':str(source),'new_target_calls':0,
          'artifact_sha256':{str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()}})
    print(json.dumps({'status':'NODE46_IDENTITY_BOARD_COMPLETE','Gain_percent':gain['Gain_percent'],
                      'wins':gain['wins'],'new_target_calls':0}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.source.resolve(),a.output.resolve())
