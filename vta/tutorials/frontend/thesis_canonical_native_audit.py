#!/usr/bin/env python3
"""Independently re-read native outputs, failed-run costs, and typed live frontiers."""
import argparse
import json
from pathlib import Path
import statistics
import numpy as np

from thesis_supplement_run import sha256, write_new_json
from thesis_supplement_audit import reconcile


def verify_sealed(directory):
    manifest=json.loads((directory/'artifact_hashes.json').read_text())
    actual={str(p.relative_to(directory)) for p in directory.rglob('*')
            if p.is_file() and p.name!='artifact_hashes.json'}
    if actual!=set(manifest):raise ValueError('Sealed file inventory changed')
    for relative,expected in manifest.items():
        p=directory/relative
        if p.stat().st_size!=expected['bytes'] or sha256(p)!=expected['sha256']:
            raise ValueError('Sealed artifact changed: '+str(p))
    cost=reconcile(directory)
    if cost!=json.loads((directory/'analysis/cost_audit.json').read_text()) or cost['unfinished_action_ids']:
        raise ValueError('Cost ledger drift or unfinished action')
    rows=sum(len(p.read_text().splitlines()) for p in directory.glob('raw/*/result.jsonl'))
    probes=len(list(directory.glob('raw/carry_*_*.json')))
    return {'run_id':directory.name,'status':json.loads((directory/'summary.json').read_text())['status'],
            'actual_completed_full_native_rows':rows,'actual_carry_actions':probes,
            'measured_action_seconds':cost['measured_action_total_seconds'],
            'material_preparation_seconds':json.loads((directory/'analysis/material_preparation_cost.json').read_text())['wall_seconds'],
            'failed_actions_preserved':cost['failed_actions_preserved'],
            'artifact_manifest_sha256':sha256(directory/'artifact_hashes.json')}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',nargs=3,type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); output=args.output
    rows=[verify_sealed(p) for p in args.runs]
    if [r['status'] for r in rows]!=['FAILED_NO_CERTIFICATE','FAILED_NO_CERTIFICATE','NATIVE_CPU_PREQUALIFICATION_PASS']:
        raise ValueError('Unexpected source-run outcomes')
    accepted=args.runs[-1]
    interfaces=json.loads((accepted/'raw/native_transport_interface.json').read_text())['stage_interfaces']
    lookup={(r['begin'],r['end_exclusive']):r for r in interfaces}
    programs={'full':[(0,21)],'integrated':[(0,3),(3,18),(18,21)],'units':[(i,i+1) for i in range(21)]}
    checks=[]; integer=0; boundaries=0
    for program,ranges in programs.items():
        finals={}
        for transport in ('copy','input_zero_copy'):
            directory=accepted/'raw'/(program+'_'+transport)
            records=[json.loads(line) for line in (directory/'result.jsonl').read_text().splitlines()]
            if len(records)!=3:raise ValueError('Wrong completed call count')
            for record,case in zip(records,('zeros','uniform','normal')):
                if record['stage_count']!=len(ranges):raise ValueError('Wrong executed stage count')
                raw=Path(record['raw_output_files'][0]['path']).read_bytes()
                reference=np.load(accepted/'raw/inputs'/case/'reference_logits.npy',allow_pickle=False)
                if raw!=reference.tobytes():raise ValueError('Final full bytes mismatch')
                finals[(transport,case)]=raw
                if len(record['stage_raw_outputs'])!=len(ranges):raise ValueError('Incomplete stage snapshots')
                if program!='full':
                    for stage,interval in zip(record['stage_raw_outputs'],ranges):
                        sources=lookup[interval]['output_sources']
                        if len(stage['output_files'])!=len(sources):raise ValueError('Incomplete live bundle')
                        for item,source in zip(stage['output_files'],sources):
                            raw=Path(item['path']).read_bytes()
                            reference=(accepted/'raw/inputs'/case/f'node_{source["node_id"]}.bin').read_bytes()
                            if raw!=reference or len(raw)!=item['nbytes']:raise ValueError('Live boundary bytes mismatch')
                            dtype=np.dtype(item['dtype']); boundaries+=1
                            integer+=dtype.kind in 'iu'
                checks.append({'program':program,'transport':transport,'input':case,'final_bytes_exact':True})
        for case in ('zeros','uniform','normal'):
            if finals[('copy',case)]!=finals[('input_zero_copy',case)]:raise ValueError('Transport byte mismatch')
    controls=[]
    for begin in (3,6):
        interface=lookup[(begin,begin+1)]
        graph=json.loads((accepted/f'raw/compiled/units_{begin:02}_{begin:02}/graph.json').read_text())
        for case in ('zeros','uniform','normal'):
            for mode in ('naive','routed'):
                wrong=[]
                for i,source in enumerate(interface['output_sources']):
                    raw=(accepted/'raw'/f'carry_{begin:02}_{case}_{mode}'/f'output{i}.bin').read_bytes()
                    expected=(accepted/'raw/inputs'/case/f'node_{source["node_id"]}.bin').read_bytes()
                    head=graph['heads'][i]; eid=graph['node_row_ptr'][head[0]]+head[1]
                    dtype=np.dtype(graph['attrs']['dltype'][1][eid])
                    if len(raw)!=len(expected):raise ValueError('Probe binary length mismatch')
                    count=int(np.count_nonzero(np.frombuffer(raw,dtype)!=np.frombuffer(expected,dtype)))
                    if source['kind']=='live_input':wrong.append(count)
                    elif count:raise ValueError('Computed output changed by poison')
                if mode=='naive' and not all(wrong):raise ValueError('Negative getter unexpectedly matched')
                if mode=='routed' and any(wrong):raise ValueError('Actual incoming route mismatch')
                controls.append({'unit':begin,'input':case,'mode':mode,'carry_mismatching_elements':wrong})
    if len(checks)!=18 or boundaries!=216 or integer!=204 or len(controls)!=12:
        raise ValueError('Verification denominator changed')
    frontier=[]
    for begin in range(1,21):
        record=lookup[(begin,begin+1)]
        frontier.append({'before_unit':begin,'live_tensor_count':len(record['input_node_ids']),
                         'live_bundle_bytes':sum((accepted/'raw/inputs/zeros'/f'node_{n}.bin').stat().st_size
                                                 for n in record['input_node_ids'])})
    write_new_json(output/'analysis/source_run_audit.json',rows)
    write_new_json(output/'analysis/native_independent_checks.json',checks)
    write_new_json(output/'analysis/native_independent_carry.json',controls)
    write_new_json(output/'analysis/live_frontiers.json',frontier)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,3.5),layout='constrained')
    x=[r['before_unit'] for r in frontier]
    ax.step(x,[r['live_tensor_count'] for r in frontier],where='mid',color='tab:blue')
    ax.set_xticks(x); ax.set_yticks([1,2]); ax.set_ylim(.75,2.25)
    ax.set_xlabel('Boundary before canonical unit'); ax.set_ylabel('Live tensors',color='tab:blue')
    right=ax.twinx()
    right.plot(x,[r['live_bundle_bytes']/1024 for r in frontier],'o-',color='tab:orange',markersize=3)
    right.set_yscale('log');right.set_ylabel('Complete live bundle (KiB)',color='tab:orange')
    ax.set_title('ResNet18 once-quantized typed frontiers: complete residual dependencies')
    for extension in ('svg','pdf'):fig.savefig(output/'figures'/('canonical_live_frontiers.'+extension))
    plt.close(fig)
    paid=sum(r['measured_action_seconds']+r['material_preparation_seconds'] for r in rows)
    summary={'status':'NATIVE_HOST_RESULTS_INDEPENDENTLY_AUDITED','formal_phase02_completed':False,
             'hardware_calls':0,'real_images':0,'accepted_full_native_calls':18,
             'accepted_boundary_outputs':216,'accepted_integer_bytes_checks':204,
             'accepted_carry_probes':12,'naive_wrong_cases':6,'routed_correct_cases':6,
             'source_completed_full_calls_including_failed_runs':sum(r['actual_completed_full_native_rows'] for r in rows),
             'source_carry_actions_including_failed_runs':sum(r['actual_carry_actions'] for r in rows),
             'source_paid_actions_plus_material_preparation_seconds':paid,
             'source_costs_charged_again_to_this_run':False,'source_runs':rows}
    write_new_json(output/'summary.json',summary)
    links='\n'.join(f'- [{r["run_id"]}](../{r["run_id"]}/REPORT.md)' for r in rows)
    (output/'REPORT.md').write_text(
        '# 阶段02 native本机语义预资格与独立复算\n\n'
        '正式阶段02尚未完成：01板端认证、真实ImageNet图、Q-board/K2和任务准确率仍缺。'
        '用户授权先开展其他实验，本轮补齐CPU native执行入口，未改变05未暴露任务的信息集。\n\n'
        '生产native runner从实际graph JSON heads识别plain live-input输出，保留incoming NDArray所有者；'
        '计算输出仍快照，避免下一帧重用executor池。新增CPU-only serial输入零拷贝模式，'
        '不能与VTA、pipeline或managed模式组合。默认copy路径也采用实际live-input路由。\n\n'
        '同一预训练、一次量化224图的全图/三段/21单元LLVM库，copy和输入零拷贝各执行3个冻结诊断输入。'
        '18个完整native输出均与Q参考逐字节相同，216个live边界输出全部逐字节相同，其中204个整数、12个浮点。'
        '单位03/06各3输入原输入池取反：native naive-getter6/6出错，实际incoming路由6/6正确，'
        '计算输出保持正确；负对照仅清空来源映射，不改算子或输入。完整bin/NPY参考已保存并独立复算。\n\n'
        'accepted run独立执行完整矩阵，不拼接旧前缀。第一次失败为探针const metadata编译错误，0个完整调用；'
        '第二次18个完整调用通过，但探针借用文本WriteFile导致bin多换行，保留整run失效，随后新run修正探针重做。'
        f'全部来源实际完成{summary["source_completed_full_calls_including_failed_runs"]}个CPU全图调用，'
        f'执行{summary["source_carry_actions_including_failed_runs"]}个carry probe action；计分仍仅接受run的18/12。\n\n'
        'managed及legacy K2源码另加入live-input→outgoing slot copy后统一Publish，incoming token持有至复制结束；'
        'managed trace计入live_input_copy_bytes与framework materialization，不伪报0复制。'
        '该板端路径本轮未执行，必须在重构建后重新资格；也尚未实现/认证正式生命周期latch及Q1证书。\n\n'
        'host link guard不提供FPGA分配/模拟器；VTA内存/地址调用即exit86，成功CPU路线不触发guard。'
        '本轮不是实际VTA lowering/FSim/DMA/FINISH/coherence，也没有真实准确率或推理性能主张。\n\n'
        '真实typed前沿图：[SVG](figures/canonical_live_frontiers.svg)、[PDF](figures/canonical_live_frontiers.pdf)，'
        '来源为实际接口node IDs和冻结完整字节，不能把残差边界当单tensor链。\n\n'
        f'三个来源已付action＋材料准备成本合计{paid:.6f}s，包含失败；本次核验/绘图另计，'
        '开发和人工等待unknown/null，不重复归入来源成本或C3在线预算。sealed文件清单/hash和成本复算一致。\n\n'
        '所有来源运行：\n\n'+links+'\n\n'
        '下一步仍为恢复01板卡访问和原图授权入口，再正式完成02各backend及1000图任务质量；'
        '当前有限结论仅是共同量化图在所测CPU native完整链和input-zero-copy下语义保持。\n')
    print(json.dumps({'accepted_full_calls':18,'integer_checks':204,'formal_phase02_completed':False}))


if __name__=='__main__':main()
