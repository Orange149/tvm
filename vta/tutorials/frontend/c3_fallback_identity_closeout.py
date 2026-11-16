"""Read-only closure, tables and standalone figures for the F/T/U attempt."""
import argparse
import csv
from pathlib import Path
from c3_tps_sync_release_local import HOLD, read, write, sha, verify, require
from c3_fallback_identity_protocol import analyse
from c3_fallback_identity_compare import LOCAL


def closeout(source, output):
    require(not output.exists(),'OUTPUT_EXISTS')
    require(read(source/'completion_manifest.json')['status']=='F_T_U_COMPLETE','ONLY_COMPLETE_ATTEMPT')
    sealed=read(source/'completion_manifest.json')['artifact_sha256']
    verify({str(source/k):v for k,v in sealed.items()})
    verify(read(source/'freeze.json')['bindings'])
    ledger=read(source/'call_manifest.json');rows=read(source/'board/results.json')
    stats=analyse(ledger,rows)
    require(stats==read(source/'result.json')['statistics'],'STATISTICS_CHANGED')
    account=read(source/'result.json')['accounting']
    require(account['registered_attempts']==account['helper_started_observed_lower_bound']==
            account['helper_completed_observed_lower_bound']==account['certified']==360 and
            not account['invalid_records'] and not account['execution_unresolved_units'],'INCOMPLETE_CALL_CLOSURE')
    boot=read(source/'board/preflight.json')['boot_id']
    require(read(source/'board/final_state.json')['boot_id']==boot,'BOOT_CHANGED')
    for i in range(1,61):
        folder=source/'board'/('session_%03d'%i)
        require(read(folder/'preflight.json')['boot_id']==boot and
                read(folder/'summary.json')['status']=='COMPLETE_FRESH_UNIT_SUCCESS','SESSION_OR_BOOT_UNEXPLAINED')
    output.mkdir()
    write(output/'statistics.json',stats)
    table=[]
    text=['# YOLO224两个完整融合算子的fallback、TPS与身份复用对照','',
        'F是冻结本地AXU5EVB编译器的实际默认安全fallback；T是原已认证的固定oc2/h1本地适配TPS；'
        'U在同一T上只做内容等价μop缓存身份复用。没有使用TopHub记录生成F，也没有复用旧时延样本。','',
        '|节点|F fallback ms|T TPS ms|U TPS加身份复用 ms|T/F改善|U/T改善|U/F改善|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for item in stats['objects']:
        node=int(item['object_id'].replace('yolo224_node',''))
        p=item['pairwise'];t=item['T_ms']
        text.append('|%d|%.6f|%.6f|%.6f|%.4f%%|%.4f%%|%.4f%%|' %
                    (node,t['F'],t['T'],t['U'],p['T/F']['Gain_percent'],p['U/T']['Gain_percent'],p['U/F']['Gain_percent']))
        for v in ('F','T','U'):
            table.append({'node':node,'program':v,'full_call_ms':t[v]})
    text += ['', '改善率均为六轮逐轮比值的中位数，不由表中两个总体时延相除代替。'
             '各轮三seed各正式执行一次，先取轮内中位数，再取六轮中位数。', '',
             'T/F反映既有TPS路线相对fallback的变化；U/T反映本文身份复用的增量；'
             'U/F反映两者组合的总变化。算子倒数不标作整网FPS。','',
             '两个算子合计360次完整FPGA调用、60个fresh RPC单元、108个正式样本均通过。'
             '三seed、同实际权重存储W0→W1→W0、输出哨兵、只读参数、实际存储、完整提交、'
             '逐μop/缓存与有限FIFO认证闭合。计时覆盖命令组织、提交与等待，输出检查在外。', '',
             '本轮不新增整图推理，不能据算子时延直接填写F/T/U整图FPS。'
             'YOLO416已有TopHub正负对照保留；此处未改名或覆盖其结论。','',
             '|节点及对照|六轮改善%|严格胜场|节省ms中位数|改善范围%|改善MAD%|',
             '|---|---|---:|---:|---|---:|']
    for item in stats['objects']:
        for pair,p in item['pairwise'].items():
            text.append('|%s %s|%s|%d/6|%.6f|%s|%.6f|' %
                (item['object_id'],pair,', '.join('%.4f'%v for v in p['round_gains_percent']),
                 p['strict_wins'],p['median_absolute_ms'],p['range_percent'],p['MAD_percent']))
    text += ['', '|节点/版本|乘加GEMM|RESET|ALU|UOP LOAD|累计UOP字节|序列化字节|回绕|提交|',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    organization={}
    for session in ledger['sessions']:
        if session['phase']!='diagnostic' or session['rotation']!=0:continue
        ordinal=ledger['sessions'].index(session)+1
        o=read(source/'board'/('session_%03d'%ordinal)/'organization_04.json')
        key=session['object_id']+'/'+session['variant'];organization[key]=o
        text.append('|%s|%d|%d|%d|%d|%d|%d|%d|%d|' % (key,o['MAC_GEMM'],o['RESET_GEMM'],o['ALU'],
            o['UOP_LOAD'],o['UOP_LOAD_bytes'],o['serialized_uop_bytes'],o['UOP_SRAM_wraps'],o['submissions']))
    write(output/'organization.json',organization)
    with (output/'latencies.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    with (output/'formal_samples.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['object_id','variant','round','seed','elapsed_ms'])
        writer.writeheader();writer.writerows({k:r[k] for k in writer.fieldnames} for r in rows if r['timed'])
    text += ['', '默认fallback保留原始数据安排，仅采用已冻结的必要完成及DMA合法化适配。'
             'T/U保持Tile、线程、张量请求及融合语义相同；F/T允许既有TPS方法改变这些安排。'
             '当前T不是已证明的TPS全局最优程序；更宽线程域C及TopHub参照仍在原报告中。', '',
             '本轮为同启动受控比较，不称统计显著性。完成后暂停。', '',
             '[逐调用原始证据](%s)；[本地fallback资格](%s)。' %
             (str(source/'result.json'),str(LOCAL/'status.json'))]
    (output/'REPORT.md').write_text('\n'.join(text)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(1,2,figsize=(9,4),layout='constrained')
    for ax,item in zip(axes,stats['objects']):
        vals=[item['T_ms'][v] for v in ('F','T','U')]
        ax.bar(['Fallback','TPS','TPS + identity'],vals,color=['#777777','#4477aa','#228833'])
        ax.set_title('YOLO224 node '+item['object_id'].split('node')[-1]);ax.set_ylabel('Complete fused call (ms)')
        for i,value in enumerate(vals):ax.text(i,value,'%.3f'%value,ha='center',va='bottom')
        ax.set_ylim(0,max(vals)*1.18)
    for suffix in ('png','pdf','svg'):fig.savefig(output/('F_T_U_full_calls.'+suffix),dpi=180)
    plt.close(fig)
    write(output/'completion_manifest.json',{'status':'READONLY_F_T_U_CLOSEOUT_COMPLETE','paused':True,
        'new_target_calls':0,'boot_id':boot,'verified_source_artifacts':len(sealed),
        'source_manifest':{'path':str(source/'completion_manifest.json'),'sha256':sha(source/'completion_manifest.json')},
        'artifact_sha256':{str(p.relative_to(output)):sha(p) for p in output.iterdir() if p.is_file()}})
    print({'status':'READONLY_F_T_U_CLOSEOUT_COMPLETE','report':str(output/'REPORT.md')})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();closeout(a.source.resolve(),a.output.resolve())
