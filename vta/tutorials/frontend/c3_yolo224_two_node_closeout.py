"""Independent, zero-target-call whole-graph result reconciliation.

Recompute paired statistics from raw per-unit records; check every diagnostic
FINISH has all material instructions as actual FIFO/token ancestors. The
completed experiment and its frozen tools remain read-only.
"""
import argparse
import math
from pathlib import Path
from statistics import median
from c3_tps_sync_release_local import HOLD, HERE, read, write, sha, require, verify
from c3_official_window_native_binding import material_ancestor_payloads
from c3_five_tradeoff_commands import decode_submissions


def closeout(root,out):
    out.mkdir(exist_ok=False)
    seal=read(root/'completion_manifest.json')
    require(seal['success'] and seal['status']=='WHOLE_YOLO224_TWO_NODE_B_A_COMPLETE','COMPLETE_ATTEMPT_ONLY')
    verify({str(root/p):v for p,v in seal['artifact_sha256'].items()})
    verify(read(root/'freeze.json')['bindings'])
    ledger=read(root/'call_manifest.json');rows=read(root/'board/results.json')
    require(len(rows)==120 and len(ledger['sessions'])==20,'EXACT_BUDGET')
    flattened=[];closure=[];started=completed=0
    for i,session in enumerate(ledger['sessions'],1):
        folder=root/'board'/('session_%03d'%i)
        summary=read(folder/'summary.json')
        require(summary['status']=='COMPLETE_FRESH_UNIT_SUCCESS' and summary['session']==session,'SESSION_IDENTITY')
        unit=summary['results'];require(len(unit)==6,'SIX_CALLS_PER_SESSION')
        require(unit==[read(folder/('call_%02d_end.json'%j)) for j in range(1,7)],'UNIT_END_RECORDS_DIFFER')
        flattened+=unit
        last=unit[-1]['helper']['metadata'];started+=last['started_calls'];completed+=last['completed_calls']
        for j,row in enumerate(unit,1):
            h=row['helper'];m=h['metadata'];sub=h['submits']
            require(h['helper_returned'] and h['counts']==m['started_calls']==m['completed_calls']==j and
                    m['packed_calls_per_invocation']==1 and m['adds_synchronize'] is False,'ONE_GRAPH_CALL')
            require(sub['attempted_submits']==sub['recorded_submits']==len(sub['submits'])==22 and not sub['overflow'],
                    'ALL_22_SUBMISSIONS')
            require(all(s['completion'] and s['timeout']==0 and s['reason']=='explicit_sync' for s in sub['submits']),
                    'SUBMIT_NOT_COMPLETE')
            require(row['status']=='PASS' and row['target_calls']==1 and
                    row['output_checks']['all_8_raw_outputs_equal'],'OUTPUT_OR_TARGET_COUNT')
            if row['timed']:
                require(row['collection']=='minimal' and row['collection_status']['detailed_clock_reads']==0 and
                        type(row['elapsed_ms']) in (int,float) and math.isfinite(row['elapsed_ms']) and
                        row['elapsed_ms']>0,'INVALID_FORMAL_SAMPLE')
            if session['phase']!='diagnostic':continue
            native=read(folder/('native_%02d.json'%j));storage=read(folder/('storage_%02d.json'%j))
            require(native['status']=='ALL_22_SUBMISSIONS_AND_12_FUNCTION_CALLS_CERTIFIED' and
                    len(native['calls'])==12 and storage['status']=='GRAPH_NATIVE_STORAGE_REUSE_BOUND',
                    'FULL_NATIVE_OR_STORAGE_CERTIFICATE_MISSING')
            require(len(read(folder/('integer_comparisons_%02d.json'%j)))==12,'INTEGER_INTERMEDIATE_COVERAGE')
            dump=folder/('native_%02d.txt'%j);blocks=decode_submissions(dump.read_text())
            require(len(blocks)==22,'DUMP_SUBMISSION_COUNT')
            for ordinal,records in enumerate(blocks,1):
                mask=[]
                for k,r in enumerate(records):
                    control=r['description'] in {'NOP-MEMORY-STAGE','NOP-COMPUTE-STAGE','NOP-STORE-STAGE','FINISH'}
                    mask.append(0 if control else 1<<k)
                ancestors=material_ancestor_payloads(records,mask);material=sum(mask)
                require(records[-1]['description']=='FINISH' and ancestors[-1]&material==material,
                        'FINISH_PRECEDES_MATERIAL_COMPLETION')
            closure.append({'session':i,'position':j,'dump_sha256':sha(dump),'completed_submissions':22,
                            'all_material_precedes_FINISH':True})
    require(flattened==rows,'GLOBAL_ARRAY_DIFFERS_FROM_UNIT_RECORDS')
    require(started==completed==120 and len(closure)==36,'ACTUAL_CALL_OR_DIAGNOSTIC_BUDGET')
    reference_cases={}
    for row in rows:
        identity=row['reference_identity'];key=(row['seed'],row['weight'])
        case={k:identity[k] for k in ('data_sha256','weight_sha256','expected_head_sha256')}
        require(key not in reference_cases or reference_cases[key]==case,'B_A_INPUT_WEIGHT_OR_REFERENCE_CHANGED')
        reference_cases[key]=case
        require(row['output_checks']['head_sha256']==case['expected_head_sha256'],'ACTUAL_GRAPH_HEAD_HASH')
    samples={(r['round'],r['variant'],r['seed']):r['elapsed_ms'] for r in rows if r['timed']}
    require(len(samples)==36,'NO_DUPLICATE_OR_MISSING_FORMAL_SAMPLE')
    seeds=(0,20250901,20260910)
    b=[median(samples[r,'TH',s] for s in seeds) for r in range(1,7)]
    a=[median(samples[r,'P',s] for s in seeds) for r in range(1,7)]
    gains=[100*(1-y/x) for x,y in zip(b,a)];saved=[x-y for x,y in zip(b,a)]
    stats={'B_ms':median(b),'A_ms':median(a),'round_B_ms':b,'round_A_ms':a,
           'round_gains_percent':gains,'paired_gain_percent':median(gains),
           'strict_wins':sum(y<x for x,y in zip(b,a)),'median_saved_ms':median(saved),
           'gain_range_percent':[min(gains),max(gains)],
           'gain_MAD_percent':median(abs(g-median(gains)) for g in gains)}
    existing=read(root/'result.json')['measurement']
    require(existing['T_ms']=={'B':stats['B_ms'],'A':stats['A_ms']} and
            existing['A_over_B_gain_percent']==stats['paired_gain_percent'] and
            existing['A_strict_wins']==stats['strict_wins'] and existing['round_gains_percent']==gains,
            'INDEPENDENT_STATISTICS_DIFFER')
    first=read(root/'board/preflight.json');last=read(root/'board/final_state.json')
    require(first['boot_id']==last['boot_id'] and not last['accelerator_occupancy_scan'],
            'BOOT_CHANGED_OR_OWN_SERVER_LEFT_RUNNING')
    result={'status':'COMPLETE_INDEPENDENTLY_RECONCILED','network':'YOLOv3-tiny-224',
            'only_changed_graph_nodes':[34,46],'B':'two original TPS kernels',
            'A':'same TPS kernels with UOP cache identity reuse',
            'all_other_programs_and_parameters_fixed':True,'not_full_network_TPS_or_TopHub':True,
            'FPGA_graph_calls':120,'formal_samples':36,'fresh_RPC_units':20,
            'diagnostic_calls':36,'diagnostic_submissions_with_FINISH_proof':len(closure)*22,
            'new_validation_target_calls':0,'statistics':stats,'boot_id':first['boot_id'],
            'raw_experiment':str(root),'raw_manifest_sha256':sha(root/'completion_manifest.json')}
    write(out/'result.json',result);write(out/'FINISH_closure.json',closure)
    lines=['# YOLO224固定图：节点34、46身份复用的整图净收益','',
        '同一份YOLOv3-tiny-224图，仅将节点34、46的TPS B换成身份复用A；其余原ARM程序、图结构与训练参数固定。','',
        '| 程序 | 整图完整调用时延 |','|---|---:|',
        '| B：优化前 | %.6f ms |'%stats['B_ms'],'| A：优化后 | %.6f ms |'%stats['A_ms'],'',
        '六轮配对改善中位数 **%.4f%%**，严格胜 **%d/6**，每轮节省时间中位数 **%.6f ms**。'%(
            stats['paired_gain_percent'],stats['strict_wins'],stats['median_saved_ms']),
        '六轮改善范围 %.4f%%～%.4f%%，MAD %.4f个百分点。'%(
            *stats['gain_range_percent'],stats['gain_MAD_percent']),'',
        '| 轮次 | B ms | A ms | 改善 % |','|---|---:|---:|---:|']
    for r,(x,y,g) in enumerate(zip(b,a,gains),1):lines.append('| %d | %.6f | %.6f | %.4f |'%(r,x,y,g))
    lines += ['', '24次新attempt完整融合FSim资格通过；板端120次整图调用全部通过，含36个正式样本、20个fresh RPC单元。',
        '每次检查8个输出及只读参数；36次详细诊断还核对12个整数中间结果、全部22次提交、μop、缓存、存储与有限FIFO。',
        '独立回读20份单元记录与全局结果逐项一致；36份诊断dump的792次提交均确认所有实质指令先于FINISH。',
        '正式计时只执行一次完整GraphExecutor.run，覆盖全部命令组织、提交和等待；RPC、输入更新、初始化和输出检查在外。',
        '', '这次结果回答两节点身份复用在固定YOLO224图上的实际增量；不是全网TPS对TopHub的比较。',
        '41.67%/18.40%来自另一个C/A算子比较，未并入本轮样本或改善率；本轮没有改Tile、线程或张量安排。',
        '本次为单启动六轮受控比较，不称统计显著性、独立启动复验或整网FPS。','',
        '准备中的local_run01因检查器全序要求过强，在首次helper完成后停止并封存；修正仅离线回放原dump。',
        '新local_run02从头完成24次资格；旧失败调用不拼接，也不算新attempt资格成功。',
        '本轮新增FSim完整调用共25次（旧attempt完成1次但证书失败，新attempt合格24次）；验证脚本不执行目标。',
        '', '原始整图账本、样本与冻结身份：[实验报告](../'+root.name+'/REPORT.md)。完成后暂停。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    write(out/'completion_manifest.json',{'status':result['status'],'paused':True,
        'source_script_sha256':sha(Path(__file__)),
        'artifact_sha256':{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}})
    print(stats,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('experiment',type=Path);p.add_argument('output',type=Path)
    args=p.parse_args();closeout(args.experiment.resolve(),args.output.resolve())
