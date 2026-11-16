"""Seal fresh YOLO416 local stages without rerunning any target function."""
import argparse
from pathlib import Path
import shutil
from c3_tps_sync_release_local import HERE,read,write,sha,verify,require


def seal_operators(root):
    require(not (root/'completion_manifest.json').exists(),'ALREADY_SEALED')
    status=read(root/'status.json')['status'];frozen=read(root/'freeze.json')['bindings']
    verify(frozen);items=read(root/'objects.json');account=read(root/'call_accounting.json')
    success=status=='ALL_22_FUNCTIONS_LOCAL_QUALIFIED_GRAPH_QUALIFICATION_PENDING'
    objects=[];totals={'registered':0,'helper_returned':0,'certified':0};methods=set()
    for i,obj in enumerate(items):
        unit=root/'objects'/('%02d'%i);counts=dict.fromkeys(totals,0)
        for mode in ('detailed','minimal'):
            folder=unit/('R_'+mode)
            if success:
                require(read(root/('%02d__qualify_%s_worker.json'%(i,mode)))['returncode']==0,
                        'QUALIFICATION_WORKER_FAILED')
                summary=read(folder/'summary.json')
                require(summary['status']=='FULL_LOCAL_QUALIFICATION_PASSED' and summary['target_calls']==7,
                        'QUALIFICATION_SUMMARY_CHANGED')
            for path in sorted(folder.glob('call_*_begin.json')):
                begin=read(path);index=begin['call_index'];counts['registered']+=1
                ep=folder/('call_%02d_execution.json'%index)
                if ep.exists():
                    execution=read(ep)
                    if execution.get('helper_returned') is True:
                        meta=execution['metadata']
                        require(execution['counts']==meta['started_calls']==meta['completed_calls']==index,
                                'HELPER_EXECUTION_COUNTS_CHANGED')
                        counts['helper_returned']+=1
                end=folder/('call_%02d_end.json'%index)
                if end.exists():
                    row=read(end);require(row['status']=='PASS' and row['correct'] is True,'INVALID_END_RECORD')
                    require(row['output_sha256']==row['reference_sha256'] and
                            row['native_dump_sha256']==sha(folder/('native_call_%02d.txt'%index)),
                            'END_OUTPUT_OR_DUMP_IDENTITY_CHANGED')
                    require(all(row.get(k)==obj[k] for k in ('variant','object_id')),'OBJECT_MAPPING_CHANGED')
                    progress=read(folder/('progress_%02d.json'%index))
                    require(progress['status']=='FINITE_FIFO_PROGRESS_PROVED' and
                            progress['method']=='finite_fifo_action_graph','PROOF_ADOPTION_CHANGED')
                    methods.add(progress['method']);counts['certified']+=1
        if success:
            require(counts=={'registered':14,'helper_returned':14,'certified':14},'OBJECT_CALL_BUDGET_NOT_CLOSED')
            require(len(read(unit/'R_mode_equivalence.json')['calls'])==7,'MISSING_MODE_EQUIVALENCE')
            require(read(root/('%02d__prepare_worker.json'%i))['returncode']==0,'PREPARATION_FAILED')
        objects.append({k:obj[k] for k in ('variant','object_id','nodes','symbol')}|counts)
        for k in totals:totals[k]+=counts[k]
    require(totals['registered']==account['registered'] and totals['certified']==account['all_certificates_passed'],
            'INDEPENDENT_ACCOUNTING_DISAGREEMENT')
    if success:
        require(totals=={'registered':308,'helper_returned':308,'certified':308},'FULL_BUDGET_NOT_CLOSED')
        require(not account['execution_count_unresolved_workers'],'UNRESOLVED_EXECUTION')
        require(not (root/'stop.json').exists() and not list(root.glob('failure*.json')),'STOP_STATE_PRESENT')
    write(root/'closeout_validation.json',{'status':status,'objects':objects,'totals':totals,
          'frozen_bindings_verified':len(frozen),'proof_methods':sorted(methods),'FPGA_calls':0,
          'whole_graph_calls':0,'performance_samples':0,'old_attempt_adopted_calls':0})
    text=['# YOLO416 TopHub / TPS 完整融合本地资格','',
          '本次是独立新attempt，采纳已封存的有限FIFO动作图证明；不采纳旧attempt的成功调用。',
          '目标Tile、线程、命令、runtime及参考计算不随证明方法改变。','',
          '|版本|函数数|helper返回|全部证书通过|','|---|---:|---:|---:|']
    for variant in ('TH','P'):
        subset=[x for x in objects if x['variant']==variant]
        text.append('|%s|%d|%d|%d|'%(variant,len(subset),sum(x['helper_returned'] for x in subset),
                                   sum(x['certified'] for x in subset)))
    text+=['', '状态：'+status+'。',
        '登记%d次，helper已返回%d次，全部证书通过%d次。'%(totals['registered'],totals['helper_returned'],totals['certified']),
        '开始/完成证据与未决情况另见call_accounting.json；call_begin不直接作为执行完成。','']
    if success:
        text+=['**能不能执行？** TH与P各11个完整融合函数通过三seed、同实际存储W0→W1→W0、定向用例及两模式资格；22组模式编码一致。',
          '**边界在哪里？** 本地认证覆盖逻辑、实际指令地址、μop内容、缓存、存储与有限FIFO。仍是单函数证据，尚不替代完整图的跨函数存储/缓存和全部提交认证。',
          '**值不值得？** FPGA调用和性能样本均为0，不能据FSim时长判断TPS相对TopHub快慢。',
          '下一阶段仅从同一已冻结最终IR构建完整TH/P图，并检查8个原始输出和12次VTA调用。I=P的去重结论保持，不额外制造冗余。']
    else:
        text+=['本次在首次错误或资源限制处停止。原始stop/failure记录保留，不重试、不补调用、不拼接资格；不形成性能结论。']
    (root/'REPORT.md').open('x').write('\n\n'.join(text)+'\n')
    snapshots=root/'source_snapshots';snapshots.mkdir(exist_ok=False)
    for p,d in frozen.items():
        p=Path(p)
        if p.parent==HERE and p.suffix in ('.py','.cc'):
            shutil.copyfile(p,snapshots/p.name);require(sha(snapshots/p.name)==d,'SNAPSHOT_MISMATCH')
    verify(frozen)
    write(root/'completion_manifest.json',{'status':'SEALED_'+status,
          'FSim_helper_returned':totals['helper_returned'],'all_certificates_passed':totals['certified'],
          'FPGA_calls':0,'whole_graph_calls':0,'performance_samples':0,
          'artifact_sha256':{str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file()}})
    return success


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('operators',));p.add_argument('output',type=Path)
    a=p.parse_args();print({'success':seal_operators(a.output.resolve())})
