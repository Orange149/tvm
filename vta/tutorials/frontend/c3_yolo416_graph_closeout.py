"""Read-only validation and immutable closeout of whole-graph local stages."""
import argparse
from pathlib import Path
import shutil
from c3_tps_sync_release_local import HERE,read,write,sha,verify,require


def seal(root,stage):
    require(not (root/'completion_manifest.json').exists(),'ALREADY_SEALED')
    status=read(root/'status.json')['status'];freeze=root/'freeze.json'
    bindings=read(freeze)['bindings'] if freeze.exists() else {}
    verify(bindings);success=False;detail={}
    if stage=='prepare':
        success=status=='WHOLE_TH_P_GRAPHS_AND_HELPER_BUILT_LOCAL_QUALIFICATION_PENDING'
        if success:
            require(read(root/'prepare_complete.json')['status']==status,'GRAPH_PREPARE_COMPLETION_CHANGED')
            for variant in ('TH','P'):
                require(read(root/(variant+'_worker.json'))['returncode']==0,'GRAPH_BUILD_WORKER_FAILED')
                require(read(root/variant/'result.json')['graph_calls']==12 and
                        read(root/variant/'result.json')['raw_outputs']==8,'GRAPH_BUILD_COVERAGE_CHANGED')
                require(sha(root/variant/'reference_fsim.so')==sha(root/variant/'graph_fsim.so'),
                        'REFERENCE_BINARY_CHANGED')
                detail[variant]=read(root/variant/'result.json')
        counts={'registered':0,'helper_returned':0,'certified':0}
    else:
        success=status=='WHOLE_TH_P_GRAPH_LOCAL_QUALIFIED_ARM_AND_FPGA_QUALIFICATION_PENDING'
        account=read(root/'call_accounting.json')
        counts={'registered':0,'helper_returned':0,'certified':0}
        for variant in ('TH','P'):
            for mode in ('detailed','minimal'):
                folder=root/(variant+'_'+mode);unit=dict.fromkeys(counts,0)
                for p in sorted(folder.glob('call_*_begin.json')):
                    i=read(p)['call_index'];unit['registered']+=1
                    ep=folder/('call_%02d_execution.json'%i)
                    if ep.exists():
                        e=read(ep)
                        if e.get('helper_returned') is True:
                            require(e['metadata']['started_calls']==e['metadata']['completed_calls']==i,
                                    'GRAPH_EXECUTION_COUNT_CHANGED')
                            unit['helper_returned']+=1
                    end=folder/('call_%02d_end.json'%i)
                    if end.exists():
                        row=read(end);require(row['status']=='PASS' and row['correct'] is True and
                            row['all_8_raw_heads_elementwise_equal'] is True and row['native_submit_count']==12,
                            'GRAPH_END_NOT_QUALIFIED')
                        require(row['native_dump_sha256']==sha(folder/('native_call_%02d.txt'%i)),
                                'GRAPH_DUMP_CHANGED')
                        progress=read(folder/('progress_%02d.json'%i))['fragments']
                        require(len(progress)==12 and all(x['status']=='FINITE_FIFO_PROGRESS_PROVED'
                                and x['method']=='finite_fifo_action_graph' for x in progress),
                                'GRAPH_FRAGMENT_PROGRESS_UNPROVED')
                        if mode=='detailed':
                            require(row['intermediate_integer_comparisons']==12,'GRAPH_INTEGER_COVERAGE_CHANGED')
                        unit['certified']+=1
                if success:
                    require(unit=={'registered':6,'helper_returned':6,'certified':6},'GRAPH_WORKER_BUDGET_OPEN')
                    require(read(root/(variant+'_'+mode+'_worker.json'))['returncode']==0,'GRAPH_WORKER_FAILED')
                detail[variant+'_'+mode]=unit
                for k in counts:counts[k]+=unit[k]
            if success:require(len(read(root/(variant+'_mode_equivalence.json'))['calls'])==6,
                               'GRAPH_MODE_EQUIVALENCE_MISSING')
        require(counts['registered']==account['registered'] and
                counts['certified']==account['all_certificates_passed'],'GRAPH_ACCOUNTING_DISAGREEMENT')
        if success:require(counts=={'registered':24,'helper_returned':24,'certified':24},'GRAPH_FULL_BUDGET_OPEN')
    if success:require(not (root/'stop.json').exists() and not list(root.glob('failure*.json')),'STOP_STATE_PRESENT')
    write(root/'closeout_validation.json',{'status':status,'success':success,'detail':detail,'counts':counts,
          'FPGA_calls':0,'performance_samples':0})
    text=['# YOLO416整图'+('构建' if stage=='prepare' else '本地资格')+'封存','',
          'TH为精确TopHub重实例化，P为固定解析规则选择的本地适配TPS；I=P，不追加候选。','',
          '状态：'+status+'。',
          '完整GraphExecutor登记%d次，helper返回%d次，全部证书通过%d次。'%(counts['registered'],counts['helper_returned'],counts['certified'])]
    if success and stage=='prepare':
        text+=['完整TH/P图保留11个VTA函数、12次VTA调用、21个CPU函数及8个原始输出。',
               '编译、保存/载入和initializer绑定通过，独立参考库为字节相同的不同文件。未执行整图。']
    elif success:
        text+=['两版各12次整图调用通过，包含三seed与同实际参数存储W0→W1→W0，两种采集模式一致。',
               '全部8个原始输出逐元素一致；详细模式逐调用核对12个中间整数结果及只读输入。',
               '全部实际提交、跨函数缓存与图存储复用、有限FIFO进度闭合；不是把孤立单函数证书当作整图证书。',
               '参考使用独立NumPy整数卷积及明确受信的共同CPU代码，未执行第二个GraphExecutor作参考。',
               '尚无ARM/FPGA资格或正式时延，不能由本地时长判断TPS胜负。']
    else:
        text+=['首次失败或资源限制处停止。原始failure/stop保留；不重试、不补调用、不拼接，不形成性能结论。']
    (root/'REPORT.md').open('x').write('\n\n'.join(text)+'\n')
    snapshots=root/'source_snapshots';snapshots.mkdir(exist_ok=False)
    for p,d in bindings.items():
        p=Path(p)
        if p.parent==HERE and p.suffix in ('.py','.cc'):
            shutil.copyfile(p,snapshots/p.name);require(sha(snapshots/p.name)==d,'SNAPSHOT_CHANGED')
    verify(bindings)
    write(root/'completion_manifest.json',{'status':'SEALED_'+status,'GraphExecutor_helper_returned':counts['helper_returned'],
          'all_certificates_passed':counts['certified'],'FPGA_calls':0,'performance_samples':0,
          'artifact_sha256':{str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file()}})
    return success


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('prepare','qualify'));p.add_argument('output',type=Path)
    a=p.parse_args();print({'success':seal(a.output.resolve(),a.stage)})
