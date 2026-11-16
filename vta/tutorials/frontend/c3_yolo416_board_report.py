"""Report only a complete successful TH/P attempt; otherwise seal failure."""
import argparse
from pathlib import Path
import shutil
from c3_tps_sync_release_local import HERE,read,write,sha,verify,require
from c3_yolo416_comparison_protocol import analyse


def report(root):
    require(not (root/'completion_manifest.json').exists(),'ATTEMPT_ALREADY_SEALED')
    summary_path=root/'board/summary.json'
    require(summary_path.exists() or (root/'prepare_failure.json').exists(),'ATTEMPT_NOT_FINISHED')
    summary=read(summary_path) if summary_path.exists() else read(root/'prepare_failure.json')
    success=summary.get('status')=='COMPLETE_TH_P_OPERATORS_AND_GRAPH_SINGLE_BOOT_ATTEMPT'
    text=['# YOLO416：TopHub与本地适配TPS的算子及整图对照','',
          'TH：PYNQ来源官方TopHub精确缓存配置，在AXU5EVB上重实例化和重测；原搜索预算未知，不称当前板全空间最优。',
          'P：TPS解析规则选取的作者schedule本地适配，不是完整作者编译器/runtime的移植，也不保证本板最快。',
          'I与P的执行表示相同，未做I/P差异板测，不填伪实测0%。','']
    if success:
        frozen=read(root/'freeze.json');verify(frozen['bindings'])
        ledger=read(root/'call_manifest.json');rows=read(root/'board/results.json')
        account=read(root/'board/call_accounting.json');n=ledger['total_target_calls']
        require(account['registered_attempts']==account['helper_started_observed_lower_bound']==
                account['helper_completed_observed_lower_bound']==account['certified']==n and
                not account['execution_unresolved_units'] and not account['invalid_records'],
                'CALL_ACCOUNTING_NOT_CLOSED')
        stats=analyse(ledger,rows);write(root/'statistics.json',stats)
        require(all(r['output_checks']['correct'] is True for r in rows),'OUTPUT_QUALIFICATION_INCOMPLETE')
        cold=[r for r in rows if r['phase']=='qualification' and r['position']==1]
        write(root/'qualification_first_calls.json',cold)
        artifacts=read(root/'artifacts.json')
        text+=['## 能不能执行','',
            f"{n}次完整目标调用全部通过；{len(ledger['sessions'])}个fresh RPC单元，{ledger['formal_samples']}个正式样本。",
            '三个seed、同实际权重存储W0→W1→W0、输出哨兵和只读参数检查通过。完整图保留12次VTA调用、21个CPU函数及8个原始输出，详细诊断逐调用核对中间整数结果。',
            '参考的整数卷积独立计算；CPU浮点路径使用同一冻结ARM CPU代码，是明确受信的共同组件。','',
            '## 边界在哪里','',
            '比较保持网络结构、训练参数、融合语义及CPU/VTA分工。TH/P允许使用各自冻结Tile及装载安排，性能差异属于既有TPS复现的整体效果。',
            '原始TPS采用解析代价最小的固定支持域代表，没有按板端时延挑选Tile、线程或局部层。仅结构相同者去重，不补候选。','',
            '## 完整调用值不值','',
            '|对象|TH T / ms|P T / ms|P/TH逐轮改善中位数 / %|严格胜场|逐轮绝对节省中位数 / ms|',
            '|---|---:|---:|---:|---:|---:|']
        for row in stats['objects']:
            label='完整GraphExecutor.run' if row['kind']=='graph' else row['object_id']
            if row['status']!='MEASURED_FULL_CALL':
                text.append('|'+label+'|同一执行表示，未做差异板测|||||');continue
            text.append('|%s|%.6f|%.6f|%.4f|%d/6|%.6f|'%(label,row['T_ms']['TH'],row['T_ms']['P'],
                row['Gain_P_over_TH_percent'],row['strict_wins'],row['median_absolute_ms']))
        text+=['', 'T为六个轮内三seed中位数的中位数；改善率先逐轮取比值再取中位数，不用两个T相除替代。全部对象、负结果、六轮值、范围、MAD及离散程度见statistics.json。',
               '完整图时间直接来自一次GraphExecutor.run，包含CPU函数、所有VTA函数及必要等待，不是单算子时延之和。',
               'RPC、输入准备、参数更新、bitstream初始化、输出读取及检查在计时外。正式最小模式仍含固定模式检查和逐提交记录成本，不称零仪器开销。',
               '详细资格首调用成本单列，不与最小模式相减解释缓存成本；正式关闭的profile字段不冒充正式实测。',
               '本次是同一Linux启动的六轮受控比较，不称统计显著性、独立复验或整网FPS。P/TH属于既有方法的基线强弱证据，不能当作I/P的新创新收益。']
        organization=[]
        for i,session in enumerate(ledger['sessions'],1):
            if session['phase']!='diagnostic':continue
            for position in range(1,7):
                path=root/'board'/('session_%03d'%i)/('organization_%02d.json'%position)
                organization.append({'session':session,'position':position,**read(path)})
        write(root/'organization.json',organization)
        text+=['', 'GEMM/RESET/ALU/NOP、真实UOP LOAD、载荷、缓存分配/重装/回绕及提交计数见organization.json。计数不直接等于性能原因，无法分解的成本不强行归因。']
    else:
        counts=summary.get('accounting',{})
        failure=read(root/'board/failure.json') if (root/'board/failure.json').exists() else summary
        text+=['## 本次停止','',str(summary.get('status','PREPARATION_FAILED')),
            '登记尝试%d次，helper完成证据下界%d次，完整证书通过%d次。'%(counts.get('registered_attempts',0),
                counts.get('helper_completed_observed_lower_bound',0),counts.get('certified',0)),
            '停止原因：'+str(failure.get('message',failure.get('type','见原始失败记录'))),
            '本attempt未完成，不能形成性能结论；已完成调用和失败证据保留，不重试、不补样、不拼接。']
    (root/'REPORT.md').open('x').write('\n'.join(text)+'\n')
    snapshots=root/'source_snapshots';snapshots.mkdir(exist_ok=False)
    mismatches=[]
    if (root/'freeze.json').exists():
        for p,d in read(root/'freeze.json')['bindings'].items():
            p=Path(p)
            if p.parent==HERE and p.suffix in ('.py','.cc'):
                actual=sha(p) if p.exists() else None
                if actual!=d:
                    mismatches.append({'source':str(p),'expected_sha256':d,'actual_sha256':actual})
                    if success:raise ValueError('CLOSEOUT_SOURCE_CHANGED')
                    if p.exists():
                        folder=snapshots/'unverified_current';folder.mkdir(exist_ok=True)
                        shutil.copyfile(p,folder/p.name)
                else:shutil.copyfile(p,snapshots/p.name)
    if mismatches:write(root/'source_identity_failures.json',mismatches)
    write(root/'completion_manifest.json',{'status':'SEALED_'+('COMPLETE_TH_P_COMPARISON' if success else 'STOPPED_TH_P_COMPARISON'),
        'performance_conclusion_allowed':success,
        'artifact_sha256':{str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}})
    return success


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args();print({'success':report(a.output.resolve())})
