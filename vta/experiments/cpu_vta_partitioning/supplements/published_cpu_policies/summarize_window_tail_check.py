"""Report the fixed four-layout tail test without changing scores or selection.

Compute software-VTA occupancy in the exact completion window. Per-stage mean
service and whole-session TID CPU counters retain their different scopes.
"""
from pathlib import Path
import json
import os
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from cpu_vta_partitioning.common import read,write,sha256,verify_inventory
from scheduler_gap_audit.curves import csv_write
from run_scope_rank import assert_plan,inputs_for
from model_ablation.model import score
from audit_formal_window_resources import complement,ns

ROOT=Path(__file__).resolve().parent
PARENT=ROOT/'artifacts/window_tail_check_20261112_01'
OUT=PARENT/'report'
REPORT=ROOT.parents[4]/'paper/my_paper/analysis/20261112_window_tail_check'


def main():
    if OUT.exists() or REPORT.exists():raise FileExistsError('preserve this accepted diagnostic report')
    result_path=PARENT/'analysis/RESULTS.json';result=read(result_path)
    pool=read(PARENT/'POOL.json');protocol=read(PARENT/'PROTOCOL.json');plan=read(PARENT/'PLAN.json')
    old=read(plan['old_result_path'])
    if sha256(plan['old_result_path'])!=plan['old_result_sha256']:
        raise ValueError('old short result changed')
    rank_plan=assert_plan()
    costs={n:inputs_for(rank_plan,n,'original')[1] for n in pool}
    sources={str(result_path):sha256(result_path),plan['old_result_path']:sha256(plan['old_result_path']),
             str(PARENT/'FREEZE.json'):sha256(PARENT/'FREEZE.json'),
             str(ROOT/'audit_formal_window_resources.py'):sha256(ROOT/'audit_formal_window_resources.py')}
    rows=[];stage_rows=[]
    for selected in result['rows']:
        network,key=selected['network'],selected['configuration'];cfg=pool[network][key]
        p=Path(selected['receipt'])
        if sha256(p)!=selected['receipt_sha256']:raise ValueError('tail receipt changed')
        receipt=read(p);verify_inventory(receipt['raw'],receipt['files'])
        if receipt['boot_id']!=result['same_boot']:raise ValueError('tail boot changed')
        raw=[json.loads(s) for s in (Path(receipt['raw'])/'result.jsonl').read_text().splitlines()]
        if len(raw)!=48:raise ValueError('frozen tail frame count differs')
        summary=receipt['summary'];previous=next(r for r in old['rows'] if r['network']==network and r['configuration']==key)
        start,end=[ns(raw[i]['completion_ms']) for i in (8,23)]
        all_holds=[];isolated_VTA=0;isolated_host_CPU=0;isolated_total_CPU=0;caller_session_CPU=0
        for i,s in enumerate(cfg['stages']):
            c=costs[network].get(s,cfg)
            isolated_total_CPU+=c['run_core_ms']+c['boundary_core_ms']
            stage_rows.append(dict(network=network,configuration=key,stage_index=i,program=s['program'],device=s['device'],
                threads=s['threads'],mask=s['mask'],isolated_service_ms=c['service_ms'],
                old_short_service_ms=previous['stage_service_ms'][i],new_tail_service_ms=summary['stage_service_ms'][i],
                source_key=c['source_key'],cost_assumptions=c['assumptions'],
                old_stage_counts=previous['stage_completion_counts_at_window'][i]['counts'],
                new_stage_counts=summary['stage_completion_counts_at_window'][i]['counts']))
            if s['device']!='vta':continue
            isolated_VTA+=c['service_ms'];isolated_host_CPU+=c['run_core_ms']+c['boundary_core_ms']
            held=[(ns(r[f'stage{i}_vta_mutex_acquire_ms']),ns(r[f'stage{i}_vta_mutex_release_ms'])) for r in raw]
            all_holds.extend(held)
            activity=[r for r in summary['thread_ledger'] if r.get('stage')==s['program'] and r['role']=='caller']
            if len(activity)!=1 or activity[0]['scope']!='whole_session_including_warmup_and_tail':
                raise ValueError('VTA caller activity scope differs')
            caller_session_CPU+=activity[0]['cpu_time_delta_ms']
        busy,gaps=complement(start,end,all_holds)
        formal=score(cfg,costs[network],'complete')
        item=dict(network=network,configuration=key,
            role='static_first' if read(PARENT/'LISTS.json')[network]['static_first'][0]['id']==key else 'previous_short_Top20_winner',
            old_short_FPS=selected['old_short_FPS'],new_tail_FPS=selected['new_tail_FPS'],
            observed_FPS_change_percent=100*(selected['new_tail_FPS']/selected['old_short_FPS']-1),
            old_II_ms=previous['II_ms'],new_II_ms=summary['II_ms'],
            formal_predicted_II_ms=formal['score_ns']/1e6,formal_terms_ns=formal['terms_ns'],binding_terms=formal['binding_terms'],
            isolated_VTA_service_sum_ms=isolated_VTA,
            VTA_software_mutex_occupancy_percent=100*busy/(end-start),
            VTA_held_ms_per_completion=busy/15/1e6,VTA_unheld_ms_per_completion=sum(b-a for a,b in gaps)/15/1e6,
            isolated_host_CPU_ms=isolated_host_CPU,
            whole_session_host_CPU_ms_per_call_proxy=caller_session_CPU/len(raw),
            whole_session_host_CPU_not_scored_window_measurement=True,
            CPU_ms_per_completion=summary['process_CPU_ms_per_completion'],isolated_total_CPU_ms=isolated_total_CPU,
            any_stage_exhausted=any(r['stage_exhausted_before_window_end'] for r in selected['inventory']['stages']),
            inventory_changes=[r['inventory_change'] for r in selected['inventory']['stages']],
            all_stage_counts=[r['completed_in_window'] for r in selected['inventory']['stages']],
            physical_VTA_compute_utilization_unknown=True,receipt=str(p),receipt_sha256=sha256(p))
        rows.append(item);sources[str(p)]=sha256(p)
    if len(rows)!=4:raise ValueError('four-layout audit incomplete')
    no_exhaustion=all(not r['any_stage_exhausted'] for r in rows)
    OUT.mkdir();REPORT.mkdir(parents=True)
    write(OUT/'RESULTS.json',dict(rows=rows,stage_rows=stage_rows,sources=sources,
        new_native_sessions=8,new_DNN_inferences=196,model_or_rank_unchanged=True,
        early_exhaustion_removed_in_all_four=no_exhaustion,
        no_new_Top20_best_inferred_from_four_configurations=True,
        separate_processes_and_time_blocks_not_pure_tail_causal_attribution=True,
        observed_no_early_exhaustion_not_a_stationarity_proof=True))
    csv_write(REPORT/'four_configuration_comparison.csv',rows)
    csv_write(REPORT/'stage_cost_and_service.csv',stage_rows)
    lines=[
        '四方案尾部长度匹配验证：保持8帧预热、16时间戳，只把尾部4改为24。',
        '同一boot、同一程序/运行库/阶段模块/输入/绑定/门控，完成8个会话、196次完整推理。未重启、未重载FPGA，未改评分或DP。',
        '全部输出、实际绑定、原始文件哈希和帧计数通过；性能窗口仍是15个完成间隔。',
        '四项均未出现任何阶段提前做完全部任务。两个ResNet配置各阶段窗口完成数均为15；YOLO第一名也均为15，旧短窗口优胜配置有一个VTA阶段为14，其余为15。',
        '', '配置角色 / 原短窗口FPS / 尾部24FPS / 本次变化：',
    ]
    for r in rows:
        role='静态第一名' if r['role']=='static_first' else '旧短窗口Top20优胜配置'
        lines.append('%s %s %s / %.6f / %.6f / %+.3f%%' %
            (r['network'],role,r['configuration'],r['old_short_FPS'],r['new_tail_FPS'],r['observed_FPS_change_percent']))
    lines.extend([
        '', '可支持的结论：',
        '尾部24消除了提前耗尽这一具体边界；ResNet两项FPS变化约0.1–0.2%，仍保留约13FPS的已测配置表现。',
        'YOLO两项本次观察下降约4.8%/6.3%；前段与末段工作计数更一致。由于前后分进程、分时间块，不能将变化全部归为尾部，更不能据此宣称稳态或显著性。',
        '旧短窗口5.435/5.503不覆盖或改写。此次只复核两个YOLO代表，不能把5.159称为新Top20最佳；本次静态第一名5.176略高于旧短窗口优胜配置5.159。',
        '', '剩余误差具体落点：',
    ])
    for r in rows:
        if r['role']!='static_first':continue
        lines.append('%s静态第一名：原完整模型II %.3f ms，本次实际%.3f ms；VTA软件锁在同一窗口占用%.3f%%，每完成间隔持有%.3f ms。' %
            (r['network'],r['formal_predicted_II_ms'],r['new_II_ms'],r['VTA_software_mutex_occupancy_percent'],r['VTA_held_ms_per_completion']))
        if r['network']=='yolov3_tiny416':
            lines.append('  隔离VTA服务之和%.3f ms；原模型由该累计资源项主导。本次VTA锁内时间已接近整个实际II。锁内包含主机、复制、设备和输出，99%%软件锁占用不能称为99%%纯设备计算利用率。'%r['isolated_VTA_service_sum_ms'])
            lines.append('  所有VTA主机隔离CPU需求合计%.3f核毫秒；本次全48帧会话的主机TID计数除以调用数约%.3f核毫秒。后者含初始化/暖尾部且为粗粒度计数，只作工作量线索，不能直接与正式窗口墙钟相减作定量归因。'%
                (r['isolated_host_CPU_ms'],r['whole_session_host_CPU_ms_per_call_proxy']))
    lines.extend([
        '这把下一步重点收敛到：VTA主机与CPU工作竞争时，为什么已串行持有的软件VTA资源服务变长。显式计入主机CPU总量，并不自动预测主机失去CPU期间对VTA锁的延长占用。',
        '当前证据不能区分CPU失去运行机会与设备/DDR等待增长；仍需匹配的设备完成时序。已有旧切图的硬件时序不能直接套到当前vta_01_01/vta_05_06/vta_09_11。',
        '不把本次慢化比例或整段非运行时间回填为通用静态成本；不自动启动全域标定、重搜或新Top20。',
        '', '文献对应：',
        'Pipe-it原文第8页式(11)以最长阶段服务决定流水吞吐，第10页实验报告50图像流平均吞吐：https://arxiv.org/pdf/1903.05898 。该论文没有给出本项目的尾部24规则；这里依据当前队列源码推导并验证边界，属于测量口径核查。',
    ])
    (REPORT/'FINDINGS_zh.txt').write_text('\n'.join(lines)+'\n')
    draw(rows)
    paths=[OUT/'RESULTS.json',*REPORT.iterdir()]
    write(OUT/'MANIFEST.json',dict(source_sha256=sha256(__file__),
        files={str(p):sha256(p) for p in paths if p.is_file()},sources=sources,
        approved_scope_executed_exactly=True,new_native_sessions=8,new_DNN=196))
    print(json.dumps(dict(rows=[{k:r[k] for k in ('network','role','old_short_FPS','new_tail_FPS','any_stage_exhausted','VTA_software_mutex_occupancy_percent')} for r in rows],
                         report=str(REPORT/'FINDINGS_zh.txt')),ensure_ascii=False))


def draw(rows):
    os.environ.setdefault('MPLCONFIGDIR','/tmp/cpu_vta_tail_check_mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    plt.rcParams.update({'font.family':['DejaVu Sans','Droid Sans Fallback'],
        'axes.unicode_minus':False,'svg.fonttype':'none','pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(12,5.7))
    for ax,network,title in zip(axes,('resnet18','yolov3_tiny416'),('ResNet18','YOLOv3-tiny')):
        chosen=[r for r in rows if r['network']==network]
        for key,shift,color in [('old_short_FPS',-.18,'#3F79AA'),('new_tail_FPS',.18,'#CC8A32')]:
            bars=ax.bar([i+shift for i in range(2)],[r[key] for r in chosen],width=.33,color=color)
            ax.bar_label(bars,labels=['%.3f'%r[key] for r in chosen],padding=4,fontsize=11)
        ax.set_xticks(range(2),['静态第一名','旧短窗口Top-20\n优胜配置'],fontsize=11)
        ax.set_ylim(0,15 if network=='resnet18' else 6.5);ax.set_ylabel('本次实测FPS（越高越好）')
        ax.set_title(title,fontsize=16,pad=15);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
        for s in ('top','right'):ax.spines[s].set_visible(False)
    fig.suptitle('保持16个时间戳，仅改变尾部长度',fontsize=19,y=.98)
    fig.legend(handles=[Patch(facecolor='#3F79AA',label='原尾部4帧'),Patch(facecolor='#CC8A32',label='本次尾部24帧')],
               loc='upper center',bbox_to_anchor=(.5,.92),ncol=2,frameon=False,fontsize=12)
    fig.subplots_adjust(left=.075,right=.97,bottom=.27,top=.78,wspace=.28)
    fig.text(.5,.11,'四项均消除了阶段提前耗尽；同一启动、不同进程及时间块，不宣称纯尾部因果效应或稳态。',ha='center',fontsize=10)
    fig.text(.5,.045,'只复核既定代表，未重测整个Top-20；静态模型、排名、程序与输入均未改变。',ha='center',fontsize=10,color='#5A6876')
    for ext in ('png','svg','pdf'):fig.savefig(REPORT/('tail_comparison.'+ext),dpi=180,facecolor='white')
    plt.close(fig)


if __name__=='__main__':main()
