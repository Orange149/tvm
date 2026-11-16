"""Explain the sealed current seven-stage trace without fitting a predictor."""
from pathlib import Path
import json
import os
import statistics

import run_current_pipeline_timeline as native
import run_current_host_lock_replay as model
from current_timeline_metrics import cpu_calls,host_calls
from cpu_vta_partitioning.common import read,write,sha256,digest,verify_inventory
from scheduler_gap_audit.curves import csv_write

ROOT=native.ROOT
OUT=native.OUT/'analysis'
REPORT=ROOT.parents[4]/'paper/my_paper/analysis/20261112_current_pipeline_timeline'


def jsonl(path):return [json.loads(s) for s in Path(path).read_text().splitlines()]


def load_receipt(path,summarize,bundle,sources):
    value=read(path);verify_inventory(value['raw'],value['files'])
    if digest(summarize(value['raw'],value['task'],bundle))!=digest(value['summary']):raise ValueError('accepted summary changed')
    raw=Path(value['raw'])
    if (raw/'boot_id').read_text().strip()!=value['boot_id'] or (raw/'task_hash').read_text().strip()!=digest(value['task']):
        raise ValueError('raw boot or task changed')
    sources[str(path)]=sha256(path);return value


def main():
    if OUT.exists() or REPORT.exists():raise FileExistsError('preserve the matched timeline report')
    native.validate();model.check();state=read(native.RUN/'state.json')
    if state['status']!='COMPLETE' or state['native_sessions']!=3 or state['new_DNN']!=97:raise ValueError('three-session timeline incomplete')
    sources={str(native.OUT/'PLAN.json'):sha256(native.OUT/'PLAN.json'),
        str(native.OUT/'TASKS.json'):sha256(native.OUT/'TASKS.json'),str(Path(__file__)):sha256(__file__),
        str(ROOT/'current_timeline_metrics.py'):sha256(ROOT/'current_timeline_metrics.py')}
    items={};count=0
    for mode in ('qualify','clean','observed'):
        doc=native.task(mode);path=native.RUN/'accepted'/state['boot_id']/(doc['task_id']+'.json')
        item=load_receipt(path,native.summarize,native.BUNDLE,sources)
        if item['task']!=doc or item['boot_id']!=state['boot_id']:raise ValueError('timeline identity changed')
        items[mode]=item;count+=len(jsonl(Path(item['raw'])/'result.jsonl'))
    if count!=97:raise ValueError('timeline frame count differs')
    if len({digest(v['summary']['output_fingerprints']) for v in items.values()})!=1:raise ValueError('complete outputs differ')
    observed=items['observed'];raw=Path(observed['raw']);phases=jsonl(raw/'phase_trace.jsonl');events=jsonl(raw/'wait_trace.jsonl')
    files=read(native.BUNDLE/'manifest.json')['files']
    pipeline=cpu_calls('cpu_02_04',phases,events,files,8,16,[1,2,3])
    isolated={}
    for name in native.isolated.CASES:
        doc=native.isolated.task(dict(scope='component',case=name,mode='observed'))
        path=native.isolated.RUN/'accepted'/state['boot_id']/(doc['task_id']+'.json')
        isolated[name]=load_receipt(path,native.isolated.summarize,native.isolated.BUNDLE,sources)
    isoCPU=isolated['cpu_02_04'];iraw=Path(isoCPU['raw'])
    isolation=cpu_calls('cpu_02_04',jsonl(iraw/'phase_trace.jsonl'),jsonl(iraw/'wait_trace.jsonl'),
        read(native.isolated.BUNDLE/'manifest.json')['files'],2,8,[1,2,3])
    if pipeline['signature']!=isolation['signature']:raise ValueError('different CPU kernel sequence/identity')
    replay=None;replay_path=None
    for leaf,h in read(model.OUT/'REPLAY_FREEZE.json')['files'].items():
        path=model.OUT/leaf
        if sha256(path)!=h:raise ValueError('previous prediction changed')
        value=read(path)
        if value['spec']==dict(configuration='cfg_2787a77188dd5f9363d00b15',CPU='finite',completion='independent'):
            replay=value['result'];replay_path=path;sources[str(path)]=h
    if replay is None:raise ValueError('previously specified prediction missing')
    kernels=[]
    for i,(a,b) in enumerate(zip(isolation['kernels'],pipeline['kernels'])):
        joins=[v for v in replay['kernel_joins'] if v['stage']==2 and v['kernel']==i]
        if len(joins)!=16:raise ValueError('model kernel coverage differs')
        kernels.append(dict(kernel_human=i+1,kernel=i,module_sha256=a['module_sha256'],function_offset=a['function_offset'],
            isolated_compute_CPU_ms=a['means_ms']['compute_CPU_ms'],pipeline_compute_CPU_ms=b['means_ms']['compute_CPU_ms'],
            isolated_callback_span_ms=a['means_ms']['callback_span_ms'],pipeline_callback_span_ms=b['means_ms']['callback_span_ms'],
            isolated_entry_skew_ms=a['means_ms']['entry_skew_ms'],pipeline_entry_skew_ms=b['means_ms']['entry_skew_ms'],
            pipeline_caller_tail_CPU_ms=b['means_ms']['caller_tail_CPU_ms'],
            predicted_join_span_ms=statistics.mean(v['join_ns']-v['ready_ns'] for v in joins)/1e6,
            predicted_first_service_skew_ms=statistics.mean(v['first_service_skew_ns'] for v in joins)/1e6,
            isolated_last_slots=a['last_slots'],pipeline_last_slots=b['last_slots'],
            entry_and_model_first_service_are_approximate_correspondences=True))
    host_rows=[];host_details={}
    for name in ('vta_01_01','vta_05_06','vta_09_11'):
        item=isolated[name];path=Path(item['raw'])
        a=host_calls(name,jsonl(path/'phase_trace.jsonl'),item['summary']['absolute_device']['mapped'],2,8)
        b=host_calls(name,phases,observed['summary']['absolute_device']['mapped'],8,16)
        host_details[name]=dict(isolated=a,pipeline=b)
        host_rows.append(dict(program=name,**{'isolated_'+k:v for k,v in a['means_ms'].items()},
            **{'pipeline_'+k:v for k,v in b['means_ms'].items()}))
    clean=items['clean']['summary'];measured=observed['summary']
    iso_cost=read(native.isolated.OUT/'projection/ISOLATED_PHASES.json')['profiles']['cpu_02_04']['clean_cost']
    clean_gap=abs(clean['stage_service_ms'][2]-iso_cost['service_ms'])
    probe_delta=abs(measured['stage_service_ms'][2]-clean['stage_service_ms'][2])
    quality=dict(clean_CPU2_4_service_ms=clean['stage_service_ms'][2],observed_CPU2_4_service_ms=measured['stage_service_ms'][2],
        isolated_clean_service_ms=iso_cost['service_ms'],absolute_probe_service_delta_ms=probe_delta,
        absolute_isolation_pipeline_gap_ms=clean_gap,limit_ms=.2*clean_gap,passed=probe_delta<=.2*clean_gap,
        clean_FPS=clean['FPS'],observed_FPS_not_formal=measured['FPS'],
        performance_II_probe_difference_percent=100*(measured['II_ms']/clean['II_ms']-1))
    # Check the original-executable clean only as a separate bridge reference.
    bridge=read(ROOT/'artifacts/matched_device_profile_20261112_01/PLAN.json')['clean_reference']
    if sha256(bridge['receipt'])!=bridge['receipt_sha256']:raise ValueError('original program clean bridge changed')
    previous=read(bridge['receipt']);verify_inventory(previous['raw'],previous['files']);sources[bridge['receipt']]=bridge['receipt_sha256']
    result=dict(isolated_CPU=isolation,pipeline_CPU=pipeline,kernels=kernels,host_rows=host_rows,host_details=host_details,
        probe_quality=quality,sources=sources,new_native_sessions=3,new_DNN=97,no_reboot=True,no_reprogram=True,
        original_executable_bridge_FPS=1000/previous['summary']['II_ms'],current_phase_executable_clean_FPS=clean['FPS'],
        separate_program_identity_and_time_blocks_not_performance_improvement=True,
        native_counts_at_completion_window=clean['stage_completion_counts_at_window'],
        model=replay['replay'],model_source=str(replay_path),
        no_model_or_DP_or_Top20_changed=True,no_wait_sum_added_to_II=True,
        controls_and_scheduler_queue_not_directly_measured=True,whole_goal_complete=False)
    OUT.mkdir();REPORT.mkdir(parents=True)
    write(OUT/'RESULTS.json',result);csv_write(REPORT/'CPU_kernel_comparison.csv',kernels);csv_write(REPORT/'VTA_host_comparison.csv',host_rows)
    report(result);draw(result)
    write(OUT/'MANIFEST.json',dict(sources=sources,files={str(p):sha256(p) for p in [OUT/'RESULTS.json',*REPORT.iterdir()] if p.is_file()}))
    print(json.dumps(dict(quality=quality,CPU_isolated=isolation['means_ms'],CPU_pipeline=pipeline['means_ms'],
        host_rows=host_rows,report=str(REPORT/'FINDINGS_zh.txt')),ensure_ascii=False))


def report(r):
    a,b=r['isolated_CPU']['means_ms'],r['pipeline_CPU']['means_ms'];q=r['probe_quality']
    totals={side:{k:sum(row[side+'_'+k] for row in r['host_rows']) for k in
        ('host_wall_ms','host_CPU_ms','host_nonrunning_ms','driver_wall_ms')} for side in ('isolated','pipeline')}
    lines=['当前YOLO七阶段匹配时间线：CPU单元2–4与VTA主机',
        '固定cfg_2787a77188dd5f9363d00b15；CPU0→VTA1→CPU2–4→VTA5–6→CPU7–8→VTA9–11→CPU12–13。',
        '本批3会话、97次完整网络推理；单帧资格＋同一阶段时钟程序的无探针／联合探针对照。8预热、16时间戳、24尾部。',
        '原等待、原核心映射、队列2、输入0；运行库和模块原哈希；未重启或重载FPGA，不修改评分或DP。',
        '全部八输出、实际线程映射、帧、boot和原始文件哈希验收通过；6060条线程记录和288次设备调用完整，无溢出。',
        '', '探针质量与性能口径：',
        '无探针%.6f FPS，探针%.6f FPS；CPU2–4服务%.3f/%.3f ms。绝对扰动%.3f ms，小于事先规定的隔离／流水差值20%%阈值%.3f ms，通过=%s。'%
        (q['clean_FPS'],q['observed_FPS_not_formal'],q['clean_CPU2_4_service_ms'],q['observed_CPU2_4_service_ms'],q['absolute_probe_service_delta_ms'],q['limit_ms'],q['passed']),
        '原正式程序先前为%.6f FPS；本批阶段时钟程序%.6f FPS，程序身份和时间块不同，不把差值称为优化收益。'%
        (r['original_executable_bridge_FPS'],r['current_phase_executable_clean_FPS']),
        '下述隔离取帧2–9、流水取帧8–23的阶段调用均值；不同于按网络完成窗口裁剪。单启动短窗口不证明稳态。',
        '', '1. CPU中段究竟增加了什么：',
        'CPU2–4 run墙钟 %.3f→%.3f ms，增%.3f ms；所有worker计算CPU量 %.3f→%.3f核ms（%+.2f%%）。'%
        (a['run_wall_ms'],b['run_wall_ms'],b['run_wall_ms']-a['run_wall_ms'],a['compute_CPU_ms'],b['compute_CPU_ms'],100*(b['compute_CPU_ms']/a['compute_CPU_ms']-1)),
        '调用线程完成等待及返回CPU量 %.3f→%.3f核ms；这是同一个调用线程上互不重叠的12次启动尾部之和，不能直接加到流水II。'%
        (a['caller_wait_and_return_CPU_ms'],b['caller_wait_and_return_CPU_ms']),
        '因此，当前前段慢化并非计算CPU工作增加约54ms。启动错位、线程失去CPU和完成等待占据了显著份额；非运行时间仍不能全部命名为调度排队。',
        '', '最清楚的短任务例子：']
    for i in (1,7,11):
        v=r['kernels'][i]
        lines.append('内核%d：流水所有worker计算共%.3f核ms，回调覆盖跨度%.3fms，worker进入错位%.3fms，调用线程尾部CPU%.3f核ms；最后完成者分布%s。'%
            (i+1,v['pipeline_compute_CPU_ms'],v['pipeline_callback_span_ms'],v['pipeline_entry_skew_ms'],v['pipeline_caller_tail_CPU_ms'],v['pipeline_last_slots']))
    lines += ['', '2. VTA主机段新增的墙钟是否只是工作变多：']
    for side,label in (('isolated','隔离'),('pipeline','流水')):
        v=totals[side];lines.append('%s三个VTA图运行、设备调用之外的合计：墙钟%.3fms，调用线程CPU%.3f核ms，非运行%.3fms。'%
            (label,v['host_wall_ms'],v['host_CPU_ms'],v['host_nonrunning_ms']))
    lines += [
        '这直接区分了“工作CPU时间”与“同样主机工作在持锁时迟迟无法完成的墙钟”。该非运行量也可能包含睡眠／阻塞，尚不能全归调度排队。',
        '设备调用仍单列；CPU与设备等待没有重复相加，也没有把VTA软件锁利用率称为纯计算利用率。',
        '', '3. 已有时序模型哪里仍不可靠：',
        '模型确实已经保留每次并行完成依赖，也确实生成了部分启动错位，不是简单假定所有worker同时运行。',
        '但具体长尾落点仍与实测不同：内核1、2、8的真实启动错位显著；模型又把一个约12ms长尾放在内核12，而本批该内核跨度约1ms。',
        '几个大卷积核的回调跨度也仍被低估。总耗时接近并不证明逐内核机制准确，误差存在抵消。',
        '这说明下一步不应直接把局部回放包装成完整DP修正，也不应根据这12个核的实测差额逐项加惩罚。',
        '', '4. 本轮结论与停止边界：',
        '本轮支持共享执行中的完成拖尾、调用线程等待CPU占用以及主机持锁时非运行，是隔离／流水差异的重要组成；尚未恢复Linux的真实任务就绪与调度次序。',
        '已有容量评分和按隔离工作量回放均有明确近似。继续调时间片使这一个案例贴近FPS，无法证明未测候选选点变好。',
        '暂停增加调度模拟器分支与微基准；保留此次误差定位为边界证据。下一阶段优先补齐公平的两网络完整流程共同对比，明确已有最强模型与输入份额修正的真实选点差异。',
        '若完整流程没有新收益，报告负向结果与输入成本，不以这个局部时序工具完成代替整体目标完成。']
    (REPORT/'FINDINGS_zh.txt').write_text('\n'.join(lines)+'\n')


def draw(result):
    os.environ.setdefault('MPLCONFIGDIR','/tmp/current_pipeline_timeline_mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':['DejaVu Sans','Droid Sans Fallback'],'axes.unicode_minus':False,'svg.fonttype':'none','pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(13.4,6.4))
    a,b=result['isolated_CPU']['means_ms'],result['pipeline_CPU']['means_ms']
    ax=axes[0]
    fields=[('run_wall_ms','run墙钟时间'),('compute_CPU_ms','worker计算核时'),('caller_wait_and_return_CPU_ms','调用线程尾部核时')]
    for key,shift,color in [('isolated',-.18,'#4B83B4'),('pipeline',.18,'#D59645')]:
        values=a if key=='isolated' else b
        bars=ax.bar([i+shift for i in range(3)],[values[k] for k,_ in fields],width=.33,color=color,label='隔离' if key=='isolated' else '流水')
        ax.bar_label(bars,labels=['%.2f'%values[k] for k,_ in fields],padding=3,fontsize=10)
    ax.set_xticks(range(3),[v for _,v in fields]);ax.set_ylim(0,420);ax.set_ylabel('ms／核ms（不同计量，不能相加）');ax.legend(frameon=False);ax.set_title('CPU2–4：计算量相近，完成过程变长',pad=13)
    ax=axes[1];kernels=result['kernels']
    ax.plot(range(1,13),[k['isolated_callback_span_ms'] for k in kernels],marker='o',color='#4B83B4',label='隔离回调跨度')
    ax.plot(range(1,13),[k['pipeline_callback_span_ms'] for k in kernels],marker='o',color='#D59645',label='流水回调跨度')
    ax.plot(range(1,13),[k['predicted_join_span_ms'] for k in kernels],marker='x',ls='--',color='#6A9A79',label='已有有限粒度模型完成跨度')
    ax.set_xticks(range(1,13));ax.set_xlabel('CPU2–4内部并行启动序号');ax.set_ylabel('ms');ax.set_ylim(0,110)
    ax.set_title('总量更接近，仍存在长尾落点错误',pad=13);ax.legend(frameon=False,fontsize=9)
    for ax in axes:
        ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
        for side in ('top','right'):ax.spines[side].set_visible(False)
    fig.suptitle('当前冻结YOLO方案：隔离profile与流水实测差异的直接证据',fontsize=17,y=.98)
    fig.subplots_adjust(left=.07,right=.98,bottom=.25,top=.79,wspace=.3)
    fig.text(.5,.12,'回调跨度／模型完成跨度仅作结构对应；非运行不等于已测调度排队，等待核时不直接叠加到II。',ha='center',fontsize=10)
    fig.text(.5,.055,'同一执行器的关闭／开启探针对照；本轮未改评分、DP或Top-20，不表示吞吐优化成功。',ha='center',fontsize=10,color='#566474')
    for ext in ('png','svg','pdf'):fig.savefig(REPORT/('current_pipeline_timeline.'+ext),dpi=180,facecolor='white')
    plt.close(fig)


if __name__=='__main__':main()
