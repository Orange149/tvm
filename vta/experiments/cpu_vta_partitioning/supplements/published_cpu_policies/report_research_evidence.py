"""Generate a Chinese evidence synthesis without changing predictions or data."""
from collections import Counter
from pathlib import Path
import csv
import json
import shutil
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from cpu_vta_partitioning.common import now,read,sha256,write
from research_evidence_audit import ROOT,OUTPUT,CROSS

REPORT=ROOT.parents[4]/'paper/my_paper/analysis/20261111_research_evidence'


def generate():
    r=read(OUTPUT/'AUDIT.json')
    if r['audit_source_sha256']!=sha256(ROOT/'research_evidence_audit.py'):
        raise ValueError('re-run audit after implementation change')
    lines=['CPU–VTA共享执行：方法、双网络证据及最新修正结论','',
        '当前结论：原资源模型和完整DP流程仍然可复现；调用级回放曾在共同候选池中改善选点，但没有准确解释共享执行。最新有限CPU服务修正改善了部分完成次序，却没有改善两个网络的新布局选点，因此不接入完整DP，不替换原评分。',
        '本文件是分批次证据汇总。下面分别列出旧共同Top-20、本次开发诊断和新布局验证；不同boot的FPS不相减，不合并为一项增益。没有宣布共享预测问题已经解决。','',
        '一、原方法究竟是什么',
        'ResNet18有21单元、172模块、775项组件成本；YOLOv3-tiny416有14单元、159模块、714项组件成本。CPU从合法子图中分别选择1–4线程与核心集合，逐线程固定；最多三个VTA岛共用一个物理加速器。两个网络使用相同公式，各自真实成本。',
        'CPU逐核需求：D_c = Σ_{i,j:m_ij=c} U_i w_ij + Σ_{i:caller_i=c} B_i。U是CPU计算核时，w是已有会话线程份额；CPU交接核时B只归调用线程。极短签名沿用已记录的均分回退。',
        'VTA主机需求H是各VTA阶段运行CPU核时与输入/输出CPU核时之和。设备等待的整段墙钟不充作CPU核时。',
        '对15个非空核心子集S计算 ceil((Σ_{c∈S}D_c + H·1[M⊆S])/|S|)，M为主机允许集合。共享主机采用可迁移容量近似，不代表真实线程会即时选择最空闲的核心。',
        '原主评分取CPU最大阶段服务、所有VTA阶段累计服务、DDR代理、核心子集容量、真实同组串行累计和线程组服务项的最大值。线程组项为 max_c Σ_{i:c∈A_i}s_i；开放并发时它是排序启发项，不是严格Linux响应时间下界。主机集中约束关闭。',
        'Tarnawski平台适配采用单线程物理CPU映射、同核阶段服务累计、单VTA累计及共同DDR项。融合区间成本、单VTA复用和Top-K是平台适配内容，不称原作者完整系统复现。',
        '公式—代码—成本来源表：../20261111_static_revision对应原项目artifacts/static_revision_01/model_formula_inventory.csv；源码staged_ablation/model.py、cpu_policies/open_resources/costs.py及resource_envelope/model.py。','',
        '二、重新核验了什么',
        '本次重新读取源码及成本，回放完整域检查点的输入指纹、Python/C++来源与枚举凭据，核对全局Top-20及标量评分。没有把缓存检查称为重新冷枚举。']
    for n,d in r['complete_searches'].items():
        lines.append(f"{n}：主模型{d['main']['count']:,}项资源组合；Tarnawski适配{d['tarnawski']['count']:,}项映射；真实组件记录{d['real_component_records']}项。")
    c=r['common_comparison']
    lines += [f"旧共同池{c['native_configurations']}个不同配置的资格与性能凭据，以及{c['long_sessions']}个较长窗口结果，合计{len(c['source_receipts'])}份接受凭据均重新核验原始文件、输出、线程位置与窗口。", 
        '这验证已有结果可追溯，不增加统计重复，也不是本次重新板测。旧共同池boot：'+c['boot_id']+'。','',
        '三、已有双网络共同Top-20：预算必须相同',
        '短窗口为8预热、16完成时间戳、4尾帧，按15个间隔计算FPS。时序方法只对完整DP生成的固定候选并集重排，不是全域时序目标的最优Top-20。']
    labels={'current':'原资源容量静态模型','tarnawski':'Tarnawski单线程平台适配',
            'fixed_family':'固定不重叠计算核组家族','temporal_replace':'隔离调用级时序池内重排'}
    for network in ('resnet18','yolov3_tiny416'):
        lines+=['',network+'：方法 | 预算1 FPS | 预算5 FPS | 预算20 FPS']
        for key,label in labels.items():
            values={x['budget']:x for x in c['budgets'] if x['network']==network and x['method']==key}
            lines.append(label+' | '+' | '.join(f"{values[k]['FPS']:.3f}" for k in (1,5,20)))
    lines += ['','较长窗口仅复核各方法冻结的短窗口预算20优胜配置；8/64/32，按63个间隔计算。不能将它写成预算1、5也做过较长复核。']
    final=read(CROSS/'FINAL_LONG_RESULTS.json')
    for network in ('resnet18','yolov3_tiny416'):
        methods=final['networks'][network]
        lines.append(network+'：方法 | 冻结选点较长窗口FPS | 配置')
        for key,label in labels.items():
            item=methods[key]
            lines.append(f"{label} | {item['FPS']:.3f} | {item['configuration_id']}")
    lines += ['两网络时序Top-20的优胜配置都与固定核组家族优胜相同。这支持固定组是有价值的候选类型，却没有证明额外调用级采集优于直接测固定家族Top-20。固定家族限制了执行策略，必须单独报告其贡献。',
        '原先的主机CPU需求与CPU细化消融保留在../20261109_staged_ablation/。该批预算20：粗CPU/不计主机11.432，粗CPU/计主机12.084，细CPU/不计主机10.485，细CPU/计主机13.417 FPS。细化与主机贡献存在交互，不能把不同条件的百分比相加。该批不与上表混算收益。','',
        '四、为什么ResNet18和YOLO的结果不能简单互推',
        '已重新核对全部166个CPU编译子图的1–4线程真实曲线，共664行；全部保留，包括非单调点，没有删候选。表格同时给出服务、核时增长、边际收益与来源。']
    for network in ('resnet18','yolov3_tiny416'):
        table=[x for x in r['real_thread_curves']['rows'] if x['network']==network]
        programs=sorted({x['program'] for x in table})
        nonmonotonic=sum(x['nonmonotonic_step'] for x in table)
        lines.append(f'{network}：{len(programs)}个CPU子图，线程数增加却更慢的相邻点有{nonmonotonic}项；单次微小差异不能作为裁剪依据。')
        for name in ('cpu_00_02','cpu_20_20' if network=='resnet18' else 'cpu_13_13'):
            group=sorted([x for x in table if x['program']==name],key=lambda x:x['threads'])
            lines.append(name+'：'+ '；'.join(f"{x['threads']}线程服务{x['service_ms']:.3f}ms、计算核时{x['compute_CPU_ms']:.3f}核ms" for x in group))
    lines += ['相同四线程前段CPU0–2：ResNet为13次并行启动，worker计算173.999核ms，最长内核20.190ms；YOLO为12次启动、447.184核ms、最长内核44.329ms。YOLO不是启动更多，而是本例计算任务更长、串行及主机工作更多。',
        '三线程前段也不是均分：ResNet三个逻辑槽约64.79/64.64/43.91核ms；YOLO约170.62/169.97/103.25核ms。具体形状、TVM调度和编译实现共同决定线程收益，不按网络名称制定规则。',
        'YOLO存在双检测头、多输入frontier及八输出；结果统计代码段约9.5核ms/帧，ResNet约0.7。该值不是纯哈希开销。旧诊断补控制工作只让前段预测变化约1.11ms，不能解释主要差距；也不能直接从II扣除这项CPU时间。',
        '这些是负载特征及关联证据。输入尺寸、切图和编译内核没有同时匹配，不能把差异全部归因于DNN架构。','',
        '五、更正后的真实轨迹与模型错在哪里',
        '旧环形缓冲截断批次不再用于全窗口归因。修复批次保留62,743个切换事件，仅对完整覆盖的帧12–23、144次启动、576个回调作定量诊断。探针对前段约9.96ms扰动单列；正式FPS仍取关闭探针的会话。',
        'YOLO四线程重叠：新隔离前段约139.91ms，原理想共享预测约182.07ms，无探针流水约220.26ms。这里有真实模型缺口，不能因整体II偶然接近就宣布解释正确。',
        '在完全相同144次启动上，调用线程最后完成比例：实际6.25%；原逐事件理想共享71.53%；有限服务主结果25.00%。有限服务改善了方向，仍未恢复真实完成次序。',
        '模型首次服务错位逐内核均值之和：原理想共享0，有限服务主结果约10.96ms；实际计算体入口错位约18.55ms。两个时钟包络不同，真实入口还含唤醒、分派和探针；这些和也不能再作为惩罚加到完成间隔。',
        '有限服务主结果生成调用线程等待约56.60核ms/帧；真实尾部约66.44核ms，后者还含返回及观测工作。接近一个总量不代表资源位置、次序及其他配置已经预测准确。','',
        '六、最新修正做了什么，验证结果如何',
        '有限服务修正保留隔离工作量、fork/join、深度2队列与VTA互斥，每核一次只执行一个线程，并保留虚拟运行历史。参数来自独立零DNN忙负载：2/3/4/5/6线程竞争一核的连续服务中位数约12/8/8/4/4ms，不称内核真实时间片参数。混合短任务平台检查通过，不能替代DNN验证。',
        '开发回放中YOLO前段约191.48→208.91ms，更接近220.26ms；主机严格不迁移的敏感性版本约219.19ms，但会让另一分组方案明显偏慢，不能事后选最接近的版本。',
        '之后分别冻结两个网络各17项局部资源代表，模型选点并集各3项，才开始共同实测。共12会话、174次完整推理，输入0，8/16/4，同一新boot。固定切图，只验证新线程/核心布局，不冒充全域Top-20或新切图盲测。']
    for row in r['prospective_results']['methods']:
        lines.append(f"{row['network']} | {row['label']} | {row['fps']:.4f} FPS | {row['ii_ms']:.3f}ms | {row['selected_id']}")
    lines += ['有限服务相对原资源静态评分：ResNet选点下降3.64%；YOLO选择相同配置，差异为0。故不支持把这一修正接入完整DP。不能只展示它相对另一较差回放的正增益。',
        '七、额外核验：等待中的Yield不是不占CPU',
        '对实际部署libtvm_runtime反汇编确认Yield跳转sched_yield。Linux 5.4公平类会在公平性条件下尝试跳过让出者，不等同于阻塞。新零DNN自测仅一进程、三个有限对照、不访问FPGA或系统参数。']
    for x in r['zero_DNN_wait_result']['cases']:
        lines.append(f"{x['case']}：共同窗口等待线程占CPU {100*x['waiter_CPU_fraction']:.3f}%；目标80核ms任务墙钟{x['target_wall_ns']/1e6:.3f}ms。")
    lines += ['同核忙等51.33%、Yield等待51.12%，仅约0.21个百分点差异，没有重复或显著性证据；不据此引入低优先级等待规则。单独占核的Yield仍接近满核，不能把等待线程改为零CPU需求。原程序及评分未修改。','',
        '八、文献借鉴到哪里，不能照搬什么',
        'Tarnawski第5节以最大设备负载构造稳态划分目标；本项目适配其物理单核和设备累计语义。原文没有给当前TVM重叠线程组的Linux共享响应预测。',
        'https://proceedings.neurips.cc/paper/2020/file/b14680dec683e744ada1f2fe08614086-Paper.pdf',
        'Pipe-it把计算并行限制在分配的核组，利用层预测平衡流水。这里借鉴固定计算组和资源预算；非相邻子图复用组、同组互斥及VTA主机共享属于本项目适配，不宣称复现了其全部系统。',
        'https://arxiv.org/pdf/1903.05898',
        'GCAPS图2及第5/6节分开加速器相关CPU操作、纯设备执行及忙等/挂起。在本项目可借鉴资源分解，不能把其固定优先级响应时间公式直接作为CFS流水节拍。',
        'https://drops.dagstuhl.de/storage/00lipics/lipics-vol298-ecrts2024/LIPIcs.ECRTS.2024.14/LIPIcs.ECRTS.2024.14.pdf',
        'Daydream第4节用细粒度任务、依赖和线程进度回放；这里复用表示方法，不声称其已解决多个固定CPU组重叠争用。',
        'https://www.cs.toronto.edu/~pekhimenko/Papers/Daydream-ATC_20.pdf',
        'Packrat附录A观察隔离/并发差距，验证干扰项不改变其配置选择后未在优化器建模。本项目不能照搬“排序不变”：必须做独立选点检查；此次有限服务结果正说明预测更接近不保证更好选点。',
        'https://proceedings.mlr.press/v267/bhardwaj25a.html；本地用户PDF paper/VTA/Packrat.pdf。',
        'Linux上游源码是语义参考，供应商内核参数与完整调度状态未恢复。此次有限近似不是完整CFS复现。',
        'https://raw.githubusercontent.com/torvalds/linux/v5.4/kernel/sched/fair.c','',
        '九、收益、成本及未完成项',
        '主机CPU需求与线程组细化的选点贡献见共同消融；固定家族贡献与时序重排分别列出；有限服务修正本次没有额外选点收益；本次执行机制未改变，不能声称运行时加速。',
        '历史共同池的调用级补采84种签名，约22.7分钟；候选时序评分约34.23秒。共同短窗口158项、4,582次完整推理，测量/回收约54分钟；较长窗口9项、936次推理另计。它们不是本次新增板测。',
        '本次有限服务平台标定一会话、零DNN；局部候选评分约27.12秒；新布局实测12会话、174次推理，测量/回收约225.63秒。Yield审计一会话、零DNN。所有旧采集成本均保留原boot与来源，不把缓存检查耗时冒充冷搜索。',
        '没有修改阶段模块、运行库、原正式流水执行器、原完整DP域；最新代码是独立模型、验证与审计入口。没有重启开发板。',
        '尚未获得：可可靠迁移到未测配置、同时改善两个网络选点的共享调度修正。主机实际迁移、worker释放/唤醒顺序、有限服务与历史状态之间的耦合仍未充分建模；缓存等因素未完全排除。',
        '因此当前不能宣布隔离profile到共享流水的预测问题已解决。也不应继续无依据扩大Top-20或全域profile。下一项修改必须明确修复哪个已证实的语义缺口，并在新配置上验证；本报告不把负向结果改写为成功。','',
        '十、可复现入口（均不会重测已验收任务）',
        'python run_yield_semantics.py validate',
        'python audit_finite_service_alignment.py',
        'python research_evidence_audit.py',
        'python report_research_evidence.py',
        'python run_common_top20.py validate',
        'python run_pool_temporal.py validate',
        '本报告配套AUDIT.json、same_cohort_budgets.csv、thread_configuration_table.csv以及逐内核对照；AUDIT包含每份原始接受凭据的文件哈希。',
        '不提供重新执行旧boot批次的命令。']
    OUTPUT.mkdir(parents=True,exist_ok=True)
    (OUTPUT/'FINDINGS_zh.txt').write_text('\n'.join(lines)+'\n')
    plot(r)
    REPORT.mkdir(parents=True,exist_ok=True)
    for p in [OUTPUT/'AUDIT.json',OUTPUT/'same_cohort_budgets.csv',OUTPUT/'thread_configuration_table.csv',OUTPUT/'FINDINGS_zh.txt',
              ROOT/'artifacts/yield_semantics_01/alignment/model_completion_summary.csv',
              ROOT/'artifacts/yield_semantics_01/alignment/matched_kernel_alignment.csv',
              OUTPUT/'same_cohort_budget_comparison.png',OUTPUT/'same_cohort_budget_comparison.pdf',
              OUTPUT/'prospective_selection.png',OUTPUT/'thread_curve_examples.png']:
        shutil.copy2(p,REPORT/p.name)
    write(OUTPUT/'DELIVERY.json',dict(files={str(p):sha256(p) for p in REPORT.iterdir() if p.is_file()},
        source_sha256=sha256(__file__),audit_sha256=sha256(OUTPUT/'AUDIT.json'),
        main_model_replacement_supported=False,full_goal_completion_claimed=False,recorded_at=now()),replace=True)
    return dict(report=str(REPORT/'FINDINGS_zh.txt'),goal_complete=False)


def plot(r):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    networks=('resnet18','yolov3_tiny416')
    methods=('current','tarnawski','fixed_family','temporal_replace')
    labels=('Original static','Tarnawski adaptation','Fixed compute groups','Temporal pool rerank')
    colors=('#4c78a8','#a7a7a7','#72b7b2','#f58518')
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for i,network in enumerate(networks):
        for j,budget in enumerate((1,5,20)):
            ax=axes[i,j]
            ys=[next(x['FPS'] for x in r['common_comparison']['budgets']
                 if x['network']==network and x['method']==method and x['budget']==budget) for method in methods]
            bars=ax.bar(range(4),ys,color=colors)
            ax.set_title(f'{network} / budget {budget}')
            ax.set_xticks(range(4),labels,rotation=25,ha='right',fontsize=8)
            ax.set_ylabel('FPS');ax.set_ylim(0,max(ys)*1.2)
            for bar,y in zip(bars,ys):
                ax.text(bar.get_x()+bar.get_width()/2,y,f'{y:.2f}',ha='center',va='bottom',fontsize=8)
    fig.suptitle('Historical common cohort; temporal ranking is pool-local; short window')
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(OUTPUT/('same_cohort_budget_comparison.'+ext),dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    methods=('static_fine_host','fluid_instant','fluid_wake_boundary','finite_wake_boundary')
    labels=('Original static','Original fluid','Fluid with history','Finite service')
    for ax,network in zip(axes,networks):
        ys=[next(x['fps'] for x in r['prospective_results']['methods']
             if x['network']==network and x['method']==method) for method in methods]
        bars=ax.bar(range(4),ys,color=colors)
        ax.set_xticks(range(4),labels,rotation=25,ha='right',fontsize=8)
        ax.set_title(network+' / new layouts, top-1');ax.set_ylabel('FPS');ax.set_ylim(0,max(ys)*1.2)
        for bar,y in zip(bars,ys):
            ax.text(bar.get_x()+bar.get_width()/2,y,f'{y:.3f}',ha='center',va='bottom',fontsize=8)
    fig.suptitle('Separate prospective cohort; two fixed topologies, no Top-20 claim')
    fig.tight_layout();fig.savefig(OUTPUT/'prospective_selection.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    for i,network in enumerate(networks):
        programs=('cpu_00_02','cpu_20_20' if network=='resnet18' else 'cpu_13_13')
        for ax,program in zip(axes[i],programs):
            rows=sorted([x for x in r['real_thread_curves']['rows'] if x['network']==network and x['program']==program],
                        key=lambda x:x['threads'])
            ax.plot([x['threads'] for x in rows],[x['service_ms'] for x in rows],'o-',label='Isolated service (ms)')
            ax.plot([x['threads'] for x in rows],[x['compute_CPU_ms'] for x in rows],'s--',label='Compute CPU work (core-ms)')
            ax.set_title(network+' / '+program);ax.set_xticks((1,2,3,4));ax.set_xlabel('Configured threads')
            ax.set_ylabel('ms / core-ms');ax.set_ylim(bottom=0);ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle('Real isolated costs; curves do not determine shared-pipeline performance')
    fig.tight_layout();fig.savefig(OUTPUT/'thread_curve_examples.png',dpi=180);plt.close(fig)


if __name__=='__main__':print(json.dumps(generate(),ensure_ascii=False,indent=2))
