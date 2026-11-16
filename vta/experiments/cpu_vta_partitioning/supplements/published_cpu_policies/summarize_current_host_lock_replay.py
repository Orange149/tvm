"""Known-case evaluation after both isolation-only replay lists are sealed."""
from pathlib import Path
import json
import os

import run_current_host_lock_replay as model
import run_current_output_replay as output
from cpu_vta_partitioning.common import read,write,sha256,verify_inventory
from scheduler_gap_audit.curves import csv_write

ROOT=model.ROOT
OUT=model.OUT/'analysis'
REPORT=ROOT.parents[4]/'paper/my_paper/analysis/20261112_current_host_lock_replay'


def exact(value):
    return value['numerator']/value['denominator'] if isinstance(value,dict) else value


def load_seal(folder):
    seal=read(folder/'REPLAY_FREEZE.json');rows=[]
    for leaf,h in seal['files'].items():
        p=folder/leaf
        if sha256(p)!=h:raise ValueError('replay changed before native comparison')
        rows.append((p,read(p)))
    return rows


def run():
    if OUT.exists() or REPORT.exists():raise FileExistsError('preserve this known-case report')
    model.check();output.check()
    model_rows=load_seal(model.OUT);output_rows=load_seal(output.OUT)
    if len(model_rows)!=8 or len(output_rows)!=4:raise ValueError('all declared variants must be evaluated')
    # Native values are first opened below, after every replay hash is checked.
    native_path=ROOT/'artifacts/window_tail_check_20261112_01/analysis/RESULTS.json'
    evidence=read(native_path);inputs=read(model.OUT/'ISOLATED_INPUTS.json')
    sources={str(native_path):sha256(native_path),str(Path(__file__)):sha256(__file__),
        str(model.OUT/'PLAN.json'):sha256(model.OUT/'PLAN.json'),
        str(output.OUT/'PLAN.json'):sha256(output.OUT/'PLAN.json')}
    native={}
    for c in evidence['rows']:
        if c['network']!='yolov3_tiny416':continue
        if sha256(c['receipt'])!=c['receipt_sha256']:raise ValueError('clean pipeline receipt changed')
        d=read(c['receipt']);verify_inventory(d['raw'],d['files'])
        if d['task']['configuration']!=inputs['cases'][c['configuration']]['configuration']:
            raise ValueError('current native layout differs from replay')
        native[c['configuration']]=d
        sources[c['receipt']]=c['receipt_sha256']
    rows=[]
    for folder,values in ((model.OUT,model_rows),(output.OUT,output_rows)):
        for path,entry in values:
            spec=entry['spec'];key=spec['configuration'];r=entry['result']['replay'];n=native[key]['summary']
            stage=entry['result']['actor_totals'];hostindices=[i for i,s in enumerate(inputs['cases'][key]['configuration']['stages']) if s['device']=='vta']
            row=dict(configuration=key,CPU=spec.get('CPU','finite'),completion=spec.get('completion','independent'),
                output_ring=spec.get('ring',0),predicted_II_ms=r['interval_ns']/1e6,
                clean_native_II_ms=n['II_ms'],clean_native_FPS=1000/n['II_ms'],
                static_II_ms=inputs['cases'][key]['static_score']['score_ns']/1e6,
                interval_absolute_error_ms=abs(r['interval_ns']/1e6-n['II_ms']),
                predicted_stage_ms=[v/1e6 for v in r['stage_service_ns']],native_stage_ms=n['stage_service_ms'],
                predicted_VTA_service_sum_ms=sum(r['stage_service_ns'][i] for i in hostindices)/1e6,
                collector_CPU_ms_per_scored_frame=sum(exact(t['CPU']) for t in stage if t['role']=='_collector')/16/1e6,
                file=str(path),sha256=sha256(path),known_case_development_only=True)
            rows.append(row);sources[str(path)]=sha256(path)
    comparisons=[]
    for key,case in inputs['cases'].items():
        chosen=next(r for r in rows if r['configuration']==key and r['CPU']=='finite' and r['completion']=='independent' and r['output_ring']==0)
        baseline=chosen['static_II_ms'];actual=chosen['clean_native_II_ms']
        fluid=next(r for r in rows if r['configuration']==key and r['CPU']=='fluid' and r['completion']=='independent')
        variants=[r for r in rows if r['configuration']==key and r['output_ring']]
        comparisons.append(dict(configuration=key,original_static_II_ms=baseline,actual_II_ms=actual,
            primary_replay_II_ms=chosen['predicted_II_ms'],original_absolute_error_ms=abs(baseline-actual),
            primary_absolute_error_ms=chosen['interval_absolute_error_ms'],
            fluid_independent_II_ms=fluid['predicted_II_ms'],
            finite_fluid_difference_ms=fluid['predicted_II_ms']-chosen['predicted_II_ms'],
            output_ring1_II_ms=next(r['predicted_II_ms'] for r in variants if r['output_ring']==1),
            output_ring4_II_ms=next(r['predicted_II_ms'] for r in variants if r['output_ring']==4),
            new_Top20_or_selection_gain_proven=False))
    # All methods select from just these two known representatives here.
    selection=[]
    for cpu,completion,ring in [('finite','independent',0),('fluid','independent',0),('finite','independent',1),('finite','independent',4)]:
        group=sorted([r for r in rows if (r['CPU'],r['completion'],r['output_ring'])==(cpu,completion,ring)],
                     key=lambda r:(r['predicted_II_ms'],r['configuration']))
        selection.append(dict(CPU=cpu,completion=completion,output_ring=ring,
            known_two_layout_first=group[0]['configuration'],
            selected_existing_clean_FPS=group[0]['clean_native_FPS'],not_a_global_Top20=True))
    cost=read(output.COST)['costs']['yolov3_tiny416_ring1']
    result=dict(comparisons=comparisons,all_variants=rows,known_pool_choices=selection,sources=sources,
        new_isolated_native_sessions=15,new_component_calls=105,new_DNN_inferences=0,
        offline_replays=12,missing_two_CPU_profiles_and_three_VTA_splits_now_measured=True,
        independent_output_CPU_ms=cost['CPU_ns']/1e6,
        no_new_formula_or_DP_or_Top20=True,not_an_unseen_validation=True,
        cannot_choose_scheduler_variant_by_observed_error=True,whole_goal_complete=False)
    OUT.mkdir();REPORT.mkdir(parents=True)
    write(OUT/'RESULTS.json',result);csv_write(REPORT/'prediction_comparison.csv',comparisons)
    csv_write(REPORT/'all_variants.csv',rows);report(result);draw(result)
    write(OUT/'MANIFEST.json',dict(files={str(p):sha256(p) for p in [OUT/'RESULTS.json',*REPORT.iterdir()] if p.is_file()},sources=sources))
    print(json.dumps(dict(comparisons=comparisons,choices=selection,report=str(REPORT/'FINDINGS_zh.txt')),ensure_ascii=False))


def report(result):
    profiles=read(model.NEW)['profiles']
    lines=['当前YOLO隔离时序补齐与VTA持锁回放（开发回放，不是新Top20）',
        '上轮已验收的4项尾部验证不重做；本轮只补5个组件，每项资格1次、无探针2+8次、探针2+8次。',
        '已完成15会话、105次组件调用，新增完整网络推理为0；未重启、未重编运行库／阶段模块。',
        '复用已冻结的阶段时钟执行器与原等待观察库，输出、线程、帧、原始哈希及boot均复核通过。',
        '', '1. 补齐了什么，而不是又换一种评分：',
        '新增CPU2–4三线程的12次并行启动结构，以及CPU12–13三线程的5次并行启动结构。每个worker核时、调用线程串行工作与完成汇合分别保留。',
        '三个当前VTA子图的设备调用之前、之间、之后的主机CPU工作均已按同一TID时钟实测，不再依靠总量任意分摊。',
        '此前按份额换算的某些单线程预算超过阶段墙钟，说明份额近似不能冒充精确线程时序。本轮没有改写旧成本或旧结果。',
        '', 'VTA子图 / 设备调用前、之间、后的隔离主机CPU核毫秒：']
    for k,v in profiles.items():
        if v['device']=='vta':lines.append(k+' / '+', '.join('%.3f'%(exact(x)/1e6) for x in v['stages']['host_serial_CPU']))
    lines += ['', '2. 复用什么模型：',
        '直接复用既有CompletionFinite、CompletionFluid与IsolatedOutputFinite；没有新增调度算法、拟合慢化系数或挑选最接近FPS的参数。',
        '两个固定布局各测有限粒度／理想均分，以及聚合设备等待／独立设备完成，共8次离线回放；随后另冻结4次已有独立输出成本对照。',
        '设备完成随墙钟推进；主机CPU只有获得CPU服务才推进；二者按实测顺序位于同一VTA软件资源租约内。完成等待由依赖产生，隔离等待核时不再重复加作计算。',
        '沿用48帧（8预热、16时间戳、24尾部），同一队列2与线程映射。输出处理以独立CPU执行单元竞争，不直接加到II上。',
        '未知项仍包括Linux真实调度历史、生产和其他控制工作、单线程非运行区间的具体位置、干扰下CPU/DDR需求变化。',
        '', '3. 当前结果（毫秒，越接近实际越好）：',
        '配置 / 原静态预测 / 事先指定有限粒度＋独立设备 / 理想均分＋独立设备 / 无探针实际']
    for r in result['comparisons']:
        lines.append('%s / %.3f / %.3f / %.3f / %.3f'%(r['configuration'],r['original_static_II_ms'],
            r['primary_replay_II_ms'],r['fluid_independent_II_ms'],r['actual_II_ms']))
    lines += ['',
        '事先指定的有限粒度模型减少了这两个已知案例的绝对误差，但仍偏乐观约10–11ms；理想均分则偏保守约16–18ms。不能根据已知FPS选择更好看的调度近似。',
        '在这两个代表组成的小池中，新旧方法仍选同一个静态第一名；没有证明新的选点收益，更不是完整域Top20结果。',
        '', '4. 已有输出处理能否解释剩余II：',
        '独立profile的YOLO输出序列化约%.3f核毫秒；把它作为收集线程工作加入后，两种缓存口径下的完成间隔均未改变。'%result['independent_output_CPU_ms'],
        '这会改变前段及个别阶段时间，但本次没有改变决定完成节拍的CPU中段瓶颈。不能用“还差约10ms，所以加9ms输出开销”解释。',
        '本轮不再为输出处理新增微基准，也不把它当成已经解决剩余误差。',
        '', '5. 下一步与接回完整流程的边界：',
        '当前的输入缺口已缩小：最关键的两项多线程结构与三个VTA主机段已有真实隔离输入。剩余重点是这些相同工作在真实流水中何时获得CPU、在哪次并行启动拖尾。',
        '优先在当前冻结七阶段切图获取一次匹配线程时间线，使用已有无探针对照检查扰动；不再拿另一切图CPU0–2四线程的轨迹替代当前CPU2–4三线程。',
        '只有真实执行次序支持某种近似，并能在未测配置中改善选点，才考虑候选重排及全域所需输入成本。',
        '先保留完整合法域与原静态模型，不把此处两个代表的排序包装为完整DP，不自动启动全域profile或新Top20板测。',
        '', '文献依据与区别：',
        'GCAPS第4节图2区分主机CPU操作Gm与无需CPU干预的设备执行Ge：',
        'https://drops.dagstuhl.de/storage/00lipics/lipics-vol298-ecrts2024/LIPIcs.ECRTS.2024.14/LIPIcs.ECRTS.2024.14.pdf',
        '本项目复用这种资源区分和已有依赖回放；Linux时间共享、VTA软件锁和TVM多线程并非其优先级GPU任务模型，不能称为完整复现或严格响应时间上界。']
    (REPORT/'FINDINGS_zh.txt').write_text('\n'.join(lines)+'\n')


def draw(result):
    os.environ.setdefault('MPLCONFIGDIR','/tmp/current_host_lock_mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    plt.rcParams.update({'font.family':['DejaVu Sans','Droid Sans Fallback'],
        'axes.unicode_minus':False,'svg.fonttype':'none','pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(12.8,5.9))
    fields=[('original_static_II_ms','原静态模型','#8895A4'),('primary_replay_II_ms','匹配隔离时序＋有限粒度','#4B83B4'),
        ('fluid_independent_II_ms','同一工作量＋理想均分','#DAA14A'),('actual_II_ms','无探针实际','#529879')]
    for ax,row,title in zip(axes,result['comparisons'],['当前静态第一名','原短窗口Top-20优胜配置']):
        bars=ax.bar(range(4),[row[k] for k,_,_ in fields],color=[c for _,_,c in fields],width=.65)
        ax.bar_label(bars,labels=['%.2f'%row[k] for k,_,_ in fields],padding=4,fontsize=11)
        ax.set_xticks(range(4),['原静态','有限粒度','理想均分','实际']);ax.set_ylim(0,240)
        ax.set_ylabel('平均完成间隔（ms）');ax.set_title(title,pad=12,fontsize=14)
        ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
        for side in ('top','right'):ax.spines[side].set_visible(False)
    fig.suptitle('补齐当前子图隔离时序后：误差缩小，但调度近似仍影响很大',fontsize=16,y=.98)
    fig.subplots_adjust(left=.07,right=.98,bottom=.25,top=.78,wspace=.28)
    fig.text(.5,.12,'加入已有约9.13核毫秒的独立输出处理成本后，两项有限粒度预测II均未改变。',ha='center',fontsize=11)
    fig.text(.5,.055,'两个已知方案的开发回放；未改DP、未生成新Top-20，不能据此宣称选点或吞吐提升。',ha='center',fontsize=10,color='#566474')
    for ext in ('png','svg','pdf'):fig.savefig(REPORT/('current_host_lock_comparison.'+ext),dpi=180,facecolor='white')
    plt.close(fig)


if __name__=='__main__':run()
