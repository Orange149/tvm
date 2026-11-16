"""Evidence report after all known-case predictions have been frozen."""
from pathlib import Path
import json,shutil,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from cpu_vta_partitioning.common import now,read,sha256,write
from scheduler_gap_audit.curves import csv_write
import run_device_completion_replay as study

ROOT=study.ROOT
REPORT=ROOT.parents[4]/'paper/my_paper/analysis/20261111_device_completion'


def describe(cfg):
    return ' → '.join(('CPU' if s['device']=='cpu' else 'VTA')+
        f"{s['begin']}–{s['end']-1}"+(f"（{s['threads']}线程、核心{s['mask']}）" if s['device']=='cpu' else '')
        for s in cfg['stages'])


def report():
    study.assert_plan(study.OUTPUT);seal=read(study.OUTPUT/'REPLAY_FREEZE.json')
    for leaf,h in seal['files'].items():
        if sha256(study.OUTPUT/leaf)!=h:raise ValueError('completion study seal invalid')
    profile=read(study.PROFILE);result=read(study.OUTPUT/'analysis/RESULTS.json')
    inputs=read(study.OUTPUT/'INPUTS.json');REPORT.mkdir(parents=True,exist_ok=True)
    configs={key:item['configuration'] for key,item in inputs['cases'].items()}
    decisions=[]
    for network in ('resnet18','yolov3_tiny416'):
        candidates=[c for c in result['choices'] if c['network']==network and c['CPU']=='finite' and c['point']=='midpoint']
        base=next(c for c in candidates if c['completion']=='aggregate')
        new=next(c for c in candidates if c['completion']=='independent')
        decisions.append(dict(network=network,baseline_selected_FPS=base['selected_native_FPS'],
            independent_selected_FPS=new['selected_native_FPS'],
            selected_FPS_change_percent=100*(new['selected_native_FPS']/base['selected_native_FPS']-1),
            baseline_id=base['selected_id'],independent_id=new['selected_id'],
            candidate_pool_size=3,known_development_only=True))
    table=[]
    for row in result['rows']:
        table.append({**row,'configuration':describe(configs[row['network']+'/'+row['configuration_id']])})
    csv_write(REPORT/'predictions.csv',table);csv_write(REPORT/'selection.csv',decisions)
    csv_write(REPORT/'isolated_completion_brackets.csv',profile['device_table'])
    lines=['设备独立完成与主机轮询：受控开发验证','',
        '问题：原回放将驱动调用内CPU工作和不运行时长串成固定序列。真实设备在主机未获CPU时仍可运行，主机读到完成即退出，不能要求它总做完隔离时的查询次数。',
        '本轮不是修改CPU调度参数，也不是根据流水FPS拟合慢化比例。',
        '方法依据：GCAPS ECRTS2024第4节区分GPU段内主机CPU操作与无需CPU介入的设备执行。这里借用资源分离表示，未复现其优先级调度器。',
        '原文：https://drops.dagstuhl.de/storage/00lipics/lipics-vol298-ecrts2024/LIPIcs.ECRTS.2024.14/LIPIcs.ECRTS.2024.14.pdf','',
        '实测：5个VTA子图，15个原生会话、105次VTA组件调用、0次完整DNN。输入0，每项资格1次，探针关闭/开启各2预热+8计入。不重启，当前boot '+profile['native_state']['boot_id']+'。',
        '原执行器、运行库、模块和位流身份保持；仅可选适配库转发原有寄存器读写，不增加MMIO读取、休眠或轮询。队列计数明确关闭，零值不是无排队证据。',
        '探针在原查询前后读单调时钟，保留最后一次未完成和第一次完成的读取包络。启动也保留fetch启动至全部engine启动的写入包络。',
        '设备时长下界=max(0,最后未完成读取前时刻−启动写包络末时刻)，上界=首次完成读取后时刻−启动写包络首时刻。它是隔离条件下的观测区间，不能宣称共享执行的严格上下界。','']
    for row in profile['component_table']:
        lines.append(f"{row['network']} {row['program']}：设备完成区间合计{row['total_lower_ms']:.3f}–{row['total_upper_ms']:.3f}ms；干净run {row['clean_run_ms']:.3f}ms/{row['clean_run_CPU_ms']:.3f}核ms；探针CPU {row['probe_CPU_change_percent']:+.2f}%、墙钟{row['probe_wall_change_percent']:+.2f}%，质量条件通过={row['probe_quality_passed']}。")
    lines+=['','修正定义：',
        '每次设备调用先执行提交CPU工作，然后设备按隔离完成区间的中点独立推进；主机按同一隔离记录的轮询CPU与非运行间隔执行，首次查询时刻不早于设备完成即退出。设备推进不占CPU，也不随主机被抢占而暂停。',
        '主机CPU需求=提交CPU+实际模拟查询次数×每次查询CPU；次数是回放结果，不强制等于隔离次数。设备调用外的主机CPU工作保持实测，未删除。',
        '主比较固定使用原有限服务CPU调度近似和完成区间中点。区间两端只作敏感性；连续均分CPU模型是预定次要对照，不根据网络结果挑选。',
        '提交CPU仍用提交MMIO墙钟封顶代理，轮询CPU按均值分配，休眠使用观测非运行时长代理；未建模访存争用对设备自身耗时的变化、全部控制CPU或真实CFS迁移。',
        '评分仍为max(原静态容量评分,回放完成间隔)。候选域、正式执行机制均未改变。','',
        '验证范围：两个网络各3项已有配置，共36次离线回放；所有预测与源码先冻结再由分析入口读取既有流水结果。这些配置的结果已知，因此只属于开发回放，不是新的独立收益验证。','']
    for d in decisions:
        lines.append(f"{d['network']}：固定驱动序列选中{d['baseline_selected_FPS']:.6f}FPS，独立设备完成选中{d['independent_selected_FPS']:.6f}FPS，变化{d['selected_FPS_change_percent']:+.3f}%。")
        lines.append('原选择：'+describe(configs[d['network']+'/'+d['baseline_id']]))
        lines.append('新选择：'+describe(configs[d['network']+'/'+d['independent_id']]))
    supported=all(d['selected_FPS_change_percent']>=0 for d in decisions) and any(d['selected_FPS_change_percent']>0 for d in decisions)
    lines+=['','判断：'+('达到预设开发门槛，仍需冻结未测配置再验证，不能直接宣称改善。' if supported else '未达到“两网络静态第一名均不更差、至少一个改善”的预设开发门槛；不接入完整DP，也不自动新增流水测量。'),
        '它修正了设备/主机推进语义，但不能由语义更准确推断选点必然改善。阶段误差、整体间隔和选择质量必须分别评价。',
        '本轮不声称完全解释YOLO共享慢化。尤其CPU线程调度与启动次序、主机实际落核、控制CPU工作、共享访存仍有未辨识部分。',
        '准备与运行成本：板端'+str(profile['native_state']['seconds'])+'秒；离线回放'+str(read(study.OUTPUT/'COST.json')['seconds'])+'秒；组件资格、组件测量与完整DNN分开计数。',
        '完整原始索引：'+str(study.PROFILE)+'；'+str(study.OUTPUT/'REPLAY_FREEZE.json'),
        '复现：python run_completion_profile.py validate；python analyze_completion_profile.py；python run_device_completion_replay.py replay；python run_device_completion_replay.py analyze；python report_completion_study.py。',
        '重放已冻结结果不需要连接板子。不要再次执行已完成的board批次。']
    (REPORT/'FINDINGS_zh.txt').write_text('\n'.join(lines)+'\n')
    delivery=dict(report=str(REPORT/'FINDINGS_zh.txt'),development_gate_passed=supported,
        decisions=decisions,goal_complete=False,recorded_at=now(),
        files={str(p):sha256(p) for p in REPORT.iterdir() if p.is_file()})
    write(study.OUTPUT/'DELIVERY.json',delivery,replace=True)
    progress_path=ROOT/'artifacts/static_revision_01/GOAL_PROGRESS.json';progress=read(progress_path)
    progress.update(updated_at=now(),goal_status='ACTIVE',full_goal_completion_proven=False,
        last_goal_turn='progress: 15 new accepted isolated MMIO-completion sessions, 105 VTA calls, 36 frozen development replays, no reboot or new complete DNN',
        current_turn_new_native_sessions=15,current_turn_new_DNN_inferences=0,
        current_turn_VTA_component_calls=105,no_DNN_inferences_this_goal_turn=True,
        phase='independent device completion development evaluated; see frozen decision and limitations',
        device_completion_report=str(REPORT/'FINDINGS_zh.txt'),
        device_completion_results=str(study.OUTPUT/'analysis/RESULTS.json'),
        next_action='audit remaining CPU-start/completion and host placement identifiability against measured brackets; fresh candidate test only for evidence-supported revision',
        latest_device_completion_development_gate_passed=supported)
    write(progress_path,progress,replace=True)
    return delivery

if __name__=='__main__':print(json.dumps(report(),ensure_ascii=False,indent=2))
