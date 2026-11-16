"""Consolidate attribution and the bounded host-only intervention in Chinese."""
from pathlib import Path
import json
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from cpu_vta_partitioning.common import read,write,sha256,now
from scheduler_gap_audit.curves import csv_write

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/isolation_gap_focus_01'
REPORT=ROOT.parents[4]/'paper/my_paper/analysis/20261111_isolation_gap_focus'


def run():
    paths=dict(clocks=ROOT/'artifacts/clock_scope_audit_01/analysis/RESULTS.json',
        worker=ROOT/'artifacts/worker_dispatch_gap_01/RESULTS.json',
        matched=ROOT/'artifacts/host_gap_intervention_01/matching_native_window/RESULTS.json',
        native=ROOT/'artifacts/host_gap_intervention_01/analysis/RESULTS.json',
        state=ROOT/'runs/host_gap_intervention_01/state.json',
        searches=ROOT/'artifacts/current_model_gap_01/existing_host_matches/RESULTS.json')
    d={k:read(p) for k,p in paths.items()}
    if d['state']['status']!='COMPLETE' or d['state']['accepted_sessions']!=6 or d['state']['DNN_inferences']!=87:
        raise ValueError('bounded native intervention not complete')
    native={r['role']:r for r in d['native']['rows']};model={r['role']:r for r in d['matched']['rows']}
    labels=dict(shared='VTA主机允许四核迁移',model_hosts_2_1='两个VTA主机固定核心2／1',
                observed_modal_hosts_3_2='两个VTA主机固定核心3／2')
    rows=[]
    for role in labels:
        n,m=native[role],model[role]
        rows.append(dict(strategy=labels[role],configuration_id=n['configuration_id'],
            native_FPS=n['fps'],native_II_ms=n['ii_ms'],native_front_ms=n['stage_services_ms'][0],
            post_hoc_replay_front_ms=m['predicted_stage_ms'][0],
            replay_front_error_ms=m['front_error_ms'],native_receipt=n['receipt'],
            interpretation='single-boot short-window mechanism check; post-hoc replay, not independent prediction validation'))
    a,b=native['model_hosts_2_1'],native['observed_modal_hosts_3_2']
    fixed_gap=abs(a['stage_services_ms'][0]-b['stage_services_ms'][0])
    fraction=max(r['fps'] for r in native.values())/min(r['fps'] for r in native.values())-1
    verdict=dict(no_obvious_native_throughput_gain_observed=True,statistical_equivalence_not_claimed=True,
        host_physical_core_number_alone_not_supported_as_large_gap_explanation=True,
        freely_reassigned_model_vs_fixed_model_differs_more_than_native=True,
        no_new_static_model_adopted=True,no_full_search_resumed=True,
        overall_goal_complete=False,
        fixed_pair_native_front_difference_ms=fixed_gap,native_FPS_spread_percent=fraction*100)
    result=dict(rows=rows,verdict=verdict,sources={str(p):sha256(p) for p in paths.values()},
        source_sha256=sha256(__file__),recorded_at=now(),new_DNN=87,new_VTA_component_calls=0,
        zero_DNN_self_test_processes=1,native_DNN_sessions=6,
        original_profiles_and_snapshots_unchanged=True)
    write(OUT/'RESULTS.json',result,replace=True);csv_write(OUT/'host_intervention.csv',rows)
    text=['隔离profile与共享流水误差：本轮完成项及结论','',
        '范围：仅排查YOLO固定切图CPU0–2→VTA3–6→CPU7–8→VTA9–12→CPU13。CPU配置始终4线程0123／3线程123／1线程0，控制线程四核共享，CPU门控关闭。',
        '新增一次零DNN计时自测，以及三个主机放置方案的共同实测：6个DNN会话、87次完整推理。未重启、未重编运行库/阶段模块/执行器，未恢复基准桥接或完整域搜索。','',
        '一、先把几种CPU计时的口径区分开',
        '短睡眠循环（600次约5us忙段＋请求1us睡眠）关闭切换记录时：线程CPU钟7.665ms，perf task-clock 5.976ms；开启记录时：8.847ms、7.560ms，IN→OUT重建5.837ms。',
        '因此主机run中CPU钟与调度轨迹之间的差额，在没有DNN的程序中也存在，不能直接认定为新增算子计算。没有将每次切换的观测差额拟合成固定惩罚，也没有更换旧profile。',
        '两个记录开关情形各执行一次，忙段按墙钟预算而非固定指令数，不将其差值认定为稳定的探针成本。','',
        '二、已定位到启动前等待，但没有重复加到模型上',
        '在旧有效轨迹的12个完整前段帧、144次并行启动中，每次仅看最后完成的worker：在caller已经进入计算、自己尚未进入计算的累计间隔约17.851ms/帧。',
        '其中约0.404ms为运行，0.034ms为抢占标记off，17.412ms为非抢占离开后尚未恢复。入口计数读取包络重合0.367ms，不能直接解释其余长间隔。',
        '另一个累计CPU排队计数在稍宽窗口内增长17.461ms/帧，支持启动前CPU排队存在；它可含先前未结算等待，不能当作同一窗口的精确分解。',
        '先前以非抢占方式离开CPU的线程，之后可能已经被唤醒并在排队；没有sched_wakeup事件就不能把整段都叫作睡眠。',
        '旧有限服务回放已经包含部分启动延迟，但与原生最后worker身份仅18.75%一致。最后与倒数第二完成时间差的中位数1.052ms，并非全部近乎并列。以上不是把17.851ms直接加到阶段耗时的依据。','',
        '三、主机位置对照：本轮没有观察到明显吞吐收益',
        '匹配测量前核查既有11,730份凭据；同切图、同CPU配置的38份匹配记录只有共享主机，缺少固定位置对照。随后按冻结顺序共同实测三个方案。',
        '方案 | 实测FPS | 实测前段ms | 事后同窗口回放前段ms']
    for r in rows:text.append(f"{r['strategy']} | {r['native_FPS']:.4f} | {r['native_front_ms']:.3f} | {r['post_hoc_replay_front_ms']:.3f}")
    text += [f"三项FPS最大差幅约{fraction*100:.3f}%；两个固定方案前段相差约{fixed_gap:.3f}ms。样本为单输入、单启动、每项一轮短窗口，不宣称稳定收益或统计等效。",'',
        '四、这次排除了什么，留下什么',
        '不能解释为“某个核心本身更快”或“把主机移到旧观测核心就能恢复吞吐”。实际两个固定方案相近；其匹配回放也都约211.4ms。',
        '较大的预测变化来自允许模型在潜在唤醒点重选位置，与始终固定位置之间的差异。自由调度方案的回放仍明显偏乐观；真正执行中允许迁移，也不代表线程会按模型的理想化规则迁移。',
        '固定位置后回放前段误差约5.44–6.43ms，较共享方案约24.94ms小，但这是事后诊断：仍复用了共享主机的隔离时序形状、保留同构核近似、没有建模全部控制开销，不能称为可部署静态预测已通过独立验证。',
        '其他阶段仍有误差，例如尾部CPU阶段回放26.56ms、本轮实测约15.74–16.61ms。因此尚未解决全部阶段误差，也没有FPS提升或新Top20结论。','',
        '五、下一步收敛方向',
        '不再从“总核时漏了一项”出发，也不把这次诊断的17.851ms或主机落点回填为通用成本。',
        '先检查已有三线程前段＋单线程中段/尾段的匹配历史。在前段固定012、中段与尾段固定3时，仅改变VTA主机：落在较忙计算worker、较轻计算worker或前段核组外。具体热点由该子图真实隔离份额决定，不凭核心编号或网络名称。',
        '这项对照仍使用已有host_shared_1/2/3等上下文，能够检验线程负载不均与主机竞争，而在四核前段占满四核时仅交换主机核心编号，无法构造“避开整个前段核组”的干预。先核对已有证据与真实profile，再冻结必要的最小对照，不直接修改评分。',
        'Pipe-it的固定计算核组用于减少计算阶段间竞争，GCAPS的CPU主机工作/设备工作区分用于解释主机仍会参与CPU竞争；这里是策略和资源语义适配，不称完整复现。',
        '也核查了RT-Gang，但不将其内核级one-gang-at-a-time及带宽管理直接等同于现有用户态CPU门控；本轮没有实施该新路线。',
        '若需要新的资源策略，必须保留原域作为参照、单独记录执行策略贡献，先验证机制和匹配profile，再决定如何接回完整DP。','',
        '依据：',
        'Linux v5.4线程CPU钟与perf事件计时源码：https://raw.githubusercontent.com/torvalds/linux/v5.4/kernel/time/posix-cpu-timers.c；https://raw.githubusercontent.com/torvalds/linux/v5.4/kernel/events/core.c',
        'CPU排队累计口径：https://raw.githubusercontent.com/torvalds/linux/v5.4/kernel/delayacct.c；https://raw.githubusercontent.com/torvalds/linux/v5.4/kernel/sched/stats.h',
        'Pipe-it原论文：https://arxiv.org/pdf/1903.05898',
        'GCAPS原论文：https://drops.dagstuhl.de/storage/00lipics/lipics-vol298-ecrts2024/LIPIcs.ECRTS.2024.14/LIPIcs.ECRTS.2024.14.pdf',
        'RT-Gang原论文：https://ittc.ku.edu/~heechul/papers/rtgang-rtas2019-camera.pdf',
        'RT-Gang作者代码：https://github.com/CSL-KU/rt-gang','',
        '已有ResNet18全域约13FPS成果与本轮固定YOLO布局无直接替代关系，保留原身份和原口径。完整目标仍未完成。']
    (OUT/'FINDINGS_zh.txt').write_text('\n'.join(text)+'\n')
    REPORT.mkdir(parents=True,exist_ok=True)
    for p in OUT.iterdir():
        if p.is_file():shutil.copy2(p,REPORT/p.name)
    return dict(report=str(REPORT/'FINDINGS_zh.txt'),verdict=verdict,new_DNN=87)


if __name__=='__main__':print(json.dumps(run(),ensure_ascii=False,indent=2))
