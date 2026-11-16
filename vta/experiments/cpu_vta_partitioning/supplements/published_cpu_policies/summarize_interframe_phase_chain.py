"""Explain the frozen phase audit; never use observed timing for ranking."""
from collections import defaultdict
from pathlib import Path
import csv
import hashlib
import json
import os
import sys

os.environ.setdefault('MPLCONFIGDIR', '/tmp/cpu_vta_phase_chain_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1] / 'src'))
from cpu_vta_partitioning.common import read, write, sha256
from scheduler_gap_audit.curves import csv_write

PARENT = ROOT / 'artifacts/interframe_phase_chain_01'
OUT = PARENT / 'analysis'
REPORT = ROOT.parents[4] / 'paper/my_paper/analysis/20261112_interframe_phase_chain'


def main():
    if OUT.exists():
        raise FileExistsError('preserve the sealed diagnostic explanation')
    for path, expected in read(PARENT/'MANIFEST.json')['files'].items():
        if sha256(path) != expected:
            raise ValueError('phase audit changed')
    with (PARENT/'phase_chain.csv').open(encoding='utf-8-sig') as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for k in row.keys()-{'source','stage'}:
            row[k] = int(row[k])
    freeze = read(PARENT/'MODEL_FREEZE.json')
    result = read(PARENT/'RESULTS.json')
    names = [s['program'] for s in result['configuration']['stages']]
    selected = result['compared_frames']
    blocks = [('startup_0_3',list(range(4))), ('startup_4_7',list(range(4,8))),
              ('accepted_diagnostic_frames_12_23',selected)]
    summaries=[]
    for block, frames in blocks:
        for source in ('native_observed','sealed_model'):
            for name in names:
                group=[r for r in rows if r['source']==source and r['stage']==name and r['frame'] in frames]
                if len(group)!=len(frames):
                    raise ValueError('diagnostic phase coverage incomplete')
                summaries.append(dict(block=block,source=source,stage=name,frames=len(group),
                    run_ms=sum(r['run_ns'] for r in group)/len(group)/1e6,
                    service_ms=sum(r['service_ns'] for r in group)/len(group)/1e6,
                    predecessor_to_start_ms=sum(r['predecessor_to_start_ns'] for r in group)/len(group)/1e6))
    exposure=freeze['queue_exposure_ns']
    full=sum(x['duration_ns'] for x in exposure if x['kind']=='queue_full_while_waiting_for_lock')
    blocked=sum(x['duration_ns'] for x in exposure if x['kind']=='predecessor_output_blocked_by_locked_full_queue')
    if full or blocked:
        conclusion='The source-order discrepancy is activated; no correction is authorized by this audit.'
    else:
        conclusion='No locked-full-queue backpressure was activated in this sealed case; source-order mismatch is not evidence of its timing-gap cause.'
    OUT.mkdir()
    csv_write(OUT/'startup_and_steady.csv',summaries)
    write(OUT/'RESULTS.json',dict(parent_result_sha256=sha256(PARENT/'RESULTS.json'),
        summaries=summaries, queue_lock_full_ns=full, predecessor_backpressure_ns=blocked,
        source_order_conclusion=conclusion,
        relative_phase_difference_is_not_an_independent_cause=True,
        native_is_probed_diagnostic_not_formal_performance=True,
        no_prediction_fix_or_selection_gain_claimed=True,new_DNN=0,new_native_sessions=0))

    plt.rcParams.update({'font.family':['DejaVu Sans','Droid Sans Fallback'],
        'axes.unicode_minus':False,'svg.fonttype':'none','pdf.fonttype':42})
    fig=plt.figure(figsize=(13.5,10))
    grid=fig.add_gridspec(2,2,height_ratios=[.9,1.15],hspace=.55,wspace=.22)
    top=fig.add_subplot(grid[0,:])
    colors={'native_observed':'#C97927','sealed_model':'#3478AE'}
    labels={'native_observed':'原生诊断轨迹（探针开启）','sealed_model':'冻结模型回放'}
    for source in colors:
        points=sorted([r for r in rows if r['source']==source and r['stage']==names[0]],key=lambda r:r['frame'])
        top.plot([r['frame'] for r in points],[r['run_ns']/1e6 for r in points],
                 color=colors[source],marker='o',markersize=3.5,label=labels[source])
    top.axvspan(selected[0]-.3,selected[-1]+.3,color='#e5ebef',alpha=.55,zorder=-1)
    top.set_title('A  前段从首帧到后续帧：模型很早进入重复节拍，原生耗时继续变化',loc='left',fontsize=13,pad=12)
    top.set_xlabel('流水帧号');top.set_ylabel('CPU 0–2 的 run 墙钟时间（ms）')
    top.legend(loc='lower right',frameon=False);top.grid(alpha=.18)
    top.text(.02,.93,'灰色区域：沿用已验收的第12–23帧，未按误差大小选帧',
             transform=top.transAxes,fontsize=10,color='#5c6a77')

    anchor_frame=selected[0]
    short=['前段 CPU 0–2','第一段 VTA 3–6','中段 CPU 7–8','第二段 VTA 9–12','尾段 CPU 13']
    for column,source in enumerate(colors):
        ax=fig.add_subplot(grid[1,column])
        index={(r['stage'],r['frame']):r for r in rows if r['source']==source}
        anchor=index[names[0],anchor_frame]['run_start_ns']
        current=index[names[0],anchor_frame]
        ax.barh(0,current['run_ns']/1e6,left=0,color='#8C78AA',height=.5)
        for i,name in enumerate(names[1:],1):
            row=index[name,anchor_frame-1]
            ax.barh(i,row['run_ns']/1e6,left=(row['run_start_ns']-anchor)/1e6,
                    color=colors[source],height=.5)
            ax.text((row['run_start_ns']-anchor)/1e6+row['run_ns']/2e6,i,
                    f"{(row['run_start_ns']-anchor)/1e6:.1f}–{(row['run_end_ns']-anchor)/1e6:.1f}",
                    fontsize=8.7,ha='center',va='center',color='white')
        ax.axvline(0,color='#777',lw=.8)
        ax.set_xlim(-5,310);ax.set_ylim(4.7,-.8)
        ax.set_yticks(range(5),['当前帧：'+short[0]]+['上一帧：'+s for s in short[1:]],fontsize=9.2)
        ax.set_title(('B  原生轨迹' if column==0 else 'C  模型回放')+f' · 以第{anchor_frame}帧前段开始为零',loc='left',fontsize=12,pad=15)
        ax.set_xlabel('相对当前帧前段 run 起点（ms）');ax.grid(axis='x',alpha=.18)
        ax.set_axisbelow(True)
    fig.suptitle('隔离 profile → 共享流水：误差与跨帧相位一起演化',fontsize=18,y=.985)
    fig.text(.5,.035,
        '下图分别在各自时钟内对齐；条形仅为 run 区间。探针轨迹用于解释，不能作为正式 FPS 或静态成本。\n'
        '相位错位不等于已找到独立原因；本次没有修改评分、DP、Top-20，也没有新增板测。',
        ha='center',fontsize=10,color='#526170')
    fig.subplots_adjust(left=.16,right=.98,top=.91,bottom=.14)
    for suffix in ('png','svg','pdf'):
        fig.savefig(REPORT/('phase_chain_comparison.'+suffix),dpi=180,facecolor='white')
    plt.close(fig)
    lookup={(r['source'],r['stage']):r for r in summaries if r['block']=='accepted_diagnostic_frames_12_23'}
    findings=[
        '本轮结论：已定位到“跨帧相位不同”，尚未建立可以改善选点的新修正。',
        '1. 排除了一个具体解释：虽然源码在取队列/申请锁的顺序上不同，但当前冻结模型等待VTA锁时队列最多1帧，容量为2；由此导致的满队列反压为0。不能宣称只修改这个顺序就能解决慢化。',
        '2. 模型从第4帧后进入重复的前段run节拍约192.527 ms；原生前段仍变化。第2帧时二者反而接近，说明只看一个稳态平均相位会掩盖偏差如何演化。',
        '3. 对已验收第12–23帧，逐阶段run/前驱get结束到本阶段set开始的间隙（原生；模型），单位ms：',
    ]
    for name in names:
        a,b=lookup['native_observed',name],lookup['sealed_model',name]
        findings.append('   %s: run %.3f；%.3f，前驱后间隙 %.3f；%.3f。' %
                        (name,a['run_ms'],b['run_ms'],a['predecessor_to_start_ms'],b['predecessor_to_start_ms']))
    findings.extend([
        '4. 模型前段偏快的同时，第一VTA段在模型中等待前驱之后的间隙更长。阶段误差有正有负，整机完成间隔接近不能证明阶段模型正确。这个间隙含资源与线程唤醒等开销，本表不把它全部归为调度等待。',
        '5. 首帧到后续帧的变化不支持把“中段晚了约43 ms”直接作为独立慢化项叠加。前段速度也会改变上一帧设备/中段位置；需要区分因果与反馈。',
        '6. 当前可支持的最小缺口仍是真实线程在共享执行中的CPU供给/启动完成历史，不能用已知流水相位或CPU供给缺口作为未测候选输入。未从此审计得到可部署的新参数。',
        '本轮新增板测、推理为0；观察器完整返回值与冻结模型摘要一致，4项阶段链测试通过。',
        '此处原生为探针开启且CPU排队查询关闭的诊断会话；正式性能仍使用关闭探针的独立会话。',
    ])
    (REPORT/'CONCLUSIONS_zh.txt').write_text('\n'.join(findings)+'\n')
    paths=[*OUT.iterdir(),REPORT/'CONCLUSIONS_zh.txt',
           *[REPORT/('phase_chain_comparison.'+s) for s in ('png','svg','pdf')]]
    write(OUT/'MANIFEST.json',dict(source_sha256=sha256(__file__),
        files={str(p):sha256(p) for p in paths if p.is_file()},
        parent_manifest_sha256=sha256(PARENT/'MANIFEST.json'),new_native_sessions=0))
    print(json.dumps(dict(report=str(REPORT/'CONCLUSIONS_zh.txt'),
                         plot=str(REPORT/'phase_chain_comparison.png'),
                         locked_full_queue_backpressure_ns=blocked),ensure_ascii=False))


if __name__=='__main__':
    main()
