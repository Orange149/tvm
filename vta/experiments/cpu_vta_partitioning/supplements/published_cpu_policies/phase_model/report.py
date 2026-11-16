"""Export bounded development evidence; no ranking reads this module."""
from pathlib import Path
import shutil

from cpu_vta_partitioning.common import read
from workload_model.common import OUTPUT,REPORT


def report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font=Path('/mnt/c/Windows/Fonts/msyh.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family']=font_manager.FontProperties(fname=font).get_name()
    plt.rcParams['axes.unicode_minus']=False
    source=OUTPUT/'phase_profile/temporal_development'
    data=read(source/'RESULTS.json');REPORT.mkdir(parents=True,exist_ok=True)
    for leaf in ('pipeline_comparison.csv','CPU_stages.csv','RESULTS.json','DECISION.json','HYPOTHESIS.json'):
        shutil.copy2(source/leaf,REPORT/('phase_'+leaf))
    shutil.copy2(OUTPUT/'EXECUTOR_IDENTITY_CORRECTION.json',REPORT/'EXECUTOR_IDENTITY_CORRECTION.json')
    shutil.copy2(OUTPUT/'phase_profile/model_inputs/component_accounting.csv',REPORT/'phase_components.csv')
    labels=['ResNet18\n四线程重叠','ResNet18\n三线程分组','YOLO\n四线程重叠','YOLO\n三线程分组']
    rows=data['rows'];fig,ax=plt.subplots(figsize=(9,4.6));x=list(range(len(rows)))
    for offset,key,label,color in [(-.25,'original_predicted_ms','原容量模型','#9d9d9d'),
                                    (0,'runnable_host_before_predicted_ms','开发时序模型','#4a81b8'),
                                    (.25,'measured_II_ms','无探针实测','#dc8745')]:
        ax.bar([i+offset for i in x],[r[key] for r in rows],width=.24,label=label,color=color)
    ax.set_xticks(x,labels);ax.set_ylabel('平均完成间隔（毫秒，越低越好）')
    ax.legend(frameon=False);ax.set_ylim(0,250);ax.grid(axis='y',alpha=.18)
    fig.tight_layout();fig.savefig(REPORT/'phase_prediction.png',dpi=180);plt.close(fig)
    front=[r for r in data['CPU_stages'] if r['program']=='cpu_00_02']
    fig,ax=plt.subplots(figsize=(9,4.5))
    for offset,values,label,color in [(-.18,[r['pipeline_observed_callback_CPU_ms']/r['isolated_callback_CPU_ms'] for r in front],
                                      '计算回调核时比例（探针会话）','#4a81b8'),
                                     (.18,[r['pipeline_clean_service_ms']/r['isolated_clean_service_ms'] for r in front],
                                      '阶段墙钟服务比例（无探针）','#dc8745')]:
        ax.bar([i+offset for i in x],values,width=.34,label=label,color=color)
    ax.axhline(1,color='black',linewidth=.8);ax.set_xticks(x,labels);ax.set_ylabel('流水 / 隔离')
    ax.legend(frameon=False);ax.grid(axis='y',alpha=.18);fig.tight_layout()
    fig.savefig(REPORT/'phase_work_and_service.png',dpi=180);plt.close(fig)
    return dict(directory=str(REPORT),figures=2)


if __name__=='__main__':print(report())
