"""Standalone figures; measured wall spans and ideal work remain labelled."""
from pathlib import Path


def plot(target, extraction, diagnostic, replay, comparisons):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import font_manager
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    font = Path('/usr/share/fonts/truetype/arphic-gbsn00lp/gbsn00lp.ttf')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'axes.unicode_minus':False,'svg.hashsalt':'dependency-audit-fixed'})
    target = Path(target)
    def save(fig,name):
        fig.tight_layout()
        for ext in ('svg','png','pdf'):
            metadata = {'Date':None} if ext=='svg' else {'CreationDate':None,'ModDate':None} if ext=='pdf' else {}
            fig.savefig(target/(name+'.'+ext),dpi=170,metadata=metadata)
        plt.close(fig)
    if not comparisons:
        return
    labels = {'overlap4':'四线程','split3':'三线程'}
    fig,axes = plt.subplots(1,2,figsize=(10,4))
    for j,(key,label) in enumerate((('aggregate_ms','整段取最大'),('join_ms','逐次完成依赖'))):
        bars=axes[0].bar([i+(j-.5)*.32 for i in range(len(comparisons))],
            [r[key] for r in comparisons],width=.3,label=label)
        axes[0].bar_label(bars,fmt='%.3f',padding=3,fontsize=8)
    axes[0].set_xticks(range(len(comparisons)),[labels[r['case']] for r in comparisons])
    axes[0].set_ylabel('隔离计算工作回放 ms/帧（计入8帧）')
    axes[0].legend(loc='lower right')
    bars=axes[1].bar(range(len(comparisons)),[r['additional_dependency_ms'] for r in comparisons],color='#d97706')
    axes[1].bar_label(bars,fmt='%.6f',padding=3)
    axes[1].set_xticks(range(len(comparisons)),[labels[r['case']] for r in comparisons])
    axes[1].set_ylabel('两种求和方式之差 ms/帧')
    save(fig,'dependency_work')
    colors=plt.get_cmap('tab10')
    for case in replay['cases']:
        coremap=replay['cases'][case]['core_map']
        n=len(coremap)
        scored=next(f for f in replay['cases'][case]['frames'] if f['phase']=='scored')
        # The complete twelve-launch DAG; no graphviz or other process needed.
        fig,ax=plt.subplots(figsize=(18,3.8))
        kernels=len(extraction['cases'][case]['sequence'])
        for k in range(kernels):
            for worker in range(n):
                x=k*2; y=n-1-worker
                ax.add_patch(Rectangle((x,y-.22),.78,.44,facecolor=colors(coremap[worker]),alpha=.75))
                ax.text(x+.39,y,f'k{k} w{worker}',ha='center',va='center',fontsize=7)
                ax.annotate('',xy=(x+1.22,(n-1)/2),xytext=(x+.78,y),
                            arrowprops=dict(arrowstyle='->',lw=.55,color='#555555'))
                if k:
                    ax.annotate('',xy=(x,y),xytext=(x-.78,(n-1)/2),
                                arrowprops=dict(arrowstyle='->',lw=.55,color='#555555'))
            ax.plot(k*2+1.22,(n-1)/2,marker='D',color='#333333',markersize=4)
        ax.set_xlim(-.15,kernels*2-.45);ax.set_ylim(-.65,n-.35)
        ax.set_yticks(range(n),[f'worker{n-1-y} → core{coremap[n-1-y]}' for y in range(n)])
        ax.set_xticks([]);ax.set_title(labels[case]+'：每次启动的所有worker完成后，才能进入下一次启动')
        ax.set_xlabel('箭头表示依赖；菱形为完成汇合。节点精确核时与完整时序保存在DOT／REPLAY.json。')
        save(fig,'dependency_graph_'+case)
        if not diagnostic[case]['frame_mapping']:
            continue
        fig,axes=plt.subplots(3,1,figsize=(11,7),sharex=True)
        for node in scored['timeline']:
            if node['kind']=='callback':
                axes[0].broken_barh([(node['start_ns']/1e6,node['duration_ns']/1e6)],
                    (node['worker']-.32,.64),facecolors=colors(coremap[node['worker']]))
        axes[0].set_title('理想单阶段计算工作回放：不含完整启动、首尾串行及调度开销')
        for ax,tr,frame_id,title in ((axes[1],extraction['cases'][case],2,'隔离探针记录：帧2'),
                                     (axes[2],diagnostic[case],8,'流水探针记录：帧8')):
            frame=tr['frames'][frame_id];origin=frame['launches'][0]['wall_start_ns']
            for kernel in frame['launches']:
                for w,(a,b) in enumerate(zip(kernel['worker_wall_start_ns'],kernel['worker_wall_end_ns'])):
                    ax.broken_barh([((a-origin)/1e6,(b-a)/1e6)],(w-.32,.64),facecolors=colors(coremap[w]))
            ax.set_title(title+'；仅显示回调墙钟区间，含未细分的非运行时间')
        for ax in axes:
            ax.set_yticks(range(n),[f'w{i}/c{coremap[i]}' for i in range(n)])
            ax.set_ylim(-.7,n-.3);ax.grid(axis='x',alpha=.2)
        axes[-1].set_xlabel('各行独立从首次计算启动起算 ms；没有把不同钟系拼接为绝对阶段时间')
        save(fig,'callback_timeline_'+case)

