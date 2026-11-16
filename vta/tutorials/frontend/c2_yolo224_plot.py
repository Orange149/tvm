#!/usr/bin/env python3
"""Export reproducible scientific figures from reconciled C2 measurements."""
import argparse,json,statistics
from pathlib import Path
from c2_yolo224_prepare import write,sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--evidence',type=Path,required=True)
    a=p.parse_args();e=a.evidence;data=json.loads((e/'reconciliation.json').read_text())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    modes=['M0','M1','M2','M3'];ax=axes[0,0];bottom=np.zeros(4)
    allocation={}
    for row in data['physical_allocations']:allocation.setdefault(row['mode'],row)
    fields=[('graph_pool_bytes','Graph pools'),('new_slots_aligned_bytes','New shared slots'),
        ('runtime_workspace_bytes','Runtime workspace'),('command_backing_bytes','Command backing')]
    for field,label in fields:
        values=np.array([allocation[m][field]/2**20 for m in modes])
        ax.bar(modes,values,bottom=bottom,label=label,width=.65);bottom+=values
    for i,n in enumerate(bottom):ax.text(i,n+.7,f'{n:.2f}',ha='center',fontsize=8)
    ax.set_ylabel('Actual u-dma-buf pool high water (MiB)');ax.set_ylim(0,85)
    ax.set_title('(a) Pool allocation demand; fixed 192 MiB reservation unchanged')
    ax.legend(fontsize=7,loc='center right')
    # collect() appends the master, boot02 and boot03 in deployment order.
    boots=list(dict.fromkeys(x['boot_id'] for x in data['boot_statistics']))
    for ax,modes,title in ((axes[0,1],['M0','M1','M2','M3'],'(b) Main experiment'),
                           (axes[1,0],['L1','L2','CP','E1'],'(c) Input-lease experiment')):
        for i,boot in enumerate(boots):
            lookup={x['mode']:x['fps'] for x in data['boot_statistics'] if x['boot_id']==boot}
            xs=[j+(i-(len(boots)-1)/2)*.13 for j,m in enumerate(modes) if m in lookup]
            ys=[lookup[m] for m in modes if m in lookup]
            ax.scatter(xs,ys,s=30,label=f'Boot {i+1}',zorder=3)
        ax.set_xticks(range(4),modes);ax.set_ylabel('FPS, from median block II per boot')
        ax.set_title(title+'; each point is one Linux boot')
        ax.set_ylim(0,2.15);ax.grid(axis='y',alpha=.25);ax.legend(fontsize=8)
    ax=axes[1,1];wanted=[('M0','M1'),('M1','M2'),('M2','M3'),('L2','E1')]
    lookup={(x['old'],x['new']):x for x in data['contrasts']}
    names=[]
    for i,pair in enumerate(wanted):
        names.append(' → '.join(pair));x=lookup.get(pair)
        if not x:continue
        mean=x['ii_reduction_percent_mean'];interval=x['ii_reduction_percent_CI95']
        if interval:
            ax.errorbar(mean,i,xerr=[[mean-interval[0]],[interval[1]-mean]],fmt='o',capsize=4,color='tab:blue')
        else:ax.scatter([mean],[i],marker='x',color='tab:gray')
    ax.axvline(0,color='black',linestyle='--',linewidth=.8);ax.set_yticks(range(4),names)
    ax.invert_yaxis();ax.set_xlabel('II reduction (%)   positive = faster')
    ax.set_title('(d) Paired boot estimates; t 95% CI only with 3 boots')
    ax.grid(axis='x',alpha=.25)
    lo,hi=ax.get_xlim();padding=max(hi-lo,.01)*.1;ax.set_xlim(min(lo,0)-padding,max(hi,0)+padding)
    fig.suptitle('YOLOv3-tiny-224 C2 · same graph, weights, inputs, stage order and runtime',fontsize=12)
    for suffix in ('png','pdf','svg'):
        target=e/('c2_yolo224_results.'+suffix)
        if target.exists():raise FileExistsError(target)
        fig.savefig(target,dpi=300)
    write(e/'figure_provenance.json',{'generator_sha256':sha(Path(__file__)),
        'reconciliation_sha256':sha(e/'reconciliation.json'),'matplotlib':matplotlib.__version__,
        'independent_boots':len(boots),'boot_point_order':{f'Boot {i+1}':boot for i,boot in enumerate(boots)},
        'figures':{suffix:sha(e/('c2_yolo224_results.'+suffix)) for suffix in ('png','pdf','svg')}})
    print(json.dumps({'figure':str(e/'c2_yolo224_results.pdf'),'boot_points':len(boots)}))


if __name__=='__main__':main()
