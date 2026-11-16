"""Post-measurement diagnostics; no imports from this module in search/ranking."""
import csv
import json
from pathlib import Path
import statistics
from itertools import combinations
from run_resource_behavior import ROOT,api
from cpu_policies.open_resources.costs import Costs
from .model import score

def audit(run):
    c=api();run=Path(run).resolve();state=c.read(run/'state.json')
    if state['status']!='COMPLETE':raise ValueError('diagnostics require a completed campaign')
    bundle=Path(state['bundle']);pool=c.read(bundle/'plans/static_pool.json')
    ranked=c.read(bundle/'plans/selection_rankings.json');main=c.read(run/'analysis/analysis.json')
    probe=ROOT/'runs/resnet18_thread_work_probe_01';probe_ids=set();probe_provenance=None
    if (probe/'state.json').is_file():
        probe_state=c.read(probe/'state.json')
        if probe_state['status']=='COMPLETE':
            from datetime import datetime
            if datetime.fromisoformat(probe_state['created'])<datetime.fromisoformat(state['created']):
                probe_bundle=Path(probe_state['bundle'])
                configs=c.read(probe_bundle/'plans/configurations.json');probe_ids={v['id'] for v in configs.values()}
                probe_provenance={'run_id':probe.name,'bundle_manifest_sha256':c.sha256(probe_bundle/'manifest.json'),
                    'note':'diagnostic first-choice evidence became known after list freeze and before formal union measurement'}
    settings=c.read(bundle/'plans/source_provenance.json').get('model_settings',
        {'cpu_distribution':'balanced','score_variant':'combined'})
    costs=Costs(c.read(bundle/'plans/isolated_costs.json')['costs'],programs=c.read(bundle/'plans/programs.json'),
        ddr=c.read(bundle/'plans/ddr.json'),distribution=settings['cpu_distribution'])
    stages=[];configurations=[]
    for path in sorted((run/'accepted'/state['boot_id']).glob('*.json')):
        receipt=c.read(path);task=receipt['task']
        if task['kind']!='performance':continue
        cfg=pool[task['candidate_id']];raw=run/receipt['raw']/'result.jsonl'
        rows=[json.loads(line) for line in raw.read_text().splitlines() if line][8:24]
        for i,s in enumerate(cfg['stages']):
            iso=costs.get(s,cfg);service=statistics.mean(row[f'stage{i}_ms'] for row in rows)
            stages.append({'id':cfg['id'],'program':s['program'],'device':s['device'],'threads':s['threads'],
                'mask':str(s['mask']),'host_mask':str(cfg['host_mask']),'gate':cfg['execution_mode'],
                'isolated_service_ms':iso['service_ms'],'pipeline_service_ms':service,
                'service_ratio':service/iso['service_ms'],'isolated_run_ms':iso['run_ms'],
                'pipeline_run_ms':statistics.mean(row[f'stage{i}_run_ms'] for row in rows),
                'vta_lock_wait_ms':statistics.mean(row[f'stage{i}_vta_mutex_wait_ms'] for row in rows)
                    if all(row[f'stage{i}_vta_mutex_wait_ms'] is not None for row in rows) else None,
                'measurement_scope':'same frame indices, not a strict common wall-clock resource ledger'})
        prediction=score(cfg,costs);measured=receipt['summary']['ii_ms']
        configurations.append({'id':cfg['id'],'context':cfg['context_id'],'gate':cfg['execution_mode'],
            'measured_II_ms':measured,'fps':1000/measured,
            'measured_in_diagnostic_probe_before_this_batch':cfg['id'] in probe_ids,
            'old_uniform_II_ms':cfg['original_score_ns']/1e6,'selected_model_II_ms':cfg['score_ns']/1e6,
            **{variant+'_II_ms':prediction[variant]/1e6 for variant in ('original','host_concentration','team_occupancy','combined')}})
    if len(configurations)!=len(pool):raise ValueError('incomplete result set')
    output=run/'analysis'
    def write(name,rows):
        with (output/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write('stage_profile_comparison.csv',stages);write('four_scores_after_measurement.csv',configurations)
    measured={r['id']:r['measured_II_ms'] for r in configurations}
    ranking=[]
    best_pool=min(measured.values())
    def average_ranks(values):
        ordered=sorted(range(len(values)),key=lambda i:values[i]);ranks=[0.]*len(values);begin=0
        while begin<len(values):
            end=begin+1
            while end<len(values) and values[ordered[end]]==values[ordered[begin]]:end+=1
            for pos in ordered[begin:end]:ranks[pos]=(begin+end+1)/2
            begin=end
        return ranks
    for key in ('pure_static20','diverse20','coverage20'):
        ids=[r['id'] for r in ranked[key]];field='original_score_ns' if key=='pure_static20' else 'score_ns'
        predictions=[pool[i][field] for i in ids];actual=[measured[i] for i in ids]
        ties=discordant=concordant=0
        for i,j in combinations(range(20),2):
            if predictions[i]==predictions[j]:ties+=1;continue
            value=(predictions[i]-predictions[j])*(actual[i]-actual[j])
            discordant+=value<0;concordant+=value>0
        a,b=average_ranks(predictions),average_ranks(actual);am,bm=statistics.mean(a),statistics.mean(b)
        denominator=(sum((v-am)**2 for v in a)*sum((v-bm)**2 for v in b))**.5
        rho=sum((x-am)*(y-bm) for x,y in zip(a,b))/denominator if denominator else None
        ranking.append({'method':key,'distinct_predicted_scores':len(set(predictions)),
            'prediction_tied_pairs':ties,'comparable_pairs':190-ties,'discordant_comparable_pairs':discordant,
            'concordant_comparable_pairs':concordant,'Spearman_with_average_ties':rho,
            **{'budget'+str(n)+'_II_regret_percent_vs_union_best':
                (min(measured[i] for i in ids[:n])/best_pool-1)*100 for n in (1,5,20)}})
    write('ranking_with_ties.csv',ranking)
    errors={v:{'MAE_ms':statistics.mean(abs(row[v+'_II_ms']-row['measured_II_ms']) for row in configurations),
        'mean_signed_error_ms':statistics.mean(row[v+'_II_ms']-row['measured_II_ms'] for row in configurations)}
        for v in ('original','host_concentration','team_occupancy','combined')}
    comparison_errors={name:{'MAE_ms':statistics.mean(abs(row[field]-row['measured_II_ms']) for row in configurations),
        'mean_signed_error_ms':statistics.mean(row[field]-row['measured_II_ms'] for row in configurations)}
        for name,field in [('original_uniform','old_uniform_II_ms'),('selected_model','selected_model_II_ms')]}
    poolbest=main['pool_best']['id'];keys={'original_first':ranked['pure_static20'][0]['id'],
        'envelope_first':ranked['diverse20'][0]['id'],'behavior_first':ranked['coverage20'][0]['id'],
        'pool_best':poolbest}
    selected={key:{'id':name,'stages':[row for row in stages if row['id']==name]} for key,name in keys.items()}
    c.write(output/'resource_model_diagnostics.json',{'errors_on_same_union':errors,
        'comparison_errors_on_same_union':comparison_errors,'key_configurations':selected,
        'ranking_with_ties':ranking,
        'post_measurement_diagnostics_only':True,'scores_not_refitted':True,
        'searched_model_settings':settings,'four_variant_distribution':settings['cpu_distribution'],
        'prior_probe':probe_provenance,'known_from_prior_probe_in_this_union':sorted(set(pool)&probe_ids)},replace=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.1),constrained_layout=True)
    labels={'pure_static20':'Original static','diverse20':'New static','coverage20':'New static + behavior'}
    for key,method in main['decision']['methods'].items():
        axes[0].plot([1,5,20],[method['budgets'][str(n)]['fps'] for n in (1,5,20)],'o-',label=labels[key])
    axes[0].set(xlabel='Full-pipeline measurement budget',ylabel='Best observed throughput (FPS)',xticks=[1,5,20])
    axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
    for field,label,marker in [('old_uniform_II_ms','Original model','o'),('selected_model_II_ms','New model','x')]:
        axes[1].scatter([row[field] for row in configurations],[row['measured_II_ms'] for row in configurations],
            s=23,alpha=.65,label=label,marker=marker)
    values=[row[field] for row in configurations for field in ('old_uniform_II_ms','selected_model_II_ms','measured_II_ms')]
    lo,hi=min(values)*.95,max(values)*1.05;axes[1].plot([lo,hi],[lo,hi],'k--',lw=.8)
    axes[1].set(xlabel='Predicted completion interval (ms)',ylabel='Observed completion interval (ms)')
    axes[1].legend(fontsize=8);axes[1].grid(alpha=.2)
    fig.suptitle('ResNet18 / one input / one boot / short windows; no independent confirmation',fontsize=10)
    for suffix in ('png','pdf'):fig.savefig(output/('resource_model_overview.'+suffix),dpi=170)
    plt.close(fig)
    return {'configurations':len(configurations),'stage_comparisons':len(stages),'errors':errors}

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);args=p.parse_args()
    print(json.dumps(audit(args.run_dir),ensure_ascii=False))
