"""Development replay only: observations evaluate, never construct scores."""
import csv
from pathlib import Path
import statistics
from .model import score,VARIANTS

def run(output=None):
    from run_resource_behavior import ROOT,PARENT,SOURCE,api
    from cpu_policies.open_resources.costs import Costs
    from cpu_policies.open_resources.domain import canonical_tuple
    c=api();costs=Costs(c.read(SOURCE/'isolated_costs.json')['costs'],
        programs=c.read(PARENT/'plans/programs.json'),ddr=c.read(SOURCE/'ddr.json'))
    output=Path(output or ROOT/'artifacts/resource_envelope_development');output.mkdir(parents=True,exist_ok=True)
    datasets=[];details=[]
    for name,run_id in [('original_40','resnet18_open_cpu_resource_02'),
                        ('coverage_40','resnet18_autotvm_diverse_top20_01'),
                        ('matched_6','resnet18_resource_behavior_mechanism_01')]:
        root=ROOT/'runs'/run_id;state=c.read(root/'state.json');bundle=Path(state['bundle'])
        if name=='original_40':
            configs=c.read(root/'search_pool.json')
            observed={k:v['summary']['ii_ms'] for k,v in c.read(root/'screening.json')['measurements'].items()}
        elif name=='coverage_40':
            configs=c.read(bundle/'plans/static_pool.json')
            with (root/'analysis/configurations.csv').open() as f:
                observed={r['id']:float(r['measured_ii_ms']) for r in csv.DictReader(f)}
        else:
            configs={v['id']:v for v in c.read(bundle/'plans/configurations.json').values()}
            observed={r['id']:r['mean_ii_ms'] for r in c.read(root/'analysis/analysis.json')['cases']}
        if set(configs)!=set(observed):raise ValueError('development pool observation coverage differs')
        scores={key:score(cfg,costs) for key,cfg in configs.items()}
        metrics={}
        for variant in VARIANTS:
            ordered=sorted(configs,key=lambda key:(scores[key][variant],canonical_tuple(configs[key])))
            budget_values={str(n):max(1000/observed[k] for k in ordered[:min(n,len(ordered))]) for n in (1,5,20)}
            metrics[variant]={'mae_ms':statistics.mean(abs(scores[k][variant]/1e6-observed[k]) for k in configs),
                'first_id':ordered[0],'budget_FPS':budget_values,
                'pool_best_predicted_rank':ordered.index(min(observed,key=observed.get))+1}
        datasets.append({'dataset':name,'count':len(configs),'metrics':metrics})
        for key in configs:
            details.append({'dataset':name,'id':key,'observed_ii_ms':observed[key],
                **{v+'_predicted_ms':scores[key][v]/1e6 for v in VARIANTS}})
    result={'development_only':True,'no_fitted_coefficients':True,'datasets':datasets,
        'adopted_as_global_model':False,
        'reason':'Host guard corrects the six-case placement direction but neither new guard improves static-first selection on the broader 40-case coverage pool; combined guard misses that pool best at budget20.',
        'next_required_evidence':'controlled-execution applicability and thread-team scheduling/host-placement interaction; do not promote MAE reduction to global prediction validity'}
    c.write(output/'replay.json',result,replace=True)
    with (output/'predictions.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(details[0]));w.writeheader();w.writerows(details)
    return result

if __name__=='__main__':
    import json
    print(json.dumps(run(),ensure_ascii=False))
