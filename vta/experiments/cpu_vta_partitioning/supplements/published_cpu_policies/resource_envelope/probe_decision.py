"""Model-development decision, kept separate from frozen static scoring.

Choose the lower mean-II of two predefined weighted-model first choices.
Proceed to the already frozen Top20 union only if the chosen first choice
beats the original first choice in both reverse-order rounds. No coefficient
is fitted; this is development evidence and not an independent confirmation.
"""
from pathlib import Path
from run_resource_behavior import api,ROOT

def decide(run):
    c=api();run=Path(run).resolve();state=c.read(run/'state.json')
    if state['status']!='COMPLETE':raise ValueError('complete raw-backed probe required')
    result=c.read(run/'analysis/analysis.json');rows={r['case']:r for r in result['cases']}
    original=rows['static_first'];variants={'original':rows['weighted_original_first'],'combined':rows['weighted_guard_first']}
    winner=min(variants,key=lambda key:(variants[key]['mean_ii_ms'],key))
    gains={key:{str(i):(row['round'+str(i)+'_fps']/original['round'+str(i)+'_fps']-1)*100 for i in (0,1)}
        for key,row in variants.items()}
    from datetime import datetime
    qualifications=[d for d in c.read(run/'tasks.json') if d['kind']=='qualify_pipeline']
    last=qualifications[-1]['task_id']
    qualification_accepted=(run/'accepted'/state['boot_id']/(last+'.json')).stat().st_mtime
    freezes={name:datetime.fromisoformat(c.read(ROOT/'artifacts'/name/'frozen_selection/FREEZE.json')['frozen_at']).timestamp()
        for name in ('weighted_original_full_domain_01','weighted_envelope_full_domain_01')}
    frozen_before_performance=all(value<qualification_accepted for value in freezes.values())
    if not frozen_before_performance:raise ValueError('cannot prove both lists preceded probe performance')
    return {'chosen_variant':winner,'paired_FPS_gain_percent_vs_original':gains,
        'proceed_to_frozen_Top20_union':all(value>0 for value in gains[winner].values()),
        'criterion':'lower mean II among two predefined static first choices; winner improves baseline in both rounds',
        'independent_confirmation':False,'FPS_coefficients_fitted':False,
        'both_Top20_lists_frozen_before_probe_performance':frozen_before_performance,
        'last_qualification_receipt_mtime_utc':datetime.fromtimestamp(qualification_accepted,__import__('datetime').timezone.utc).isoformat(),
        'model_branch_selection_rule_preregistered_before_probe':False,
        'scope':'same boot two short reverse rounds; choice of model branch is part of development cost'}

if __name__=='__main__':
    import argparse,json
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);args=p.parse_args()
    result=decide(args.run_dir);api().write(args.run_dir/'analysis/MODEL_DEVELOPMENT_DECISION.json',result,replace=True)
    print(json.dumps(result,ensure_ascii=False))
