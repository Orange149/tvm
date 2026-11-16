"""Prospective verdict for a frozen three-item pool per network, never Top-20."""
from pathlib import Path

from cpu_vta_partitioning.common import read,sha256,verify_inventory
from phase_profile.runtime import make_task,summary
from policy_branch_eval.analysis import csv_write
from workload_model.common import freeze
from .campaign import specs


def analyze(output,run_dir):
    output,run_dir=Path(output).resolve(),Path(run_dir).resolve()
    state=read(run_dir/'state.json')
    if state['status']!='PIPELINES_COMPLETE':raise ValueError('all frozen holdout pipelines must complete')
    tasks=specs(output,'pipelines');predictions=read(output/'PREDICTIONS.json')
    source={};measured={};rows=[]
    for spec in tasks:
        bundle=output/spec['network']/'capsule';doc=make_task(bundle,spec)
        path=run_dir/'accepted'/state['boot_id']/(doc['task_id']+'.json');item=read(path)
        if item['task']!=doc or item['boot_id']!=state['boot_id']:raise ValueError('holdout task or boot differs')
        verify_inventory(item['raw'],item['files'])
        if summary(item['raw'],doc,bundle)!=item['summary']:raise ValueError('holdout evidence does not replay')
        source[str(path)]=sha256(path)
        if spec['phase']=='clean':measured[spec['network'],spec['case']]=item['summary']
    if len(measured)!=6:raise ValueError('all six frozen configurations are required')
    findings={}
    for network,cases in predictions['networks'].items():
        choices={};results={}
        for method,ranking in predictions['rankings'][network].items():
            first=measured[network,ranking[0]]
            choices[method]=dict(case=ranking[0],FPS=first['fps'],II_ms=first['ii_ms'])
            results[method]={str(b):max(measured[network,c]['fps'] for c in ranking[:b]) for b in (1,2,3)}
        proposed=choices['interleaved_before'];baseline=choices['fresh_capacity']
        findings[network]=dict(choices=choices,budget_FPS=results,
            primary_vs_refreshed_capacity_percent=100*(proposed['FPS']/baseline['FPS']-1),
            host_order_changes_choice=proposed['case']!=choices['interleaved_after']['case'],
            best_in_pool=max(cases,key=lambda c:measured[network,c]['fps']))
        for case,value in cases.items():
            performance=measured[network,case];row=dict(network=network,case=case,
                configuration_id=value['configuration']['id'],measured_II_ms=performance['ii_ms'],FPS=performance['fps'],
                old_capacity_ms=value['old_capacity']['score_ns']/1e6,
                fresh_capacity_ms=value['fresh_capacity']['score_ns']/1e6,
                phase_services_ms=performance['stage_service_ms'],
                process_CPU_ms_per_completion=performance['process_core_ms_per_completion'])
            for name,result in value['simulations'].items():
                row[name+'_predicted_ms']=result['score_ns']/1e6
                row[name+'_absolute_percentage_error']=100*abs(result['score_ns']/1e6/performance['ii_ms']-1)
            row['fresh_capacity_absolute_percentage_error']=100*abs(row['fresh_capacity_ms']/performance['ii_ms']-1)
            rows.append(row)
    changes=[f['primary_vs_refreshed_capacity_percent'] for f in findings.values()]
    uncertain=any(f['host_order_changes_choice'] for f in findings.values())
    pass_merit=all(c>=0 for c in changes) and any(c>0 for c in changes) and not uncertain
    result=dict(status='PROSPECTIVE_SMALL_POOL_COMPLETE',findings=findings,rows=rows,
        predeclared_selection_rule_passed=pass_merit,host_order_uncertain=uncertain,
        native_sessions=24,CPU_stage_calls=84,DNN_inferences=174,new_compilations=0,
        single_boot=True,boot_id=state['boot_id'],unseen_scope='resource combinations absent from audited Goal development pool; cuts remain known',
        source_receipts=source,prediction_seal_sha256=sha256(output/'PREDICTION_FREEZE.json'),
        full_domain_Top20_done=False,statistical_significance_claimed=False,goal_complete=False)
    freeze(output/'RESULTS.json',result)
    csv_write(output/'holdout_pipelines.csv',rows)
    return dict(status=result['status'],predeclared_selection_rule_passed=pass_merit,
        findings=findings,goal_complete=False)
