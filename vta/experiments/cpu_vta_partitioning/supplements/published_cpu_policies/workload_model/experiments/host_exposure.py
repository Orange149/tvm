"""Test a mode-dependent host guard; do not promote it based on FPS alone."""
from pathlib import Path

from cpu_vta_partitioning.common import read,sha256,write
from cpu_policies.open_resources.costs import Costs
from cpu_policies.open_resources import domain
from staged_ablation.model import score
from policy_branch_eval.ranking import inputs,engine_from
from policy_branch_eval.analysis import cohort_specs,load_measurements,csv_write
from policy_branch_eval.common import isolated_inputs_only
from workload_model.common import ROOT,OUTPUT,freeze,progress,require_space


def exposed(cfg):
    if cfg['execution_mode']!='off':return False
    stages=[s for s in cfg['stages'] if s['device']=='cpu']
    return any(set(a['mask'])&set(b['mask']) and max(a['threads'],b['threads'])>1
               for i,a in enumerate(stages) for b in stages[i+1:])


def predict(cfg,costs):
    original=score(cfg,costs,'fine_host')['score_ns']
    guarded=score(cfg,costs,'concentrated')['score_ns']
    return dict(original=original,concentrated=guarded,
                exposure_guard=guarded if exposed(cfg) else original)


def run(output=OUTPUT):
    output=Path(output);target=output/'hypotheses/host_exposure';require_space()
    document=dict(hypothesis='Host concentration may need a conservative guard when multithread CPU groups overlap without mutual exclusion',
        formula='score=max(original_score,max_c(D_c+H*1[c in host_mask])) iff open overlapping compute groups with at least one multiworker stage; otherwise original_score',
        origin='new conditional combination of existing project constraints; not an algorithm reproduced from the cited papers',
        literature=[dict(url='https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.ECRTS.2024.14',use='separate accelerator host CPU demand from device execution'),
                    dict(url='https://arxiv.org/abs/1903.05898',use='explicit stage compute-group organization')],
        changes_execution=False,changes_components=False,pipeline_FPS_fitting=False,
        assessment='known measured-pool developmental replay; no full-domain or held-out claim',
        preexisting_counterevidence='Earlier matched group on/off tests sometimes change FPS little; a large predicted gate effect would undermine the proposed explanation',
        decision='Report selected-point quality, prediction error, and matched gate effects separately; do not infer mechanism validity from a better picked configuration',
        sources={str(p.relative_to(ROOT)):sha256(p) for p in
            [Path(__file__),ROOT/'run_workload_hypothesis.py',ROOT/'tests/test_workload_hypothesis.py']},
        parent_feature_freeze_sha256=sha256(output/'FEATURE_FREEZE.json'))
    freeze(target/'HYPOTHESIS.json',document)
    predictions={};pools={};engines={}
    with isolated_inputs_only():
        for spec in cohort_specs():
            programs,records,ddr,_,root,_=inputs(spec['network'])
            costs=Costs(records,programs=programs,ddr=ddr,distribution='session_weights')
            pool=read(spec['bundle']/'plans'/spec['pool']);pools[spec['id']]=pool
            engine=engine_from(read(root/'proof.json'),spec['network']);engines[spec['id']]=engine
            table={name:predict(cfg,costs) for name,cfg in pool.items()}
            lists={key:sorted(pool,key=lambda n:engine._row_key({**pool[n],'score_ns':table[n][key]}))[:20]
                   for key in ('original','concentrated','exposure_guard')}
            predictions[spec['id']]=dict(scores=table,lists=lists,
                pool_sha256=sha256(spec['bundle']/'plans'/spec['pool']))
        freeze(target/'PREDICTIONS.json',dict(hypothesis_sha256=sha256(target/'HYPOTHESIS.json'),cohorts=predictions,
            pipeline_performance_read=False))
    result={};rows=[];pairs=[]
    for spec in cohort_specs():
        key=spec['id'];pool=pools[key];measured,evidence=load_measurements(spec,pool)
        selected={}
        for method,names in predictions[key]['lists'].items():
            choices={}
            for budget in (1,5,20):
                if len(names)<budget:continue
                best=min(names[:budget],key=lambda n:(measured[n]['ii_ms'],names.index(n)))
                choices[str(budget)]=dict(id=best,FPS=measured[best]['fps'],ii_ms=measured[best]['ii_ms'])
            errors=[abs(predictions[key]['scores'][name][method]/1e6/measured[name]['ii_ms']-1) for name in pool]
            selected[method]=dict(budgets=choices,mean_absolute_relative_error=sum(errors)/len(errors))
            for budget,choice in choices.items():
                rows.append(dict(cohort=key,method=method,budget=budget,**choice,
                                 mean_absolute_relative_error=selected[method]['mean_absolute_relative_error']))
        for name,cfg in pool.items():
            if cfg['execution_mode']!='resource_group':continue
            off=domain.candidate(cfg['stages'],cfg['context_id'],'off')
            if off['id'] not in pool:continue
            p_on=predictions[key]['scores'][name];p_off=predictions[key]['scores'][off['id']]
            pairs.append(dict(cohort=key,on=name,off=off['id'],
                measured_off_vs_on_II_ratio=measured[off['id']]['ii_ms']/measured[name]['ii_ms'],
                original_off_vs_on_II_ratio=p_off['original']/p_on['original'],
                exposure_off_vs_on_II_ratio=p_off['exposure_guard']/p_on['exposure_guard'],
                affected=exposed(off)))
        result[key]=dict(methods=selected,receipt_hashes=evidence['receipts'])
    write(target/'RESULTS.json',dict(cohorts=result,matched_pairs=pairs,known_pool_bias=True,
        new_board_sessions=0,root_cause_proven=False,adopted=False),replace=True)
    csv_write(target/'selection_and_error.csv',rows);csv_write(target/'matched_gate_pairs.csv',pairs)
    lines=['# 条件化主机集中约束：开发回放','',
        '现有约束的条件化组合，不是新实现主机CPU计费。只在开放重叠计算时启用已有集中项；没有参数拟合、成本回填、完整搜索或新板测。',
        '已知池回放不能代表全域Top-20。选点、误差与机制配对分别判断。','',
        '| 批次 | 评分 | 一次FPS | 五次FPS | 二十次FPS | 平均绝对相对误差 |','|---|---|---:|---:|---:|---:|']
    labels=dict(original='原主模型',concentrated='集中项始终开启',exposure_guard='仅开放重叠时开启集中项')
    for cohort,value in result.items():
        for method,v in value['methods'].items():
            cells=[f"{v['budgets'][n]['FPS']:.4f}" if n in v['budgets'] else '预算不足' for n in ('1','5','20')]
            lines.append('| '+cohort+' | '+labels[method]+' | '+' | '.join(cells)+f" | {100*v['mean_absolute_relative_error']:.2f}% |")
    lines+=['','## 同分配、仅门控改变的匹配反例','',
        '| 批次 | 实测关闭／开启II | 原评分关闭／开启 | 条件评分关闭／开启 |','|---|---:|---:|---:|']
    for p in pairs:lines.append(f"| {p['cohort']} | {p['measured_off_vs_on_II_ratio']:.4f} | {p['original_off_vs_on_II_ratio']:.4f} | {p['exposure_off_vs_on_II_ratio']:.4f} |")
    lines+=['','这项回放不自动批准新模型。若预测门控变化远大于实测，说明按执行模式判断主机集中可能只是选点代理，尚不能作为慢化机制解释。']
    (target/'REPORT_zh.md').write_text('\n'.join(lines)+'\n')
    progress(output,'hypothesis',evidence=dict(name='conditional_host_concentration',
        frozen_hypothesis_sha256=sha256(target/'HYPOTHESIS.json'),result=str(target/'RESULTS.json'),
        status='DEVELOPMENT_REPLAY_NOT_ADOPTED'),
        next_action='Review matched gate contradictions before spending a full-domain search or board budget')
    return dict(report=str(target/'REPORT_zh.md'),cohorts={k:v['methods'] for k,v in result.items()},matched_pairs=len(pairs))
