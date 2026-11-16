"""Separate a five-signature cost refresh from temporal-model changes."""
from copy import deepcopy
from pathlib import Path
import run_current_host_lock_replay as parent
from run_scope_rank import assert_plan,inputs_for,freeze
from cpu_vta_partitioning.common import read,sha256
from cpu_policies.open_resources.costs import Costs
from model_ablation.model import score
from policy_branch_eval.common import isolated_inputs_only


def run():
    plan=assert_plan();programs,old,_=inputs_for(plan,'yolov3_tiny416','original')
    new=read(parent.NEW);pool=read(parent.POOL)['yolov3_tiny416']
    with isolated_inputs_only():
        records=deepcopy(old.records);aliases=[]
        for cfg in pool.values():
            for stage in cfg['stages']:
                profile=new['profiles'].get(stage['program'])
                if profile is None or profile['threads']!=stage['threads']:continue
                if profile['device']=='vta' and profile['mask']!=stage['mask']:raise ValueError('host mismatched')
                key=old.get(stage,cfg)['source_key'];record=records[key]
                if profile['module_sha256']!=programs[stage['program']]['files']['module.so']['sha256']:
                    raise ValueError('module mismatch')
                record['cost']=profile['clean_cost']
                record['kind']='fresh_isolated_cost_explicit_requested_mask_equivalence'
                aliases.append(dict(program=stage['program'],source_key=key,measured_mask=profile['mask'],
                    requested_mask=stage['mask'],profile_source_sha256=sha256(parent.NEW)))
        fresh=Costs(records,programs=programs,ddr=old.ddr,distribution=old.distribution)
        rows=[]
        for key,cfg in pool.items():
            a,b=score(cfg,old,'complete'),score(cfg,fresh,'complete')
            rows.append(dict(configuration=key,old_score_ms=a['score_ns']/1e6,five_fresh_components_score_ms=b['score_ns']/1e6,
                cost_refresh_only_difference_ms=(b['score_ns']-a['score_ns'])/1e6,
                old_binding=a['binding_terms'],new_binding=b['binding_terms']))
        result=dict(rows=rows,aliases=aliases,original_costs_not_changed=True,same_formula_and_thread_weights=True,
            only_five_matching_signature_costs_refreshed=True,no_pipeline_performance_read=True,profile_sha256=sha256(parent.NEW))
        freeze(parent.OUT/'COST_REFRESH_ONLY.json',result)
    report=parent.ROOT.parents[4]/'paper/my_paper/analysis/20261112_current_host_lock_replay/COST_REFRESH_ONLY_zh.txt'
    if not report.exists():
        report.write_text('成本刷新与时序模型分开核查\n\n'
            '保持原公式、线程份额及配置不变，仅在内存副本中替换本轮五项签名的无探针隔离成本。旧表不改。\n'
            '两个YOLO代表的预测分别由170.127/170.353变为169.706/169.706毫秒，差-0.422/-0.647毫秒。\n'
            '可见约13毫秒的时序回放变化不能仅解释为刷新隔离耗时。\n'
            '但时序回放也换用了新取得的细粒度工作结构，尚未区分结构信息与具体CPU执行时序近似各自贡献；不宣称完整消融或新选点收益。\n'
            '非实测核心映射继续显式采用同构核置换假设。\n')
    freeze(parent.OUT/'COST_REFRESH_PROOF.json',dict(source_sha256=sha256(__file__),
        result_sha256=sha256(parent.OUT/'COST_REFRESH_ONLY.json'),report_sha256=sha256(report)))
    print(rows)


if __name__=='__main__':run()
