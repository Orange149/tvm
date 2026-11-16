"""Freeze workload features before joining historical pipeline evidence."""
import json
from pathlib import Path
import shutil
import statistics

from cpu_vta_partitioning.common import read,sha256,verify_inventory,write
from cpu_policies.open_resources.costs import Costs
from staged_ablation.model import score
from policy_branch_eval.ranking import inputs,engine_from,replay_original
from policy_branch_eval.analysis import cohort_specs,load_measurements,csv_write
from policy_branch_eval.common import isolated_inputs_only
from .common import NETWORKS,ROOT,freeze,progress,require_space,source_identity,REPORT
from .features import isolated_features,placement_features

REPORTS=('20261109_staged_ablation','20261109_yolo_validation','20261109_yolo_cpu_policy_check',
         '20261110_yolo_wait_diagnosis','20261110_yolo_affinity_check',
         '20261110_yolo_dependency_audit','20261110_policy_branch_eval')


def audit(output,report_dir=REPORT):
    output=Path(output);space=require_space();features={};proofs={};catalogs={}
    for network in NETWORKS:
        with isolated_inputs_only():
            programs,records,ddr,hashes,root,units=inputs(network)
            costs=Costs(records,programs=programs,ddr=ddr,distribution='session_weights')
            proof=read(root/'proof.json');engine=engine_from(proof,network)
            original,_,replayed=replay_original(engine,programs,costs,ddr,hashes,root,units)
            # Final measurement capsules contain only selected candidates.
            # Features require the original full component capsule, never
            # silently omit modules absent from a selected-candidate bundle.
            if network=='resnet18':
                from host_work_ablation.common import PARENT
                bundle=PARENT
            else:
                bundle=Path(read(ROOT/'artifacts/yolo_validation_01/PREPARATION.json')['profile_bundle'])
            graphs={};graph_hashes={}
            manifest=read(bundle/'manifest.json')
            catalog_path=bundle/'plans/programs.json'
            if sha256(catalog_path)!=manifest['files']['plans/programs.json']['sha256']:
                raise ValueError('full component program catalog changed')
            feature_programs=read(catalog_path)
            for name,p in feature_programs.items():
                if p['device']!='cpu':continue
                path=bundle/'programs'/name/'graph.json'
                if sha256(path)!=p['files']['graph.json']['sha256']:
                    raise ValueError('compiled CPU graph differs from program catalog')
                graphs[name]=read(path);graph_hashes[str(path)]=sha256(path)
            feature_rows=isolated_features(network,feature_programs,records,ddr,graphs)
            features[network]=feature_rows
            freeze(output/network/'isolated_features.json',dict(features=feature_rows,
                component_hashes=hashes,graph_hashes=graph_hashes,
                full_catalog_sha256=sha256(catalog_path),pipeline_performance_used=False))
            proofs[network]=dict(unit_count=units,programs=len(feature_programs),
                search_programs=len(programs),atomic_curve_only_programs=sorted(set(feature_programs)-set(programs)),components=len(records),
                score_switches=dict(thread_weights=True,host_cpu_demand=True,team_guard=True,host_guard=False),
                implementation=proof['implementation'],original_search=replayed,
                strict_Top20_gate_counts={mode:sum(c['execution_mode']==mode for c in original)
                                         for mode in ('off','resource_group')},
                main_first=original[0]['id'],runner_sha256=next(v['sha256'] for k,v in manifest['files'].items()
                    if k.startswith('runner/')),component_hashes=hashes)
            catalogs[network]=(programs,records,ddr)
    freeze(output/'FEATURE_FREEZE.json',dict(sources=source_identity(),
        inputs={n:sha256(output/n/'isolated_features.json') for n in NETWORKS},proofs=proofs,
        pipeline_results_read_before_feature_freeze=False))
    source_reports={}
    for directory in REPORTS:
        p=ROOT.parents[4]/'paper/my_paper/analysis'/directory/'README.md'
        source_reports[str(p)]=sha256(p)
    progress(output,'inheritance',evidence=dict(proofs=proofs,historical_reports=source_reports),
        next_action='Join historical measured pools only after immutable isolated feature freeze')
    cohorts={};stage_rows=[];candidate_rows=[]
    for spec in cohort_specs():
        programs,records,ddr=catalogs[spec['network']]
        costs=Costs(records,programs=programs,ddr=ddr,distribution='session_weights')
        pool=read(spec['bundle']/'plans'/spec['pool'])
        measured,evidence=load_measurements(spec,pool)
        cohorts[spec['id']]=evidence
        for name,cfg in pool.items():
            m=measured[name];detail=score(cfg,costs,'fine_host')
            candidate_rows.append(dict(cohort=spec['id'],network=spec['network'],candidate=name,
                predicted_ii_ms=detail['score_ns']/1e6,measured_ii_ms=m['ii_ms'],FPS=m['fps'],
                relative_error=detail['score_ns']/1e6/m['ii_ms']-1,
                binding_terms=detail['binding_terms'],**placement_features(cfg,costs)))
            cpu_index=0
            for i,stage in enumerate(cfg['stages']):
                iso=costs.get(stage,cfg);p=programs[stage['program']]
                service=m['stage_service_ms'][i]
                stage_rows.append(dict(cohort=spec['id'],network=spec['network'],candidate=name,
                    program=stage['program'],device=stage['device'],stage=i,CPU_stage_index=cpu_index if stage['device']=='cpu' else None,
                    threads=stage['threads'],mask=stage['mask'],gate=cfg['execution_mode'],context=cfg['context_id'],
                    isolated_service_ms=iso['service_ms'],pipeline_service_ms=service,
                    slowdown_ratio=service/iso['service_ms'],slowdown_ms=service-iso['service_ms'],
                    isolated_run_core_ms=iso['run_core_ms'],isolated_boundary_core_ms=iso['boundary_core_ms'],
                    thread_weights=iso['cpu_core_weights'],input_tensors=len(p['input_types']),
                    output_tensors=len(p['output_types'])))
                if stage['device']=='cpu':cpu_index+=1
    write(output/'HISTORICAL_EVIDENCE.json',dict(cohorts=cohorts,
        development_replay=True,independent_new_validation=False,new_native_sessions=0),replace=True)
    write(output/'historical_candidates.json',candidate_rows,replace=True)
    csv_write(output/'historical_stages.csv',stage_rows)
    csv_write(output/'historical_candidates.csv',[{k:(json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v)
                                                   for k,v in r.items()} for r in candidate_rows])
    csv_write(output/'isolated_CPU_curves.csv',[{k:(json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v)
        for k,v in row.items() if k not in ('recognized_dense_convolutions','graph_functions')}
        for rows in features.values() for row in rows])
    progress(output,'workload_features',evidence=dict(CPU_signatures={n:len(v) for n,v in features.items()},
        historical_receipts={k:len(v['receipts']) for k,v in cohorts.items()},feature_freeze_sha256=sha256(output/'FEATURE_FREEZE.json')),
        next_action='Choose one workload-dependent mechanism from isolated features and matched historical counterexamples; freeze hypothesis before new measurements')
    render(output,features,proofs,stage_rows,candidate_rows)
    report_dir=Path(report_dir);report_dir.mkdir(parents=True,exist_ok=True)
    for leaf in ('REPORT_zh.md','FEATURE_FREEZE.json','HISTORICAL_EVIDENCE.json','PROGRESS.json',
                 'historical_stages.csv','historical_candidates.csv','isolated_CPU_curves.csv'):
        shutil.copy2(output/leaf,report_dir/leaf)
    return dict(status='AUDIT_AND_WORKLOAD_FEATURES_COMPLETE',goal_complete=False,
        CPU_signatures={n:len(v) for n,v in features.items()},historical_candidates=len(candidate_rows),
        storage=space,report=str(report_dir/'REPORT_zh.md'))


def render(output,features,proofs,stages,candidates):
    lines=['# 双网络静态模型：继承核验与工作负载特征','',
        '本阶段只核验历史证据、编译图和隔离成本，未新增板端推理，也未修改评分。Goal尚未完成。',
        '特征在读取历史流水测量之前冻结；下列回放只用于提出假设，不作为独立验证。','',
        '| 网络 | 单元 | 模块 | 成本签名 | 原Top-20开放／串行 |','|---|---:|---:|---:|---|']
    for n,p in proofs.items():
        g=p['strict_Top20_gate_counts'];lines.append(f"| {n} | {p['unit_count']} | {p['programs']} | {p['components']} | {g['off']}／{g['resource_group']} |")
    lines+=['','两网络搜索与C++枚举器源码哈希：'+('相同。' if proofs[NETWORKS[0]]['implementation']==proofs[NETWORKS[1]]['implementation'] else '不同，需要核查。'),
        '运行执行器哈希：'+('相同。' if proofs[NETWORKS[0]]['runner_sha256']==proofs[NETWORKS[1]]['runner_sha256'] else '不同，需要核查。'),
        'YOLO另含cpu_01_01、cpu_12_12两个仅用于单元线程曲线的CPU模块；全曲线包含它们，搜索仍按原合法域使用157个模块。',
        '', '## CPU前段0–2的真实曲线','',
        '| 网络 | 线程 | 服务ms | 计算核ms | 单线程加速比 | 核时相对单线程 | 输入MiB | 输出MiB | 图存储代理MiB |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for n,rows in features.items():
        for r in rows:
            if r['program']!='cpu_00_02':continue
            lines.append(f"| {n} | {r['threads']} | {r['service_ms']:.3f} | {r['run_core_ms']:.3f} | {r['isolated_speedup']:.3f} | {r['run_core_growth']:.3f} | {r['input_frontier_bytes']/2**20:.3f} | {r['output_frontier_bytes']/2**20:.3f} | {r['graph_planned_storage_proxy_bytes']/2**20:.3f} |")
    lines+=['','图存储代理根据graph.json的storage_id规划估计，包含图内参数/张量安排，不代表缓存占用、额外工作区或真实DDR流量。图函数数量不等于并行启动数量。',
        '', '## 已知反例与解释边界','',
        '- 隔离服务、运行CPU核时和流水墙钟分列，不能把并发阶段墙钟相加。',
        '- 当前只有两种不同网络、不同输入尺寸、不同阶段切图；这些因素混杂，不能直接把差异归因于残差结构或双检测头。',
        '- 固定核组及同组串行已经实现；本次不会把重新找到历史配置称为新吞吐突破。',
        '- 计算工作、主机工作与线程就绪时序的区别仍需有针对性的机制验证。',
        '- 下一步依据全阶段线程曲线、图边界/工作量特征和已知匹配反例确定有限假设；不直接拟合流水FPS系数。',
        '', '## 磁盘与执行边界','',
        'Windows C盘可用空间优先于WSL虚拟可用空间；保留至少20 GiB C盘余量和10 GiB Linux余量。复用不可变资产，紧凑保存搜索记录，不复制历史原始目录。',
        '禁止重启开发板；新执行批次使用新boot身份。旧测量只按各自来源boot引用，不冒充同批数据。',
        '', '进度见PROGRESS.json；原始凭据哈希见HISTORICAL_EVIDENCE.json；完整曲线和阶段差距见CSV。']
    (output/'REPORT_zh.md').write_text('\n'.join(lines)+'\n')
