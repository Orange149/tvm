"""Developmental replay, with explicit missing coverage and separate cohorts."""
import csv
import json
import math
from pathlib import Path
import shutil
import statistics

from cpu_vta_partitioning.common import digest, read, sha256, verify_inventory, write
from cpu_policies.open_resources import domain
from cpu_policies.open_resources.costs import Costs
from staged_ablation.model import score
from .common import ROOT, freeze, require_ranked
from .selection import fixed_family, mixed

LABELS = {'original':'原完整域排序', 'fixed':'固定核组分支', 'mixed':'双分支各取10项'}


def budgets(rows, measured):
    result = {}
    for n in (1,5,20):
        if len(rows) < n:
            continue
        ids = [r['id'] for r in rows[:n]]
        missing = [name for name in ids if name not in measured]
        known = [name for name in ids if name in measured]
        best = min(known,key=lambda name:(measured[name]['ii_ms'],ids.index(name))) if known else None
        result[str(n)] = dict(covered=len(known), required=n, missing=missing,
            complete=not missing,
            # Partial observations are not a measured budget result.
            best_FPS=measured[best]['fps'] if best and not missing else None,
            best_id=best if not missing else None,
            observed_subset_best_FPS=measured[best]['fps'] if best else None)
    return result


def cohort_specs():
    return [dict(id='resnet_staged_short',network='resnet18',
        run=ROOT/'runs/resnet18_staged_ablation_01',
        bundle=ROOT/'artifacts/bundles/e542003fc5a64fb17ee1e751',
        pool='static_pool.json',window=(8,16,4),validator='resnet',expected=116),
      dict(id='yolo_validation_short',network='yolov3_tiny416',
        run=ROOT/'runs/yolov3_tiny416_validation_01/executions/5bc79220-c082-4068-b524-c1e6b223bb98/pipeline',
        bundle=ROOT/'artifacts/yolo_validation_01/pipeline_bundle',
        pool='pool.json',window=(8,16,4),validator='yolo',expected=60),
      dict(id='yolo_policy_long',network='yolov3_tiny416',
        run=ROOT/'runs/yolov3_tiny416_cpu_policy_check_01',
        bundle=ROOT/'artifacts/yolo_cpu_policy_check_01/pipeline_bundle',
        pool='pool.json',window=(8,64,32),validator='yolo',expected=8)]


def validate_rows(rows, task):
    count = sum(task[k] for k in ('warmup','scored','tail'))
    if len(rows) != count:
        raise ValueError('incomplete frame window')
    for i,row in enumerate(rows):
        if (type(row.get('frame_id')) is not int or type(row.get('completion_index')) is not int
            or row['frame_id']!=i or row['completion_index']!=i or row.get('input_index')!=0):
            raise ValueError('duplicate, missing or incorrect frame/input')
    times=[r['completion_ms'] for r in rows]
    if any(not math.isfinite(t) for t in times) or any(b<=a for a,b in zip(times,times[1:])):
        raise ValueError('invalid completion timestamps')
    first=task['warmup']; last=first+task['scored']-1
    return (times[last]-times[first])/(task['scored']-1)


def load_measurements(spec, pool):
    """Replay original raw validators; never run their transport or analyzers."""
    run,bundle=spec['run'],spec['bundle'];state=read(run/'state.json')
    if state['status']!='COMPLETE':
        raise ValueError('historical cohort is not complete')
    boot=state['boot_id']; manifest=read(bundle/'manifest.json')
    if digest({k:v for k,v in manifest.items() if k!='bundle_id'})[:24]!=manifest['bundle_id']:
        raise ValueError('source bundle identity changed')
    verify_inventory(bundle,{n:v for n,v in manifest['files'].items()
        if n.startswith(('plans/','references/','runtime/','runner/'))})
    if spec['validator']=='resnet':
        from cpu_policies.open_resources.runtime import validate_summary
        # The frozen staged controller adds only this derived FPS field to
        # the shared raw validator (staged_ablation/runtime.py, lines 26-39).
        def validate(raw,task,bundle):
            result=validate_summary(raw,task,bundle)
            if task['kind']=='performance':
                result['fps']=1000/result['ii_ms']
            return result
    else:
        from run_yolo_validation_gate_compat import adapted_summary
        from yolo_validation.pipeline import work_boundaries
        base=adapted_summary()[0]
        def validate(raw,task,bundle):
            result=base(raw,task,bundle)
            if task['phase']!='qualify':
                result['window_work']=work_boundaries([json.loads(s) for s in (Path(raw)/'result.jsonl').read_text().splitlines()],task)
            return result
    measured,qualified,receipts={},set(),{}
    for path in sorted((run/'accepted'/boot).glob('*.json')):
        item=read(path);task=item['task'];name=task['candidate_id']
        qualification=task['kind'].startswith('qualify')
        if not qualification and (task['warmup'],task['scored'],task['tail'])!=spec['window']:
            continue  # Long follow-up sessions are a different cohort.
        if name not in pool or domain.configuration(task['configuration'])!=domain.configuration(pool[name]):
            raise ValueError('receipt execution differs from frozen source pool')
        if item['boot_id']!=boot or task['bundle_id']!=manifest['bundle_id']:
            raise ValueError('receipt boot/bundle differs')
        if task['executor_sha256'] not in [v['sha256'] for v in manifest['files'].values()]:
            raise ValueError('executor is not bound by source manifest')
        raw=Path(item['raw']); relative=not raw.is_absolute()
        raw=run/raw if relative else raw
        verify_inventory(run if relative else raw,item['files'])
        if ((raw/'boot_id').read_text().strip()!=boot or (raw/'task_hash').read_text().strip()!=digest(task)):
            raise ValueError('raw boot/task hash differs')
        calculated=validate(raw,task,bundle)
        if calculated!=item['summary']:
            raise ValueError('original accepted summary does not replay')
        receipts[str(path)]=sha256(path)
        if qualification:
            if name in qualified or not calculated['correct']:
                raise ValueError('duplicate or incorrect qualification')
            qualified.add(name)
        else:
            if name in measured:
                raise ValueError('duplicate performance session in one cohort')
            rows=[json.loads(s) for s in (raw/'result.jsonl').read_text().splitlines()]
            ii=validate_rows(rows,task)
            if abs(ii-calculated['ii_ms'])>1e-8:
                raise ValueError('FPS does not reconstruct from retained timestamps')
            measured[name]=dict(ii_ms=ii,fps=1000/ii,raw=str(raw),
                stage_service_ms=[statistics.mean(r[f'stage{i}_ms'] for r in rows[task['warmup']:task['warmup']+task['scored']])
                                  for i in range(len(task['stages']))])
    if set(measured)!=set(pool) or qualified!=set(pool) or len(pool)!=spec['expected']:
        raise ValueError('source cohort coverage differs from declared pool')
    return measured,dict(boot_id=boot,window=list(spec['window']),configurations=len(pool),
        receipts=receipts,source_manifest_sha256=sha256(bundle/'manifest.json'),raw_validators_replayed=True)


def pool_rank(spec):
    """Within-pool selection also uses components only; no FPS participates."""
    from .ranking import inputs,engine_from
    programs,records,ddr,hashes,root,units=inputs(spec['network'])
    engine=engine_from(read(root/'proof.json'),spec['network'])
    costs=Costs(records,programs=programs,ddr=ddr,distribution='session_weights')
    pool=read(spec['bundle']/'plans'/spec['pool'])
    rows=[]
    for name,cfg in pool.items():
        domain.validate_candidate(cfg,programs,unit_count=units)
        rows.append({**domain.configuration(cfg),'id':name,'score_ns':score(cfg,costs,'fine_host')['score_ns']})
    original=sorted(rows,key=engine._row_key)
    fixed=[r for r in original if fixed_family(r)]
    blend,reasons=mixed(original,fixed)
    return pool,dict(lists={'original':original[:20],'fixed':fixed[:20],'mixed':blend},
                     mixed_reasons=reasons,source_pool_sha256=sha256(spec['bundle']/'plans'/spec['pool']),
                     known_pool_bias=True,component_hashes=hashes,performance_used=False)


def csv_write(path, rows):
    if not rows: return
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def analyze(output, report_dir):
    output=Path(output);require_ranked(output)
    # Freeze every known-pool ranking before opening any performance receipt.
    from .common import isolated_inputs_only
    with isolated_inputs_only():
        preparations={s['id']:pool_rank(s) for s in cohort_specs()}
        for spec in cohort_specs():
            if (preparations[spec['id']][1]['component_hashes']
                    !=read(output/spec['network']/'rankings.json')['component_hashes']):
                raise ValueError('pool replay and full ranking use different component costs')
        freeze(output/'POOL_RANK_FREEZE.json',dict(parent_sha256=sha256(output/'RANK_FREEZE.json'),
            cohorts={key:value[1] for key,value in preparations.items()}))
    cohorts={}; coverage_rows=[]; replay_rows=[]; candidate_rows=[]
    for spec in cohort_specs():
        pool,pool_lists=preparations[spec['id']]
        measured,evidence=load_measurements(spec,pool)
        full=read(output/spec['network']/'rankings.json')['lists']
        full_budgets={key:budgets(rows,measured) for key,rows in full.items()}
        replay={key:budgets(rows,measured) for key,rows in pool_lists['lists'].items()}
        cohorts[spec['id']]=dict(network=spec['network'],full_domain=full_budgets,
            known_pool_replay=replay,evidence=evidence,measurements=measured)
        for kind,table,target in [('full_domain',full_budgets,coverage_rows),('known_pool',replay,replay_rows)]:
            for key,values in table.items():
                for n,value in values.items():
                    target.append(dict(cohort=spec['id'],scope=kind,method=key,budget=n,
                        covered=value['covered'],required=value['required'],complete=value['complete'],
                        best_FPS=value['best_FPS'],best_id=value['best_id'],
                        observed_subset_best_FPS=value['observed_subset_best_FPS'],missing=';'.join(value['missing'])))
        for key,rows in full.items():
            for i,cfg in enumerate(rows,1):
                m=measured.get(cfg['id'])
                candidate_rows.append(dict(cohort=spec['id'],network=spec['network'],method=key,rank=i,
                    candidate=cfg['id'],fixed_family=fixed_family(cfg),prediction_ms=cfg['score_ns']/1e6,
                    measured_FPS=m['fps'] if m else None,measured_ii_ms=m['ii_ms'] if m else None,
                    context=cfg['context_id'],gate=cfg['execution_mode'],
                    configuration=' → '.join(s['program']+(' t'+str(s['threads'])+' '+str(s['mask']) if s['device']=='cpu' else '')
                                               for s in cfg['stages'])))
    result=dict(scope='offline known-data developmental replay; separate historical cohorts',
        new_board_sessions=0,new_inferences=0,score_formula_changed=False,
        selection_changed=True,full_domain_top20_not_jointly_validated=True,cohorts=cohorts)
    write(output/'RESULTS.json',result,replace=True)
    csv_write(output/'coverage_and_budgets.csv',coverage_rows)
    csv_write(output/'known_pool_replay.csv',replay_rows)
    csv_write(output/'top20_with_available_measurements.csv',candidate_rows)
    plots(output,result)
    report(output,result)
    report_dir=Path(report_dir);report_dir.mkdir(parents=True,exist_ok=True)
    for leaf in ('README.md','RESULTS.json','RANK_FREEZE.json','POOL_RANK_FREEZE.json',
                 'coverage_and_budgets.csv','known_pool_replay.csv','top20_with_available_measurements.csv'):
        shutil.copy2(output/leaf,report_dir/leaf)
    for network in ('resnet18','yolov3_tiny416'):
        shutil.copy2(output/network/'rankings.json',report_dir/(network+'_rankings.json'))
    for path in output.glob('*.png'):
        shutil.copy2(path,report_dir/path.name)
    for path in output.glob('*.svg'):
        shutil.copy2(path,report_dir/path.name)
    return dict(report=str(report_dir/'README.md'),cohorts={key:dict(full_domain=v['full_domain'],
        known_pool_replay=v['known_pool_replay']) for key,v in cohorts.items()})


def report(output,result):
    lines=['# 固定核组分支：离线静态选点评估','',
        '修改内容：保留原完整域评分与Top-20，新增至少两个不重叠计算核组的严格Top-20；混合名单按原分支、固定组分支交替取项，各争取10个不同配置，重复项跳过，不足时补齐。',
        '固定核组沿用Pipe-it的资源分组思想，采用本项目的同组串行执行。没有修改核时、主机、DDR或线程组评分公式，没有连接开发板，也没有新增测量。',
        '这是已知数据上的开发回放。相同方法规则用于两个网络，旧FPS不参与新名单生成；不同历史批次分别分析，不拼接提升。','',
        '## 完整域搜索结果','',
        '| 网络 | 原域组合 | 固定组组合 | 两份Top-20重合 | 原域第一名预测ms | 固定组第一名预测ms |',
        '|---|---:|---:|---:|---:|---:|']
    for network in ('resnet18','yolov3_tiny416'):
        ranked=read(output/network/'rankings.json');a=ranked['lists']['original'];b=ranked['lists']['fixed']
        lines.append(f"| {network} | {ranked['original_proof']['count']:,} | {ranked['fixed_proof']['count']:,} | {len({r['id'] for r in a}&{r['id'] for r in b})}/20 | {a[0]['score_ns']/1e6:.3f} | {b[0]['score_ns']/1e6:.3f} |")
    lines+=['','## 新名单的已有测量覆盖','',
        '只有相应前缀全部在同一个历史批次测过，才报告该预算的最佳FPS。否则显示“未齐备”，已知子集最佳值仅保留在CSV中。','',
        '| 历史批次 | 名单 | 预算1：覆盖／FPS | 预算5：覆盖／FPS | 预算20：覆盖／FPS |','|---|---|---|---|---|']
    for key,c in result['cohorts'].items():
        for method,values in c['full_domain'].items():
            cells=[]
            for n in ('1','5','20'):
                v=values[n];text=f"{v['covered']}/{n}；"+(f"{v['best_FPS']:.4f}" if v['complete'] else '未齐备')
                cells.append(text)
            lines.append('| '+key+' | '+LABELS[method]+' | '+' | '.join(cells)+' |')
    lines+=['','## 仅限已测池的开发回放','',
        '以下把候选限制在各历史批次已经测过的池中，再用相同组件评分和分支规则选点。该池本身带有旧选点偏差，不能当作新完整域Top-20的板测结果。固定组不足20项时不补造配置；不足预算显示“无该预算”。','',
        '| 历史批次 | 名单 | 可选项数 | 预算1 FPS | 预算5 FPS | 预算20 FPS |','|---|---|---:|---:|---:|---:|']
    pool_freeze=read(output/'POOL_RANK_FREEZE.json')
    for key,c in result['cohorts'].items():
        for method,values in c['known_pool_replay'].items():
            cells=[f"{values[n]['best_FPS']:.4f}" if n in values else '无该预算' for n in ('1','5','20')]
            length=len(pool_freeze['cohorts'][key]['lists'][method])
            lines.append('| '+key+' | '+LABELS[method]+f' | {length} | '+' | '.join(cells)+' |')
    lines+=['','## 结论与边界','',
        '固定组分支可以保留原评分排序靠后的资源组织方式。它解决候选名额被同类方案占满的问题，不等于解决开放共享配置的隔离／流水耗时偏差。',
        '若两个网络结果不同，应保留两种资源策略的适用范围；不能将固定组设为所有网络唯一默认，也不能据此宣称模型预测已经准确。',
        '完整域的新Top-20尚有未测项时，二十项预算提升仍然未知。此次不自动启动板测。后续如需验证，优先测新旧名单差集，并在同一批次保留共同参照。',
        '', '静态名单、逐项配置和部分实测对应见两个网络rankings.json及top20_with_available_measurements.csv；原始凭据哈希及重放信息见RESULTS.json。']
    lines+=['','## 同批历史数据中的静态第一名','',
        '这里的两个第一名均由隔离组件成本产生，不使用流水FPS选点。各网络只在指定的同一个历史批次内比较；这是开发回放，不是新实验。','']
    for key in ('resnet_staged_short','yolo_policy_long'):
        values=result['cohorts'][key]['full_domain']
        a=values['original']['1'];b=values['fixed']['1']
        if a['complete'] and b['complete']:
            gain=100*(b['best_FPS']/a['best_FPS']-1)
            lines.append(f"- {key}：原静态第一名{a['best_FPS']:.4f} FPS，固定组分支第一名{b['best_FPS']:.4f} FPS，差距{gain:+.2f}%。这是整个配置的差异，不能全部归因于同组串行。")
    lines+=['','![历史同批第一名对照](static_first_replay.png)',
        '', '![静态预测及已有测量覆盖](static_scores_and_coverage.png)', '',
        '混合名单保留原第一项，因此预算一次的性能不会变化；它将固定组第一名提前到第二项。不能将固定组分支第一名的收益称为混合名单预算一次的收益。']
    (output/'README.md').write_text('\n'.join(lines)+'\n')


def plots(output,result):
    """Standalone figures; only matched historical results, never imputed FPS."""
    import os
    os.environ.setdefault('MPLCONFIGDIR','/tmp/cpu-policy-branch-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font=Path('/usr/share/fonts/truetype/arphic-gbsn00lp/gbsn00lp.ttf')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family']=['AR PL SungtiL GB','DejaVu Sans']
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,key,title in zip(axes,('resnet_staged_short','yolo_policy_long'),('ResNet18：历史短窗口','YOLO：历史匹配长窗口')):
        values=result['cohorts'][key]['full_domain']
        heights=[values[m]['1']['best_FPS'] for m in ('original','fixed')]
        if any(h is None for h in heights): continue
        bars=ax.bar(['原静态第一名','固定组第一名'],heights,color=['#74899d','#278b80'])
        ax.bar_label(bars,fmt='%.3f');ax.set_ylim(0,max(heights)*1.17)
        ax.set_title(title);ax.set_ylabel('FPS（同批历史数据回放）')
    fig.tight_layout()
    for ext in ('png','svg'): fig.savefig(output/('static_first_replay.'+ext),dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,network,cohort in zip(axes,('resnet18','yolov3_tiny416'),('resnet_staged_short','yolo_policy_long')):
        ranked=read(output/network/'rankings.json')['lists'];measured=result['cohorts'][cohort]['measurements']
        for method,color in (('original','#74899d'),('fixed','#278b80')):
            rows=ranked[method];values=[r['score_ns']/1e6 for r in rows]
            ax.plot(range(1,21),values,color=color,label=LABELS[method])
            known=[i for i,r in enumerate(rows) if r['id'] in measured]
            ax.scatter([i+1 for i in known],[values[i] for i in known],color=color,s=24)
        ax.set_title(network+'：圆点为本批已有测量')
        ax.set_xlabel('分支内静态名次');ax.set_ylabel('预测完成间隔（ms）');ax.legend()
    fig.tight_layout()
    for ext in ('png','svg'): fig.savefig(output/('static_scores_and_coverage.'+ext),dpi=160)
    plt.close(fig)
