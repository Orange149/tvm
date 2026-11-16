"""Archive completed evidence, export readable lists, and replay the copy."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .common import ROOT,MODELS,settings,api

def archive(run,output):
    started=time.monotonic();run=Path(run).resolve();output=Path(output).resolve()
    c=api();state=c.read(run/'state.json')
    if state['status']!='COMPLETE':raise ValueError('only complete campaigns can be archived')
    if output.exists():raise FileExistsError('archive already exists')
    bundle=Path(state['bundle']);c=api(bundle);output.mkdir(parents=True)
    copied_bundle=output/'bundles'/bundle.name
    copied_run=output/'runs'/run.name
    shutil.copytree(bundle,copied_bundle);shutil.copytree(run,copied_run)
    original_state=(copied_run/'state.json').read_bytes()
    (copied_run/'state.original.json').write_bytes(original_state)
    relocated=c.read(copied_run/'state.json');relocated['bundle']=str(copied_bundle)
    if isinstance(relocated.get('analysis'),dict) and 'report' in relocated['analysis']:
        relocated['analysis']['report']=str(copied_run/'analysis/REPORT_zh.md')
    c.write(copied_run/'state.json',relocated,replace=True)
    controller=copied_bundle/'sources/ablation_controller'
    replay=subprocess.run([sys.executable,str(controller/'run_model_ablation.py'),'analyze','--run-dir',str(copied_run)],
        cwd=controller,capture_output=True,text=True,timeout=360,
        env=dict(os.environ,PYTHONPATH='',PYTHONDONTWRITEBYTECODE='1'))
    c.write(output/'ARCHIVE_REPLAY.json',dict(passed=replay.returncode==0,stdout=replay.stdout,stderr=replay.stderr,
        original_state_sha256=c.sha256(run/'state.json'),archived_original_state_sha256=c.sha256(copied_run/'state.original.json'),
        relocation_fields=['bundle','analysis.report'],new_measurements=0))
    if replay.returncode:raise RuntimeError(replay.stdout+replay.stderr)
    ranked=c.read(copied_bundle/'plans/selection_rankings.json')
    with (copied_run/'analysis/configurations.csv').open() as f:configs={r['id']:r for r in csv.DictReader(f)}
    selected=output/'top20';selected.mkdir()
    for key,*_ in MODELS:
        rows=[dict(model=settings(key)['label'],static_order=i+1,predicted_ii_ms=row['score_ms'],**configs[row['id']])
            for i,row in enumerate(ranked[key])]
        with (selected/(key+'.csv')).open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    from cpu_policies.open_resources.costs import Costs
    costs=Costs(c.read(copied_bundle/'plans/isolated_costs.json')['costs'],programs=c.read(copied_bundle/'plans/programs.json'),
        ddr=c.read(copied_bundle/'plans/ddr.json'))
    pool=c.read(copied_bundle/'plans/static_pool.json');stages=[]
    for receipt in (copied_run/'accepted'/state['boot_id']).glob('*.json'):
        accepted=c.read(receipt);task=accepted['task']
        if task['kind']!='performance':continue
        cfg=pool[task['candidate_id']]
        raw=copied_run/accepted['raw']/'result.jsonl'
        frames=[json.loads(line) for line in raw.read_text().splitlines() if line][8:24]
        for i,s in enumerate(cfg['stages']):
            iso=costs.get(s,cfg);wall=sum(r[f'stage{i}_ms'] for r in frames)/16
            stages.append(dict(configuration=cfg['id'],program=s['program'],device=s['device'],threads=s['threads'],
                core_mask=str(s['mask']),host_mask=str(cfg['host_mask']),gate=cfg['execution_mode'],
                isolated_service_ms=iso['service_ms'],pipeline_service_ms=wall,service_ratio=wall/iso['service_ms'],
                isolated_run_ms=iso['run_ms'],pipeline_run_ms=sum(r[f'stage{i}_run_ms'] for r in frames)/16,
                scope='same scored frame indices; overlapping stage wall times must not be summed'))
    with (output/'stage_profile_comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(stages[0]));writer.writeheader();writer.writerows(stages)
    result=c.read(copied_run/'analysis/analysis.json');methods=result['decision']['methods']
    rel=copied_run.relative_to(output)
    from .analysis import plot
    plot(output,result['decision'])
    lines=['# ResNet18三因素消融：八组Top-20共同实测','',
        '八组使用同一完整候选域、相同隔离成本与正式执行器；不新增profile，不使用旧流水FPS选点。',
        f"名单冻结后共同测量{result['costs']['configurations']}项，全部通过输出、绑核、哈希及帧完整性校验。",'',
        '|模型|一次预算FPS|五次预算FPS|二十次预算FPS|','|---|---:|---:|---:|']
    for key,*_ in MODELS:
        row=methods[key];lines.append('|'+row['label']+'|'+'|'.join(f"{row['budgets'][n]['fps']:.3f}" for n in ('1','5','20'))+'|')
    lines+=['','![消融结果](ablation_summary.png)','',
        '主效应平均另外两个开关全部四种状态下的对数FPS差；逐项删除只比较完整模型附近的变化，两者含义不同。',
        '全部结果限于单输入、单启动、短窗口；八种方法、候选与帧均不作为独立重复，不报告显著性或等效性。',
        '阶段耗时表复用正式日志，没有增加探针或板端调用；并发阶段的墙钟时间不能相加。','',
        f'[完整报告]({rel}/analysis/REPORT_zh.md) · [全部配置]({rel}/analysis/configurations.csv) · '
        f'[删项效果]({rel}/analysis/deletion_effects.csv) · [交互]({rel}/analysis/factorial_effects.csv) · [阶段耗时](stage_profile_comparison.csv)',
        '', '八份可读名单：'+ ' · '.join(f"[{settings(key)['label']}](top20/{key}.csv)" for key,*_ in MODELS),
        '', '归档仅调整运行记录中的部署包／报告路径，原state.json单独保留；任务、程序和原始测量不变。',
        '已通过归档副本完整分析回放。原775项标定及87项DDR记录的历史原始数据仍由来源索引指向旧批次，未冒称新测量。']
    (output/'README.md').write_text('\n'.join(lines)+'\n')
    validation=ROOT/'artifacts/model_ablation_01'
    for name in ('HOST_TESTS.json','RELOCATION_CHECK.json','PREPARATION_ISSUES.json'):
        if (validation/name).exists():shutil.copy2(validation/name,output/name)
    c.write(output/'EVIDENCE_INDEX.json',dict(source_run=str(run),source_bundle=str(bundle),
        archived_run=str(copied_run.relative_to(output)),archived_bundle=str(copied_bundle.relative_to(output)),
        bundle_manifest_sha256=c.sha256(copied_bundle/'manifest.json'),
        ranking_freeze_sha256=c.sha256(copied_bundle/'plans/ranking_freeze.json'),
        original_component_source=c.read(copied_bundle/'plans/component_source_provenance.json'),
        no_extra_board_measurements=True,archive_seconds=time.monotonic()-started))
    c.write(output/'ARCHIVE_INVENTORY.json',dict(files=c.inventory(output)))
    return dict(archive=str(output),report=str(output/'README.md'),raw_evidence_replayed=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(archive(a.run_dir,a.output),ensure_ascii=False))

if __name__=='__main__':main()
