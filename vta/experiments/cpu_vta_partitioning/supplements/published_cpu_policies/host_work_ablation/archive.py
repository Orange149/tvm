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
    controller=copied_bundle/'sources/host_ablation_controller'
    replay=subprocess.run([sys.executable,str(controller/'run_host_work_ablation.py'),'analyze','--run-dir',str(copied_run)],
        cwd=controller,capture_output=True,text=True,timeout=360,
        env=dict(os.environ,PYTHONPATH='',PYTHONDONTWRITEBYTECODE='1'))
    c.write(output/'ARCHIVE_REPLAY.json',dict(passed=replay.returncode==0,stdout=replay.stdout,stderr=replay.stderr,
        original_state_sha256=c.sha256(run/'state.json'),archived_original_state_sha256=c.sha256(copied_run/'state.original.json'),
        relocation_fields=['bundle','analysis.report'],new_measurements=0))
    if replay.returncode:raise RuntimeError(replay.stdout+replay.stderr)
    ranked=c.read(copied_bundle/'plans/selection_rankings.json')
    with (copied_run/'analysis/configurations.csv').open(encoding='utf-8-sig') as f:
        configs={r['id']:r for r in csv.DictReader(f)}
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
    result=c.read(copied_run/'analysis/analysis.json')
    for source in (copied_run/'analysis').iterdir():
        if source.is_file():shutil.copy2(source,output/source.name)
    shutil.copy2(copied_run/'analysis/REPORT_zh.md',output/'README.md')
    with (output/'README.md').open('a') as stream:
        stream.write('\n三份严格Top-20：'+' · '.join(f"[{settings(key)['label']}](top20/{key}.csv)" for key,*_ in MODELS)+'\n')
        stream.write('\n归档保留原始state.original.json，仅迁移部署包和报告路径；复制后的原始凭据已完整分析回放。\n')
    validation=ROOT/'artifacts/host_work_ablation_01'
    for name in ('HOST_TESTS.json','RELOCATION_CHECK.json','PREPARATION_ISSUES.json','OLD_CAPSULE_COMPATIBILITY.json','PRE_IMPLEMENTATION_IDENTITIES.json'):
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
