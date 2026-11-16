"""Portable archive, original-state retention and raw-backed analysis replay."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from .common import MODELS,settings,api

def archive(run,output):
    c=api();run=Path(run).resolve();output=Path(output).resolve();state=c.read(run/'state.json')
    if state['status']!='COMPLETE':raise ValueError('only complete evidence can be archived')
    if output.exists():raise FileExistsError('archive exists')
    output.mkdir(parents=True);bundle=Path(state['bundle']);b=output/'bundles'/bundle.name;r=output/'runs'/run.name
    shutil.copytree(bundle,b);shutil.copytree(run,r)
    shutil.copy2(r/'state.json',r/'state.original.json')
    relocated=c.read(r/'state.json');relocated['bundle']=str(b)
    if relocated.get('analysis'):relocated['analysis']['report']=str(r/'analysis/REPORT_zh.md')
    c.write(r/'state.json',relocated,replace=True)
    controller=b/'sources/staged_ablation_controller'
    result=subprocess.run([sys.executable,str(controller/'run_staged_ablation.py'),'analyze','--run-dir',str(r)],
        cwd=controller,capture_output=True,text=True,timeout=360,
        env=dict(os.environ,PYTHONPATH='',PYTHONDONTWRITEBYTECODE='1'))
    c.write(output/'ARCHIVE_REPLAY.json',dict(passed=result.returncode==0,stdout=result.stdout,stderr=result.stderr,
        original_state_sha256=c.sha256(run/'state.json'),archived_original_state_sha256=c.sha256(r/'state.original.json')))
    if result.returncode:raise RuntimeError(result.stdout+result.stderr)
    for p in (r/'analysis').iterdir():
        if p.is_file():shutil.copy2(p,output/p.name)
    shutil.copy2(output/'REPORT_zh.md',output/'README.md')
    shutil.copytree(b/'provenance/evidence',output/'evidence')
    ranked=c.read(b/'plans/selection_rankings.json')
    with (output/'configurations.csv').open(encoding='utf-8-sig') as f:configs={x['id']:x for x in csv.DictReader(f)}
    for key,*_ in MODELS:
        rows=[dict(model=settings(key)['label'],static_order=i+1,predicted_ii_ms=row['score_ms'],**configs[row['id']])
            for i,row in enumerate(ranked[key])]
        path=output/'lists'/(key+'.csv');path.parent.mkdir(exist_ok=True)
        with path.open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    c.write(output/'EVIDENCE_INDEX.json',dict(source_run=str(run),source_bundle=str(bundle),
        archived_run=str(r.relative_to(output)),archived_bundle=str(b.relative_to(output)),
        frozen_manifest_sha256=c.sha256(b/'manifest.json'),ranking_freeze_sha256=c.sha256(b/'plans/ranking_freeze.json'),
        no_extra_board_measurements=True))
    c.write(output/'ARCHIVE_INVENTORY.json',dict(files=c.inventory(output)))
    return dict(archive=str(output),raw_analysis_replayed=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(archive(a.run_dir,a.output),ensure_ascii=False))

if __name__=='__main__':main()
