#!/usr/bin/env python3
"""Archive the completed evidence and frozen algorithms without duplicating ELF assets."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from run_resource_behavior import ROOT,api

def archive(output,runs=None,developments=None):
    c=api();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    runs=runs or ('resnet18_resource_behavior_mechanism_01','resnet18_resource_envelope_top20_01')
    records=[]
    for name in runs:
        run=ROOT/'runs'/name;state=c.read(run/'state.json')
        if state['status']!='COMPLETE' or state.get('active_boot') is not None:
            raise ValueError('archive requires a completed idle run: '+name)
        bundle=Path(state['bundle']);kind=('behavior_controller' if (bundle/'sources/behavior_controller').is_dir()
                                          else 'envelope_controller')
        entry='run_resource_behavior.py' if kind=='behavior_controller' else 'run_envelope_top20.py'
        check=subprocess.run([sys.executable,str(bundle/'sources'/kind/entry),'analyze','--run-dir',str(run)],
            cwd=bundle/'sources'/kind,capture_output=True,text=True,timeout=180)
        if check.returncode:raise ValueError(check.stdout+check.stderr)
        dest=output/'runs'/name
        shutil.copytree(run,dest,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.lock'))
        bdest=output/'bundles'/bundle.name
        bdest.mkdir(parents=True,exist_ok=True)
        for leaf in ('manifest.json','OFFLINE_READINESS.json'):shutil.copy2(bundle/leaf,bdest/leaf)
        for folder in ('plans','sources','provenance'):
            if (bundle/folder).exists():shutil.copytree(bundle/folder,bdest/folder,dirs_exist_ok=True)
        manifest=c.read(bundle/'manifest.json')
        assets={n:v for n,v in manifest['files'].items() if n.startswith(
            ('programs/','runtime/','runner/','firmware/','inputs/','references/'))}
        c.write(bdest/'EXECUTION_ASSET_INDEX.json',{'original_capsule':str(bundle),'files':assets,
            'note':'ELF, firmware and input bytes remain in the original frozen capsule; this archive binds their hashes and retains all new raw outputs'},replace=True)
        records.append({'run_id':name,'bundle_id':bundle.name,'boot_id':state['boot_id'],
            'state_sha256':c.sha256(run/'state.json'),'manifest_sha256':c.sha256(bundle/'manifest.json'),
            'replayed_analysis_stdout':check.stdout})
    for directory in (developments or ('resource_envelope_development','resource_envelope_full_domain_01')):
        source=ROOT/'artifacts'/directory
        shutil.copytree(source,output/'development'/directory,dirs_exist_ok=True)
    from resource_envelope.cost_audit import audit
    profile_cost=audit();c.write(output/'INHERITED_PROFILE_COST.json',profile_cost,replace=True)
    for item in profile_cost['sources']:
        path=Path(item['events_path']);dest=output/'inherited_profile_events'/path.parent.name
        dest.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest/path.name)
    shutil.copy2(ROOT/'resource_envelope/cost_audit.py',output/'cost_audit.py')
    shutil.copytree(ROOT/'resource_envelope',output/'evaluation_scripts/resource_envelope',
        dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
    for path in (ROOT/'docs').glob('*_audit.json'):
        dest=output/'literature_audits';dest.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest/path.name)
    shutil.copytree(ROOT/'third_party/pipeit_reference',output/'literature_audits/pipeit_reference',dirs_exist_ok=True)
    shutil.copy2(Path(__file__),output/'archive_resource_envelope.py')
    if (ROOT/'run_thread_work_probe.py').is_file():
        shutil.copy2(ROOT/'run_thread_work_probe.py',output/'evaluation_scripts/run_thread_work_probe.py')
    c.write(output/'EVIDENCE_INDEX.json',{'runs':records,'board_rebooted':False,
        'new_component_profiles':0,'runtime_changes':False,
        'development_replay_is_not_independent_validation':True},replace=True)
    inventory={n:v for n,v in c.inventory(output).items() if n!='ARCHIVE_INVENTORY.json'}
    c.write(output/'ARCHIVE_INVENTORY.json',{'files':inventory,'file_count':len(inventory),
        'bytes':sum(v['bytes'] for v in inventory.values()),'inventory_sha256':c.digest(inventory)},replace=True)
    c.verify_inventory(output,inventory)
    return {'archive':str(output),'files':len(inventory),'bytes':sum(v['bytes'] for v in inventory.values())}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--run-id',action='append');parser.add_argument('--development',action='append')
    args=parser.parse_args();print(json.dumps(archive(args.output,args.run_id,args.development),ensure_ascii=False))
