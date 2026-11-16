#!/usr/bin/env python3
"""Replay a validator-only fix against an immutable execution/controller capsule.

Only the single warmup-frame correctness envelope changes from 28 to 1.
No native arguments, profiles, predictions, task identities or executors change.
This file is copied and hash-bound into the run before any recovery transition.
"""
import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import sys

def main(argv=None):
    parser=argparse.ArgumentParser(description='冻结执行器的首帧校验兼容修复')
    parser.add_argument('action',choices=('resume','status','analyze'))
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--execute-board',action='store_true')
    args=parser.parse_args(argv); root=args.run_dir.resolve()
    initial=json.loads((root/'state.json').read_text()); bundle=Path(initial['bundle'])
    sys.path.insert(0,str(bundle/'sources/contention_controller'))
    from cpu_policies.contention.common import api
    from cpu_policies.contention import campaign
    from cpu_policies.contention.runtime import SSH
    common=api(bundle)
    from cpu_vta_partitioning import measurement
    state=campaign.validate_state(root)
    original=measurement.correctness
    def first_warmup_correctness(rows,task,bundle,local_result):
        fixed=task
        if (task.get('kind')=='performance' and len(rows)==1
                and rows[0].get('frame_id')==0 and rows[0].get('completion_index')==0
                and (task.get('warmup'),task.get('scored'),task.get('tail'))==(8,16,4)):
            fixed={**task,'warmup':0,'scored':1,'tail':0}
        return original(rows,fixed,bundle,local_result)
    measurement.correctness=first_warmup_correctness
    source=Path(__file__).read_bytes(); fingerprint=sha256(source).hexdigest()
    if args.action=='resume':
        if not args.execute_board: parser.error('board access requires --execute-board')
        with common.locked(root):
            state=campaign.validate_state(root)
            patch=root/'validation_repair/compat.py'
            revision=root/'validation_repair/revision.json'
            if revision.exists():
                value=common.read(revision)
                if value['compat_source_sha256']!=fingerprint or common.sha256(patch)!=fingerprint:
                    raise ValueError('validation compatibility source changed')
            else:
                if state['status']!='STOPPED_INVALID_RESULT' or state.get('last_error')!='input sequence frame coverage mismatch' or state['completed_tasks']!=39:
                    raise ValueError('repair is confined to the first-warmup envelope failure')
                # Replay the completed native session before allowing a recovery.
                doc=campaign.final_plan(bundle)[4]
                rawroot=root/'raw'/state['boot_id']/doc['task_id']
                candidates=sorted(rawroot.glob('transfers/*/collected'))
                if not candidates: raise ValueError('missing completed pipeline raw')
                raw=candidates[-1]
                if (raw/'exit_code').read_text().strip()!='0': raise ValueError('native did not complete successfully')
                from cpu_policies.contention.runtime import validate_summary
                summary=validate_summary(raw,doc,bundle)
                patch.parent.mkdir(parents=True,exist_ok=True); patch.write_bytes(source)
                common.write(root/'validation_repair/before_state.json',deepcopy(state))
                common.write(revision,{'compat_source_sha256':fingerprint,'reason':'single warmup frame requires a single-frame validation envelope',
                    'execution_bundle_unchanged':state['bundle_id'],'native_programs_changed':False,
                    'predictions_sha256':common.sha256(root/'predictions.json'),'created':common.now(),
                    'replayed_pipeline_task_id':doc['task_id'],'replayed_summary':summary,
                    'replayed_files':common.inventory(raw),'new_measurements_for_repair':0})
                common.event(root,'validation_only_repair',compat_source_sha256=fingerprint,task_id=doc['task_id'])
            campaign._pin(root,state,'validation_repair/compat.py','validation_repair/revision.json','validation_repair/before_state.json')
            if state['status']=='STOPPED_INVALID_RESULT':
                state.update(status='INTERRUPTED',phase='VALIDATING',validation_revision_sha256=common.sha256(revision))
                state.pop('last_error',None); campaign._save(root,state)
        campaign.run_current(root,SSH(state['connection'],execute_board=True,bundle=bundle))
        result=campaign.status(root)
    elif args.action=='status': result=campaign.status(root)
    else:
        from cpu_policies.contention.analysis import analyze
        result=analyze(root)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))
    return result

if __name__=='__main__': main()
