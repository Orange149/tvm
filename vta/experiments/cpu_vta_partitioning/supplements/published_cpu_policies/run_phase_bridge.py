#!/usr/bin/env python3
"""Two explicit same-boot original-runner bridges; no automatic repeats."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))


def main(argv=None):
    from workload_model.common import OUTPUT,ROOT,freeze,require_space
    from cpu_vta_partitioning.common import read,sha256,now,write,locked
    from phase_profile.prepare import validate
    from phase_profile.runtime import make_task,matched_ssh
    from phase_profile.campaign import accept
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute-board',action='store_true')
    args=p.parse_args(argv)
    if not args.execute_board:raise ValueError('--execute-board is required')
    require_space();output=OUTPUT/'phase_profile';run=ROOT/'runs/cross_network_phase_profile_01'
    state=read(run/'state.json')
    if state['status']!='COMPONENTS_COMPLETE':raise ValueError('bounded component batch must be complete first')
    specs=[dict(network=n,scope='component',case='cpu_00_02_t4',phase='clean',probe=False,runner='old')
           for n in ('resnet18','yolov3_tiny416')]
    freeze(output/'BRIDGE_PLAN.json',dict(purpose='separate optional-executor overhead from changes since old profile',
        specs=specs,native_sessions=2,CPU_stage_calls=20,new_DNN_inferences=0,no_reboot=True,
        no_threshold_rejection_or_automatic_remeasurement=True,
        controller_sha256=sha256(Path(__file__))))
    ssh=matched_ssh(read(OUTPUT/'board.local.json')['board'],execute_board=True)
    if ssh.boot_id()!=state['boot_id']:raise ValueError('bridge boot differs from component boot')
    results=[]
    with locked(run):
        for spec in specs:
            bundle=output/spec['network']/'capsule';validate(bundle)
            doc=make_task(bundle,spec)
            qdoc=make_task(bundle,{**spec,'runner':'phase','phase':'qualify','probe':True})
            receipt=run/'accepted'/state['boot_id']/(qdoc['task_id']+'.json')
            q=read(receipt)
            # Replay qualification before using its output fingerprints.
            from phase_profile.runtime import summary
            from cpu_vta_partitioning.common import verify_inventory
            verify_inventory(q['raw'],q['files'])
            if summary(q['raw'],qdoc,bundle)!=q['summary']:raise ValueError('bridge qualification no longer replays')
            remote=ssh.config['remote_base']+'/bundles/'+read(bundle/'manifest.json')['bundle_id']
            ssh.wait_idle(state['boot_id'])
            item=accept(ssh,doc,bundle,remote,run,state['boot_id'],q['summary']['output_fingerprints'])
            newdoc=make_task(bundle,{**spec,'runner':'phase'})
            new=read(run/'accepted'/state['boot_id']/(newdoc['task_id']+'.json'))
            results.append(dict(network=spec['network'],old_cost=item['summary']['cost'],
                new_clean_cost=new['summary']['cost'],
                new_over_old_service_ratio=new['summary']['cost']['service_ms']/item['summary']['cost']['service_ms'],
                old_receipt_sha256=sha256(run/'accepted'/state['boot_id']/(doc['task_id']+'.json'))))
            print('原程序桥接完成',spec['network'],results[-1]['new_over_old_service_ratio'],flush=True)
    result=dict(status='COMPLETE',boot_id=state['boot_id'],native_sessions=2,CPU_stage_calls=20,
                new_DNN_inferences=0,results=results,completed_at=now())
    write(run/'BRIDGE_RESULTS.json',result,replace=True)
    return result


if __name__=='__main__':print(json.dumps(main(),ensure_ascii=False))
