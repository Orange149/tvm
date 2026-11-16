#!/usr/bin/env python3
"""Finish the approved independent boots, then explore Q2, without overlap.

This controller waits for the already-running first formal campaign. It never
repeats an unknown native attempt or silently counts a same-boot repeat as a
new independent unit. Every child invocation has a separate immutable log.
"""
import argparse,json,subprocess,sys,time
from pathlib import Path
from c2_yolo224_prepare import HERE,write,sha


def step(master,name,arguments,timeout=5400):
    status=master/'orchestration'/(name+'.json')
    if status.exists():
        previous=json.loads(status.read_text())
        if previous['returncode']!=0:
            boot=master/'boots'/name.split('_')[0];recovered=boot/'reboot_resume_complete.json'
            if name.endswith('_reboot') and recovered.exists():
                proof=json.loads(recovered.read_text())
                if proof['additional_reboot_issued'] or proof['reboot_complete_sha256']!=sha(boot/'reboot_complete.json'):
                    raise ValueError('reconnect recovery identity differs')
                marker=master/'orchestration'/(name+'_recovered.json')
                if not marker.exists():write(marker,{'original_failure_retained':True,'recovery':str(recovered),
                    'recovery_sha256':sha(recovered),'reboot_not_repeated':True})
                return
            raise RuntimeError('previous orchestration step failed: '+name)
        return
    logfile=status.with_suffix('.log')
    if logfile.exists():raise RuntimeError('previous step has unknown completion: '+name)
    command=[sys.executable,*map(str,arguments)]
    print(json.dumps({'starting':name,'argv':command}),flush=True)
    start=time.time()
    with logfile.open('x') as stream:
        try:result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=timeout)
        except subprocess.TimeoutExpired:
            write(status,{'argv':command,'returncode':-1,'completion_unknown':True,'seconds':time.time()-start})
            raise
    write(status,{'argv':command,'returncode':result.returncode,'seconds':time.time()-start})
    if result.returncode:raise RuntimeError('orchestration step failed; inspect '+str(logfile))
    print(json.dumps({'completed':name}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--master',type=Path,required=True)
    p.add_argument('--host',default='root@192.168.1.247');a=p.parse_args();master=a.master.resolve()
    (master/'orchestration').mkdir(exist_ok=True)
    deadline=time.monotonic()+7200
    while not (master/'formal_complete.json').exists():
        log=master/'formal_campaign.log'
        if 'Traceback (most recent call last)' in log.read_text():raise RuntimeError('first formal campaign failed')
        if time.monotonic()>deadline or time.time()-log.stat().st_mtime>600:
            raise RuntimeError('first formal campaign completion unknown; do not reboot')
        time.sleep(30)
    previous=master
    for index in (2,3):
        target=master/'boots'/f'boot{index:02d}'
        step(master,f'boot{index:02d}_reboot',[HERE/'c2_yolo224_reboot.py','--master',master,
            '--previous',previous,'--out',target,'--host',a.host],timeout=600)
        step(master,f'boot{index:02d}_qualification',[HERE/'c2_yolo224_campaign.py','boot_check',
            '--master',master,'--out',target,'--host',a.host])
        step(master,f'boot{index:02d}_formal',[HERE/'c2_yolo224_campaign.py','formal',
            '--master',master,'--out',target,'--host',a.host])
        previous=target
    # Q2 runs after all three main/lease formal campaigns have completed.
    step(master,'boot03_Q2',[HERE/'c2_yolo224_campaign.py','q2',
        '--master',master,'--out',previous,'--host',a.host])
    if not (master/'orchestration_complete.json').exists():
        write(master/'orchestration_complete.json',{'three_independent_formal_campaigns_complete':True,
            'q2_exploratory_boot':str(previous),'controller_sha256':sha(Path(__file__))})


if __name__=='__main__':main()
