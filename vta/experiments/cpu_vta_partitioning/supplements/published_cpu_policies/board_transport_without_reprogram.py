"""Reuse durable SSH jobs with read-only hardware qualification.

Existing controllers and scripts remain unchanged. A new campaign must bind
this source and the setup policy into its capsule identity before using it.
"""
from pathlib import Path
from types import FunctionType

from cpu_vta_partitioning.common import write
from cpu_vta_partitioning.transport import SSH
from cpu_vta_partitioning.transport.scripts import session_script as original_script


def session_script(base,remote,boot,identity,argv):
    text=original_script(base,remote,boot,identity,argv)
    lines=text.splitlines();removed=[];kept=[]
    destinations=('/sys/class/fpga_manager/fpga0/flags',
                  '/sys/class/fpga_manager/fpga0/firmware')
    for line in lines:
        if line.startswith('printf ') and any(' > '+path+' ' in line for path in destinations):
            removed.append(line)
        else:
            kept.append(line)
    if len(removed)!=2 or not all(sum(' > '+path+' ' in line for line in removed)==1 for path in destinations):
        raise ValueError('audited per-session FPGA programming anchors changed')
    result='\n'.join(kept)+'\n'
    if any(' > '+path in result for path in destinations):
        raise ValueError('FPGA programming write remains')
    return result


def qualify_environment(self,remote,boot,run_dir,bundle):
    from yolo_validation.components import hardware_check
    result=hardware_check(self,bundle,boot)
    write(Path(run_dir)/'hardware-read-only.json',result,replace=True)
    return result


def transport(config,native_command,*,execute_board=False):
    if not callable(native_command):
        raise ValueError('explicit frozen native command builder required')
    measure=FunctionType(SSH.measure.__code__,{
        **SSH.measure.__globals__,'native_command':native_command,'session_script':session_script})
    return type('NoReprogramCampaignSSH',(SSH,),{
        'measure':measure,'qualify_environment':qualify_environment})(config,execute_board=execute_board)
