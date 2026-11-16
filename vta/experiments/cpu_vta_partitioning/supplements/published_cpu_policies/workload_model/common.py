"""Separate study identity; old experiments remain read-only."""
from pathlib import Path
import shutil

from cpu_vta_partitioning.common import digest, now, read, sha256, write

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'artifacts/cross_network_model_01'
REPORT=ROOT.parents[4]/'paper/my_paper/analysis/20261110_cross_network_model'
NETWORKS=('resnet18','yolov3_tiny416')


def storage():
    result={}
    for name,path in [('workspace',ROOT),('windows_C',Path('/mnt/c'))]:
        if path.exists():
            value=shutil.disk_usage(path)
            result[name]=dict(path=str(path),free_bytes=value.free,total_bytes=value.total)
    result['interpretation']='WSL virtual free space is not Windows backing-volume free space'
    return result


def require_space():
    value=storage()
    if value['workspace']['free_bytes']<10*2**30:
        raise RuntimeError('workspace reserve below 10 GiB; do not create large artifacts')
    if value.get('windows_C',{}).get('free_bytes',100*2**30)<20*2**30:
        raise RuntimeError('Windows C reserve below 20 GiB; use existing evidence only')
    return value


def source_identity():
    paths=[ROOT/'run_workload_model.py',*sorted(Path(__file__).parent.glob('*.py')),
           ROOT/'tests/test_workload_model.py']
    return {str(p.relative_to(ROOT)):sha256(p) for p in paths}


def freeze(path,value):
    path=Path(path)
    if path.exists():
        if read(path)!=value:raise ValueError('study freeze differs: '+str(path))
    else:write(path,value)
    return value


def progress(output,phase,*,evidence=None,next_action=None):
    path=Path(output)/'PROGRESS.json'
    document=read(path) if path.exists() else dict(goal='cross-network static model and Top20 validation',
        board_reboot_forbidden=True,stages={k:dict(status='PENDING') for k in
            ('inheritance','workload_features','hypothesis','implementation','local_validation',
             'mechanism_validation','frozen_Top20','common_board_validation','final_report')},events=[])
    document['stages'][phase]=dict(status='DONE',evidence=evidence)
    document['events'].append(dict(at=now(),phase=phase,evidence=evidence))
    document.update(updated_at=now(),next_action=next_action,overall_status='ACTIVE',
                    storage=storage(),goal_complete=False)
    write(path,document,replace=True)
    return document
