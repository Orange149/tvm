"""Copy only new offline findings; archived source evidence stays in place."""
from pathlib import Path
import shutil

from cpu_vta_partitioning.common import inventory,read,sha256,sync_tree,write
from .common import protocol


def publish(output,target):
    output,target=Path(output).resolve(),Path(target).resolve()
    marker=target/'DELIVERY.json'
    if target.exists() and any(target.iterdir()) and (not marker.exists() or
            read(marker).get('experiment')!=protocol()['experiment']):
        raise ValueError('report destination contains another document/experiment')
    target.mkdir(parents=True,exist_ok=True)
    for p in (output/'analysis').iterdir():
        if p.is_file():
            shutil.copy2(p,target/p.name)
    if (output/'graphs').exists():
        shutil.copytree(output/'graphs',target/'graphs',dirs_exist_ok=True)
    for name in ('AUDIT.json','INPUT_FREEZE.json','REPLAY_FREEZE.json','MODEL_INPUT.json',
                 'REPLAY.json','EXTRACTION.json','OFFLINE_GUARD.json','LOCAL_TEST_REPORT.json'):
        if (output/name).exists():
            shutil.copy2(output/name,target/name)
    write(marker,dict(experiment=protocol()['experiment'],output=str(output),
        new_native_sessions=0,new_DNN_inferences=0,board_access=False,
        full_shared_pipeline_prediction_supported=False,
        replay_sha256=sha256(output/'REPLAY.json')),replace=True)
    write(target/'ARTIFACT_MANIFEST.json',dict(files=inventory(target,['ARTIFACT_MANIFEST.json'])),replace=True)
    sync_tree(target)
    return dict(report=str(target/'README.md'),artifacts=str(target/'ARTIFACT_MANIFEST.json'))
