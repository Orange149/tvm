"""One command from missing components to frozen long-window measurements."""
from pathlib import Path
import time
from cpu_vta_partitioning.common import locked,now,read,write
from .common import PROFILE,assert_sources,frozen

def status(output,run_dir):
    output,run_dir=Path(output),Path(run_dir)
    result={}
    for name,path in [('campaign',run_dir/'state.json'),('components',run_dir/'components/state.json'),
                      ('traffic',run_dir/'traffic/traffic.json'),('pipeline',run_dir/'pipeline/state.json')]:
        if path.exists():result[name]={k:v for k,v in read(path).items() if k not in ('records','costs')}
    if (output/'ranking/FREEZE.json').exists():result['ranking_frozen']=read(output/'ranking/FREEZE.json')['candidate_count']
    return result

def run(output,run_dir,local_config,*,execute_board=False):
    if not execute_board:raise ValueError('explicit --execute-board required')
    from . import components,ranking,bundle,pipeline,analysis
    from yolo_transfer import traffic,profile
    output,run_dir=Path(output).resolve(),Path(run_dir).resolve();prep=assert_sources(output)
    ssh=profile.matched_ssh(read(local_config)['board'],execute_board=True)
    started=time.monotonic()
    with locked(run_dir):
        boot=ssh.boot_id()
        frozen(run_dir/'run.json',dict(boot_id=boot,output=str(output),profile_manifest_sha256=prep['profile_manifest_sha256']))
        def state(stage,**fields):
            write(run_dir/'state.json',dict(status=stage,boot_id=boot,output=str(output),updated_at=now(),**fields),replace=True)
        try:
            state('COMPONENTS')
            components.complete(output,run_dir/'components',local_config,execute_board=True)
            state('DMA_TRAFFIC')
            traffic.run(prep['profile_bundle'],run_dir/'traffic',local_config,execute_board=True)
            if ssh.boot_id()!=boot:raise ValueError('boot changed before ranking; retain this batch')
            state('RANKING')
            ranking.prepare_inputs(output,run_dir/'traffic')
            ranking.rank(output)
            state('PREPARING_PIPELINE')
            bundle.prepare(output)
            if ssh.boot_id()!=boot:raise ValueError('boot changed before pipeline measurements')
            state('PIPELINE')
            result=pipeline.run(output/'pipeline_bundle',run_dir/'pipeline',local_config,execute_board=True)
            state('ANALYZING',pipeline=result)
            report=analysis.analyze(output,run_dir)
            state('COMPLETE',pipeline=result,report=report,elapsed_seconds=time.monotonic()-started)
            return status(output,run_dir)
        except Exception as error:
            state('STOPPED_ON_ERROR',error_type=type(error).__name__,error=str(error))
            raise
