"""Freeze unchanged-model predictions before this batch's DNN measurements."""
from pathlib import Path
from cpu_vta_partitioning.common import read,sha256
from cpu_policies.open_resources.costs import Costs
from policy_branch_eval.ranking import inputs
from policy_branch_eval.common import isolated_inputs_only
from pool_temporal.ranking import model_stages
from pool_temporal.simulator import PoolReplay
from staged_ablation.model import score
from .common import PARENT,freeze
from .prepare import configurations


def freeze_predictions(output):
    output=Path(output);result={};path=PARENT/'pool_temporal/INTRINSICS.json'
    with isolated_inputs_only():
        intrinsic=read(path)
        for network in ('resnet18','yolov3_tiny416'):
            programs,records,ddr,hashes,_,_=inputs(network)
            costs=Costs(records,programs=programs,ddr=ddr,distribution='session_weights')
            for name,cfg in configurations(network)['pipelines'].items():
                stages,assumptions=model_stages(network,cfg,programs,costs,intrinsic)
                short=PoolReplay(stages,gate='off',host_order='before').run()
                matched=PoolReplay(stages,gate='off',host_order='before',frames=104,warmup=8,scored=64).run()
                result[network+'/'+name]=dict(configuration=cfg,component_hashes=hashes,
                    isolated_costs=[costs.get(s,cfg) for s in cfg['stages']],
                    capacity=score(cfg,costs,'fine_host'),original_short_replay=short,
                    matched_long_replay=matched,assumptions=assumptions)
    return freeze(output/'PREDICTIONS_FREEZE.json',dict(cases=result,intrinsic_sha256=sha256(path),
        source_sha256=sha256(Path(__file__)),pipeline_FPS_read=False,modified_formulas=False))
