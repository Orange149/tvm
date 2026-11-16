"""Component-only complete-domain search; no pipeline observations are opened."""
from pathlib import Path
import time
from .common import ROOT, api
from .domain import pool_from_rankings

def build(base,source_run,output):
    from cpu_policies.open_resources.prepare import validate_bundle
    from cpu_policies.open_resources.campaign import validate_state, load_costs, load_traffic
    from cpu_policies.open_resources.search import rank_all
    from cpu_policies.open_resources.selection import select
    base,source_run,output=map(lambda x:Path(x).resolve(),(base,source_run,output));c=api(base)
    start=time.monotonic();validate_bundle(base,replay_imports=True)
    state=validate_state(source_run,validate_payload=False)
    records=load_costs(source_run,state,replay=True);traffic=load_traffic(source_run,state,replay=True)
    if len(records)!=775 or len(traffic)!=87 or c.read(source_run/'costs.json')['costs']!=records:
        raise ValueError('source calibration coverage or values differ')
    output.mkdir(parents=True,exist_ok=True)
    previous=c.read(output/'source_provenance.json') if (output/'source_provenance.json').is_file() else None
    ddr=c.read(source_run/'ddr.json');programs=c.read(base/'plans/programs.json')
    ranked=rank_all(programs,records,ddr=ddr,output_dir=output/'search',verify=True)
    ranked['diverse20'],ranked['selection_protocol']=select(ranked['pure_static20'],c.read(output/'search/stratified_pool.json'))
    # Original rankings contain only pre-measurement scores and configurations.
    old=c.read(source_run/'rankings.json')
    if any(ranked[k]!=old[k] for k in ('pure_static20','diverse20')):
        raise ValueError('same-cost DP differs from original; investigate before board use')
    pool=pool_from_rankings(ranked)
    c.write(output/'static_rankings.json',ranked,replace=True);c.write(output/'static_pool.json',pool,replace=True)
    c.write(output/'isolated_costs.json',{'costs':records},replace=True);c.write(output/'ddr.json',ddr,replace=True)
    provenance={'source_run':str(source_run),'source_bundle':str(base),
        'source_state_sha256':c.sha256(source_run/'state.json'),'source_manifest_sha256':c.sha256(base/'manifest.json'),
        'source_costs_sha256':c.sha256(source_run/'costs.json'),'source_ddr_sha256':c.sha256(source_run/'ddr.json'),
        'original_rankings_sha256':c.sha256(source_run/'rankings.json'),'isolated_costs_replayed':775,
        'traffic_records_replayed':87,'full_domain_counts':ranked['counts'],'verification':ranked['verification'],
        'same_pure20_and_diverse20_as_original':True,'pool_size':len(pool),
        'elapsed_seconds':time.monotonic()-start,'pipeline_FPS_used':False}
    if previous:
        for key in ('source_state_sha256','source_costs_sha256','source_ddr_sha256','source_manifest_sha256'):
            if previous[key]!=provenance[key]:raise ValueError('static cache sources changed; use new output directory')
        provenance['elapsed_seconds']+=previous['elapsed_seconds']
    c.write(output/'source_provenance.json',provenance,replace=True)
    return output
