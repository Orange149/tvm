"""Local checks and receipt replays, without implicit board calls."""
from pathlib import Path
from cpu_vta_partitioning.common import digest,read,sha256,verify_inventory
from cpu_policies.open_resources.domain import configuration,validate_candidate
from .common import METHODS,assert_sources,protocol

def validate_lists(lists,pool):
    if set(lists)!=set(METHODS):raise ValueError('unexpected methods')
    union=set()
    for key,rows in lists.items():
        if len(rows)!=20 or len({r['id'] for r in rows})!=20:raise ValueError('strict twenty distinct candidates required')
        for row in rows:
            name=row['id']
            if name not in pool or configuration(row)!=configuration(pool[name]):raise ValueError('list/config identity differs')
            if name!='cfg_'+digest(configuration(row))[:24]:raise ValueError('invalid configuration hash')
            if type(row['score_ns']) is not int or row['score_ns']<0:raise ValueError('invalid score')
            union.add(name)
    if union!=set(pool) or len(pool)>60:raise ValueError('union or budget differs')
    return True

def validate_bundle(bundle):
    bundle=Path(bundle);manifest=read(bundle/'manifest.json')
    if manifest['bundle_id']!=digest({k:v for k,v in manifest.items() if k!='bundle_id'})[:24]:
        raise ValueError('capsule seal differs')
    verify_inventory(bundle,manifest['files'])
    seal=read(bundle/'plans/ranking_freeze.json')
    for leaf,value in seal['files'].items():
        if sha256(bundle/'plans'/leaf)!=value:raise ValueError('candidate freeze changed')
    pool=read(bundle/'plans/pool.json');lists=read(bundle/'plans/rankings.json')['lists']
    validate_lists(lists,pool)
    programs=read(bundle/'plans/programs.json')
    for cfg in pool.values():validate_candidate(cfg,programs,unit_count=14,max_vta_islands=3)
    if read(bundle/'plans/protocol.json')!=protocol():raise ValueError('window/method protocol differs')
    proofs=read(bundle/'plans/proofs.json')
    for key in METHODS:
        if (not proofs[key]['verification']['passed'] or proofs[key]['pipeline_FPS_used'] is not False
                or proofs[key]['selected_count']!=(3067532 if key=='tarnawski' else 472266548)):
            raise ValueError('complete-domain independent proof missing')
    return manifest

def validate(output):
    from .evidence import replay_imports
    from .components import replay_complete
    from .ranking import verify_one
    output=Path(output);prep=assert_sources(output)
    result=dict(sources_match=True,old_sources_preserved=True,protocol=prep['protocol'])
    if (output/'isolated_costs.json').exists() and read(output/'isolated_costs.json')['status']=='COMPLETE':
        result['components_replayed']=len(replay_complete(output))
    else:result['imported_components_replayed']=len(replay_imports(output))
    if (output/'ranking/FREEZE.json').exists():
        for key in METHODS:verify_one(output,key)
        result['three_full_searches_verified']=True
    if (output/'pipeline_bundle/manifest.json').exists():
        m=validate_bundle(output/'pipeline_bundle');result.update(bundle_id=m['bundle_id'],candidates=m['candidate_count'])
    return result
