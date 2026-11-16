"""Copied into the immutable campaign as prepare.py; no source-tree mutation."""
from pathlib import Path
from .common import api,identity,protocol

def validate_bundle(bundle):
    from cpu_policies.open_resources.domain import configuration,validate_candidate
    bundle=Path(bundle).resolve();c=api(bundle);m=c.read(bundle/'manifest.json')
    if c.digest({k:v for k,v in m.items() if k!='bundle_id'})[:24]!=m['bundle_id']:raise ValueError('manifest differs')
    if m['controller_identity']!=identity():raise ValueError('controller differs; use frozen envelope_controller')
    c.verify_inventory(bundle,m['files'])
    if c.read(bundle/'plans/selection_protocol.json')!=protocol():raise ValueError('protocol differs')
    ranked=c.read(bundle/'plans/selection_rankings.json');pool=c.read(bundle/'plans/static_pool.json')
    freeze=c.read(bundle/'plans/ranking_freeze.json')
    if (freeze['rankings_sha256']!=c.sha256(bundle/'plans/selection_rankings.json')
            or freeze['pool_sha256']!=c.sha256(bundle/'plans/static_pool.json') or freeze['pipeline_FPS_used']):
        raise ValueError('ranking freeze differs')
    programs=c.read(bundle/'plans/programs.json');wanted={}
    for method in ('pure_static20','diverse20','coverage20'):
        rows=ranked[method]
        if len(rows)!=20 or len({r['id'] for r in rows})!=20:raise ValueError('twenty distinct candidates required')
        for row in rows:
            validate_candidate(row,programs)
            value=configuration(row)
            if row['id'] in wanted and wanted[row['id']]!=value:raise ValueError('configuration identity collision')
            wanted[row['id']]=value
    if set(pool)!=set(wanted) or len(pool)>60:raise ValueError('measurement union differs')
    for name,cfg in pool.items():
        if configuration(cfg)!=wanted[name]:raise ValueError('pool configuration differs')
    if set(programs)!={s['program'] for r in pool.values() for s in r['stages']}:raise ValueError('program closure differs')
    proof=c.read(bundle/'plans/source_provenance.json')
    if proof['domain_counts']['enhanced']!=966240306 or not proof['oracle_verification']['passed']:
        raise ValueError('complete-domain proof incomplete')
    search=proof['static_search_cost']
    if (search['costs_sha256']!=c.sha256(bundle/'plans/isolated_costs.json')
            or search['ddr_sha256']!=c.sha256(bundle/'plans/ddr.json')):
        raise ValueError('search profile identity differs')
    executor=c.read(bundle/'plans/executors.json')['policy']
    loader=c.read(bundle/'plans/qemu_loader.json')
    if (c.sha256(bundle/executor['path'])!=executor['sha256'] or not loader['passed']
            or loader['runner_sha256']!=executor['sha256']):raise ValueError('runner loader identity differs')
    from resource_envelope.model import score
    from cpu_policies.open_resources.costs import Costs
    settings=proof.get('model_settings',{'cpu_distribution':'balanced','score_variant':'combined'})
    if (settings['cpu_distribution'] not in ('balanced','session_weights')
            or settings['score_variant'] not in ('original','combined')):raise ValueError('unsupported model')
    if ranked['model_settings']!=settings or any(protocol()[key]!=value for key,value in settings.items()):
        raise ValueError('declared model settings differ')
    source=c.read(bundle/'plans/isolated_costs.json')['costs'];ddr=c.read(bundle/'plans/ddr.json')
    baseline=Costs(source,programs=programs,ddr=ddr)
    costs=Costs(source,programs=programs,ddr=ddr,distribution=settings['cpu_distribution'])
    for cfg in pool.values():
        result=score(cfg,costs)
        if (cfg['original_score_ns']!=score(cfg,baseline)['original']
                or cfg['score_ns']!=result[settings['score_variant']]):
            raise ValueError('static component accounting differs')
    from cpu_vta_partitioning.prepare.payload import execution_files
    if execution_files(bundle)!=c.read(bundle/'plans/board_payload.json')['files']:raise ValueError('payload differs')
    return m

def prepare(*args,**kwargs):
    raise RuntimeError('This capsule is frozen. Prepare a new capsule with the repository run_envelope_top20.py entry.')
