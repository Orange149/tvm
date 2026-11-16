"""Authenticated compact complete-domain checkpoints; component-only inputs."""
from pathlib import Path
import time
from .common import api, FULL_PARENT, STATIC_PARENT, STATIC_SOURCE, ROOT
from .selector import choose, execution_class, features
from cpu_policies.open_resources import domain, search

def compact_records(programs, checkpoints):
    graphs=search._topologies(programs,21,3)
    contexts={c['id']:c for c in domain.contexts()}
    terminals={}
    for checkpoint in checkpoints:
        rows=list(checkpoint['rankings']['enhanced'])
        rows += [r for key in ('topology_strata','resource_strata')
                 for group in checkpoint[key].values() for r in group]
        for value in rows:
            t=search.Terminal.from_json(value)
            if t.context!=checkpoint['context']['id']:raise ValueError('terminal context differs')
            key=(t.topology,t.allocation,t.context,t.gate)
            if key in terminals and terminals[key].score!=t.score:raise ValueError('conflicting terminal score')
            terminals[key]=t
    records=[]
    for key,t in terminals.items():
        ctx=contexts[t.context];options=search._masks(tuple(ctx['compute_pool']));allocation=iter(t.allocation)
        topology=[];cpu=[];stages=[]
        for name in graphs[t.topology]:
            p=programs[name];device=p['device'];mask=options[next(allocation)] if device=='cpu' else tuple(ctx['host_mask'])
            topology.append((device,p['begin'],p['end']))
            stages.append((device,p['begin'],p['end'],len(mask) if device=='cpu' else 1,mask))
            if device=='cpu':cpu.append((p['begin'],p['end'],len(mask),mask,mask[0]))
        canonical=(tuple(stages),tuple(ctx['compute_pool']),tuple(ctx['host_mask']),tuple(ctx['control_mask']),
                   tuple(ctx['reserved_cores']),t.gate,'pinned')
        records.append({'key':key,'terminal':t,'context':t.context,'score_ns':t.score,
            'class':execution_class(t.gate,[r[3] for r in cpu]),'canonical':canonical,
            'features':features(topology,ctx,t.gate,cpu)})
    def expand(row):
        t=row['terminal'];cfg=search._configuration(t,graphs,programs,contexts)
        return {**cfg,'score_ns':t.score,'score_ms':t.score/1e6}
    return records,expand

def verify_checkpoints(base, static):
    """Recompute input identities and independently enumerated result digests.

    No old pipeline measurements or screening files are opened. The previous
    capsule manifest authenticates the source hashes and component-only tables.
    """
    from cpu_policies.open_resources.costs import Costs
    base,static=Path(base),Path(static);c=api(base)
    programs=c.read(base/'plans/programs.json');records=c.read(static/'isolated_costs.json')['costs']
    ddr=c.read(static/'ddr.json');costs=Costs(records,programs=programs,ddr=ddr)
    if len(records)!=775:raise ValueError('isolated signature coverage differs')
    graphs=search._topologies(programs,21,3)
    implementation={name:c.sha256(Path(search.__file__).with_name(name)) for name in ('search.py','oracle.cc','picojson.h')}
    outputs=[]
    for ctx in domain.contexts():
        value=c.read(static/'search'/(ctx['id']+'.json'))
        if value.get('checkpoint_sha256')!=search._digest({k:v for k,v in value.items() if k!='checkpoint_sha256'}):
            raise ValueError('checkpoint content hash differs')
        payload={'programs':search._prepare(programs,costs,ctx,ddr),'topologies':graphs,'context':ctx,
                 'k':20,'score_scale':search.SCORE_SCALE,'implementation':implementation}
        if value['input_sha256']!=search._digest(payload):raise ValueError('complete-domain checkpoint input changed')
        proof=value['verification'];fields=('counts','gate_counts','rankings','topology_strata','resource_strata')
        if (not proof.get('passed') or proof['source_sha256']!=implementation['oracle.cc']
                or proof['result_sha256']!=search._digest({k:value[k] for k in fields})):
            raise ValueError('independent enumeration proof differs')
        outputs.append(value)
    return programs,outputs

def build(base=None, static_source=None, source_capsule=None, output=None, replay_components=True):
    base=Path(base or FULL_PARENT).resolve();static=Path(static_source or STATIC_SOURCE).resolve()
    capsule=Path(source_capsule or STATIC_PARENT).resolve()
    output=Path(output or ROOT/'artifacts/selection_top20_static').resolve()
    c=api(base);start=time.monotonic()
    m=c.read(capsule/'manifest.json')
    if c.digest({k:v for k,v in m.items() if k!='bundle_id'})[:24]!=m['bundle_id']:raise ValueError('static capsule identity differs')
    names=('source_provenance.json','static_rankings.json','static_pool.json','isolated_costs.json','ddr.json')
    for name in names:
        expected=m['files']['plans/'+name]
        c.verify_inventory(capsule,{'plans/'+name:expected})
        if c.sha256(static/name)!=expected['sha256']:raise ValueError('component-only static source differs: '+name)
    for ctx in domain.contexts():
        name=ctx['id']+'.json';key='provenance/static_search/'+name
        c.verify_inventory(capsule,{key:m['files'][key]})
        if c.sha256(static/'search'/name)!=m['files'][key]['sha256']:
            raise ValueError('frozen compact checkpoint differs: '+name)
    source=c.read(static/'source_provenance.json')
    if (source['source_manifest_sha256']!=c.sha256(base/'manifest.json')
            or not source['verification']['passed'] or source['traffic_records_replayed']!=87
            or not source['same_pure20_and_diverse20_as_original']):raise ValueError('static provenance incomplete')
    if replay_components:
        # Replay only original component/traffic evidence, never validate the
        # old complete campaign (which would also read screening performance).
        from cpu_policies.open_resources.campaign import load_costs,load_traffic
        run=Path(source['source_run']);state=c.read(run/'state.json')
        if c.sha256(run/'state.json')!=source['source_state_sha256']:raise ValueError('calibration source identity changed')
        costs=load_costs(run,state,replay=True);traffic=load_traffic(run,state,replay=True)
        if costs!=c.read(static/'isolated_costs.json')['costs'] or len(traffic)!=87:
            raise ValueError('raw component/traffic replay differs')
    programs,checkpoints=verify_checkpoints(base,static)
    records,expand=compact_records(programs,checkpoints)
    ordered=sorted(records,key=lambda r:(r['score_ns'],r['canonical']))
    pure=[]
    for i,r in enumerate(ordered[:20],1):
        row=expand(r);row['static_rank']=i;pure.append(row)
    old=c.read(static/'static_rankings.json')
    if pure!=old['pure_static20']:raise ValueError('pure-static replay differs')
    from cpu_policies.open_resources.selection import select
    guard=pure[0]['score_ns']*115
    # Expand only bounded representatives relevant to the old 15% guard.
    old_diverse,old_protocol=select(pure,[expand(r) for r in ordered if r['score_ns']*100<=guard])
    if old_diverse!=old['diverse20']:raise ValueError('old diversity replay differs')
    baseline={r['id'] for r in [*pure,*old_diverse]}
    if len(baseline)!=28:raise ValueError('old lists no longer have the frozen 28-entry union')
    new,coverage=choose(records,ordered[:5],[v['id'] for v in domain.contexts()],expand,fallback=ordered[:20])
    if [r['id'] for r in new[:5]]!=[r['id'] for r in pure[:5]]:raise ValueError('static first five changed')
    lists={'pure_static20':pure,'diverse20':old_diverse,'coverage20':new}
    pool={}
    for values in lists.values():
        for r in values:
            domain.validate_candidate(r,programs)
            cfg={**domain.configuration(r),'id':r['id'],'score_ns':r['score_ns'],'score_ms':r['score_ms']}
            if r['id'] in pool and pool[r['id']]!=cfg:raise ValueError('candidate identity collision')
            pool[r['id']]=cfg
    if len(pool)>43:raise ValueError('measurement budget exceeded')
    output.mkdir(parents=True,exist_ok=True)
    c.write(output/'rankings.json',{**lists,'selection_protocol':coverage,'counts':old['counts']},replace=True)
    c.write(output/'pool.json',dict(sorted(pool.items())),replace=True)
    c.write(output/'source_provenance.json',{'source':source,'source_capsule_id':m['bundle_id'],
        'compact_checkpoint_hashes':{v['context']['id']:v['checkpoint_sha256'] for v in checkpoints},
        'component_evidence_replayed':replay_components,'isolated_costs':775,'traffic_records':87,
        'baseline_union':28,'measurement_union':len(pool),'candidate_representatives':len(records),
        'same_original_lists':True,'pipeline_FPS_used':False,'elapsed_seconds':time.monotonic()-start},replace=True)
    return output
