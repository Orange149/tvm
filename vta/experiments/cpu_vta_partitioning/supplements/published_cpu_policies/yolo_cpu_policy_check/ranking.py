"""Static-only family scan and matched blocks; FPS files are never opened."""
from copy import deepcopy
import importlib.util
import itertools
from pathlib import Path
import random
import subprocess
import sys
import time
import numpy as np
from cpu_vta_partitioning.common import digest, now, read, sha256, write
from cpu_policies.open_resources import domain
from staged_ablation.model import score
from yolo_validation.ranking import inputs, costs_for
from .common import SOURCE, protocol, frozen, source_identity, assert_prepared


def load_engine():
    proof = read(SOURCE / 'ranking/fine_host/proof.json')
    directory = Path(proof['source_directory'])
    for leaf, value in proof['implementation'].items():
        if sha256(directory / leaf) != value:
            raise ValueError('frozen search source changed')
    name = '_yolo_policy_frozen_engine'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, directory / '__init__.py',
                                                     submodule_search_locations=[str(directory)])
        package = importlib.util.module_from_spec(spec); sys.modules[name] = package
        spec.loader.exec_module(package)
        spec = importlib.util.spec_from_file_location(name + '.search', directory / 'search.py')
        module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return sys.modules[name + '.search']


def verify_old(key, engine, programs, records, ddr, hashes):
    root = SOURCE / 'ranking' / key; proof = read(root / 'proof.json')
    if proof['component_hashes'] != hashes or proof['pipeline_FPS_used'] is not False:
        raise ValueError('original static input proof differs')
    original_dir = Path(proof['source_directory'])
    for leaf, value in proof['implementation'].items():
        if sha256(original_dir / leaf) != value: raise ValueError('old search implementation changed')
    # Each model's own _prepare semantics and mask domain are replayed unchanged.
    name = '_yolo_policy_old_' + key
    spec = importlib.util.spec_from_file_location(name, original_dir / '__init__.py',
                                                 submodule_search_locations=[str(original_dir)])
    package = importlib.util.module_from_spec(spec); sys.modules[name] = package; spec.loader.exec_module(package)
    spec = importlib.util.spec_from_file_location(name + '.search', original_dir / 'search.py')
    m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m; spec.loader.exec_module(m)
    costs = costs_for(key, programs, records, ddr)
    graphs = m._topologies(programs, 14, 3)
    contexts = {c['id']:c for c in domain.contexts()}; expanded = []; count = 0
    method = 'single_core' if key == 'tarnawski' else 'enhanced'
    for ctx in contexts.values():
        cp = read(root / 'search' / (ctx['id'] + '.json'))
        payload = dict(programs=m._prepare(programs,costs,ctx,ddr),topologies=graphs,
                       context=ctx,k=20,score_scale=m.SCORE_SCALE,implementation=proof['implementation'])
        fields = ('counts','gate_counts','rankings','topology_strata','resource_strata')
        if (cp['checkpoint_sha256'] != m._digest({k:v for k,v in cp.items() if k!='checkpoint_sha256'})
            or cp['input_sha256'] != m._digest(payload) or not cp['verification']['passed']
            or cp['verification']['source_sha256'] != proof['implementation']['oracle.cc']
            or cp['verification']['result_sha256'] != m._digest({k:cp[k] for k in fields})):
            raise ValueError('original full-domain checkpoint does not replay')
        count += cp['counts'][method]
        for raw in cp['rankings'][method]:
            terminal = m.Terminal.from_json(raw)
            cfg = m._configuration(terminal,graphs,programs,contexts)
            expanded.append({**cfg,'score_ns':terminal.score,'score_ms':terminal.score/1e6})
    rows = sorted(expanded,key=m._row_key)[:20]
    for i,row in enumerate(rows,1): row['static_rank']=i
    original = read(root/'rankings.json')['methods'][method]['rankings']
    if rows!=original or count!=proof['selected_count']:
        raise ValueError('original strict Top20 differs')
    for row in rows:
        if score(row,costs,key)['score_ns']!=row['score_ns']: raise ValueError('original scalar mismatch')
    return rows


def audit(output):
    from yolo_validation.common import assert_sources
    from yolo_validation.components import replay_complete
    from run_yolo_validation_resume import replay_record
    from yolo_transfer.ranking import ddr_table
    started = time.monotonic(); output = Path(output)
    original = assert_sources(SOURCE); records = replay_complete(SOURCE)
    engine = load_engine(); programs, indexed, ddr, hashes = inputs(SOURCE)
    if records != indexed or len(records)!=714: raise ValueError('isolated source table differs')
    traffic_path = SOURCE/'ranking/inputs/traffic.json'; traffic = read(traffic_path)
    full = read(Path(original['profile_bundle'])/'plans/programs.json')
    if len(traffic['records'])!=78: raise ValueError('78 DMA signatures required')
    for row in traffic['records'].values(): replay_record(original['profile_bundle'],row,traffic_path.parent)
    hardware = read(SOURCE/'ranking/inputs/hardware.json')['provenance']
    if ddr_table(full,traffic['records'],hardware)!=ddr: raise ValueError('DDR table no longer replays')
    old_lists = {key:verify_old(key,engine,programs,records,ddr,hashes)
                 for key in ('fine_host','tarnawski')}
    files = [SOURCE/'PREPARATION.json', SOURCE/'ranking/FREEZE.json', SOURCE/'ranking/ddr.json',
             Path(original['profile_bundle'])/'manifest.json',
             *[SOURCE/'ranking/inputs'/leaf for leaf in ('programs.json','isolated_costs.json','traffic.json','hardware.json')]]
    for key in old_lists:
        files += [SOURCE/'ranking'/key/'proof.json', SOURCE/'ranking'/key/'rankings.json']
        files += list((SOURCE/'ranking'/key/'search').glob('*.json'))
    seal = dict(protocol=protocol(), sources=source_identity(), profile_bundle=original['profile_bundle'],
                source_files={str(p.resolve()):sha256(p) for p in files}, component_hashes=hashes,
                original_results_read_only=True, old_pipeline_FPS_read=False)
    frozen(output/'PREPARATION.json',seal)
    frozen(output/'original_static_firsts.json',{key:rows[0] for key,rows in old_lists.items()})
    report = dict(status='AUDITED',components=714,DMA=78,static_originals_replayed=True,
                  old_pipeline_FPS_read=False,seconds=time.monotonic()-started)
    write(output/'AUDIT.json',report,replace=True)
    return report


def fixed_eligible(alloc):
    return alloc['compatible'] & (alloc['allocations'] != alloc['allocations'][:, :1]).any(axis=1)


def fixed_values(engine, topology, table, ctx, alloc):
    """Gather only compatible >=2-group paths, never evaluate the full Cartesian domain."""
    eligible = np.flatnonzero(fixed_eligible(alloc)); assignment = alloc['allocations'][eligible]
    count = len(assignment); core = np.zeros((count,4),dtype=np.int64)
    teams = np.zeros_like(core); groups = np.zeros((count,len(alloc['options'])),dtype=np.int64)
    wall = np.zeros(count,dtype=np.int64); memory = np.zeros(count,dtype=np.int64)
    host = vta = j = 0
    for name in topology:
        p = table[name]
        if not p['cpu']:
            w,u,b,d = p['costs'][0][:4]; vta+=w; host+=u+b; memory+=d; continue
        values = np.asarray(p['costs'],dtype=np.int64); m = len(values)
        cpu = np.zeros((m,4),dtype=np.int64); team = np.zeros_like(cpu)
        services = np.zeros((m,m),dtype=np.int64)
        for k,mask in enumerate(alloc['options']):
            cpu[k,list(mask)] = engine._allocate_integer(values[k,1],values[k,4:4+len(mask)])
            cpu[k,mask[0]] += values[k,2]; team[k,list(mask)] = values[k,0]
            services[k,k] = values[k,0]
        choice = assignment[:,j]; j+=1
        core += cpu[choice]; teams += team[choice]; groups += services[choice]
        wall = np.maximum(wall,values[choice,0]); memory += values[choice,3]
    demand = core.max(axis=1)
    if len(ctx['host_mask'])==1: demand=np.maximum(demand,core[:,ctx['host_mask'][0]]+host)
    elif len(ctx['host_mask'])==4: demand=np.maximum(demand,(core.sum(axis=1)+host+3)//4)
    else: raise ValueError('unexpected frozen context')
    result = np.maximum.reduce([wall,np.full(count,vta),memory,demand,teams.max(axis=1),groups.max(axis=1)])
    return eligible,result


def oracle_source(output, engine):
    """Filter allocations/gates only; keep every original score expression intact."""
    directory = Path(output)/'oracle'; directory.mkdir(parents=True,exist_ok=True)
    base = Path(engine.__file__).with_name('oracle.cc').read_text()
    replacements = [
        ('std::string context_id;', 'std::string context_id;\nstd::string policy_family;'),
        ('    context_id=context.at("id").get<std::string>();',
         '    context_id=context.at("id").get<std::string>();\n    policy_family=root.at("policy_family").get<std::string>();'),
        ('  for(int gate=0;gate<2;++gate) {',
         '  for(int gate=(policy_family=="fixed" ? 1 : 0);gate<(policy_family=="fixed" ? 2 : 1);++gate) {'),
        ('      for(const auto& allocation:allocation_tables[n])evaluate(graph,allocation);',
         '      for(const auto& allocation:allocation_tables[n]) {\n'
         '        bool distinct=false;for(int i=1;i<n;++i)distinct |= allocation.values[i]!=allocation.values[0];\n'
         '        if(policy_family=="fixed" && (!allocation.compatible || !distinct))continue;\n'
         '        evaluate(graph,allocation);\n      }')]
    changed = base
    for before,after in replacements:
        if changed.count(before)!=1: raise ValueError('frozen oracle filtering anchor changed')
        changed=changed.replace(before,after)
    p=directory/'oracle.cc'
    if p.exists() and p.read_text()!=changed: raise ValueError('family oracle source changed')
    if not p.exists():p.write_text(changed)
    header=directory/'picojson.h'; data=Path(engine.__file__).with_name('picojson.h').read_bytes()
    if header.exists() and header.read_bytes()!=data: raise ValueError('header changed')
    if not header.exists():header.write_bytes(data)
    # The existing compiler helper resolves paths from its function globals.
    from types import FunctionType
    compile_oracle=FunctionType(engine._oracle_executable.__code__,
        {**engine._oracle_executable.__globals__,'__file__':str(directory/'search.py')})
    executable=compile_oracle(directory)
    frozen(directory/'FILTER_PROOF.json',dict(original_sha256=sha256(Path(engine.__file__).with_name('oracle.cc')),
        filtered_sha256=sha256(p),score_expressions_changed=False,allocation_and_gate_filter_only=True))
    return executable


def scan(engine,programs,costs,ddr,graphs,contexts,output,*,family):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    executable=oracle_source(output,engine); global_rows=[]; total=0
    for ctx in contexts:
        started=time.monotonic(); table=engine._prepare(programs,costs,ctx,ddr)
        payload=dict(programs=table,topologies=graphs,context=ctx,k=1,score_scale=engine.SCORE_SCALE,
                     policy_family=family)
        fingerprint=digest(payload); checkpoint=output/(ctx['id']+'.json')
        if checkpoint.exists():
            saved=read(checkpoint)
            if (saved['input_sha256']!=fingerprint or saved['oracle_sha256']!=sha256(executable)
                    or saved['content_sha256']!=digest({k:v for k,v in saved.items() if k!='content_sha256'})):
                raise ValueError('family checkpoint differs')
            global_rows.extend(saved['best']);total+=saved['count'];continue
        best=[];count=0
        for gi,graph in enumerate(graphs):
            cpus=tuple(n for n in graph if table[n]['cpu'])
            alloc=engine._allocation_domain(tuple(ctx['compute_pool']),len(cpus))
            if family=='fixed': indices,values=fixed_values(engine,graph,table,ctx,alloc)
            elif family=='open':
                indices=np.arange(len(alloc['allocations']))
                values=engine._join_values(cpus,graph,table,ctx,alloc)['enhanced']
            else:raise ValueError('unknown strategy family')
            count+=len(values)
            minimum=int(values.min())
            # Within one graph/context the option Cartesian order is exactly
            # (thread count, mask) lexicographic execution order. Preserve its
            # first minimum; compare complete tuples across graphs/contexts.
            for pos in np.flatnonzero(values==minimum)[:1]:
                idx=int(indices[pos]);t=engine.Terminal(minimum,gi,tuple(map(int,alloc['allocations'][idx])),
                    ctx['id'],'resource_group' if family=='fixed' else 'off')
                cfg=engine._configuration(t,graphs,programs,{ctx['id']:ctx})
                row={**cfg,'score_ns':minimum}
                if not best or engine._row_key(row)<engine._row_key(best[0]):best=[row]
        checked=subprocess.run([str(executable)],input=__import__('json').dumps(payload),
            text=True,capture_output=True,timeout=300)
        if checked.returncode:raise RuntimeError('independent family oracle failed: '+checked.stderr)
        reference=__import__('json').loads(checked.stdout)
        expanded=[]
        for raw in reference['rankings']['enhanced']:
            t=engine.Terminal.from_json(raw)
            expanded.append({**engine._configuration(t,graphs,programs,{ctx['id']:ctx}),'score_ns':t.score})
        if expanded!=best or reference['counts']['enhanced']!=count:
            raise ValueError('vector family search / independent scalar C++ mismatch')
        for row in best:
            if score(row,costs,'fine_host')['score_ns']!=row['score_ns']:
                raise ValueError('independent Python scalar score mismatch')
        saved=dict(input_sha256=fingerprint,oracle_sha256=sha256(executable),family=family,
            count=count,best=best,verification_passed=True,seconds=time.monotonic()-started)
        saved['content_sha256']=digest(saved);frozen(checkpoint,saved)
        global_rows.extend(best);total+=count
        print('策略家族核验',family,ctx['id'],count,'配置',flush=True)
    return sorted(global_rows,key=engine._row_key)[0],dict(count=total,verification_passed=True,family=family)


def block_key(cfg):
    return (tuple((s['device'],s['begin'],s['end']) for s in cfg['stages']),cfg['context_id'])


def gate_twin(cfg):
    return domain.candidate(cfg['stages'],cfg['context_id'],'off')


def rank(output):
    output=Path(output)
    if not (output/'PREPARATION.json').exists():audit(output)
    assert_prepared(output);started=time.monotonic();engine=load_engine()
    programs,records,ddr,hashes=inputs(SOURCE);costs=costs_for('fine_host',programs,records,ddr)
    graphs=engine._topologies(programs,14,3)
    fixed,proof=scan(engine,programs,costs,ddr,graphs,domain.contexts(),output/'family_search',family='fixed')
    originals=read(output/'original_static_firsts.json')
    seeds=[('主模型静态第一名',originals['fine_host']),('固定核组家族第一名',fixed),
           ('Tarnawski静态第一名',originals['tarnawski'])]
    blocks=[];seen={};pool={};scores={};members={}
    for origin,cfg in seeds:
        key=block_key(cfg)
        if key in seen:blocks[seen[key]]['origins'].append(origin);continue
        idx=len(blocks);seen[key]=idx;ctx=domain.context(cfg['context_id'])
        graph=tuple(s['program'] for s in cfg['stages']); selected={};proofs={}
        for family in ('open','fixed'):
            row,proofs[family]=scan(engine,programs,costs,ddr,[graph],[ctx],
                output/'block_search'/str(idx)/family,family=family)
            selected[family]=row
        selected['fixed_gate_off']=gate_twin(selected['fixed'])
        block=dict(id='block_'+str(idx+1),origins=[origin],context=ctx,topology=list(graph),
            candidates={k:r['id'] for k,r in selected.items()},proofs=proofs,
            repeated_group= len({tuple(s['mask']) for s in selected['fixed']['stages'] if s['device']=='cpu'})
                         <sum(s['device']=='cpu' for s in selected['fixed']['stages']))
        blocks.append(block)
        for label,row in selected.items():
            clean={**domain.configuration(row),'id':row['id']};pool[clean['id']]=clean
            members.setdefault(clean['id'],[]).append(dict(block=block['id'],role=label))
    reference=originals['tarnawski'];pool[reference['id']]={**domain.configuration(reference),'id':reference['id']}
    members.setdefault(reference['id'],[]).append(dict(role='tarnawski_bridge'))
    if len(pool)>10:raise ValueError('candidate budget exceeded')
    for name,cfg in pool.items():
        domain.validate_candidate(cfg,programs,unit_count=14,max_vta_islands=3)
        scores[name]=score(cfg,costs,'fine_host')
    order=sorted(pool);random.Random(protocol()['seed']).shuffle(order)
    root=output/'ranking'
    docs={'pool.json':dict(sorted(pool.items())),'blocks.json':blocks,'scores.json':scores,
          'membership.json':members,'execution_order.json':order,'protocol.json':protocol(),
          'search_proof.json':dict(fixed_family=proof,original_components=hashes,
                                   pipeline_FPS_read=False,model_modified=False)}
    for leaf,doc in docs.items():frozen(root/leaf,doc)
    material=dict(files={leaf:sha256(root/leaf) for leaf in docs},candidate_count=len(pool),
                  preparation_sha256=sha256(output/'PREPARATION.json'),pipeline_FPS_read=False)
    path=root/'FREEZE.json'
    if path.exists():
        if any(read(path)[k]!=v for k,v in material.items()):raise ValueError('selection freeze changed')
    else:frozen(path,{**material,'frozen_at':now()})
    write(output/'ranking_cost.json',dict(seconds=time.monotonic()-started),replace=True)
    return dict(candidates=len(pool),blocks=len(blocks),fixed_family_count=proof['count'],freeze=str(path))
