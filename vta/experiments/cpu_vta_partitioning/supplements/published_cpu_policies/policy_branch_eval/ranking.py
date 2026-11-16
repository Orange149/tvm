"""Hash-verified complete-domain replay and new fixed-family Top-K enumeration."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from cpu_vta_partitioning.common import digest, read, sha256, write
from cpu_policies.open_resources import domain
from cpu_policies.open_resources.costs import Costs
from staged_ablation.model import score
from yolo_cpu_policy_check.ranking import fixed_values, oracle_source
from .common import ROOT, NETWORKS, freeze, isolated_inputs_only, protocol, sources
from .selection import fixed_family, mixed


def inputs(network):
    if network == 'resnet18':
        from host_work_ablation.search import inputs as load
        return (*load(), ROOT/'artifacts/staged_ablation_01/fine_host', 21)
    if network == 'yolov3_tiny416':
        from yolo_validation.ranking import inputs as load
        root = ROOT/'artifacts/yolo_validation_01'
        return (*load(root), root/'ranking/fine_host', 14)
    raise ValueError('unsupported network')


def engine_from(proof, network):
    directory = Path(proof['source_directory'])
    for leaf, expected in proof['implementation'].items():
        if sha256(directory/leaf) != expected:
            raise ValueError('frozen engine changed')
    name = '_policy_branch_frozen_' + network
    if name not in sys.modules:
        for suffix, leaf, sub in [('', '__init__.py', [str(directory)]), ('.search', 'search.py', None)]:
            spec = importlib.util.spec_from_file_location(name+suffix, directory/leaf,
                                                        submodule_search_locations=sub)
            module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
            spec.loader.exec_module(module)
    return sys.modules[name+'.search']


def replay_original(engine, programs, costs, ddr, hashes, root, units):
    proof = read(root/'proof.json')
    if proof['component_hashes'] != hashes or proof['pipeline_FPS_used'] is not False:
        raise ValueError('cost identity or no-feedback proof differs')
    graphs = engine._topologies(programs, units, 3)
    contexts = {c['id']: c for c in domain.contexts()}
    expanded, count, files = [], 0, {}
    for ctx in contexts.values():
        path = root/'search'/(ctx['id']+'.json'); cp = read(path)
        payload = dict(programs=engine._prepare(programs,costs,ctx,ddr), topologies=graphs,
            context=ctx, k=20, score_scale=engine.SCORE_SCALE, implementation=proof['implementation'])
        fields = ('counts','gate_counts','rankings','topology_strata','resource_strata')
        if (cp['checkpoint_sha256'] != engine._digest({k:v for k,v in cp.items() if k!='checkpoint_sha256'})
            or cp['input_sha256'] != engine._digest(payload) or not cp['verification']['passed']
            or cp['verification']['source_sha256'] != proof['implementation']['oracle.cc']
            or cp['verification']['result_sha256'] != engine._digest({k:cp[k] for k in fields})):
            raise ValueError('full-domain checkpoint/oracle proof differs')
        count += cp['counts']['enhanced']; files[str(path)] = sha256(path)
        for raw in cp['rankings']['enhanced']:
            t = engine.Terminal.from_json(raw)
            expanded.append({**engine._configuration(t,graphs,programs,contexts),
                             'score_ns':t.score,'score_ms':t.score/1e6})
    rows = sorted(expanded,key=engine._row_key)[:20]
    for i,row in enumerate(rows,1): row['static_rank'] = i
    if (rows != read(root/'rankings.json')['methods']['enhanced']['rankings']
            or count != proof['selected_count'] or len(graphs) != proof['counts']['topologies']):
        raise ValueError('complete-domain original list differs')
    for row in rows:
        domain.validate_candidate(row,programs,unit_count=units)
        if score(row,costs,'fine_host')['score_ns'] != row['score_ns']:
            raise ValueError('original scalar score mismatch')
    files.update({str(root/n):sha256(root/n) for n in ('proof.json','rankings.json')})
    return rows, graphs, dict(count=count, topologies=len(graphs), source_hashes=files,
                              cached_full_domain_oracle_replayed=True)


def fixed_topk(engine, programs, costs, ddr, graphs, output, k=20):
    output = Path(output); output.mkdir(parents=True,exist_ok=True)
    executable = oracle_source(output,engine)
    contexts = {c['id']:c for c in domain.contexts()}
    global_rows, total, context_proofs = [], 0, {}
    for ctx in contexts.values():
        started = time.monotonic(); table = engine._prepare(programs,costs,ctx,ddr)
        payload = dict(programs=table,topologies=graphs,context=ctx,k=k,
                       score_scale=engine.SCORE_SCALE,policy_family='fixed')
        fingerprint = digest(payload); path = output/(ctx['id']+'.json')
        if path.exists():
            cp = read(path)
            if (cp['input_sha256'] != fingerprint or cp['oracle_sha256'] != sha256(executable)
                or cp['content_sha256'] != digest({n:v for n,v in cp.items() if n!='content_sha256'})
                or not cp['verification_passed']):
                raise ValueError('fixed-family checkpoint changed')
        else:
            collector = engine.Collectors(graphs,programs,ctx,k); terminals=[]; count=0
            for gi,graph in enumerate(graphs):
                n = sum(table[p]['cpu'] for p in graph)
                alloc = engine._allocation_domain(tuple(ctx['compute_pool']),n)
                indices, values = fixed_values(engine,graph,table,ctx,alloc)
                count += len(values)
                positions = engine._first_k(values,np.ones(len(values),dtype=bool),k)
                for pos in positions:
                    t = engine.Terminal(int(values[pos]),gi,
                        tuple(map(int,alloc['allocations'][indices[pos]])),ctx['id'],'resource_group')
                    collector.insert(terminals,t,k)
            rows = [{**engine._configuration(t,graphs,programs,contexts),'score_ns':t.score}
                    for t in terminals]
            checked = subprocess.run([str(executable)],input=json.dumps(payload),text=True,
                                     capture_output=True,timeout=300)
            if checked.returncode:
                raise RuntimeError('independent fixed-family oracle failed: '+checked.stderr)
            reference = json.loads(checked.stdout); oracle_rows=[]
            for raw in reference['rankings']['enhanced']:
                t=engine.Terminal.from_json(raw)
                oracle_rows.append({**engine._configuration(t,graphs,programs,contexts),'score_ns':t.score})
            if rows != oracle_rows or count != reference['counts']['enhanced']:
                raise ValueError('fixed-family vector / independent C++ Top-K mismatch')
            for row in rows:
                if not fixed_family(row) or score(row,costs,'fine_host')['score_ns']!=row['score_ns']:
                    raise ValueError('fixed family / independent scalar score mismatch')
            cp = dict(input_sha256=fingerprint,oracle_sha256=sha256(executable),rankings=rows,
                count=count,verification_passed=True,reference_sha256=digest(reference))
            cp['content_sha256']=digest(cp); freeze(path,cp)
            write(output/(ctx['id']+'_cost.json'),dict(seconds=time.monotonic()-started),replace=True)
        global_rows.extend(cp['rankings']); total+=cp['count']
        context_proofs[ctx['id']]=dict(count=cp['count'],sha256=sha256(path))
        print('固定核组Top-20核验',output.parent.name,ctx['id'],cp['count'],flush=True)
    rows=sorted(global_rows,key=engine._row_key)[:k]
    return rows,dict(count=total,contexts=context_proofs,verification_passed=True,
                     full_fixed_family_enumerated=True,score_formula_changed=False)


def rank(output):
    output=Path(output); started=time.monotonic(); result={}
    with isolated_inputs_only():
        for network in NETWORKS:
            programs,records,ddr,hashes,root,units=inputs(network)
            proof=read(root/'proof.json'); engine=engine_from(proof,network)
            costs=Costs(records,programs=programs,ddr=ddr,distribution='session_weights')
            original,graphs,old_proof=replay_original(engine,programs,costs,ddr,hashes,root,units)
            fixed,new_proof=fixed_topk(engine,programs,costs,ddr,graphs,output/network/'fixed_search')
            blend,reasons=mixed(original,fixed)
            lists={'original':original,'fixed':fixed,'mixed':blend}
            if any(len(v)!=20 or len({r['id'] for r in v})!=20 for v in lists.values()):
                raise ValueError('each frozen list must have 20 unique execution configurations')
            freeze(output/network/'rankings.json',dict(lists=lists,mixed_reasons=reasons,
                original_proof=old_proof,fixed_proof=new_proof,component_hashes=hashes,
                original_implementation=proof['implementation'],performance_read=False))
            result[network]=dict(original_count=old_proof['count'],fixed_count=new_proof['count'],
                original_first=original[0]['id'],fixed_first=fixed[0]['id'],
                overlap=len({r['id'] for r in original}&{r['id'] for r in fixed}))
    files={str(p.relative_to(output)):sha256(p) for p in sorted(output.rglob('*.json'))
           if p.name=='rankings.json' or p.parent.name=='fixed_search' and not p.stem.endswith('_cost')
           or p.name=='FILTER_PROOF.json'}
    freeze(output/'RANK_FREEZE.json',dict(protocol=protocol(),sources=sources(),files=files,
        historical_pipeline_performance_read=False))
    write(output/'RANK_COST.json',dict(seconds=time.monotonic()-started,networks=result),replace=True)
    return result
