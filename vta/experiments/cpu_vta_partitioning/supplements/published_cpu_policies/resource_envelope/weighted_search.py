"""A separate full-domain ablation using already recorded isolated thread shares.

The earlier small fixed-partition weight correction did not improve selection.
Here the new falsifiable question is whether *joint* thread-budget and host-core
selection changes when a three-thread stage's caller/helper loads differ.
No measured pipeline service time or FPS enters the coefficients.
"""
import argparse
import importlib.util
from pathlib import Path
import sys
import time
from run_resource_behavior import api,ROOT,PARENT,SOURCE
from cpu_policies.open_resources.costs import Costs
from .model import score
from .search_adapter import generated,_replace

def original_engine(output):
    c=api();parent=ROOT/'cpu_policies/open_resources'
    source=(parent/'search.py').read_text()
    source=_replace(source,'pool = [make(row) for row in pool_refs]\n    diverse = select_diverse(pure, pool) if k == 20 else []',
        'pool = []  # compact terminal strata remain available\n    diverse = []')
    contents={'search.py':source,'oracle.cc':(parent/'oracle.cc').read_text(),
        'picojson.h':(parent/'picojson.h').read_text(),'__init__.py':'"""Isolated weighted original resource model."""\n',
        'domain.py':'from cpu_policies.open_resources.domain import *\n',
        'costs.py':'from cpu_policies.open_resources.costs import Costs, score_configuration, quantize_ns, integer_weights\n'}
    name='weighted_original_'+c.digest(contents)[:20];folder=Path(output)/'sources'/name;folder.mkdir(parents=True,exist_ok=True)
    for leaf,value in contents.items():
        path=folder/leaf
        if path.exists() and path.read_text()!=value:raise ValueError('generated source differs')
        path.write_text(value)
    spec=importlib.util.spec_from_file_location(name,folder/'__init__.py',submodule_search_locations=[str(folder)])
    package=importlib.util.module_from_spec(spec);sys.modules[name]=package;spec.loader.exec_module(package)
    spec=importlib.util.spec_from_file_location(name+'.search',folder/'search.py')
    engine=importlib.util.module_from_spec(spec);sys.modules[spec.name]=engine;spec.loader.exec_module(engine)
    c.write(Path(output)/'source_identity.json',{'generated_package':name,'generated_files':c.inventory(folder),
        'parent_sources':{leaf:c.sha256(parent/leaf) for leaf in ('search.py','oracle.cc','picojson.h')},
        'adapter_sha256':c.sha256(Path(__file__)),'domain_changed':False},replace=True)
    return engine

def run(output,variant):
    c=api();started=time.monotonic();output=Path(output).resolve()
    records=c.read(SOURCE/'isolated_costs.json')['costs'];programs=c.read(PARENT/'plans/programs.json');ddr=c.read(SOURCE/'ddr.json')
    hashes={name:c.sha256(SOURCE/name) for name in ('isolated_costs.json','ddr.json')}
    from run_resource_behavior import SELECTION
    manifest=c.read(SELECTION/'manifest.json')
    if any(hashes[name]!=manifest['files']['plans/'+name]['sha256'] for name in hashes):
        raise ValueError('isolated source cost identity differs')
    missing=[r['configuration']['stages'][0] for r in records.values()
        if r['configuration']['stages'][0]['device']=='cpu' and r['configuration']['stages'][0]['threads']>1
        and 'session_thread_weights' not in r]
    # Existing three tiny t2 signatures lack resolvable whole-session shares.
    # Their original uniform fallback remains explicit, never fabricated.
    expected={('cpu_12_12',2),('cpu_14_14',2),('cpu_17_17',2)}
    if {(r['program'],r['threads']) for r in missing}!=expected:
        raise ValueError('thread-share coverage changed; review before searching')
    costs=Costs(records,programs=programs,ddr=ddr,distribution='session_weights')
    engine=generated(output) if variant=='combined' else original_engine(output)
    ranked=engine.rank_all(programs,costs,ddr=ddr,output_dir=output/'search',verify=True)
    for row in ranked['pure_static20']:
        if score(row,costs)[variant]!=row['score_ns']:raise ValueError('weighted scalar score differs')
    if ranked['counts']['enhanced']!=966240306 or not ranked['verification']['passed']:
        raise ValueError('complete domain proof failed')
    ranked['assumptions']['cpu_distribution']='whole-session isolated thread shares rescaled to scored total CPU work'
    ranked['assumptions']['thread_share_fallback_signatures']=missing
    if any(c.sha256(SOURCE/name)!=value for name,value in hashes.items()):
        raise ValueError('isolated costs changed during search')
    c.write(output/'rankings.json',ranked,replace=True)
    c.write(output/'cost.json',{'seconds':time.monotonic()-started,'variant':variant,'cpu_distribution':'session_weights',
        'costs_sha256':c.sha256(SOURCE/'isolated_costs.json'),'ddr_sha256':c.sha256(SOURCE/'ddr.json'),
        'new_profiles':0,'thread_share_fallback':missing,'domain_counts':ranked['counts'],
        'verification':ranked['verification'],'pipeline_FPS_used':False},replace=True)
    return {'variant':variant,'first':ranked['pure_static20'][0]['id'],'counts':ranked['counts']}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--variant',choices=('original','combined'),required=True);args=p.parse_args()
    print(__import__('json').dumps(run(args.output,args.variant)))
