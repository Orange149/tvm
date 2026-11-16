"""Generate an isolated DP/oracle variant, preserving the complete legal domain.

Only terminal resource accounting is changed. The existing path/resource DP
and independent scalar enumerator are source-patched in an artifact directory,
not edited in place. Both additions consume local isolated profile fields.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
from .model import VARIANTS

def _replace(text,old,new):
    if text.count(old)!=1:raise ValueError('audited source anchor changed: '+old[:90])
    return text.replace(old,new)

def generated(output):
    from run_resource_behavior import api,ROOT
    from cpu_policies.open_resources import search
    c=api();original=Path(search.__file__).parent
    source=(original/'search.py').read_text()
    source=_replace(source,'states = {"core": np.zeros((1, 4), dtype=np.int64),',
        'states = {"reservation": np.zeros((1, 4), dtype=np.int64), "core": np.zeros((1, 4), dtype=np.int64),')
    source=_replace(source,'core = np.zeros((m, 4), dtype=np.int64)',
        'core = np.zeros((m, 4), dtype=np.int64)\n        reservation = np.zeros((m, 4), dtype=np.int64)')
    source=_replace(source,'core[j, mask[0]] += values[j, 2]',
        'core[j, mask[0]] += values[j, 2]\n            reservation[j, list(mask)] = values[j, 0]')
    source=_replace(source,'(("core", core), ("groups", groups), ("prefix", prefix))',
        '(("core", core), ("reservation", reservation), ("groups", groups), ("prefix", prefix))')
    source=_replace(source,'enhanced = np.maximum(service, demand)',
        'demand = np.maximum(demand, add("reservation").max(axis=1))\n'
        '    demand = np.maximum(demand, (core[:, host_mask] + host).max(axis=1))\n'
        '    enhanced = np.maximum(service, demand)')
    source=_replace(source,'pool = [make(row) for row in pool_refs]\n    diverse = select_diverse(pure, pool) if k == 20 else []',
        'pool = []  # compact checkpoints retain all bounded strata\n    diverse = []  # behavior selection is separate and frozen before board execution')
    cpp=(original/'oracle.cc').read_text()
    cpp=_replace(cpp,'std::array<std::int64_t,4> core{};',
        'std::array<std::int64_t,4> core{};\n  std::array<std::int64_t,4> team_core{};')
    cpp=_replace(cpp,'for (int k=0;k<widths[option];++k) core[options[option][k]]+=c.allocated[k];',
        'for (int k=0;k<widths[option];++k) { core[options[option][k]]+=c.allocated[k]; team_core[options[option][k]]+=c.wall; }')
    cpp=_replace(cpp,'auto service=std::max({cpu_max,vta,ddr});',
        'for (int id=0;id<4;++id) {\n'
        '    bool permitted=std::find(host.begin(),host.end(),id)!=host.end();\n'
        '    capacity=std::max({capacity,team_core[id],core[id]+(permitted ? host_work : 0)});\n'
        '  }\n  auto service=std::max({cpu_max,vta,ddr});')
    contents={'search.py':source,'oracle.cc':cpp,'picojson.h':(original/'picojson.h').read_text(),
        '__init__.py':'"""Frozen resource-envelope DP generated from the audited parent."""\n',
        'domain.py':'from cpu_policies.open_resources.domain import *\n',
        'costs.py':'from cpu_policies.open_resources.costs import Costs, score_configuration, quantize_ns, integer_weights\n'}
    fingerprint=c.digest(contents)[:20];name='envelope_generated_'+fingerprint
    root=Path(output)/'sources'/name;root.mkdir(parents=True,exist_ok=True)
    for leaf,text in contents.items():
        p=root/leaf
        if p.exists() and p.read_text()!=text:raise ValueError('generated source changed')
        if not p.exists():p.write_text(text)
    spec=importlib.util.spec_from_file_location(name,root/'__init__.py',submodule_search_locations=[str(root)])
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
    spec=importlib.util.spec_from_file_location(name+'.search',root/'search.py')
    engine=importlib.util.module_from_spec(spec);sys.modules[spec.name]=engine;spec.loader.exec_module(engine)
    c.write(Path(output)/'source_identity.json',{'generated_package':name,
        'parent_sources':{leaf:c.sha256(original/leaf) for leaf in ('search.py','oracle.cc','picojson.h')},
        'generated_files':c.inventory(root),'adapter_sha256':c.sha256(Path(__file__)),
        'domain_changed':False,'arithmetic':'integer nanoseconds; maxima of separate resource envelopes',
        'no_new_CPU_or_pipeline_measurements':True},replace=True)
    return engine

def run(output):
    import time
    from run_resource_behavior import api,PARENT,SOURCE
    from .model import score
    c=api();start=time.monotonic();output=Path(output).resolve();engine=generated(output)
    programs=c.read(PARENT/'plans/programs.json');records=c.read(SOURCE/'isolated_costs.json')['costs'];ddr=c.read(SOURCE/'ddr.json')
    ranked=engine.rank_all(programs,records,ddr=ddr,output_dir=output/'search',verify=True)
    from cpu_policies.open_resources.costs import Costs
    costs=Costs(records,programs=programs,ddr=ddr)
    for row in ranked['pure_static20']:
        if score(row,costs)['combined']!=row['score_ns']:raise ValueError('scalar formula and DP score differ')
    if ranked['counts']['enhanced']!=966240306 or not ranked['verification']['passed']:
        raise ValueError('complete domain or independent oracle validation incomplete')
    ranked['model_formulas']['enhanced']='max(original capacity model, max_core sum_CPU_stage_service_on_core, max_core fixed_CPU_work_plus_all_host_work_if_allowed)'
    ranked['assumptions']['resource_envelope']='conservative planning proxies, not proven actual latency bounds'
    c.write(output/'rankings.json',ranked,replace=True)
    c.write(output/'cost.json',{'seconds':time.monotonic()-start,'new_profiles':0,'costs_sha256':c.sha256(SOURCE/'isolated_costs.json'),
        'ddr_sha256':c.sha256(SOURCE/'ddr.json'),'domain_counts':ranked['counts'],'verification':ranked['verification']},replace=True)
    return ranked

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=run(args.output)
    print(json.dumps({'counts':result['counts'],'verified':result['verification'],
        'first20':[{k:r[k] for k in ('id','context_id','execution_mode','score_ns')} for r in result['pure_static20']]},ensure_ascii=False))
