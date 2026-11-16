#!/usr/bin/env python3
"""Six prospective static-choice/host-role checks using the frozen probe controller."""
from copy import deepcopy
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from run_resource_behavior import ROOT,PARENT,SELECTION,SOURCE,api
from resource_envelope.search_adapter import _replace

def plan(original_search,envelope_search):
    from cpu_policies.open_resources.costs import Costs
    from cpu_policies.open_resources.domain import candidate,configuration
    from resource_envelope.model import score
    c=api();sources={}
    for key,path in [('weighted_original_first',original_search),('weighted_guard_first',envelope_search)]:
        path=Path(path).resolve();ranked=c.read(path/'rankings.json');cost=c.read(path/'cost.json')
        if not ranked['verification']['passed'] or ranked['counts']['enhanced']!=966240306:
            raise ValueError('both complete-domain proofs required before freezing a probe')
        if cost['cpu_distribution']!='session_weights':raise ValueError('weighted cost semantics required')
        sources[key]={'configuration':ranked['pure_static20'][0],'search_cost':cost,
            'rankings_sha256':c.sha256(path/'rankings.json')}
    old=c.read(ROOT/'artifacts/bundles/afcb8f95255f7d27e637a4e2/plans/selection_rankings.json')
    configurations={'static_first':old['pure_static20'][0],'uniform_guard_first':old['diverse20'][0],
        **{name:value['configuration'] for name,value in sources.items()}}
    for host,name in [(2,'helper2_matched'),(3,'single_core_matched')]:
        rows=deepcopy(configurations['uniform_guard_first']['stages'])
        for row in rows:
            if row['device']=='vta':row['mask']=[host]
        configurations[name]=candidate(rows,'host_shared_'+str(host),'off')
    configurations={k:{**configuration(v),'id':v['id']} for k,v in configurations.items()}
    if len({r['id'] for r in configurations.values()})!=6:
        raise ValueError('prospective choices overlap; revise the task count instead of adding fillers')
    programs=c.read(PARENT/'plans/programs.json');records=c.read(SOURCE/'isolated_costs.json')['costs'];ddr=c.read(SOURCE/'ddr.json')
    tables={key:Costs(records,programs=programs,ddr=ddr,distribution=key) for key in ('balanced','session_weights')}
    predictions={name:{mode:score(cfg,table) for mode,table in tables.items()} for name,cfg in configurations.items()}
    return {'configurations':configurations,'predictions':predictions,'sources':sources,
        'labels':{'static_first':'原均分静态第一名','uniform_guard_first':'均分包络第一名／主机核心1',
            'weighted_original_first':'逐线程核时原公式第一名','weighted_guard_first':'逐线程核时包络第一名',
            'helper2_matched':'匹配放置／主机核心2','single_core_matched':'匹配放置／主机核心3'},
        'hypotheses':['isolated caller/helper load differences affect host-placement ranking',
            'joint thread-budget and host placement may differ from earlier fixed-partition weight correction'],
        'pipeline_FPS_used_for_predictions':False,'component_profiles_added':0,
        'decision_rule':'use this as model-development evidence; no score coefficients fitted and no independent confirmation claimed'}

def prepare(original_search,envelope_search):
    c=api();doc=plan(original_search,envelope_search)
    source=(ROOT/'run_resource_behavior.py').read_text()
    for name,path in [('PARENT',PARENT),('SELECTION',SELECTION),('SOURCE',SOURCE)]:
        old=next(line for line in source.splitlines() if line.startswith(name+'='))
        source=_replace(source,old,name+'=Path('+repr(str(path))+')')
    source=source.replace('resnet18_resource_behavior_mechanism','resnet18_thread_work_probe')
    source=_replace(source,'    from cpu_policies.diagnosis.prepare import _copy\n',
        '''    def _copy(src,dst,expected=None):
        src,dst=Path(src),Path(dst)
        if expected is not None and c.sha256(src)!=expected:raise ValueError('source asset hash differs')
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
''')
    start=source.index('def configurations(');end=source.index('\ndef _make_task',start)
    source=source[:start]+'''def configurations(parent=PARENT,selection=SELECTION):
    from cpu_policies.open_resources.domain import validate_candidate
    c=api(parent);programs=c.read(Path(parent)/'plans/programs.json')
    result=c.read(ROOT/'configs/thread_work_probe.json')['configurations']
    for value in result.values():validate_candidate(value,programs)
    if len(result)!=6 or len({r['id'] for r in result.values()})!=6:raise ValueError('six distinct cases required')
    return result

'''+source[end:]
    source=_replace(source,"    tests=subprocess.run([sys.executable,'-m','pytest'",
        "    c.write(work/'plans/thread_work_probe.json',c.read(ROOT/'configs/thread_work_probe.json'))\n"
        "    tests=subprocess.run([sys.executable,'-m','pytest'")
    start=source.index('    for a,b,label in [');end=source.index('        per_round=',start)
    source=source[:start]+'''    for a,b,label in [('helper2_matched','uniform_guard_first','同一计算配置：主机核心2相对核心1'),
        ('single_core_matched','uniform_guard_first','同一计算配置：主机核心3相对核心1')]:
'''+source[end:]
    start=source.index('    labels={');end=source.index('    for r in cases:',start)
    source=source[:start]+"    labels=c.read(bundle/'plans/thread_work_probe.json')['labels']\n"+source[end:]
    source=source.replace('# 资源机制六项匹配对照','# 逐线程核时及主机角色：六项验证')
    source=source.replace('历史方案仅作机制参照，不进入生产Top-20优待。','模型选择与匹配放置诊断，不作为独立确认或Top-20特殊名额。')
    source=_replace(source,"root=ROOT/'runs'/args.run_id","root=PROJECT/'supplements/published_cpu_policies/runs'/args.run_id")
    tests=(ROOT/'tests/test_resource_behavior.py').read_text()
    start=tests.index('def test_six_cases_');end=tests.index('\ndef test_round_identity',start)
    tests=tests[:start]+'''def test_six_static_choices_and_matched_host_roles():
    configs=campaign.configurations();assert len(configs)==len({r['id'] for r in configs.values()})==6
    cpu=lambda name:[s for s in configs[name]['stages'] if s['device']=='cpu']
    for name in ('helper2_matched','single_core_matched'):
        assert cpu(name)==cpu('uniform_guard_first')
    assert configs['uniform_guard_first']['host_mask']==[1]
    assert configs['helper2_matched']['host_mask']==[2]
    assert configs['single_core_matched']['host_mask']==[3]

'''+tests[end:]
    tests=tests.replace("old=ROOT/'runs/resnet18_autotvm_diverse_top20_01'",
        "old=Path("+repr(str(ROOT/'runs/resnet18_autotvm_diverse_top20_01'))+")")
    start=tests.index('\ndef test_existing_completed_campaign_identity_is_unchanged')
    tests=tests[:start]
    fingerprint=c.digest({'plan':doc,'source':source,'tests':tests})[:24]
    stage=ROOT/'artifacts/thread_work_probe_sources'/fingerprint
    if not stage.exists():
        stage.mkdir(parents=True)
        for name in ('cpu_policies','configs'):shutil.copytree(ROOT/name,stage/name,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
        shutil.copy2(ROOT/'pyproject.toml',stage/'pyproject.toml');(stage/'tests').mkdir()
        (stage/'tests/test_resource_behavior.py').write_text(tests)
        (stage/'run_resource_behavior.py').write_text(source)
        c.write(stage/'configs/thread_work_probe.json',doc)
        c.write(stage/'ADAPTER.json',{'parent_source_sha256':c.sha256(ROOT/'run_resource_behavior.py'),
            'adapter_sha256':c.sha256(Path(__file__)),'frozen_plan_sha256':c.sha256(stage/'configs/thread_work_probe.json'),
            'changes':'six frozen static/host-role cases, labels and comparisons only; unchanged transport and native executor'})
    (stage/'artifacts/bundles').mkdir(parents=True,exist_ok=True)
    result=subprocess.run([sys.executable,str(stage/'run_resource_behavior.py'),'prepare'],cwd=stage,capture_output=True,text=True,timeout=240)
    if result.returncode:raise RuntimeError(result.stdout+result.stderr)
    return json.loads(result.stdout.splitlines()[-1])['bundle']

def main():
    args=sys.argv[1:]
    if args and args[0]=='prepare':
        p=argparse.ArgumentParser();p.add_argument('--original-search',type=Path,required=True);p.add_argument('--envelope-search',type=Path,required=True)
        options=p.parse_args(args[1:]);print(json.dumps({'bundle':prepare(options.original_search,options.envelope_search)}));return
    if '--bundle' in args:
        i=args.index('--bundle')+1;bundle=Path(args[i]).resolve();args[i]=str(bundle)
    elif '--run-dir' in args:
        i=args.index('--run-dir')+1;run=Path(args[i]).resolve();args[i]=str(run);bundle=Path(json.loads((run/'state.json').read_text())['bundle'])
    else:raise ValueError('frozen --bundle or --run-dir required')
    controller=bundle/'sources/behavior_controller'
    raise SystemExit(subprocess.run([sys.executable,str(controller/'run_resource_behavior.py'),*args],cwd=controller).returncode)

if __name__=='__main__':main()
