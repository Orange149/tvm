"""Six distinct kernel boots: discover, check/profile, screen, confirm x3."""
from copy import deepcopy
from pathlib import Path
import math
import random
import statistics

from cpu_vta_partitioning.common import (read,write,sha256,digest,inventory,verify_inventory,locked,event,now,sync_tree)
from ..campaign import _save,_pin,_publish
from ..evidence import identified
from .domain import (ROOT,CORES,CUTS,FAMILIES,protocol,anchor,cut_candidate,component,component_key,
                     policy_components,complete_components,partitions,stage)
from .evidence import Evidence,FIELDS,validate_summary,validate_import
from .transport import make_task
from .search import (Costs,mirror_check,policy_selection,freeze_rule,rank_dp,enumerate_reference)


def validate_state(root):
    from .prepare import identity,validate
    root=Path(root).resolve();s=read(root/'state.json');m=read(Path(s['bundle'])/'manifest.json')
    if (s['experiment']!=protocol()['experiment'] or s['controller_identity']!=identity()
            or m['controller_identity']!=s['controller_identity'] or m['bundle_id']!=s['bundle_id']
            or sha256(Path(s['bundle'])/'manifest.json')!=s['bundle_manifest_sha256']):
        raise ValueError('campaign/controller identity changed; use its snapshot')
    if read(root/'protocol.json')!=protocol():raise ValueError('campaign protocol differs')
    verify_inventory(root,s['frozen_files'])
    if (type(s['boot_index'])is not int or not 0<=s['boot_index']<=6 or len(s['completed_boots'])!=s['boot_index']
            or len(set(s['completed_boots']))!=s['boot_index'] or set(s['completed_boots'])&set(s['excluded_boots'])):
        raise ValueError('invalid independent boot accounting')
    if s['status']=='COMPLETE' and s['boot_index']!=6:raise ValueError('incomplete campaign marked complete')
    validate(s['bundle'])
    return s


def create(bundle,root,connection,excluded_boot):
    from .prepare import validate,identity
    root=Path(root).resolve();bundle=Path(bundle).resolve();m=validate(bundle)
    if root.exists():raise FileExistsError('run exists; resume it')
    root.mkdir(parents=True)
    s={'experiment':protocol()['experiment'],'controller_identity':identity(),'bundle':str(bundle),'bundle_id':m['bundle_id'],
       'bundle_manifest_sha256':sha256(bundle/'manifest.json'),'connection':connection,'created':now(),
       'status':'WAITING_NEW_BOOT','boot_index':0,'completed_boots':[],'excluded_boots':[excluded_boot],
       'active_boot':None,'abandoned_boots':[],'completed_rounds':[],'abandoned_rounds':[],'frozen_files':{}}
    write(root/'protocol.json',protocol());_pin(root,s,'protocol.json')
    for key,item in validate_import(bundle).items():
        _publish(root,s,'components/'+key+'.json',{'kind':'imported',**item})
    return s


def task(s,kind,c,*,warmup=0,scored=4,tail=0,indices=None,index=None,**labels):
    i=s['boot_index'] if index is None else index
    return make_task(kind,c,s['bundle'],warmup=warmup,scored=scored,tail=tail,indices=indices,
                     boot_index=i,campaign_phase=protocol()['boot_phases'][i] if i>=0 else 'preflight',**labels)


def performance(s,kind,c,warmup,**labels):
    return task(s,kind,c,warmup=warmup,scored=32,tail=4,indices=[16],**labels)


class Recorder:
    def __init__(self,root,state,transport,remote):
        self.root=Path(root).resolve();self.state=state;self.transport=transport;self.remote=remote;self.boot=state['active_boot']
        self.evidence=Evidence(self.root,state['bundle'])

    def saved(self,name,doc):
        p=self.root/'records'/self.boot/(name+'.json')
        if p.exists():
            r=read(p)
            if r['document']!=doc or r['boot_id']!=self.boot:raise ValueError('record identity differs')
            value=r['summary'];self.evidence.summary(value,doc,self.boot)
        else:
            receipt=self.root/'accepted'/self.boot/(doc['task_id']+'.json')
            if receipt.exists():
                r=read(receipt);value={**r['summary'],'_receipt':{'path':str(receipt.relative_to(self.root)),'sha256':sha256(receipt)}}
                self.evidence.summary(value,doc,self.boot)
            else:
                print('[资源策略] '+name,flush=True);event(self.root,'session_start',boot_id=self.boot,task_id=doc['task_id'])
                raw=Path(self.transport.measure(doc,self.state['bundle'],self.remote,self.root,self.boot)).resolve()
                if self.transport.boot_id()!=self.boot or not raw.is_relative_to(self.root/'raw'/self.boot/doc['task_id']):
                    raise ValueError('native result belongs to another boot or directory')
                summary=validate_summary(raw,doc,self.state['bundle']);sync_tree(raw)
                files={str(raw.relative_to(self.root)/n):v for n,v in inventory(raw).items()}
                write(receipt,{'task':doc,'boot_id':self.boot,'summary':summary,'files':files})
                value={**summary,'_receipt':{'path':str(receipt.relative_to(self.root)),'sha256':sha256(receipt)}}
                self.evidence.summary(value,doc,self.boot);event(self.root,'session_accepted',boot_id=self.boot,task_id=doc['task_id'])
            write(p,{'document':doc,'summary':value,'boot_id':self.boot})
        _pin(self.root,self.state,str(p.relative_to(self.root)),value['_receipt']['path'])
        pending=self.root/'cleanup_pending'/self.boot/(doc['task_id']+'.json')
        try:self.transport.acknowledge(doc,self.root,self.boot)
        except (RuntimeError,OSError) as e:write(pending,{'document':doc,'boot':self.boot},replace=True)
        else:pending.unlink(missing_ok=True)
        return value


def load_costs(root,s,verify=True):
    root=Path(root);result={};source=read(Path(s['bundle'])/'plans/imported_profiles.json')['records']
    ev=Evidence(root,s['bundle'])
    for p in sorted((root/'components').glob('*.json')):
        r=read(p);key=p.stem
        if component_key(r['configuration'])!=key:raise ValueError('component key differs')
        if r['kind']=='imported':
            if r!={'kind':'imported',**source[key]}:raise ValueError('imported component was changed')
        else:
            if r['kind']!='measured' or len(r['blocks'])!=1:raise ValueError('invalid component provenance')
            for b in r['blocks']:
                d=b['document']
                if (d['kind']!='calibration' or d['configuration']!=r['configuration'] or d['input_indices']!=[0]
                        or (d['warmup'],d['scored'],d['tail'])!=(5,32,0)):raise ValueError('component measurement protocol differs')
                if verify:ev.summary(b['summary'],d,b['boot_id'])
            q=r['qualification']
            if q['document']['kind']!='qualify_component' or q['document']['configuration']!=r['configuration']:
                raise ValueError('component correctness qualification differs')
            if verify:ev.summary(q['summary'],q['document'],q['boot_id'])
            expected={f:statistics.mean(b['summary'][f] for b in r['blocks']) for f in FIELDS}
            if r['cost']!=expected:raise ValueError('component means disagree with raw records')
        result[key]=r
    return result


def ensure_components(root,s,rec,required):
    result=load_costs(root,s)
    for number,(key,c) in enumerate(sorted(required.items())):
        if key in result:continue
        qdoc=task(s,'qualify_component',c,component_key=key)
        q=rec.saved('components/'+key+'/qualify',qdoc)
        d=task(s,'calibration',c,warmup=5,scored=32,indices=[0],component_key=key)
        value=rec.saved('components/'+key+'/calibration',d)
        item={'kind':'measured','configuration':c,'cost':{f:value[f] for f in FIELDS},
              'blocks':[{'document':d,'summary':value,'boot_id':rec.boot}],
              'qualification':{'document':qdoc,'summary':q,'boot_id':rec.boot}}
        _publish(root,s,'components/'+key+'.json',item);result[key]=item
    return result


def qualify_pool(s,rec,pool):
    for key,c in sorted(pool.items()):rec.saved('qualify/'+key,task(s,'qualify_pipeline',c,candidate_id=key))


def warmup(root,s,rec,selection):
    if (Path(root)/'warmup.json').exists():return read(Path(root)/'warmup.json')['warmup']
    wanted=[anchor(),selection['pool'][selection['families']['current']['local_fastest']],
            selection['pool'][selection['families']['current']['fixed_groups']]]
    checks={};extended=False
    for i,c in enumerate(wanted):
        values={}
        for n in (8,24):
            d=performance(s,'warmup',c,n,warmup_anchor=i,warmup_repeat=0)
            values[str(n)]={'document':d,'summary':rec.saved(f'warmup/{i}/{n}',d),'boot_id':rec.boot}
        extended|=abs(values['8']['summary']['ii_ms']/values['24']['summary']['ii_ms']-1)>.05
        checks[str(i)]=values
    if extended:
        for i,c in enumerate(wanted):
            d=performance(s,'warmup',c,24,warmup_anchor=i,warmup_repeat=1)
            v=rec.saved(f'warmup/{i}/24_repeat',d);checks[str(i)]['repeat']={'document':d,'summary':v,'boot_id':rec.boot}
            if abs(v['ii_ms']/checks[str(i)]['24']['summary']['ii_ms']-1)>.05:
                _publish(root,s,'warmup_failure.json',checks);raise RuntimeError('extended warmup unstable; stop and investigate')
    _publish(root,s,'warmup.json',{'warmup':24 if extended else 8,'checks':checks})
    return 24 if extended else 8


def run_rounds(root,s,rec,pool,role,nwarm):
    root=Path(root);directory=root/'phases'/('boot'+str(s['boot_index']+1))/role;directory.mkdir(parents=True,exist_ok=True)
    answer=[]
    for number in range(2):
        p=directory/f'round{number}.json'
        if p.exists():
            record=read(p);validate_round(root,s,record,pool,nwarm)
            if record['boot_id']!=rec.boot:raise ValueError('copied round from wrong boot')
        else:
            for attempt in range(2):
                invalid=directory/f'invalid_round{number}_attempt{attempt}.json'
                if invalid.exists():continue
                common={'role':role,'round':number,'attempt':attempt}
                before_doc=performance(s,'anchor',anchor(),nwarm,position='before',**common)
                before=rec.saved(f'{role}/round{number}/attempt{attempt}/before',before_doc)
                order=sorted(pool);random.Random(20261104+s['boot_index']*101+attempt*1009).shuffle(order)
                if number:order.reverse()
                values={}
                for key in order:
                    d=performance(s,'performance',pool[key],nwarm,candidate_id=key,**common)
                    values[key]=rec.saved(f'{role}/round{number}/attempt{attempt}/'+key,d)
                after_doc=performance(s,'anchor',anchor(),nwarm,position='after',**common)
                after=rec.saved(f'{role}/round{number}/attempt{attempt}/after',after_doc)
                record={'boot_id':rec.boot,'boot_index':s['boot_index'],'role':role,'round':number,'attempt':attempt,
                        'order':order,'measurements':values,'before':before,'after':after}
                if abs(after['ii_ms']/before['ii_ms']-1)>.05:
                    write(invalid,{**record,'invalid_reason':'anchor_drift'});_pin(root,s,str(invalid.relative_to(root)));continue
                write(p,record);_pin(root,s,str(p.relative_to(root)));break
            else:
                s['status']='STOPPED_ANCHOR_DRIFT';_save(root,s)
                raise RuntimeError('anchor drift exhausted single replacement; failed evidence retained')
        name=str(p.relative_to(root))
        if name not in s['completed_rounds']:s['completed_rounds'].append(name);_save(root,s)
        answer.append(record)
    return answer


def executor_bridge(root,s,rec,nwarm):
    path=Path(root)/'executor_bridge.json'
    if path.exists():
        value=read(path)
        if not value['passed']:raise RuntimeError('executor bridge failed; resume cannot bypass it')
        return value
    configs=[anchor(),cut_candidate('current',(2,2),((0,1),(2,3)))]
    reports=[]
    for c in configs:
        q=task(s,'qualify_executor_bridge',c,executor_override='policy',candidate_id=c['id'])
        rec.saved('executor_bridge/'+c['id']+'/qualify',q)
        rows=[]
        for repeat in range(2):
            values={}
            for exe in (('legacy','policy') if repeat==0 else ('policy','legacy')):
                d=performance(s,'executor_bridge',c,nwarm,executor_override='policy' if exe=='policy' else None,
                              bridge_repeat=repeat,candidate_id=c['id'])
                v=rec.saved(f"executor_bridge/{c['id']}/{repeat}/{exe}",d)
                values[exe]={'document':d,'summary':v,'boot_id':rec.boot}
            rows.append(values)
        legacy=statistics.mean(r['legacy']['summary']['ii_ms'] for r in rows)
        modern=statistics.mean(r['policy']['summary']['ii_ms'] for r in rows)
        reports.append({'configuration':c,'relative_difference':abs(modern/legacy-1),'measurements':rows})
    result={'scope':'matched group-mode executors; diagnostics only, not selection or independent samples',
            'passed':all(r['relative_difference']<=.05 for r in reports),'reports':reports}
    _publish(root,s,'executor_bridge.json',result)
    if not result['passed']:
        s['status']='STOPPED_EXECUTOR_DIFFERENCE';_save(root,s)
        raise RuntimeError('executor bridge exceeds 5%; investigate binary/setup difference before attributing pinning')
    return result


def means(rounds):
    if len(rounds)!=2 or {r['round'] for r in rounds}!={0,1} or len({r['boot_id'] for r in rounds})!=1:raise ValueError('two same-boot rounds required')
    if set(rounds[0]['measurements'])!=set(rounds[1]['measurements']):raise ValueError('round coverage differs')
    return {k:statistics.mean(r['measurements'][k]['ii_ms'] for r in rounds) for k in rounds[0]['measurements']}


def validate_round(root,s,r,pool,nwarm):
    i=r['boot_index'];expected=s['completed_boots'][i] if type(i)is int and 0<=i<len(s['completed_boots']) else s['active_boot'] if i==s['boot_index'] else None
    if (type(i)is not int or not 0<=i<6 or r['boot_id']!=expected or r['round'] not in (0,1) or r['attempt'] not in (0,1)
            or sorted(r['order'])!=sorted(pool) or len(r['order'])!=len(pool) or set(r['measurements'])!=set(pool)):
        raise ValueError('round boot/coverage differs')
    ev=Evidence(root,s['bundle']);common={'role':r['role'],'round':r['round'],'attempt':r['attempt'],'index':i}
    for pos in ('before','after'):ev.summary(r[pos],performance(s,'anchor',anchor(),nwarm,position=pos,**common),r['boot_id'])
    if abs(r['after']['ii_ms']/r['before']['ii_ms']-1)>.05:raise ValueError('accepted anchor drift exceeds threshold')
    for key,value in r['measurements'].items():ev.summary(value,performance(s,'performance',pool[key],nwarm,candidate_id=key,**common),r['boot_id'])


def search_pool(rankings):
    rows=rankings['enhanced']['rankings'][:20]+rankings['single_core']['rankings'][:5]+rankings['prefix']['rankings'][:5]
    pool={r['id']:{k:r[k] for k in ('id','stages','affinity_mode','execution_mode','host_mask','control_mask','core_partition')} for r in rows}
    if len(pool)>30:raise ValueError('Top20 search pool exceeds 30')
    return pool


def final_selection(rankings,rounds,selection,rule):
    measured=means(rounds);chosen={}
    for method in ('single_core','prefix','enhanced'):
        rows=rankings[method]['rankings'];chosen[method]={}
        for budget in ((1,5,20) if method=='enhanced' else (1,5)):
            keys=[r['id'] for r in rows[:budget]]
            chosen[method][str(budget)]=min(keys,key=lambda k:(measured[k],k))
    pool=search_pool(rankings);wanted={k for v in chosen.values() for k in v.values()}
    # Keep policy counterparts independent of the full-search winner.
    for cut in CUTS:
        names=selection['families'][cut]
        for name in {'current','local_fastest','fixed_groups','same_groups_pinned',rule['family']}:wanted.add(names[name])
        for pair in selection['pairs'][cut].values():wanted.update(pair)
    all_configs={**selection['pool'],**pool}
    return {'methods':chosen,'pool':{k:all_configs[k] for k in sorted(wanted)},'screening_means':measured,
            'selection_boot_id':rounds[0]['boot_id'],'rule_family':rule['family']}


def _archive(root,s,old):
    names=[n for n in s['completed_rounds'] if read(Path(root)/n)['boot_id']==old]
    s['completed_rounds']=[n for n in s['completed_rounds'] if n not in names];s['abandoned_rounds'].extend(names)
    phase=Path('phases')/('boot'+str(s['boot_index']+1))
    to_move=[str(phase)] if (Path(root)/phase).exists() else []
    if s['boot_index']==0:to_move += [n for n in ('warmup.json','warmup_failure.json') if (Path(root)/n).exists()]
    if s['boot_index']==1:to_move += [n for n in ('resource_rule.json','full_cost_keys.json','rankings.json','search_pool.json') if (Path(root)/n).exists()]
    if s['boot_index']==2:to_move += [n for n in ('final_selection.json',) if (Path(root)/n).exists()]
    for name in to_move:
        dest='abandoned/'+old+'/'+name;(Path(root)/dest).parent.mkdir(parents=True,exist_ok=True);(Path(root)/name).rename(Path(root)/dest)
        for rel in list(s['frozen_files']):
            if rel==name or rel.startswith(name+'/'):s['frozen_files'][dest+rel[len(name):]]=s['frozen_files'].pop(rel)
        s['abandoned_rounds']=[dest+n[len(name):] if n==name or n.startswith(name+'/') else n for n in s['abandoned_rounds']]
    s['abandoned_boots'].append(old);s['active_boot']=None;_save(root,s)


def run_current(root,transport):
    from .prepare import validate
    root=Path(root).resolve()
    with locked(root):
        s=validate_state(root)
        if s['status']=='COMPLETE':return s
        boot=transport.boot_id()
        if boot in s['completed_boots']+s['excluded_boots']+s['abandoned_boots']:
            s['status']='WAITING_NEW_BOOT';_save(root,s);return s
        transport.wait_idle(boot);transport.mount_storage()
        if s['active_boot'] and s['active_boot']!=boot:
            transport.recover_abandoned(root,s['active_boot']);_archive(root,s,s['active_boot'])
        remote=transport.deploy(s['bundle'],root)
        transport.verify_payload(remote,boot);transport.qualify_environment(remote,boot,root,s['bundle'])
        s['active_boot']=boot;s['status']='RUNNING';_save(root,s);rec=Recorder(root,s,transport,remote)
        for p in sorted((root/'cleanup_pending').glob('*/*.json')):
            value=read(p);transport.acknowledge(value['document'],root,value['boot']);p.unlink()
        index=s['boot_index']
        if index==0:
            records=ensure_components(root,s,rec,policy_components())
            check=mirror_check(Costs(records,False))
            if not check['homogeneous']:
                extra={}
                for a,b in CUTS.values():
                    for begin,end in ((0,a),(b,21)):
                        for mode in ('group','pinned'):
                            for mask in sorted({g for part in partitions(CORES,True) for g in part}):
                                c=component(stage('cpu',begin,end,len(mask),mask),mode);extra[component_key(c)]=c
                records=ensure_components(root,s,rec,extra)
            if not (root/'policy_cost_keys.json').exists():_publish(root,s,'policy_cost_keys.json',sorted(records))
            subset={k:records[k] for k in read(root/'policy_cost_keys.json')}
            selection=_publish(root,s,'policy_selection.json',policy_selection(subset))
        else:selection=read(root/'policy_selection.json')
        if index<2:pool=selection['pool'];role='policy_discovery' if index==0 else 'policy_check'
        elif index==2:pool=read(root/'search_pool.json');role='top20_screening'
        else:pool=read(root/'final_selection.json')['pool'];role='confirmation'
        qualify_pool(s,rec,pool)
        nwarm=warmup(root,s,rec,selection)
        if index==0:executor_bridge(root,s,rec,nwarm)
        rounds=run_rounds(root,s,rec,pool,role,nwarm)
        if index==1:
            discovery=[read(root/n) for n in s['completed_rounds'] if read(root/n)['role']=='policy_discovery']
            rule=_publish(root,s,'resource_rule.json',freeze_rule(selection,means(discovery),means(rounds)))
            programs=read(Path(s['bundle'])/'plans/programs.json')
            required=complete_components(programs,rule['affinity_mode'],rule['reserved_host'],rule['homogeneous'])
            records=ensure_components(root,s,rec,required)
            _publish(root,s,'full_cost_keys.json',sorted(required))
            ranked={}
            for method in ('single_core','prefix','enhanced'):
                print('[资源策略] 完整切图DP与独立枚举核验: '+method,flush=True)
                dp=rank_dp(programs,records,rule,method);reference=enumerate_reference(programs,records,rule,method)
                if [(r['id'],r['score_ms']) for r in dp['rankings']]!=[(r['id'],r['score_ms']) for r in reference['rankings']]:
                    raise ValueError('complete-domain DP/independent enumeration Top20 differs')
                dp['enumeration_verified']=True;dp['enumerated_terminals']=reference['visited_terminals'];ranked[method]=dp
            _publish(root,s,'rankings.json',ranked);_publish(root,s,'search_pool.json',search_pool(ranked))
        if index==2:
            _publish(root,s,'final_selection.json',final_selection(read(root/'rankings.json'),rounds,selection,read(root/'resource_rule.json')))
        if transport.boot_id()!=boot:raise RuntimeError('boot changed before acceptance')
        transport.verify_payload(remote,boot);transport.idle_command('sync',boot=boot,timeout=120)
        s['completed_boots'].append(boot);s['boot_index']+=1;s['active_boot']=None
        s['status']='COMPLETE' if s['boot_index']==6 else 'WAITING_NEW_BOOT';_save(root,s);sync_tree(root)
        return s


def validated_rounds(root):
    root=Path(root).resolve();s=validate_state(root);records=load_costs(root,s)
    selection=read(root/'policy_selection.json') if (root/'policy_selection.json').exists() else None
    if selection:
        keys=read(root/'policy_cost_keys.json')
        if selection!=policy_selection({k:records[k] for k in keys}):raise ValueError('policy selection differs from frozen component profiles')
    ev=Evidence(root,s['bundle'])
    for path in sorted((root/'accepted').glob('*/*.json')):
        receipt=read(path)
        value={**receipt['summary'],'_receipt':{'path':str(path.relative_to(root)),'sha256':sha256(path)}}
        ev.summary(value,receipt['task'],receipt['boot_id'])
    if (root/'executor_bridge.json').exists():
        bridge=read(root/'executor_bridge.json')
        for report in bridge['reports']:
            for pair in report['measurements']:
                for e in pair.values():ev.summary(e['summary'],e['document'],e['boot_id'])
            a=statistics.mean(p['legacy']['summary']['ii_ms'] for p in report['measurements'])
            b=statistics.mean(p['policy']['summary']['ii_ms'] for p in report['measurements'])
            if report['relative_difference']!=abs(b/a-1):raise ValueError('executor bridge result differs')
        if bridge['passed']!=all(r['relative_difference']<=.05 for r in bridge['reports']):raise ValueError('executor bridge decision differs')
    if not (root/'warmup.json').exists():return []
    w=read(root/'warmup.json');nwarm=w['warmup'];extended=False
    for values in w['checks'].values():
        for entry in values.values():ev.summary(entry['summary'],entry['document'],entry['boot_id'])
        extended|=abs(values['8']['summary']['ii_ms']/values['24']['summary']['ii_ms']-1)>.05
        if 'repeat' in values and abs(values['repeat']['summary']['ii_ms']/values['24']['summary']['ii_ms']-1)>.05:raise ValueError('extended warmup unstable')
    if nwarm!=(24 if extended else 8) or any(('repeat'in v)!=extended for v in w['checks'].values()):raise ValueError('warmup rule changed')
    final=read(root/'final_selection.json') if (root/'final_selection.json').exists() else None
    result=[]
    for name in s['completed_rounds']:
        r=read(root/name);role=r['role'];i=r['boot_index']
        if role!=protocol()['boot_phases'][i]:raise ValueError('round phase differs')
        expected=str(Path('phases')/('boot'+str(i+1))/role/f"round{r['round']}.json")
        if name!=expected:raise ValueError('round path differs')
        pool=selection['pool'] if i<2 else read(root/'search_pool.json') if i==2 else final['pool']
        validate_round(root,s,r,pool,nwarm);result.append(r)
    discovery=[r for r in result if r['role']=='policy_discovery'];check=[r for r in result if r['role']=='policy_check']
    if (root/'resource_rule.json').exists():
        rule=freeze_rule(selection,means(discovery),means(check))
        if read(root/'resource_rule.json')!=rule:raise ValueError('resource rule differs from independent check')
        if (root/'rankings.json').exists():
            ranked=read(root/'rankings.json');programs=read(Path(s['bundle'])/'plans/programs.json')
            for method,value in ranked.items():
                replay=rank_dp(programs,records,rule,method)
                for key in replay:
                    if value[key]!=replay[key]:raise ValueError('static ranking differs from profile-only DP replay')
            if read(root/'search_pool.json')!=search_pool(ranked):raise ValueError('search pool changed')
            screened=[r for r in result if r['role']=='top20_screening']
            if final and final!=final_selection(ranked,screened,selection,rule):raise ValueError('final selection differs from frozen screening')
    return result


def preflight(bundle,transport):
    from .prepare import validate
    bundle=Path(bundle).resolve();m=validate(bundle);boot=transport.boot_id()
    root=ROOT/'artifacts/resource_qualification'/(m['bundle_id']+'_'+boot);root.mkdir(parents=True,exist_ok=True)
    with locked(root):
        s={'bundle':str(bundle),'bundle_id':m['bundle_id'],'boot_index':-1,'active_boot':boot,'frozen_files':{},'status':'PREFLIGHT'}
        if (root/'state.json').exists():s=read(root/'state.json');verify_inventory(root,s['frozen_files'])
        transport.wait_idle(boot);transport.mount_storage();remote=transport.deploy(bundle,root)
        transport.verify_payload(remote,boot);transport.qualify_environment(remote,boot,root,bundle);rec=Recorder(root,s,transport,remote)
        reports=[]
        for cut in CUTS:
            configs=[cut_candidate(cut,(1,1),((1,),(0,))),cut_candidate(cut,(2,2),((0,1),(2,3)),'pinned','resource_group'),
                     cut_candidate(cut,(2,1),((0,1),(2,)),'pinned','resource_group',True)]
            for c in configs:reports.append(rec.saved('pipeline/'+c['id'],task(s,'qualify_pipeline',c,candidate_id=c['id'])))
        for n in (3,4):
            c=component(stage('cpu',20,21,n,CORES[:n]),'pinned')
            reports.append(rec.saved('component/'+str(n),task(s,'qualify_component',c)))
        if transport.boot_id()!=boot:raise RuntimeError('boot changed during preflight')
        transport.verify_payload(remote,boot);transport.idle_command('sync',boot=boot)
        result={'status':'PREFLIGHT_PASSED','excluded_boot_id':boot,'bundle_id':m['bundle_id'],'reports':reports}
        _publish(root,s,'PREFLIGHT.json',result);return result
