"""Matched native evidence, including explicitly imported legacy components."""
from copy import deepcopy
from pathlib import Path
import statistics

from cpu_vta_partitioning.common import read,sha256,verify_inventory,relative_file,digest
from cpu_vta_partitioning.measurement import frames,input_sequence,correctness,calibration
from ..measurement import whole_pipeline_window,isolated_components,thread_activity
from ..evidence import Evidence as LegacyEvidence,identified
from .domain import ENV,validate_candidate,component_key,component


def validate_summary(rawdir,document,bundle):
    rawdir=Path(rawdir);rows=frames(rawdir/'result.jsonl',sum(document[k] for k in ('warmup','scored','tail')))
    input_sequence(rows,document)
    config=document['configuration'];validate_candidate(config,read(Path(bundle)/'plans/programs.json'),document['kind'] in ('calibration','qualify_component'))
    report=read(rawdir/'affinity.json')
    if len(report['stages'])!=len(config['stages']):raise ValueError('affinity stage coverage differs')
    for s,r in zip(config['stages'],report['stages']):
        if r['requested_cpus']!=s['mask'] or r['active_runtime_threads']!=s['threads']:
            raise ValueError('configured threads/core mask differ from native affinity')
        entries=r['effective_affinities'];tids=r['thread_ids']
        if len(set(tids))!=len(tids) or {e['tid'] for e in entries}!=set(tids) or r['caller_tid'] not in tids:
            raise ValueError('invalid owned thread identities')
        for e in entries:
            if not e['cpus'] or not set(e['cpus'])<=set(s['mask']):raise ValueError('thread escaped its allowed group')
        if s['device']=='cpu' and config['affinity_mode']=='pinned':
            if len(tids)!=s['threads'] or any(len(e['cpus'])!=1 for e in entries):
                raise ValueError('pinned pool is not one active slot per core')
            if sorted(e['cpus'][0] for e in entries)!=s['worker_core_map']:
                raise ValueError('pinned pool mapping is not bijective')
            caller=next(e for e in entries if e['tid']==r['caller_tid'])
            if caller['cpus']!=s['worker_core_map'][:1]:raise ValueError('worker0 caller was not pinned')
            if document['kind'].startswith('qualify'):
                slots=r.get('worker_slots',[])
                if len(slots)!=s['threads'] or sorted(v['slot'] for v in slots)!=list(range(s['threads'])):
                    raise ValueError('missing logical worker probe')
                for v in slots:
                    if v['cpus']!=[s['worker_core_map'][v['slot']]] or v['tid'] not in tids:
                        raise ValueError('logical slot mapping differs')
                if next(v for v in slots if v['slot']==0)['tid']!=r['caller_tid']:
                    raise ValueError('logical worker0 is not the caller')
    if len(config['stages'])>1 and document['executor']=='policy':
        controls=report.get('control_threads',[])
        if {r['role'] for r in controls}!={'collector','producer'} or any(r['cpus']!=config['control_mask'] for r in controls):
            raise ValueError('control thread affinity differs')
    if any(r.get('cpu_stage_gate_mode','off')!=config['execution_mode'] for r in rows):
        raise ValueError('CPU gate mode differs from scored configuration')
    activity=thread_activity(report,config['stages'])
    if document['kind'].startswith('qualify'):
        correctness(rows,document,bundle,rawdir)
        return {'correct':True,'thread_activity':activity}
    if document['kind']=='calibration':result=isolated_components(rows,document['warmup'],document['scored'])
    else:result=whole_pipeline_window(rows,document['warmup'],document['scored'],document['tail'])
    return {**result,'thread_activity':activity}


class Evidence:
    def __init__(self,root,bundle):
        self.root=Path(root).resolve();self.bundle=Path(bundle).resolve();self.bid=read(self.bundle/'manifest.json')['bundle_id']

    def summary(self,value,document,boot):
        expected=identified(document,self.bid);ref=value['_receipt'];name=f"accepted/{boot}/{expected['task_id']}.json"
        p=relative_file(self.root,name)
        if ref['path']!=name or sha256(p)!=ref['sha256']:raise ValueError('receipt hash/identity differs')
        r=read(p)
        if r['task']!=expected or r['boot_id']!=boot:raise ValueError('wrong task/boot receipt')
        raw=self.root/'raw'/boot/expected['task_id']
        if not r['files'] or any(not relative_file(self.root,n).is_relative_to(raw) for n in r['files']):
            raise ValueError('receipt raw scope differs')
        verify_inventory(self.root,r['files'])
        logs=[relative_file(self.root,n) for n in r['files'] if Path(n).name=='result.jsonl']
        if len(logs)!=1:raise ValueError('duplicate/missing native result')
        rawdir=logs[0].parent
        if str((rawdir/'affinity.json').relative_to(self.root)) not in r['files']:raise ValueError('unbound affinity file')
        if document['kind'].startswith('qualify'):
            for row in frames(logs[0],sum(document[k] for k in ('warmup','scored','tail'))):
                for dump in row['stage_raw_outputs']:
                    for output in dump['output_files']:
                        path=rawdir/'outputs'/('run_'+str(row['completion_index']))/Path(output['path']).name
                        if str(path.relative_to(self.root)) not in r['files']:raise ValueError('unbound correctness output')
        calculated=validate_summary(rawdir,expected,self.bundle)
        if calculated!=r['summary'] or calculated!={k:v for k,v in value.items() if k!='_receipt'}:
            raise ValueError('summary differs from raw data')
        return r


FIELDS=('service_ms','run_core_ms','boundary_core_ms','set_ms','run_ms','get_ms','set_core_ms','get_core_ms')


def validate_import(bundle):
    bundle=Path(bundle);plan=read(bundle/'plans/imported_profiles.json');answer={}
    for key,item in plan['records'].items():
        source=bundle/item['source_root'];old=bundle/item['source_bundle']
        original=read(old/'manifest.json')
        new=read(bundle/'manifest.json') if (bundle/'manifest.json').exists() else {'files':read(bundle/'plans/execution_assets.json')}
        for path in item['required_assets']:
            if original['files'][path]['sha256']!=new['files'].get(path,{}).get('sha256'):
                raise ValueError('imported profile execution asset differs: '+path)
        if sha256(old/'runner/vta_stage_pipeline_runner')!=sha256(bundle/'runner/vta_stage_pipeline_runner'):
            raise ValueError('legacy executor changed; component import refused')
        evidence=LegacyEvidence(source,item['source_bundle_id'],old)
        samples=[]
        for entry in item['blocks']:
            doc=entry['document']
            if (doc['kind']!='calibration' or doc['input_indices']!=[0] or (doc['warmup'],doc['scored'],doc['tail'])!=(5,32,0)
                    or doc['execution_mode']!='off' or doc['environment']['poll_sleep_ns']!=1000
                    or doc['environment']['post_start_sleep_ns']!=1000):raise ValueError('incompatible imported calibration protocol')
            evidence.summary(entry['summary'],doc,entry['boot_id']);samples.append(entry['summary'])
        q=item['qualification'];evidence.summary(q['summary'],q['document'],q['boot_id'])
        source_config=component(item['blocks'][0]['document']['stages'][0])
        if component_key(source_config)!=key or item['configuration']!=source_config:raise ValueError('imported signature was relabeled')
        cost={f:statistics.mean(v[f] for v in samples) for f in FIELDS}
        if cost!=item['cost']:raise ValueError('imported cost does not match original blocks')
        answer[key]=deepcopy(item)
    return answer
