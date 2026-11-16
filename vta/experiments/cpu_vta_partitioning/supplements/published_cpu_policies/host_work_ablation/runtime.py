"""Unchanged native calls with ablation task and affinity validation."""
from functools import lru_cache
import json
from pathlib import Path
from .common import api,protocol

def pipeline_task(bundle,name,qualification=False):
    from cpu_policies.open_resources.runtime import make_task
    cfg=api(bundle).read(Path(bundle)/'plans/static_pool.json')[name]
    doc=make_task('qualify_pipeline' if qualification else 'performance',cfg,bundle,
        warmup=0 if qualification else 8,scored=1 if qualification else 16,tail=0 if qualification else 4,
        indices=[0],campaign=protocol()['experiment'],candidate_id=name)
    if doc['environment']!=protocol()['environment']:raise ValueError('frozen environment differs')
    return doc

def validate_document(doc,bundle):
    if doc.get('kind') not in ('qualify_pipeline','performance'):raise ValueError('component sessions prohibited')
    if doc!=pipeline_task(bundle,doc['candidate_id'],doc['kind']=='qualify_pipeline'):
        raise ValueError('task/executor/window/configuration identity differs')

def native_command(doc,bundle,remote,job):
    from cpu_policies.open_resources.runtime import native_command as base
    validate_document(doc,bundle)
    return base(doc,bundle,remote,job)

def validate_summary(raw,doc,bundle):
    from cpu_policies.open_resources.runtime import validate_summary as base
    validate_document(doc,bundle);result=base(raw,doc,bundle)
    report=api(bundle).read(Path(raw)/'affinity.json')
    for stage,row in zip(doc['configuration']['stages'],report['stages']):
        if stage['device']=='vta':
            caller=next(e for e in row['effective_affinities'] if e['tid']==row['caller_tid'])
            if caller['cpus']!=stage['mask']:raise ValueError('VTA caller affinity differs from requested mask')
    if doc['kind']=='performance':
        rows=[json.loads(s) for s in (Path(raw)/'result.jsonl').read_text().splitlines() if s]
        if any(r.get('queue_timing_enabled') is not False or r.get('defer_result_write_enabled') is not True for r in rows):
            raise ValueError('performance probes enabled or logging not deferred')
        result['fps']=1000/result['ii_ms']
    return result

@lru_cache(maxsize=4)
def _ssh_class(bundle):
    api(bundle)
    from cpu_strategy.policy.transport import SSH as FrozenSSH
    from cpu_policies.open_resources.runtime import _measure
    class AblationSSH(FrozenSSH):
        def measure(self,doc,bundle,remote,root,boot):
            validate_document(doc,bundle)
            return _measure(self,doc,bundle,remote,root,boot)
    return AblationSSH

def SSH(connection,*,execute_board=False,bundle=None):
    obj=_ssh_class(str(bundle))(connection,execute_board=execute_board)
    obj.options+=['-o','ControlMaster=auto','-o','ControlPersist=60','-o','ControlPath=/tmp/cpu-vta-host-work-%C']
    return obj
