"""Host-only causal check on a known layout; not Top20 or unseen validation.

Three choices are fixed by prior CPU placement evidence, not FPS. Different
per-island placements are diagnostic execution settings, outside the current
nine-context search domain. The domain and all old controllers stay intact.
"""
from copy import deepcopy
from pathlib import Path
from types import FunctionType
import argparse
import json
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from cpu_vta_partitioning.common import read,sha256,digest
from finite_service.common import freeze
from cpu_policies.open_resources.domain import configuration,stage
import finite_service_holdout_board as parent

ROOT=parent.ROOT
POOL=ROOT/'artifacts/host_gap_intervention_01'
RUN=ROOT/'runs/host_gap_intervention_01'
CONFIG=parent.CONFIG
OUTPUT=parent.OUTPUT


def placed(base,cores):
    c=deepcopy(base)
    if cores is None:return c
    hosts=[i for i,s in enumerate(c['stages']) if s['device']=='vta']
    if len(cores)!=len(hosts) or any(type(k) is not int or k not in range(4) for k in cores):
        raise ValueError('complete valid host placement required')
    for i,core in zip(hosts,cores):
        s=c['stages'][i];c['stages'][i]=stage('vta',s['begin'],s['end'],1,[core])
    c['context_id']='diagnostic_per_island_hosts'
    c['host_mask']=sorted(set(cores))  # host pool; stage masks are authoritative
    c['id']='cfg_'+digest(configuration(c))[:24]
    return c


def protocol():
    p=parent.protocol();p.update(experiment='host_gap_intervention_01',
        only_local_pool_validation=False,known_layout_host_only_diagnostic=True,
        not_static_search_domain_extension=True,no_model_promotion=True,
        no_observer=True,host_stage_masks_authoritative=True)
    return p


def sources():
    return {**parent.sources(),**{str(p):sha256(p) for p in (
        Path(__file__),ROOT/'tests/test_host_gap_intervention.py')}}


def freeze_selection():
    source=ROOT/'artifacts/current_model_gap_01/INPUTS.json';base=read(source)['case']['configuration']
    if base['context_id']!='shared4' or base['control_mask']!=[0,1,2,3] or base['execution_mode']!='off':
        raise ValueError('diagnostic reference changed')
    evidence=ROOT/'artifacts/current_model_gap_01/observed_placement_diagnostic/INPUT_FREEZE.json'
    observed=read(evidence)
    if [observed['selected'][str(i)]['core'] for i in (1,3)]!=[3,2]:raise ValueError('observed placement source changed')
    match=ROOT/'artifacts/current_model_gap_01/existing_host_matches/RESULTS.json'
    if read(match)['host_placement_pairs']!=['[[0, 1, 2, 3], [0, 1, 2, 3]]']:
        raise ValueError('new matched historical evidence must be reviewed before additional measurement')
    configs=[placed(base,None),placed(base,[2,1]),placed(base,[3,2])]
    for c in configs:
        if [s for s in c['stages'] if s['device']=='cpu']!=[s for s in base['stages'] if s['device']=='cpu']:
            raise ValueError('CPU placement changed in host-only comparison')
    freeze(POOL/'SELECTION_FREEZE.json',dict(selected_configurations={'yolov3_tiny416':configs},
        roles=dict(zip([c['id'] for c in configs],['shared','model_hosts_2_1','observed_modal_hosts_3_2'])),
        input_source_sha256=sha256(source),observed_placement_sha256=sha256(evidence),
        historical_match_audit_sha256=sha256(match),source_sha256=sha256(__file__),
        selection_before_new_native_measurement=True,selection_uses_FPS=False,
        DNN_budget=87,max_native_sessions=6,new_component_profiles=0,
        scope='one known topology and CPU layout; two per-island host placement interventions',
        same_programs_and_waiting=True,only_host_allowed_sets_change=True))
    return dict(configurations=3,native_sessions=6,DNN=87,roles=['shared','model_hosts_2_1','observed_modal_hosts_3_2'])


def selection():
    x=read(POOL/'SELECTION_FREEZE.json')
    if x['source_sha256']!=sha256(__file__):raise ValueError('intervention protocol source changed')
    if x['DNN_budget']!=87 or len(x['selected_configurations']['yolov3_tiny416'])!=3:
        raise ValueError('host-only native budget differs')
    return x


def progress_print(*args,**kwargs):
    if args and args[0]=='未见配置共同实测':args=('主机位置匹配实测',)+args[1:]
    print(*args,**kwargs)


def adapt(name):
    fn=getattr(parent,name)
    result=FunctionType(fn.__code__,{**fn.__globals__,**globals(),'print':progress_print},fn.__name__,fn.__defaults__)
    result.__kwdefaults__=fn.__kwdefaults__
    return result


def specs():return adapt('specs')()
def prepare():return adapt('prepare')()
def validate(bundle):return adapt('validate')(bundle)
def make_task(bundle,spec):return adapt('make_task')(bundle,spec)
def native_command(doc,bundle,remote_bundle,remote_job):return adapt('native_command')(doc,bundle,remote_bundle,remote_job)
def matched_ssh(config,*,execute_board=False):return adapt('matched_ssh')(config,execute_board=execute_board)
def qemu():return adapt('qemu')()
def board(*,execute_board=False):return adapt('board')(execute_board=execute_board)
summary=parent.summary
accept=parent.accept


def analyze():
    state=read(RUN/'state.json')
    if state['status']!='COMPLETE' or state['DNN_inferences']!=87:raise ValueError('host intervention incomplete')
    from cpu_vta_partitioning.common import verify_inventory,write
    rows=[]
    for spec in specs():
        if spec['phase']!='clean':continue
        bundle=POOL/spec['network']/'capsule';doc=make_task(bundle,spec)
        p=RUN/'accepted'/state['boot_id']/(doc['task_id']+'.json');item=read(p)
        if item['task']!=doc or item['boot_id']!=state['boot_id']:raise ValueError('native receipt identity differs')
        verify_inventory(item['raw'],item['files'])
        if summary(item['raw'],doc,bundle)!=item['summary']:raise ValueError('native summary no longer replays')
        s=item['summary'];rows.append(dict(role=selection()['roles'][spec['case']],configuration_id=spec['case'],
            fps=s['fps'],ii_ms=s['ii_ms'],stage_services_ms=s['stage_service_ms'],
            process_core_ms_per_completion=s['process_core_ms_per_completion'],
            VTA_host_masks=[s['mask'] for s in doc['stages'] if s['device']=='vta'],
            receipt=str(p),receipt_sha256=sha256(p)))
    rows.sort(key=lambda r:r['role'])
    result=dict(rows=rows,boot_id=state['boot_id'],single_input_single_boot_short_window=True,
        new_DNN=87,new_native_sessions=6,only_host_allowed_sets_changed=True,
        no_static_model_updated=True,not_full_DP_or_Top20=True,source_sha256=sha256(__file__))
    write(POOL/'analysis/RESULTS.json',result,replace=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('freeze','prepare','qemu','board','analyze'))
    p.add_argument('--execute-board',action='store_true');a=p.parse_args()
    if a.command=='board' and not a.execute_board:p.error('--execute-board is required')
    fn=freeze_selection if a.command=='freeze' else globals()[a.command]
    print(json.dumps(fn(execute_board=True) if a.command=='board' else fn(),ensure_ascii=False,indent=2))
