"""An isolated-only CPU-share input variant; original timing totals stay fixed.

This is a coarse reconstruction, not directly measured per-thread run clocks.
It cannot supply Linux scheduling order or repair the temporal model's gap.
"""
from copy import deepcopy
from itertools import combinations
from pathlib import Path
import json
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from cpu_vta_partitioning.common import read,write,sha256,digest
from cpu_policies.open_resources.costs import Costs,FIELDS,allocate_core_ns,quantize_ns
from cpu_policies.open_resources.domain import stage
from audit_isolated_run_scope import INPUTS

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/run_scope_weights_01'
SCOPE='estimated_all_recorded_run_CPU_from_quiet_noncaller_session_counters'
OLD_ASSUMPTION='whole_session_thread_shares_rescaled_to_scored_run_core_not_scored_per_thread_measurement'
NEW_ASSUMPTION='run_share_estimated_from_all_recorded_process_run_CPU_minus_noncaller_session_CPU_not_direct_per_thread_measurement'
ENTRY_KEYS={'status','original_weights','estimated_thread_CPU_ns','all_recorded_run_CPU_ns',
            'threads','source_receipt','source_boot','scope'}


class RunScopeCosts(Costs):
    """Replace only valid CPU shares; retain original values and assumptions otherwise."""
    def __init__(self,records,patches,*,programs=None,ddr=None):
        super().__init__(records,programs=programs,ddr=ddr,distribution='session_weights')
        if set(patches)!={'original_records_digest','CPU_multithread_entries','scope'}:
            raise ValueError('unexpected input fields; pipeline metrics are not accepted')
        if patches['scope']!=SCOPE or patches['original_records_digest']!=digest(self.records):
            raise ValueError('run-scope input or original cost identity differs')
        self.patches=deepcopy(patches['CPU_multithread_entries'])
        expected={key for key,r in self.records.items() if
            r['configuration']['stages'][0]['device']=='cpu' and
            r['configuration']['stages'][0]['threads']>1}
        if set(self.patches)!=expected:
            raise ValueError('run-scope coverage differs from original CPU domain')
        for key,p in self.patches.items():
            if set(p)!=ENTRY_KEYS or p['scope']!=SCOPE:
                raise ValueError('unexpected run-scope entry fields')
            record=self.records[key];n=record['configuration']['stages'][0]['threads']
            if p['threads']!=n or p['original_weights']!=record.get('session_thread_weights'):
                raise ValueError('original thread identity or shares differ')
            if p['status']=='reconstruction_available':
                work=p['estimated_thread_CPU_ns'];total=p['all_recorded_run_CPU_ns']
                if (not isinstance(work,list) or len(work)!=n or
                        any(type(v) is not int or v<0 for v in work) or
                        any(v==0 for v in work[1:]) or
                        type(total) is not int or total<=0 or sum(work)!=total or
                        record.get('session_thread_weights') is None):
                    raise ValueError('reconstructed run CPU not conserved')
            elif p['status'] in ('retain_original_unobservable_noncaller',
                                  'retain_original_incompatible_remainder',
                                  'retain_original_explicit_uniform_fallback'):
                if p['estimated_thread_CPU_ns'] is not None:
                    raise ValueError('fallback must not invent worker CPU')
            else:
                raise ValueError('unsupported reconstruction status')

    def get(self,row,context_value='shared4'):
        value=super().get(row,context_value)
        row=row if isinstance(row,dict) else row.to_dict()
        if row['device']!='cpu' or row['threads']==1:
            return value
        patch=self.patches[value['source_key']]
        value['run_share_status']=patch['status']
        if patch['status']=='reconstruction_available':
            total=patch['all_recorded_run_CPU_ns']
            value['cpu_core_weights']=[v/total for v in patch['estimated_thread_CPU_ns']]
            value['assumptions']=[a for a in value['assumptions'] if a!=OLD_ASSUMPTION]
            value['assumptions'].append(NEW_ASSUMPTION)
            value['thread_weight_scope']=SCOPE
        else:
            value['assumptions'].append('original_thread_shares_retained_due_to_run_reconstruction_limit')
        self.aliases[value['requested_key']]={
            'source_key':value['source_key'],'assumptions':list(value['assumptions'])}
        return value


def project(records,audit_rows):
    entries={}
    for row in audit_rows:
        if row['threads']==1:
            continue
        key=row['component_key']
        if key in entries:
            raise ValueError('duplicate isolated component identity')
        available=row['status']=='reconstruction_available'
        entries[key]=dict(status=row['status'],original_weights=row['original_weights'],
            estimated_thread_CPU_ns=(row['reconstruction']['estimated_thread_CPU_ns'] if available else None),
            all_recorded_run_CPU_ns=row['measured_all_run_CPU_ns'],threads=row['threads'],
            source_receipt=row['source_receipt'],source_boot=row['source_boot'],scope=SCOPE)
    return dict(original_records_digest=digest(records),CPU_multithread_entries=entries,scope=SCOPE)


def validate_options(records,programs,patches):
    old=Costs(records,programs=programs,distribution='session_weights')
    new=RunScopeCosts(records,patches,programs=programs)
    cpu_options=vta_components=changed=0
    for program in programs.values():
        if program['device']!='cpu':
            continue
        for n in range(1,5):
            for mask in combinations(range(4),n):
                row=stage('cpu',program['begin'],program['end'],n,list(mask))
                a,b=old.get(row),new.get(row)
                if any(a.get(k)!=b.get(k) for k in FIELDS):
                    raise ValueError('CPU service or total run/boundary cost changed')
                total=quantize_ns(a['run_core_ms'])
                for value in (a,b):
                    vector=allocate_core_ns(total,value['cpu_core_weights'])
                    if sum(vector)!=total or min(vector)<0:
                        raise ValueError('per-core run CPU not conserved')
                cpu_options+=1;changed+=a['cpu_core_weights']!=b['cpu_core_weights']
    for record in records.values():
        cfg=record['configuration'];row=cfg['stages'][0]
        if row['device']=='vta':
            if old.get(row,cfg)!=new.get(row,cfg):
                raise ValueError('VTA cost or its accounting changed')
            vta_components+=1
    return dict(CPU_core_set_options_checked=cpu_options,VTA_components_unchanged=vta_components,
                options_with_changed_weights=changed,all_service_run_and_boundary_totals_unchanged=True,
                run_CPU_conserved_for_every_option=True,configuration_domain_restricted=False)


def prepare():
    if OUT.exists():
        raise FileExistsError('preserve the frozen input variant')
    audit_path=ROOT/'artifacts/isolated_run_scope_01/RESULTS.json'
    audit=read(audit_path)
    if (audit['pipeline_results_read'] is not False or audit['new_board_sessions']!=0 or
            audit['formal_costs_modified'] is not False):
        raise ValueError('isolated-only input audit not established')
    for path,expected in {**audit['sources'],**audit['source_files']}.items():
        if sha256(path)!=expected:
            raise ValueError('original audited evidence changed')
    source={str(audit_path):sha256(audit_path),str(Path(__file__)):sha256(__file__)}
    variants={};validation={}
    for network,(cost_path,program_path) in INPUTS.items():
        records=read(cost_path)['costs'];programs=read(program_path)
        patches=project(records,[r for r in audit['rows'] if r['network']==network])
        validation[network]=validate_options(records,programs,patches)
        variants[network]=dict(cost_path=str(cost_path),cost_sha256=sha256(cost_path),
            programs_path=str(program_path),programs_sha256=sha256(program_path),patches=patches)
        source.update({str(p):sha256(p) for p in (cost_path,program_path)})
    OUT.mkdir(parents=True)
    write(OUT/'INPUTS.json',dict(networks=variants,source_hashes=source,
        no_pipeline_performance_input=True,new_measured_costs=False,
        old_static_method_or_results_overwritten=False,
        temporal_schedule_error_claimed_solved=False))
    write(OUT/'VALIDATION.json',dict(networks=validation,no_board_sessions=True,
        no_candidate_search_or_Top20_yet=True,source_sha256=sha256(__file__)))
    write(OUT/'MANIFEST.json',dict(files={str(p):sha256(p) for p in (OUT/'INPUTS.json',OUT/'VALIDATION.json')}))
    print(json.dumps(dict(validation=validation,output=str(OUT)),ensure_ascii=False,indent=2))


if __name__=='__main__':
    prepare()
