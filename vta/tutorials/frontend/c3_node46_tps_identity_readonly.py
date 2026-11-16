"""Adopt 14 qualified node46 baseline calls and complete read-only inspection.

The original attempt remains STOPPED_IDENTITY_INSPECTION. No target, compiler
build, FPGA access or historical source mutation occurs here.
"""
import argparse
from pathlib import Path
from c3_node46_tps_local import HOLD,HERE,REPO,read,write,sha,verify,require


def load_parent(path):
    import tvm
    import vta  # Register tir.vta.* before LoadJSON in a fresh process.
    require(Path(tvm._ffi.base._LIB._name).resolve()==(REPO/'build/libtvm.so').resolve(),'COMPILER_LIBRARY_CHANGED')
    return tvm.ir.load_json(Path(path).read_text())


def run(baseline,out):
    from c3_uop_identity_applicability import inspect_function,classify
    from c3_tps_aggregation_capacity import organization
    out.mkdir(exist_ok=False)
    seal=read(baseline/'completion_manifest.json')
    require(seal['status']=='STOPPED_IDENTITY_INSPECTION' and seal['baseline_qualification_subset_complete'] is True,
            'BASELINE_SUBSET_NOT_AVAILABLE')
    bindings={str(baseline/p):v for p,v in seal['artifact_sha256'].items()}
    bindings[str(baseline/'completion_manifest.json')]=sha(baseline/'completion_manifest.json')
    bindings.update(read(baseline/'freeze.json')['bindings'])
    verify(bindings)
    account=read(baseline/'call_accounting.json')
    require(account['fully_certified']==account['started_calls']['exact']==account['completed_calls']['exact']==14,
            'BASELINE_14_CALL_ACCOUNTING_CHANGED')
    require(read(baseline/'R_mode_equivalence.json')['status']=='ACTUAL_ENCODING_EQUAL','MODE_COMPARISON_MISSING')
    for mode in ('detailed','minimal'):
        result=read(baseline/f'R_{mode}/summary.json')
        worker=read(baseline/f'qualify_{mode}_worker.json')
        require(worker['returncode']==0 and worker['limit'] is None,'QUALIFICATION_WORKER_FAILED')
        require(result['status']=='FULL_LOCAL_QUALIFICATION_PASSED' and len(result['calls'])==7 and
                all(c['status']=='PASS' and c['correct'] for c in result['calls']),'BASELINE_QUALIFICATION_MISSING')
    bindings[str(Path(__file__).resolve())]=sha(__file__)
    write(out/'freeze.json',{'bindings':bindings,'new_target_calls':0,'adopted_baseline_calls':14})
    parent=load_parent(baseline/'prepare/R/parent.json')
    require(len(parent.functions)==1,'EXPECTED_SINGLE_FUSION')
    census=inspect_function(parent[list(parent.functions)[0]])
    require(census['status']=='FINITE_DOMAIN_CONTENT_SCREEN_COMPLETE','IDENTITY_DOMAIN_UNRESOLVED')
    events=read(baseline/'prepare/R/events.json')
    classes=classify([e for e in events if e['kind']=='kernel'])
    mac=[r for r in classes if r['mode_reset']==[0,0]]
    require(sum(r['signatures'] for r in mac)==census['MAC_identities'] and
            sum(r['content_classes'] for r in mac)==census['MAC_content_classes'],'INDEPENDENT_CLASS_CENSUS_CHANGED')
    write(out/'identity_census.json',census)
    rows=[]
    for mode in ('detailed','minimal'):
        for index in range(1,8):
            unit=baseline/f'R_{mode}'
            text=(unit/f'native_call_{index:02d}.txt').read_text()
            native=read(unit/f'native_certificate_{index:02d}.json')
            org=organization(text,native)
            require(org['host_identity_base_uops_total']==census['identity_base_uops'],'ACTUAL_IDENTITY_CENSUS_CHANGED')
            rows.append({'mode':mode,'call':index,**org})
    fields=('MAC_GEMM','RESET_GEMM','ALU','all_instructions','compute_NOP','UOP_LOAD','UOP_LOAD_bytes',
            'serialized_uop_bytes','UOP_SRAM_wraps','submissions')
    require(all(tuple(x[k] for k in fields)==tuple(rows[0][k] for k in fields) for x in rows),'ORGANIZATION_CHANGED')
    write(out/'organization.json',rows)
    traffic=read(baseline/'prepare/R/tensor_requests.json')
    require(traffic['WGT']['payload_bytes']==884736 and traffic['WGT']['multiplicities']==[1],'WEIGHT_ONCE_NOT_CONFIRMED')
    write(out/'adoption.json',{'status':'NODE46_BASELINE_14_CALL_SUBSET_ADOPTED','source':str(baseline),
          'source_status_preserved':seal['status'],'source_manifest_sha256':sha(baseline/'completion_manifest.json'),
          'adopted_calls':14,'new_calls':0,'different_attempt_samples_spliced':False})
    write(out/'inspection.json',{'status':'QUALIFIED_NODE46_BASELINE_CONTENT_AUDITED',
          'MAC_identities':census['MAC_identities'],'MAC_content_classes':census['MAC_content_classes'],
          'MAC_redundant_identities':census['MAC_redundant_identities'],
          'native_capacity_pressure':rows[0]['UOP_SRAM_wraps']>0,
          'same_detector_as_node34':True,'optimized_variant_built':False,'new_target_calls':0,
          'new_compilations':0,'board_contacted':False,'performance_conclusion':None})
    verify(bindings)
    print(census['MAC_identities'],census['MAC_content_classes'],census['MAC_redundant_identities'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.baseline.resolve(),a.output.resolve())
