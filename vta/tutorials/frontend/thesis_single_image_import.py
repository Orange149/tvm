#!/usr/bin/env python3
"""Explicitly import sealed, same-program evidence preceding recovery-only failure."""
import argparse,json,shutil
from pathlib import Path
from thesis_supplement_run import RunStore,sha256,write_new_json
from run_thesis_single_image_smoke import SSH,checked

def import_evidence(store):
    r=store.directory
    source=Path(store.contract['reused_evidence_run'])
    previous=RunStore(source)
    invalid=json.loads((source/'invalid_session.json').read_text())
    snapshot_fix = store.contract.get('snapshot_fix_redeployment', False)
    recovery_only = invalid['operation'] == 'c1_reboot-restore' and 'new boot_id within180s' in invalid['reason']
    if snapshot_fix:
        summary = json.loads((source / 'summary.json').read_text())
        if store.contract.get('native_env_redeployment'):
            log = (source / 'raw/native_sessions/b3_D_integrated_correctness/native.stderr').read_text().strip()
            if summary['status'] != 'STOPPED_Q1_ENVIRONMENT_PREFLIGHT' or invalid['operation'] != 'integrate' or log != 'vta_stage_pipeline_runner error: command-capacity: missing environment VTA_INSN_BUFFER_BYTES' or summary['D_complete_frames'] != 0 or summary['D_vta_submissions'] != 0:
                raise ValueError('Environment retry accepts only the identified zero-inference Q1 preflight failure')
        else:
            log = (source / 'raw/native_sessions/b3_C_integrated_correctness/native.stderr').read_text()
            if summary['status'] != 'STOPPED_SHARED_STABILITY_SNAPSHOT_SEGFAULT' or invalid['operation'] != 'integrate' or 'VTABufferCopy' not in log or 'fault_addr=(nil)' not in log or summary['C_complete_frames'] != 0:
                raise ValueError('Snapshot redeployment accepts only the identified raw-slot inspection failure')
        if summary['C1_completed_windows'] != 48:
            raise ValueError('Source selection incomplete')
    elif not recovery_only:
        raise ValueError('Only the qualified recovery-only stop can be imported')
    for key in ('canonical_qir_sha256','image_sha256','weights_sha256','baseline_history_sha256','A_thread_vector','numeric','performance','udmabuf_bytes','bitstream_sha256','protocol_version'):
        if previous.contract[key]!=store.contract[key]:
            raise ValueError('Evidence contract identity differs: '+key)
    sealed=json.loads((source/'artifact_hashes.json').read_text())
    for relative,item in sealed.items():
        if sha256(source/relative)!=item['sha256']:
            raise ValueError('Sealed evidence changed: '+relative)
    first=json.loads((source/'analysis/c1_boot1_windows.json').read_text())
    if len(first['results'])!=16:
        raise ValueError('Incomplete original first-boot evidence')
    for relative in ('raw/c1_frozen/fair_support_contract.json','raw/c1_measurement_contract.json',
                     'analysis/native_runner_selection.json','analysis/board_artifact_bindings.json'):
        if relative not in sealed: raise ValueError('Unsealed identity: '+relative)
    files=[]
    excluded={'analysis/board_directory.json','analysis/cost_audit.json','analysis/retry_chain_cost_audit.json','analysis/imported_evidence.json'}
    if snapshot_fix:
        excluded.update({'analysis/native_runner_selection.json','analysis/c1_boot_restore_bindings.json',
            'analysis/A_fair_program_binding.json','analysis/boot3_trace_audit.json','analysis/boot3_dependency_audit.json',
            'analysis/selection_deployment_lineage.json','analysis/source_C1_runner_selection.json',
            'analysis/redeployment_kernel_binding.json','analysis/B_C_Ca_byte_exact.json',
            'raw/c1_boot_restore.tgz','raw/c1_boot3_restore.sh','raw/stop_identified_idle_source_rpc.sh',
            'raw/ABCCa_conditions.json','raw/ABCCaD_conditions.json','raw/integrated_qualification_contract.json'})
    for relative,item in sealed.items():
        if not relative.startswith(('raw/','analysis/','figures/')) or relative in excluded:
            continue
        if snapshot_fix and (relative.startswith(('raw/condition_controls/','raw/condition_packages/','raw/boot3/','raw/context_reset_','analysis/context_reset_','analysis/Ca_','analysis/D_')) or relative.startswith('raw/native_sessions/') and ('_integrated_' in relative or '_default_worker_profile' in relative)):
            continue
        target=r/relative
        if relative=='raw/executed_controller.py':
            target=r/'raw/imported_controllers'/source.name/'executed_controller.py'
        if target.exists():
            if sha256(target)!=item['sha256']: raise ValueError('Import target collision: '+relative)
        else:
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source/relative,target)
        if sha256(target)!=item['sha256']: raise ValueError('Imported file changed: '+relative)
        files.append({'path':str(target.relative_to(r)),'source_path':relative,'sha256':item['sha256'],'source_run':source.name})
    write_new_json(r/'analysis/imported_evidence.json',{
        'source_run':str(source),'source_ledger_sha256':sha256(source/'artifact_hashes.json'),
        'original_boot_id':first['boot_id'],'imported_files':files,
        'new_observations':False,'source_status':'identified inspection failure after complete original C1 selection' if snapshot_fix else 'invalid recovery transition, valid preceding model evidence',
        'original_cost_audit':str(source/'analysis/cost_audit.json'),
        'raw_action_records':'Imported records belong to original run; current timeline does not count them as new actions',
        'support_not_refrozen_after_labels':True,
        'controller_changes':store.contract.get('selector_deployment_boundary') if snapshot_fix else 'Same executable, ABI, graph, inputs, schedules, thread masks and scoring; recovery absolute path and partial-output archival fixes do not alter prior window scoring',
        'next_boot_rule':'Recollect completion and output checks before measuring new boot'})
    write_new_json(r/'analysis/board_directory.json',{'path':'/var/volatile/thesis_'+r.name})
    if snapshot_fix:
        original = source / 'analysis/source_C1_runner_selection.json'
        if not original.exists():
            original = source / 'analysis/native_runner_selection.json'
        shutil.copy2(original, r / 'analysis/source_C1_runner_selection.json')
    probe=checked(store,SSH+['cat /proc/sys/kernel/random/boot_id; cat /sys/class/fpga_manager/fpga0/state; uptime'],
                  'user_cold_reboot_identity','failed_recovery',30)
    store.event('sealed_same_graph_evidence_imported',source_run=source.name,source_boot_id=first['boot_id'],
                files=len(files),new_model_calls=0,support_sha256=sha256(r/'raw/c1_frozen/fair_support_contract.json'))
    print('IMPORTED',r,len(files),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run_dir',type=Path);args=p.parse_args()
    store=RunStore(args.run_dir)
    try:import_evidence(store)
    except BaseException as e:
        write_new_json(store.directory/'invalid_session.json',{'operation':'import_sealed_evidence','reason':str(e),'all_paid_costs_retained':True});raise
