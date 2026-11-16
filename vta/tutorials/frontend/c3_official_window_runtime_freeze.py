"""Freeze tested experiment runtime/helper identity without rebuilding targets."""
import argparse
import hashlib
import json
from pathlib import Path

import c3_official_window_runtime as build


def load(path):return json.loads(path.read_text())
def write(path,data):
    with path.open('x') as stream:json.dump(data,stream,indent=2,sort_keys=True);stream.write('\n')


def freeze(root):
    local=root/'runtime_local_final';arm=root/'runtime_arm'
    records=[load(directory/'build.json') for directory in (local,arm)]
    generated,rt_gate=build.runtime_overlay();driver,dr_gate=build.driver_overlay()
    reproduction=[]
    for directory,record in zip((local,arm),records):
        assert record['returncode']==0
        assert build.digest(record['library'])==record['binary_sha256']
        assert hashlib.sha256(generated.encode()).hexdigest()==record['runtime_overlay_sha256']
        assert hashlib.sha256(driver.encode()).hexdigest()==record['driver_overlay_sha256']
        assert (directory/'c3_official_window_runtime_generated.cc').read_text()==generated
        assert (directory/'c3_official_window_runtime_driver_generated.cc').read_text()==driver
        assert (directory/build.COLLECTION.name).read_bytes()==build.COLLECTION.read_bytes()
        assert (directory/build.QUEUE.name).read_bytes()==build.QUEUE.read_bytes()
        changes={}
        for path,old_sha in record['source_bindings'].items():
            current=build.digest(path)
            if current!=old_sha:
                # Only the ARM helper link directory was corrected after the
                # runtime build. Prove the exact earlier builder bytes and
                # reproduce both generated runtime sources with current code.
                assert Path(path).resolve()==Path(build.__file__).resolve()
                previous=Path(path).read_text().replace(
                    "    runtime_lib = REPO/'build' if target == 'sim' else REPO/'build_axu_aarch64'\n",
                    "    runtime_lib = REPO/'build' if target == 'sim' else REPO/'build_aarch64'\n"
                    "    if target != 'sim':\n"
                    "        runtime_lib = HERE/'report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout/20261013_c3_banerjee_baseline_completion_run02/runtime_board_attempt02'\n")
                assert hashlib.sha256(previous.encode()).hexdigest()==old_sha
                snapshot=directory/'builder_at_compile.py'
                if not snapshot.exists():snapshot.write_text(previous)
                assert build.digest(snapshot)==old_sha
                changes[path]={'at_compile_sha256':old_sha,'current_sha256':current,'snapshot':str(snapshot),
                               'change':'ARM helper library search path only; generated runtime/driver identical'}
        reproduction.append({'directory':str(directory),'library':record['library'],
                             'binary_sha256':record['binary_sha256'],'current_generator_reproduces_compiled_sources':True,
                             'post_build_generator_changes':changes})
    helper={}
    for target,directory in (('local',root/'helper/local_final02'),('arm',root/'helper/arm_attempt02')):
        record=load(directory/'build.json')
        assert record['returncode']==0 and record['source_sha256']==build.digest(build.HELPER)
        assert record['binary_sha256']==build.digest(record['binary']) and not record['links_VTA_runtime']
        helper[target]={'binary':record['binary'],'binary_sha256':record['binary_sha256'],
                        'source_sha256':record['source_sha256'],'build':str(directory/'build.json')}
    evidence={
        'collection_equivalence':local/'mode_equivalence02/comparison.json',
        'generic_cpu_once':root/'helper/cpu_regression03/result.json',
        'strict_native_and_queue':local/'native_queue_gate_final02/result.json',
        'real_two_explicit_syncs_detailed':local/'two_explicit_syncs_detailed01/result.json',
        'real_two_explicit_syncs_minimal':local/'two_explicit_syncs_minimal01/result.json'}
    for name,path in evidence.items():assert load(path)['status'].endswith(('PASSED','EQUAL','CERTIFIED')),(name,path)
    files=[Path(build.__file__),build.COLLECTION,build.QUEUE,build.HELPER,Path(__file__)]
    files += [build.HERE/name for name in ('c3_official_window_runtime_test.py','c3_official_window_runtime_native.py',
               'c3_official_window_runtime_native_test.py','c3_official_window_runtime_multisubmit_test.py',
               'c3_official_window_runtime_queue_test.cc','c3_valid_window_uop_values.py','c3_valid_window_native.py',
               'c3_valid_window_tps_audit.py','c3_five_tradeoff_commands.py','c3_thread_sync_audit.py',
               'audit_c3_banerjee_completion.py','c3_banerjee_baseline.py')]
    original=['vta/runtime/runtime.cc','3rdparty/vta-hw/src/axu5evb/axu5evb_driver.cc']
    files += [build.REPO/name for name in original]
    files += [build.HERE/'c3_banerjee_uop_queue.h']
    data={'schema':'c3_official_window_runtime_contract_v1','status':'LOCAL_RUNTIME_HELPER_NATIVE_GATES_PASSED',
          'runtime':reproduction,'helper':helper,'source_bindings':{str(path):build.digest(path) for path in files},
          'evidence':{name:{'path':str(path),'sha256':build.digest(path)} for name,path in evidence.items()},
          'collection_env':'C3_OFFICIAL_WINDOW_COLLECTION','modes':['detailed','minimal'],
          'minimal_gates':['runtime caller detail clocks','runtime Add methods/callers and their locks/events/histograms',
                           'driver detail clocks and AddRun','UOP/INSN transfer diagnostic clocks'],
          'minimal_keeps':['fixed <=16 submit ledger','execution/cache/dependency/barrier/AutoSync policy'],
          'common_safety_repair':'greedy new SRAM tail may not move before an intervening GEMM/ALU UOP read intersecting that tail',
          'repair_attribution':'common B/A DSO; do not attribute repair benefit to aggregation',
          'host_serialization_lifetime':'append-only whole-submit kernel list, unchanged; SRAM eviction does not discard DRAM sources',
          'support':{'single_owner_nonconcurrent_call':True,'replay_required':'disabled','dump_required_formal':False,
                     'uop_sram_entries':8192,'uop_element_bytes':4,'instruction_bytes':16,
                     'submit_limit':16,'ledger_overflow':'INVALID and external stop, never silent truncation',
                     'queue_bytes':33554432,'native_explicit_sync_fragments':True,'unknown_AutoSync_native_cuts':'refuse',
                     'cache_signature':'empty or one int32 0..98; -1/empty alias excluded',
                     'cache_definitions':'full args/loops, same caller-bound module/session; Sync resets SRAM and DRAM, not host definitions'},
          'helper_contract':{'exact_target_calls':1,'adds_sync':False,'target_lookup_outside_timer':True,
                             'counter_and_metadata_reads_outside_timer':True,'parameter_arity':'one or more tensors',
                             'all_session_invocations_counted':True},
          'tests':{'sealed_FSim_complete_targets':36,'generic_CPU_targets':12,
                   'two_sync_runtime_fixture_helper_targets':6,'two_sync_unchanged_sealed_operator_invocations':12,
                   'actual_8192_ABI_fixture_shrinks_SRAM':False,'unknown_cuts_and_warm_imm_mutations_refused':True},
          'retained_failed_attempts':['runtime_local/mode_equivalence: original helper duplicate registry before target',
             'runtime_local_final/mode_equivalence: base Python NumPy2 import failure before target',
             'helper/cpu_regression and cpu_regression02: helper load/import failures before target',
             'helper/arm: missing link search path, no executable target'],
          'bitstream_feedback_interval_status':'existing HLS/build reports exist but not bound to frozen bitstream; no invented ACC latency',
          'production_and_sealed_sources_unchanged':True,'board_contacted':False,
          'no_performance_conclusion':'Collection/queue/helper fixtures certify runtime safety and measurement; no aggregation speedup inferred'}
    write(root/'runtime_contract.json',data)
    print(json.dumps({'status':data['status'],'contract':str(root/'runtime_contract.json'),'sha256':build.digest(root/'runtime_contract.json')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    freeze(parser.parse_args().root.resolve())
