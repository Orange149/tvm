#!/usr/bin/env python3
"""Measure Q1 anchor off/on with identical programs, K=2 and complete outputs."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tarfile
import uuid
import numpy as np
from run_thesis_single_image_smoke import BASE, FRONTEND, PYTHON, SSH, checked
from thesis_supplement_run import RunStore, sha256, write_new_json
from thesis_supplement_audit import reconcile, seal
from thesis_completion_context_reset import reset, RPC_CHILDREN_CHECK
from thesis_single_image_integrated import make_package, make_anchor, deploy_controls, make_q1
from thesis_single_image_native import native, runner_name
from thesis_single_image_board_check import numeric


def create(source, retry_of=None):
    source = Path(source).resolve()
    prior = RunStore(source)
    ledger = json.loads((source / 'artifact_hashes.json').read_text())
    selected = json.loads((source / 'analysis/B_selection.json').read_text())
    renderer = json.loads((source / 'analysis/native_runner_selection.json').read_text())
    runtime = json.loads((source / 'analysis/board_artifact_bindings.json').read_text())
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_anchor_q1_pair_' + uuid.uuid4().hex[:8]
    boot = json.loads((source / 'analysis/c1_boot3_board.json').read_text())['boot_id']
    contract = dict(prior.contract, run_id=run_id, scope='actual_anchor_after_q1_paired_single_input',
        protocol_version='thesis_anchor_q1_pair_v1', audit_only_recovery=False,
        source_execution_run_id=None, retry_of=Path(retry_of).name if retry_of else None, source_artifact_run=str(source),
        new_board_calls='actual calls recorded in timeline',
        expected_boot_id=boot, independent_kernel_boots=1, extra_quantize_calls=0,
        conditions={'OFF': 'B with K2 and actual-program Q1; anchor absent',
                    'ON': 'same B with K2 and actual-program Q1; exclusive anchor enabled'},
        paired_orders=[['OFF', 'ON'], ['ON', 'OFF'], ['OFF', 'ON']],
        measurements_per_condition_per_block=10, default_peak_profile_frames_per_condition=20,
        measured_mode='pipeline_correctness; full outputs and allocations; K=2 both',
        measurement_sessions=6, primary_metric='actual DMA allocation high water and extra slot bytes',
        timeouts_seconds={'model_session':60, 'completion_mode':60, 'transfer':120},
        interpretation='single real input, three fresh contexts per condition, one kernel boot; no accuracy or throughput claim')
    store = RunStore.create(BASE / 'results/06' / run_id, contract)
    r = store.directory
    imported = []
    def copy(relative):
        if relative not in ledger or sha256(source / relative) != ledger[relative]['sha256']:
            raise ValueError('Unsealed or changed source artifact: ' + relative)
        target = r / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, target)
        imported.append({'path':relative,'sha256':sha256(target)})
    for relative in ('raw/input_image.png','raw/inputs/input.npy','raw/inputs/input.bin',
                     'raw/host_reference/reference_logits.npy','raw/host_reference/reference_boundaries.npz',
                     'analysis/native_runner_selection.json', 'raw/' + renderer['name']):
        copy(relative)
    for stage in selected['stages']:
        for name in stage['files']:
            copy('raw/' + stage['directory'] + '/' + name)
    for name in ('weights.npy','zeros_input.npy','uniform_input.npy','normal_input.npy',
                 'zeros_expected.npy','uniform_expected.npy','normal_expected.npy',
                 'zeros_sentinel.npy','uniform_sentinel.npy','normal_sentinel.npy'):
        copy('raw/' + name)
    files = {}
    bindings = {}
    for name,item in runtime.items():
        p=Path(item['host_path'])
        if sha256(p)!=item['sha256']:
            raise ValueError('Frozen runtime or hardware artifact changed: '+name)
        files[name]=p
        bindings[name]=item
    write_new_json(r/'analysis/board_artifact_bindings.json',bindings)
    write_new_json(r/'analysis/board_directory.json',{'path':'/var/volatile/thesis_'+run_id})
    write_new_json(r/'analysis/board_boot.json',{'boot_id':boot,'new_kernel_boot':False})
    write_new_json(r/'analysis/program.json',{'stages':selected['stages'],'threads':selected['candidate']['thread_vector']})
    write_new_json(r/'analysis/imported_artifacts.json',{'source_run':str(source),
        'source_ledger_sha256':sha256(source/'artifact_hashes.json'),'files':imported,
        'scope':'qualified artifacts only; all new anchor measurements collected in this run'})
    for name in (renderer['name'],): files[name]=r/'raw'/name
    files['inputs/input.bin']=r/'raw/inputs/input.bin'
    for stage in selected['stages']:
        for name in ('graph.json','module.so','params.bin'):
            files[stage['directory']+'/'+name]=r/'raw'/stage['directory']/name
    with tarfile.open(r/'raw/deployment.tgz','w:gz') as tar:
        for name,p in files.items():tar.add(p,arcname=name)
    write_new_json(r/'analysis/deployment_bindings.json',{
        name:{'host_path':str(p),'sha256':sha256(p)} for name,p in files.items()})
    # Source execution owns the existing idle RPC, even when its audit is in a new directory.
    owner=source
    if prior.contract.get('audit_only_recovery'):
        owner=source.parent/prior.contract['source_execution_run_id']
    if retry_of:
        owner=Path(retry_of).resolve()
        stopped=RunStore(owner)
        if not (owner/'invalid_session.json').exists() or stopped.contract['source_artifact_run']!=str(source):
            raise ValueError('Retry must reference a stopped run with the same frozen artifact source')
    old_board=json.loads((owner/'analysis/board_directory.json').read_text())['path']
    write_new_json(r/'analysis/source_rpc_owner.json',{'run':str(owner),'board_directory':old_board})
    source_dir=r/'raw/execution_sources';source_dir.mkdir()
    for name in ('run_thesis_anchor_q1_pair.py','thesis_single_image_integrated.py',
                 'thesis_single_image_native.py','thesis_completion_context_reset.py',
                 'build_thesis_command_capacity_certificate.py'):
        shutil.copy2(FRONTEND/name,source_dir/name)
    (r/'REPORT.md').write_text('# Q1之后anchor直接配对实测\n\n状态：准备中，尚无新增板端测量。\n')
    print(r,flush=True)


def execute(store):
    r=store.directory
    if (r/'invalid_session.json').exists() or (r/'artifact_hashes.json').exists():
        raise ValueError('Cannot resume stopped or sealed run')
    boot=store.contract['expected_boot_id']
    board=json.loads((r/'analysis/board_directory.json').read_text())['path']
    old=json.loads((r/'analysis/source_rpc_owner.json').read_text())['board_directory']
    stop=r/'raw/stop_source_owned_idle_rpc.sh'
    stop.write_text('set -eu\ntest "$(cat /proc/sys/kernel/random/boot_id)" = '+boot+'\n'
        'pid=$(cat '+old+'/rpc.pid)\ntest "$(readlink /proc/$pid/exe)" = '+old+'/tvm_rpc\n'
        +RPC_CHILDREN_CHECK+'\ntest -z "$(rpc_children)"\nkill "$pid"\n')
    checked(store,SSH+['sh -s'],'stop_source_owned_idle_rpc','one_time_preparation',30,stop)
    checked(store,SSH+['test ! -d '+board+' && mkdir '+board], 'new_ram_directory','one_time_preparation',30)
    checked(store,SSH+['cat > '+board+'/deployment.tgz'],'transfer_frozen_bundle','one_time_preparation',120,r/'raw/deployment.tgz')
    binding=json.loads((r/'analysis/deployment_bindings.json').read_text())
    lines=['set -eu','cd '+board,'tar -xzf deployment.tgz']
    for name,item in binding.items():
        lines.append('test "$(sha256sum '+name+' | cut -d \' \' -f 1)" = '+item['sha256'])
    lines += ['test "$(cat /proc/sys/kernel/random/boot_id)" = '+boot,
        'test "$(cat /sys/class/u-dma-buf/udmabuf0/size)" = 201326592',
        'chmod 755 tvm_rpc '+runner_name(r),
        'export LD_LIBRARY_PATH='+board+' LD_PRELOAD='+board+'/libvta.so',
        'export TVM_NUM_THREADS=1 VTA_REPLAY_POLICY=disabled VTA_QUEUE_DIAGNOSTICS=1 C3_COMPLETION_TRACE=1',
        'export AXU5EVB_DRIVER_POST_START_SLEEP_NS=1000 AXU5EVB_DRIVER_POLL_SLEEP_NS=1000',
        'unset VTA_INSN_BUFFER_BYTES VTA_UOP_BUFFER_BYTES VTA_INSN_SUBMIT_THRESHOLD_BYTES VTA_COMMAND_MANIFEST_ID',
        './tvm_rpc server --host=0.0.0.0 --port=9090 --port-end=9091 > rpc.log 2>&1 </dev/null &',
        'echo $! > rpc.pid','sleep 1','kill -0 "$(cat rpc.pid)"']
    script=r/'raw/deploy_frozen_bundle.sh';script.write_text('\n'.join(lines)+'\n')
    checked(store,SSH+['sh -s'],'verify_and_launch_frozen_runtime','one_time_preparation',60,script)
    for mode in ('O','I','W','IW'):
        reset(store,1,boot,mode,board)
        checked(store,[PYTHON,str(FRONTEND/'thesis_completion_board.py'),str(r),'--boot-index','1',
            '--boot-id',boot,'--mode',mode,'--board-directory',board],
            'fresh_completion_'+mode,'functional_qualification',60)
    log=checked(store,SSH+['cat '+board+'/rpc.log'],'completion_raw_log','functional_qualification',30)
    checked(store,[PYTHON,str(FRONTEND/'thesis_completion_trace_audit.py'),str(r),'--boot-index','1',
        '--log',str(r/log['stdout'])],'fresh_completion_trace_audit','functional_qualification',60)
    checked(store,[PYTHON,str(FRONTEND/'thesis_completion_dependency_audit.py'),'--run-dir',str(r),
        '--boot-index','1','--log',str(r/log['stdout']), '--output',str(r/'analysis/boot1_dependency_audit.json')],
        'fresh_store_ack_audit','functional_qualification',60)
    print('Fresh completion and STORE acknowledgement PASS',flush=True)
    program=json.loads((r/'analysis/program.json').read_text())
    stages,threads=program['stages'],program['threads']
    package=make_package(store,'program',stages,'frozen_B_anchor_pair')
    anchor=make_anchor(store,'ON',package)
    deploy_controls(store,'ON')
    (r/'raw/condition_controls/OFF').mkdir()
    caps={}
    for name,plan in [('OFF',None),('ON',anchor)]:
        caps[name]=make_q1(store,name,package,stages,threads,plan)
        print(name+': actual default-capacity worker profile and independent Q1 certificate collected',flush=True)
    certificate=[json.loads((r/'raw'/caps[n]).read_text()) for n in ('OFF','ON')]
    if certificate[0]['derivation']!=certificate[1]['derivation']:
        raise ValueError('Anchor comparison would confound different command capacities')
    for block,order in enumerate(store.contract['paired_orders']):
        for name in order:
            native(store,f'b1_pair{block}_{name}',stages,threads,1,boot,
                'pipeline_correctness',10,managed=True,anchor_plan=anchor if name=='ON' else None,capacity=caps[name])
            print(f'pair{block} {name}: 10 complete pipeline outputs and allocation snapshots saved',flush=True)
    finish(store,caps)


def finish(store,caps):
    r=store.directory;board=json.loads((r/'analysis/board_directory.json').read_text())['path']
    expected=np.load(r/'raw/host_reference/reference_logits.npy',allow_pickle=False)
    values=[];all_hashes=[]
    for block in range(3):
        for name in ('OFF','ON'):
            identity=f'b1_pair{block}_{name}';d=r/'raw/native_sessions'/identity
            rows=[json.loads(l) for l in (d/'result.jsonl').read_text().splitlines()]
            if len(rows)!=10:raise ValueError('Missing complete outputs')
            output_hashes=[]
            for row in rows:
                item=row['raw_output_files'][0]
                p=d/Path(item['path']).relative_to(Path(board)/'native_sessions'/identity)
                actual=np.frombuffer(p.read_bytes(),dtype=item['dtype']).reshape(item['shape'])
                numeric(actual,expected,store.contract['numeric']);output_hashes.append(sha256(p))
            all_hashes.append(output_hashes)
            snapshots=[json.loads(l) for l in (d/'allocations.jsonl').read_text().splitlines()]
            phases={row['phase']:row for row in snapshots}
            queues=[json.loads(l.split('[VTA_QUEUE] ',1)[1]) for l in (d/'native.stderr').read_text().splitlines() if '[VTA_QUEUE] ' in l]
            cert=json.loads((r/'raw'/caps[name]).read_text())
            if not queues or any(row['insn_capacity']!=cert['derivation']['insn_capacity_bytes'] or row['uop_capacity']!=cert['derivation']['uop_capacity_bytes'] or row['timeout'] for row in queues):
                raise ValueError('Actual Q1 queues violate certificate')
            stdout=(d/'native.stdout').read_text()
            if '[COMMAND-CAPACITY] certificate='+sha256(r/'raw'/caps[name]) not in stdout:
                raise ValueError('No actual preflight certificate identity')
            if name=='ON' and 'source=pool-anchor' not in stdout:
                raise ValueError('Anchor planned but not actually used')
            if name=='OFF' and 'source=pool-anchor' in stdout:
                raise ValueError('Off condition unexpectedly used anchor')
            values.append({'block':block,'condition':name,'session':identity,
                'high_water_bytes':phases['after_steady_state']['high_water_bytes'],
                'extra_slot_bytes':phases['slots_allocated']['high_water_bytes']-phases['graph_and_params_loaded']['high_water_bytes'],
                'graph_loaded_bytes':phases['graph_and_params_loaded']['high_water_bytes'],
                'output_sha256':output_hashes,'allocations_sha256':sha256(d/'allocations.jsonl'),
                'queue_records':len(queues),'certificate_sha256':sha256(r/'raw'/caps[name])})
    if any(x!=all_hashes[0] for x in all_hashes):raise ValueError('Off/on final outputs differ in bytes')
    offs=[v for v in values if v['condition']=='OFF'];ons=[v for v in values if v['condition']=='ON']
    if len({v['high_water_bytes'] for v in offs})!=1 or len({v['high_water_bytes'] for v in ons})!=1:
        raise ValueError('Repeated fresh-context high water is not stable; no single-value certificate')
    off,on=offs[0],ons[0];saved=off['high_water_bytes']-on['high_water_bytes']
    if any(x['graph_loaded_bytes']!=off['graph_loaded_bytes'] for x in values):
        raise ValueError('Graph loading allocation differs between conditions')
    if saved!=off['extra_slot_bytes']-on['extra_slot_bytes']:
        raise ValueError('High water change cannot be attributed to slot savings')
    write_new_json(r/'analysis/raw_recomputation.json',{'sessions':values,'all_final_bytes_equal':True,
        'repeated_high_water_identical':True,'all_other_frozen_artifacts_same':True})
    result={'status':'ANCHOR_AFTER_Q1_ACTUAL_PAIR_PASS','formal_stage_completed':False,
        'accuracy_evidence':False,'input_images':1,'kernel_boots':1,'paired_blocks':3,
        'full_outputs':60,'default_profile_frames':40,'all_output_bytes_equal':True,
        'q1_off_anchor_high_water_bytes':off['high_water_bytes'],
        'q1_on_anchor_high_water_bytes':on['high_water_bytes'],'anchor_saved_bytes':saved,
        'anchor_total_high_water_reduction_percent':100*saved/off['high_water_bytes'],
        'extra_slots_before_bytes':off['extra_slot_bytes'],'extra_slots_after_bytes':on['extra_slot_bytes'],
        'extra_slot_reduction_percent':100*saved/off['extra_slot_bytes'],
        'q1_certificates':caps,'costs':'analysis/cost_audit.json','results_are_measured_not_estimated':True}
    store.event('anchor_q1_pair_raw_audit_passed',**result)
    cost=reconcile(r)
    if cost['unfinished_action_ids']:raise ValueError('Unclosed paid actions')
    write_new_json(r/'analysis/cost_audit.json',cost)
    if store.contract.get('retry_of'):
        prior=r.parent/store.contract['retry_of']
        write_new_json(r/'analysis/retry_cost_audit.json',{'run':prior.name,'costs':reconcile(prior),
            'note':'Previous stopped preparation retained; not counted as current new measurements'})
    write_new_json(r/'summary.json',result)
    (r/'REPORT.md').write_text('# Q1之后anchor直接配对实测\n\n**实测通过，不使用反事实推算。**\n\n'
        '固定当前预训练ResNet18共同量化图的B三段程序、线程[4,1,4]、同图片、同位流/runtime/driver/runner；流水K=2。两边分别从实际20帧default程序FINISH后全worker峰值派生Q1证书，容量相同，shared plan身份分别绑定none与anchor。\n\n'
        f'Q1下关闭anchor高水位：{off["high_water_bytes"]} B（{off["high_water_bytes"]/2**20:.6f} MiB）；开启anchor：{on["high_water_bytes"]} B（{on["high_water_bytes"]/2**20:.6f} MiB）。anchor实测减少{saved} B（{saved/1024:.0f} KiB），总高水位减少{result["anchor_total_high_water_reduction_percent"]:.6f}%，新增slot减少{result["extra_slot_reduction_percent"]:.6f}%。\n\n'
        '正序、逆序、正序三配对块，各条件每块10帧完整输出；60份最终输出与共同参考比较通过，关闭/开启及各块之间最终字节相等，三次新上下文的高水位逐字节相同。所有Q1调用的实际队列日志、证书preflight SHA及ON的pool-anchor实际标记独立核验。新部署四模式完成及STORE握手先行复测。\n\n'
        '本实验只直接测量当前共同图，不把旧拓扑A/B/C/D的反事实数值升级为实测；不报告准确率或吞吐改善。DMA池物理保留192MiB不变，结论仅为所测单输入、单kernel boot下的分配高水位。\n\n'
        '[运行合同](contract.json)、[原始数据复算](analysis/raw_recomputation.json)、[实测汇总](summary.json)、[成本](analysis/cost_audit.json)、[时间线](timeline.jsonl)。原始输出/分配日志位于raw/native_sessions/b1_pair{0,1,2}_{OFF,ON}/。\n')
    if store.contract.get('retry_of'):
        with (r/'REPORT.md').open('a') as stream:
            stream.write('\n之前准备阶段失败及已付成本保留：[失败报告](../'+store.contract['retry_of']+'/REPORT.md)、[重试成本](analysis/retry_cost_audit.json)。\n')
    seal(r)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('operation',choices=['create','execute']);p.add_argument('directory',type=Path);p.add_argument('--retry-of',type=Path)
    args=p.parse_args()
    if args.operation=='create':create(args.directory,args.retry_of)
    else:
        store=RunStore(args.directory)
        try:execute(store)
        except BaseException as error:
            if not (store.directory/'artifact_hashes.json').exists():
                if not (store.directory/'invalid_session.json').exists():
                    write_new_json(store.directory/'invalid_session.json',{'operation':'anchor_q1_actual_pair',
                        'reason':str(error),'all_paid_costs_retained':True,'retry_requires_new_run':True})
                store.event('anchor_actual_pair_stopped',reason=str(error))
                (store.directory/'REPORT.md').write_text('# Q1之后anchor直接配对实测\n\n停止，未签发实测通过结论。\n\n'+str(error)+'\n\n失败和全部已付成本保留，重试须另建目录。\n')
            raise
