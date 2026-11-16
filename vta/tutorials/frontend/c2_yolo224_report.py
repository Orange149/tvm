#!/usr/bin/env python3
"""Reconcile saved evidence and publish the YOLO224 C2 research report.

Publication requires three actual boot IDs and completed frozen formal blocks.
An unfinished campaign can only produce a separately named progress snapshot.
"""
from __future__ import annotations
import argparse,csv,datetime,json,math,re,statistics
from pathlib import Path
from c2_yolo224_prepare import ROOT,HERE,write,sha
from c2_yolo224_analyze import reference,metrics,read

MODES=('M0','M1','M2','M3','L1','L2','CP','E1')
CONTRASTS=(('M0','M1'),('M1','M2'),('M2','M3'),('L1','L2'),('L1','CP'),('L1','E1'),('L2','E1'))


def csv_file(path,rows):
    if not rows:return
    with path.open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def ci95(values):
    if len(values)!=3:return None
    mean=statistics.mean(values);half=4.3026527299*statistics.stdev(values)/math.sqrt(3)
    return [mean-half,mean+half]


def profile(directory):
    value=read(directory/'result.json');p=value['runtime_profile'];n=len(value['frames'])
    records=[json.loads(line[len('[VTA_QUEUE] '):]) for line in (directory/'stderr').read_text().splitlines()
        if line.startswith('[VTA_QUEUE] ')]
    measured=[r for r in value['frames'] if r['measured']]
    return {'total_frames':n,'submits_per_frame':p['synchronize_calls']/n,
        'device_wait_ms_per_frame':p['device_run_wait_us']/n/1000,
        'driver_mmio_us_per_frame':p['driver_submit_mmio_us']/n,
        'load_bytes_per_frame':p['synchronize_load_bytes']/n,
        'store_bytes_per_frame':p['synchronize_store_bytes']/n,
        'driver_timeout_calls':p['driver_timeout_calls'],
        'stage1_run_ms_mean':statistics.mean(r['stage_end_ms'][1]-r['stage_begin_ms'][1] for r in measured),
        **{'diagnostic_'+key+'_per_frame':sum(r[key] for r in records)/n if records else None
            for key in ('insn_bytes','uop_bytes','finalize_us','uop_pack_us','uop_copy_us','insn_copy_us','queue_cache_us')}}


def collect(master,recheck):
    directories=[master,*sorted((master/'boots').glob('boot*'))] if (master/'boots').exists() else [master]
    directories=[d for d in directories if (d/'board_state.json').exists()]
    ref=reference(master/'runs/reference_M0_01') if recheck else None
    inventory=[];blocks=[];boot_stats=[];allocations=[];submissions=[];faults=[];environments=[]
    expected=read(master/'formal_protocol.json')
    q1=read(master/'q1/command_capacity_certificate.json')['derivation']
    actual_boots=set();command_checks=[]
    for directory in directories:
        state=read(directory/'board_state.json');boot=state['boot_id']
        if (directory/'formal_complete.json').exists():
            from c2_yolo224_qualification import reconcile
            reconcile(directory)
        if boot in actual_boots:raise ValueError('same Linux boot must not appear as two independent units')
        actual_boots.add(boot)
        if sha(directory/'package/c2_yolo224_runner')!=sha(master/'package/c2_yolo224_runner'):
            raise ValueError('program changed between boots')
        if read(directory/'deployment_inventory.json')!=read(master/'deployment_inventory.json'):
            raise ValueError('graph, parameters, inputs or runtime package changed between boots')
        program=read(directory/'program.json');manifest=read(master/'package/manifest.json')
        from c2_yolo224_command import reconcile as reconcile_commands
        commands=reconcile_commands(directory)
        if commands:
            if command_checks and commands['command_signature_sha256']!=command_checks[0]['command_signature_sha256']:
                raise ValueError('command work changed between independent boots')
            for value in commands['modes'].values():
                if value['Q1_capacities']!=[q1['insn_capacity_bytes'],q1['uop_capacity_bytes']]:
                    raise ValueError('complete peaks do not reproduce frozen Q1 capacities')
            command_checks.append({'boot_id':boot,**commands})
        for actual,frozen in zip(program['stages'],manifest['stages']):
            if actual['runtime_num_threads']!=frozen['runtime_num_threads'] or any(
                    actual[key]!=value for key,value in frozen['artifact_sha256'].items()):
                raise ValueError('formal program differs from the frozen stage identity/thread settings')
        if [x['sha256'] for x in program['inputs']]!=[x['sha256'] for x in read(master/'package/inputs.json')]:
            raise ValueError('formal input sequence changed')
        completed={};all_verified={}
        for run in sorted((directory/'runs').iterdir()):
            if not (run/'exitcode').exists():continue
            code=int((run/'exitcode').read_text());relative=str(run.relative_to(master))
            row={'directory':relative,'boot_id':boot,'exitcode':code,
                'result_sha256':sha(run/'result.json') if (run/'result.json').exists() else None,
                'raw_outputs_sha256':sha(run/'outputs.bin') if (run/'outputs.bin').exists() else None}
            if code:
                if run.name.startswith('Q2_') and not (run/'result.json').exists():
                    row.update(status='failed_candidate',error='native process exited without joined INVALID result; retained and recovered separately')
                    faults.append(row);inventory.append(row);continue
                result=read(run/'result.json')
                if result['status']!='INVALID':raise ValueError('failed execution lacks joined INVALID result')
                row.update(status='expected_invalid' if (run/'verified.json').exists() else 'unqualified',
                    error=result['error']);faults.append(row);inventory.append(row);continue
            if recheck:
                verified=metrics(run,ref)
                if (run/'verified.json').exists() and read(run/'verified.json')!=json.loads(json.dumps(verified)):
                    raise ValueError('independent replay differs: '+relative)
            elif (run/'verified.json').exists():verified=read(run/'verified.json')
            else:continue
            if verified['boot_id']!=boot:raise ValueError('raw result boot identity differs')
            row.update(status='byte_exact',frames=verified['frames'],outputs=verified['raw_outputs_checked'],
                bytes=verified['raw_bytes_checked'],mode=verified['mode'])
            inventory.append(row);all_verified[run.name]=verified
            if run.name.startswith('formal_') and run.name!='formal_manual':
                contract=read(run/'contract.json');env=contract['environment']
                if contract['program_sha256']!=sha(directory/'program.json'):raise ValueError('formal program contract drift')
                if env['VTA_QUEUE_DIAGNOSTICS']!='0' or any(k.startswith('C2_ADDRESS') for k in env):
                    raise ValueError('formal timing has diagnostic instrumentation')
                group='lease' if verified['mode'] in ('L1','L2','CP','E1') else 'main'
                capacity=[int(env[k]) for k in ('VTA_INSN_BUFFER_BYTES','VTA_UOP_BUFFER_BYTES')]
                wanted=[1<<25,1<<25] if verified['mode'] in ('M0','M1','M2') else [q1['insn_capacity_bytes'],q1['uop_capacity_bytes']]
                if capacity!=wanted or 'VTA_INSN_SUBMIT_THRESHOLD_BYTES' in env:
                    raise ValueError('formal command capacity or scheduling threshold drift')
                for phase in ('before','after'):
                    text=read(directory/'transport'/(run.name+'_'+phase+'.json'))['stdout']
                    freq=re.search(r'\n([0-9]+)\n([a-z]+)\n',text)
                    clock=re.search(r'^\s*pl0_ref\s+\d+\s+\d+\s+\d+\s+(\d+)',text,re.M)
                    if not freq or not clock:raise ValueError('formal frequency evidence is incomplete')
                    hardware={'boot_id':boot,'label':run.name,'phase':phase,
                        'CPU_kHz':int(freq[1]),'governor':freq[2],'PL0_Hz':int(clock[1])}
                    if environments and any(hardware[k]!=environments[0][k] for k in ('CPU_kHz','governor','PL0_Hz')):
                        raise ValueError('formal CPU or PL frequency changed')
                    environments.append(hardware)
                p=expected[group]
                if verified['measured_frames']!=p['frames'] or verified['frames']!=p['frames']+p['warmup']+p['settle']:
                    raise ValueError('formal frame count drift')
                if verified['completion_window_ms']<p['minimum_planned_window_ms']:
                    raise ValueError('formal completion window too short')
                raw=read(run/'result.json');sub=profile(run)
                if sub['driver_timeout_calls'] or sub['submits_per_frame']!=10:
                    raise ValueError('main fixed program command accounting changed')
                if verified['steady_pool_growth_bytes']!=0:raise ValueError('formal pool grows after warmup')
                measured=[r for r in raw['frames'] if r['measured']]
                measured_ids={r['frame_id'] for r in measured};pending={};slot_wait=[0.,0.]
                for e in raw['events']:
                    key=(e['edge'],e['frame'])
                    if e['event']=='producer_wait':pending[key]=e['time_ms']
                    elif e['event']=='producer_acquired' and key in pending:
                        elapsed=e['time_ms']-pending.pop(key)
                        if e['frame'] in measured_ids:slot_wait[e['edge']]+=elapsed
                block={'boot_id':boot,'label':run.name,'mode':verified['mode'],
                    'measured_frames':verified['measured_frames'],'ii_ms':verified['ii_ms'],'fps':verified['fps'],
                    'window_ms':verified['completion_window_ms'],'handoff_ms':verified['handoff_ms_median'],
                    'stage0_run_ms_median':statistics.median(r['stage_end_ms'][0]-r['stage_begin_ms'][0] for r in measured),
                    'stage1_run_ms_median':statistics.median(r['stage_end_ms'][1]-r['stage_begin_ms'][1] for r in measured),
                    'stage2_run_ms_median':statistics.median(r['stage_end_ms'][2]-r['stage_begin_ms'][2] for r in measured),
                    **{'input_wait_stage'+str(i)+'_ms':verified['input_queue_wait_ms_mean'][i] for i in range(3)},
                    **{'producer_slot_wait_edge'+str(i)+'_ms_mean':slot_wait[i]/len(measured) for i in range(2)},
                    'overlap_pairs':verified['overlap']['eligible_pairs'],
                    'next_start_overlap':verified['overlap']['next_production_starts_before_old_consumer_end'],
                    'next_complete_overlap':verified['overlap']['next_production_finishes_before_old_consumer_end'],
                    'pool_high_water_bytes':verified['pool_high_water_bytes'],**sub}
                blocks.append(block);completed.setdefault(verified['mode'],[]).append(block)
            if run.name.startswith('qual_'):
                result=read(run/'result.json');snap={s['phase']:s for s in result['allocation_snapshots']}
                command=sum(int(read(run/'contract.json')['environment'][key]) for key in
                    ('VTA_INSN_BUFFER_BYTES','VTA_UOP_BUFFER_BYTES'))
                value={'boot_id':boot,'mode':verified['mode'],'graph_pool_bytes':snap['graphs_loaded']['high_water_bytes'],
                    'new_shared_slots':len([x for x in result['slots'] if not x['copy_staging'] and not x.get('borrowed')]),
                    'borrowed_input_bytes':verified['borrowed_pool_payload_bytes'],
                    'new_slots_aligned_bytes':verified['slot_external_actual_bytes'],
                    'command_backing_bytes':command,
                    'runtime_workspace_bytes':verified['pool_high_water_bytes']-snap['slots_allocated']['high_water_bytes']-command,
                    'actual_pool_high_water_bytes':verified['pool_high_water_bytes'],
                    'pool_allocation_count':snap['after_steady_state']['allocation_count'],
                    'host_staging_payload_bytes':verified['host_copy_staging_bytes'],
                    'kernel_reserved_pool_bytes':verified['fixed_reserved_pool_bytes']}
                if sum(value[k] for k in ('graph_pool_bytes','new_slots_aligned_bytes','command_backing_bytes','runtime_workspace_bytes'))!=value['actual_pool_high_water_bytes']:
                    raise ValueError('physical pool reconciliation failed')
                allocations.append(value)
                if verified['submission_records']:
                    submissions.append({'boot_id':boot,'label':run.name,'mode':verified['mode'],
                        'complete_records':verified['submission_records'],'global_sync_calls':verified['synchronize_calls'],
                        'instruction_peak_bytes':verified['instruction_peak_bytes'],'uop_peak_bytes':verified['uop_peak_bytes'],
                        **profile(run)})
        complete=(directory/'formal_complete.json').exists()
        for mode in MODES:
            rows=completed.get(mode,[])
            if complete and len(rows)!=4:raise ValueError('completed boot lacks four balanced blocks: '+mode)
            if len(rows)==4:
                ii=statistics.median(r['ii_ms'] for r in rows)
                boot_stats.append({'boot_id':boot,'mode':mode,'blocks':4,'ii_ms':ii,'fps':1000/ii,
                    'ii_ms_min':min(r['ii_ms'] for r in rows),'ii_ms_max':max(r['ii_ms'] for r in rows),
                    'block_ii_range_percent':100*(max(r['ii_ms'] for r in rows)-min(r['ii_ms'] for r in rows))/ii,
                    'handoff_ms':statistics.median(r['handoff_ms'] for r in rows),
                    'next_start_overlap':sum(r['next_start_overlap'] for r in rows),
                    'next_complete_overlap':sum(r['next_complete_overlap'] for r in rows),
                    'overlap_pairs':sum(r['overlap_pairs'] for r in rows)})
    contrasts=[]
    for old,new in CONTRASTS:
        a={x['boot_id']:x for x in boot_stats if x['mode']==old}
        b={x['boot_id']:x for x in boot_stats if x['mode']==new}
        keys=sorted(a.keys()&b.keys());improvement=[100*(a[k]['ii_ms']-b[k]['ii_ms'])/a[k]['ii_ms'] for k in keys]
        if keys:contrasts.append({'old':old,'new':new,'independent_boots':len(keys),
            'ii_reduction_percent_mean':statistics.mean(improvement),'ii_reduction_percent_CI95':ci95(improvement),
            'handoff_reduction_ms_mean':statistics.mean(a[k]['handoff_ms']-b[k]['handoff_ms'] for k in keys),
            'handoff_reduction_ms_CI95':ci95([a[k]['handoff_ms']-b[k]['handoff_ms'] for k in keys]),
            'per_boot_ii_reduction_percent':dict(zip(keys,improvement))})
    return dict(boot_directories=[str(d.relative_to(master)) for d in directories],
        boot_ids=sorted(actual_boots),run_inventory=inventory,formal_blocks=blocks,boot_statistics=boot_stats,
        contrasts=contrasts,physical_allocations=allocations,complete_submission_ledger=submissions,negative_cases=faults,
        hardware_environments=environments,
        command_reconciliation=command_checks,raw_rechecked=recheck)


def report(master,data,final):
    manifest=read(master/'package/manifest.json');q1=read(master/'q1/command_capacity_certificate.json')['derivation']
    n=len(set(r['boot_id'] for r in data['boot_statistics']))
    host_label='host_protocol_extended' if (master/'host_protocol_extended/result.json').exists() else 'host_protocol'
    host_checks=read(master/host_label/'result.json')['native_result']['checks']
    status='正式交付' if final else ('三次主对照与归还机制正式测量完成，Q2或最终交付复核仍在进行'
        if len(data['formal_blocks'])==96 and len(data['boot_statistics'])==24 else '独立启动正式测量尚未全部完成')
    rows=['# YOLOv3-tiny-224 C2 实验报告' if final else '# YOLOv3-tiny-224 C2 进行中证据快照',
        '',f'状态：{status}。统计单位为真实 Linux boot；当前完整模式数据来自 {n} 个 boot，各模式实际 boot 数见结果表。',
        '', '## 问题动机', '',
        '嵌入式 CPU 与单个物理 VTA 共享外部 DDR。固定图的边界形状、布局、生产消费关系和生命周期，允许管理共享输入使用权和命令 backing，减少框架复制和池内新增分配。共享路径是否提高吞吐，需要由同程序、相同调度的配对测量回答。',
        '', '历史 416 三阶段包采用分阶段量化；已有 224 C3 包为整图。二者均不能充当本次共享三阶段路径的数值参考。本次新建共同量化的 224 三阶段部署，全部比较只使用该冻结程序。未开展 C1 搜索、未引入 C3 优化、未修改 Word。',
        '', '## 方案', '',
        '全图量化一次后保存共同 Relay，再按真实表达式提取 CPU data→pool2 / VTA pool2→两个 logits / CPU logits→全部输出。量化 global_scale=23、跳过首卷积、低位输出、移位舍入。阶段活动线程 3/1/4，环境 TVM_NUM_THREADS=4，队列深度 2，亲和性均为 CPU0–3。该环境上限作用于每个调用线程池，进程还有闲置 helper 和调度线程，不误记为进程总共仅 4 个线程；实际线程及亲和性逐进程保存在 result.affinity。所有配置使用同一 native 可执行文件、DSO、图、参数、输入序列和预处理。',
        '', '冻结 64 张按内容哈希去重的真实图片（person、dog、62 张本地猫图），采用 YOLO load_image 的 RGB CHW letterbox224 预处理。原始图片位置、内容哈希及预处理输入哈希保存在 [inputs.json](package/inputs.json)。未使用 ResNet 归一化。',
        '', '| 边界 | 真实形状与类型 | 逻辑载荷/帧 |', '|---|---|---:|',
        '| CPU→VTA | float32 [1,64,28,28] | 200,704 B |',
        '| VTA→CPU，整 bundle | int8 [1,255,14,14] + [1,255,7,7] | 49,980 + 12,495 = 62,475 B |',
        '', '两个检测头共享同一个 frame/generation token。全部成员绑定且生产完成才能发布；异常、旧代次和未完成成员禁止发布。正常路径要求所有槽最后均为 FREE。',
        '', '| 配置 | CPU→VTA / VTA→CPU | 命令容量 |', '|---|---|---|',
        '| M0 | 普通 get/set 复制，双槽 staging | 默认 |',
        '| M1 | 外部双槽 / 外部双槽 bundle | 默认 |',
        '| M2 | 合格输入池借作 slot0，另一个外部槽 / 外部双槽 | 默认 |',
        '| M3 | 与 M2 相同 | Q1 |',
        '| manual | 与 M2 相同 | 手工指定与 Q1 完全相同的数字 |',
        '', '输入池复用准入检查布局、dtype、形状、连续大小、256 B 对齐、完整物理范围、非参数、SID 无别名、使用关系和地址互斥。CPU→VTA 输入满足条件。VTA→CPU 输入属于普通 CPU 池，未通过现有物理池准入，整边保留外部双槽；没有为增加收益绕过准入。[共享计划](shared_buffer_plan.json) 和 [完整静态存储审计](storage_audit.json) 保存逐张量原因。',
        '', f'Q1 完整提交峰值为 instruction 745,792 B、UOP 264 B。分别应用 align256(max(4096, 2×峰值))，得到 {q1["insn_capacity_bytes"]:,} B 和 {q1["uop_capacity_bytes"]:,} B。超运行时上限拒绝，不截断余量。',
        '', '峰值来自工作线程实际提交日志，覆盖冷启动、全部 64 种输入、依赖收尾和 FINISH。当前冻结运行时实际使用一个共享队列实例；按 queue_id 收集全部记录，并与全局 synchronize_calls 对账，未通过主线程查询新建空队列。Q1 证书绑定程序、运行库、bitstream、共享计划和峰值来源。源代码检查与小容量负例确认：两块 backing 都在写入前进行完整容量检查。',
        '', '另外独立检查每个实际队列的提交编号连续、逐次峰值正确，再按 reason/instruction/UOP/LOAD/STORE 及出现次数比较各主配置与各独立启动；完整命令需求签名必须一致。[命令复核账本](evidence/command_reconciliation.json) 保存逐模式结果，提交完整性与需求漂移负例使用真实日志副本验证。',
        '', '最后直接读者为 VTA 图 node1 的 CPU 量化/重排函数，原输入为 node0/SID0，唯一直接使用和唯一 SID 成员。函数返回前通过 TVMBackendParallelLaunch 的实际 ARM pending-jobs acquire-load 等待完成 CPU 任务；回调无硬件提交或外部调用。输出为独立 int8 [1,4,28,28,1,16]（50,176 B），后续图节点只读打包结果。',
        '', '静态图、生成 LLVM/ARM、同步语义及动态范围共同构成资格条件。动态资格逐帧检查原输入与打包输出的虚拟/物理范围分离，后续每帧 73 个真实 DLTensor 参数均与原输入分离；图输入仍持有底层分配，工作区由独立分配器分配，回调不保存原输入指针。资格观察器仅透明调用真实地址解析器并记录，不参与任何正式计时。[静态证据](last_reader_certificate.json)、[ARM/LLVM](reader_evidence/)、[实际地址证据](dynamic_address_certificate.json)。',
        '', '提前归还组统一 Q1 容量、下游双槽 bundle，CPU→VTA 统一外部槽：L1 单槽阶段结束归还；L2 双槽阶段结束归还；CP 单槽复制到原独立消费者私有输入后立即归还；E1 单槽最后读者返回后归还。四者共用执行器顺序和最小读者回调。E1 仅归还输入使用权，保持底层分配、中间结果以及整个 VTA 阶段的物理设备互斥锁；阶段结束不得再次归还旧 token。',
        '', '## 受控实验', '',
        '先建立串行 M0：64 张输入各重复两次、保存 8 项全部原始输出，自身逐字节一致且 float 输出有限。其后所有正例逐帧逐输出与该参考逐字节比较，容差预先固定为零。8 项真实输出包含两个 float32 检测张量及各自 mask/bias/属性元数据，不只检查检测框或部分输出。',
        '', '| 输出顺序 | 形状 | 类型 |', '|---|---|---|']
    for i,x in enumerate(manifest['stages'][2]['output_schema']['slots']):
        rows.append(f'| {i} | {x["shape"]} | {x["dtype"]} |')
    rows += ['', f'同一次启动先完成资格、延迟消费者/满槽、错误代次、重复归还、不完整 bundle、abort 唤醒、安全 join、健康新执行器以及提前归还前后事件屏障。主机 ASAN/UBSAN {host_checks} 项覆盖真实协议方法和旧 token 在槽复用后的拒绝；扩展负例在正确 token 原本可归还的 Using 状态下，分别隔离错误 edge/slot/frame/generation 检查。板端错误代次使用 generation0 注入。两者证据范围分别记录，不把主机模拟当成设备数值验证。',
        '', '可控交错分别证明：最后读取前，下一生产者确实在同一满槽阻塞；最后读取完成归还后，下一帧完整 bundle 发布可以发生在旧消费者继续执行时，输出仍一致。自然重叠与强制屏障重叠分别统计，开始生产与完整生产不合并为一项。',
        '', '同 boot 配对试点后冻结 [正式协议](formal_protocol.json) 和 [分析口径](formal_analysis_policy.json)：每组每 boot 4 轮平衡顺序，每块 5 帧预热排空、16 帧流水稳定前缀、128 个正式帧。完成窗口以 (末完成时间−首完成时间)/(N−1) 计算 II，FPS=1000/II；每窗口至少 20 秒。保存输出的最终写盘和核对在计时外；所有模式均保留相同的预分配输出暂存和逐帧快照。',
        '', '正式计时关闭逐提交诊断和地址观察器，保留相同的有界事件缓冲。完成点包含全部三阶段及八输出快照；JPEG 解码、letterbox 预处理、模型/输入预载、写盘和 NMS 不计入 FPS。每 boot 每模式用四块 II 的中位数；跨 boot 使用 3 个独立值的配对差异均值和 t(df=2) 95% 区间。同 boot 多块和多帧不冒充独立启动。没有预设共享一定提高 FPS，也未设置事后等效界限。',
        '', '手工同容量对照每 boot 另外运行一块，验证数值、分配与提交工作相同；它没有四轮平衡设计，不据此宣称自动定容与手工配置具有稳定性能差异。',
        '', '## 结果', '', '### 正确性及生命周期', '']
    positive=[x for x in data['run_inventory'] if x['status']=='byte_exact']
    rows += [f'本快照已保存并核对 {len(positive)} 个正例进程、{sum(x["frames"] for x in positive):,} 帧、{sum(x["outputs"] for x in positive):,} 项原始输出、{sum(x["bytes"] for x in positive):,} 字节。全部已核对正例保持逐字节一致。正常排空的 acquire/publish/consume/release 数目逐边守恒，槽代次递增，全部归还。完整清单见 [运行账本](evidence/run_inventory.json)。',
        '', '输入池坏 entry 负例整边安全回退到 M1 分配，64 帧全部输出一致。相同图仅改变 JSON 空白：手工同容量配置正确执行；旧 Q1 证书在图加载和分配前拒绝，说明自动证书提供身份与来源约束，空间收益来自相同数字而非“自动”标签。过小 instruction=256 B 负例安全拒绝，故障后健康新执行器通过。',
        '', '### 复制与物理分配', '',
        'M0 每帧两个边界逻辑载荷合计 263,179 B，普通 get_output/set_input 各搬一次，框架复制载荷共 526,358 B/帧；M1/M2/M3 降为零。CP 仍复制 CPU→VTA 的 200,704 B/帧。该口径是边界框架复制，不把首次参数上传、公共输入/输出快照或 VTA 内部 LOAD/STORE 混入，也不把 API 字节数称为实测 AXI 流量。',
        '', '| 模式 | 新增共享分配数 | 新增槽实际对齐 B | 借用既有池 B | 命令 backing B | 物理池高水位 B |', '|---|---:|---:|---:|---:|---:|']
    first={x['mode']:x for x in data['physical_allocations'] if x['boot_id']==read(master/'board_state.json')['boot_id']}
    for mode in (*MODES,'manual'):
        if mode not in first:continue
        x=first[mode];rows.append(f'| {mode} | {x["new_shared_slots"]} | {x["new_slots_aligned_bytes"]:,} | {x["borrowed_input_bytes"]:,} | {x["command_backing_bytes"]:,} | {x["actual_pool_high_water_bytes"]:,} |')
    pointer=master/'allocation_trace_location.json'
    if pointer.exists():
        traced=Path(read(pointer)['directory'])
        link=(traced/'allocation_trace_reconciliation.json').relative_to(master)
        rows += ['',f'另外启用现有驱动 INFO 的独立资格运行，逐笔记录分配请求、256 B 对齐大小、虚拟及物理地址、高水位，覆盖全部九种模式。图池、共享槽、两块命令 backing 和工作区均按实际地址对账；借用视图不会重复计为物理分配。[完整地址账本]({link})。',
            '', '驱动复制日志另行确认 M0 每帧 H2D 边界载荷 200,704 B、D2H 双头载荷 62,475 B；另外两笔 CPU 端 get/set 复制由统一 runner 的 API 账本记录。共享各模式的边界驱动复制为零，CP 保留 200,704 B H2D。动态加载器记录实际 DSO 初始化路径，加载文件哈希另保存在该 boot 的 transport 中。所有驱动/加载诊断均排除于正式性能。']
    rows += ['', '池内物理分配按真实共享槽范围去重，借用输入池只算一次，CPU/VTA 包装视图不当作两次分配。每个正例进程的分配计数与高水位增量对账，正式预热后无继续增长。共同图池 9,745,920 B，运行时工作区 405,504 B；分项之和严格等于高水位。[逐启动分配 CSV](evidence/physical_allocations.csv)。',
        '', '逻辑 bundle 槽与物理成员分配分别计数：M1/M2 的两条边均为 [2,2] 个逻辑槽，共四个 bundle 槽。M1 需要六个新成员分配；M2 借用其中一个输入成员，只新增五个。E1 的逻辑槽为 [1,2]，共三个 bundle 槽、五个新成员分配。双头 bundle 的两个成员不会误算成两个帧槽。',
        '', 'M1→M2 少分配一个 200,704 B 槽，6 次外部分配降至 5 次。M2→M3 命令 backing 从 67,108,864 B 降至 1,495,808 B，减少 65,613,056 B；完整提交内容/次数与 M2 及手工同容量对照相同，基线为 10 次提交/帧。手工同数字具有相同空间需求，Q1 的额外价值是自动收集完整需求、绑定资产与拒绝漂移；未声称相同数字下 Q1 比手工更省。',
        '', '外部单槽 E1 相对外部双槽 L2 再少 200,704 B；但 E1 与借用输入池的 M3 恰好具有相同新增空间，不叠加宣称两项各 200,704 B 收益，也未验证“借用池＋提前归还”的新组合。',
        '', '本表是 u-dma-buf 池内实际分配需求。VTAMemAlloc 记录是同一个物理池内的子分配，并非各自新增一次 CMA 申请；映射视图也不重复计入物理内存。内核固定预留仍为 201,326,592 B（192 MiB），CMA 总量仍为 256 MiB，片上 UOP SRAM 仍为 32 KiB；没有声称 Linux 可用 RAM 随高水位等量增加。M0 另外有 526,358 B 普通 CPU staging 有效载荷（6 个 NDArray），已独立列账，不计入物理池；libc 元数据、线程栈和进程总 RSS 未作为已测的物理节省量。每块 149 帧的公共输出暂存约 35.5 MiB，同样不计入 C2 节省。',
        '', '### 正式吞吐与交接服务时间', '',
        '| 模式 | 完整 boot 数 | 跨 boot 平均 II ms | FPS（平均 II 的倒数） | 边界交接中位数 ms |', '|---|---:|---:|---:|---:|']
    for mode in MODES:
        xs=[x for x in data['boot_statistics'] if x['mode']==mode]
        if xs:
            ii=statistics.mean(x['ii_ms'] for x in xs)
            rows.append(f'| {mode} | {len(xs)} | {ii:.6f} | {1000/ii:.6f} | {statistics.mean(x["handoff_ms"] for x in xs):.6f} |')
    rows += ['', '| 相邻对照 | 独立 boot 数 | II 降低 %（正值较快） | 95% 区间 % |', '|---|---:|---:|---|']
    for x in data['contrasts']:
        interval=x['ii_reduction_percent_CI95'];text='未满 3 个 boot' if interval is None else f'[{interval[0]:.5f}, {interval[1]:.5f}]'
        rows.append(f'| {x["old"]}→{x["new"]} | {x["independent_boots"]} | {x["ii_reduction_percent_mean"]:.5f} | {text} |')
    dispersed=sorted(data['boot_statistics'],key=lambda x:x['block_ii_range_percent'],reverse=True)[:3]
    if dispersed:
        rows += ['', '同 boot 四块仍有离散，全部原值均保留，未按吞吐筛选或追加选择性重跑。以下列出相对 II 跨度最大的三个 boot/模式组合，属于描述性离散度，不改变预定的每 boot 中位数及跨 boot 配对口径；完整范围见 boot_statistics.csv。',
            '', '| boot ID | 模式 | 四块 II 范围 ms | 相对中位数跨度 % |', '|---|---|---:|---:|']
        for x in dispersed:
            rows.append(f'| `{x["boot_id"]}` | {x["mode"]} | {x["ii_ms_min"]:.5f}–{x["ii_ms_max"]:.5f} | {x["block_ii_range_percent"]:.4f} |')
    handoff=next((x for x in data['contrasts'] if (x['old'],x['new'])==('M0','M1')),None)
    if handoff:
        interval=handoff['handoff_reduction_ms_CI95']
        confidence='独立启动尚未满三次' if interval is None else f'逐 boot 配对 t 95% 区间 [{interval[0]:.6f}, {interval[1]:.6f}] ms'
        rows += ['',f'M0→M1 的交接服务中位数平均减少 {handoff["handoff_reduction_ms_mean"]:.6f} ms/帧（{confidence}）。这是框架交接服务时间，不含等候上游输入/空槽的时间，不以该项降低直接推断完整流水 FPS 提升。']
    base_blocks=[x for x in data['formal_blocks'] if x['mode']=='M0']
    if base_blocks:
        ranges=['{:.3f}–{:.3f}'.format(min(x[f'stage{i}_run_ms_median'] for x in base_blocks),
            max(x[f'stage{i}_run_ms_median'] for x in base_blocks)) for i in range(3)]
        rows += ['',f'M0 已完成块的阶段运行中位数范围：CPU前段 {ranges[0]} ms、VTA阶段 {ranges[1]} ms、CPU后段 {ranges[2]} ms。这些是描述性阶段计时；共享减少的交接服务时间须结合长阶段和完整完成窗口解释，不能直接换算为同幅度吞吐收益。']
    if final:rows += ['', '![三次真实启动的 C2 结果](evidence/c2_yolo224_results.png)', '', '[可导出 PDF](evidence/c2_yolo224_results.pdf)。每个散点代表一个真实 Linux boot，区间来自预定的逐 boot 配对统计。']
    rows += ['', '原始逐块 II/FPS、三个阶段时间、输入等待、提交/同步、交接服务时间保存在 [formal_blocks.csv](evidence/formal_blocks.csv)，各 boot 中位数保存在 [boot_statistics.csv](evidence/boot_statistics.csv)。输入队列 Pop 等待与生产者空槽 Acquire 等待分别列账。E1 回调内归还的时间包含在 stage1.run 中；表中主路径交接计时不另加这一项，不把移动计时位置解释为开销消失。跨归还机制的性能比较使用完整完成窗口。区间只用于这套固定程序；不显著差异不证明性能等效。',
        '', '| 模式 | 自然重叠对数 | 下一帧开始生产重叠 | 下一帧完整发布重叠 |', '|---|---:|---:|---:|']
    for mode in ('L1','L2','CP','E1'):
        xs=[x for x in data['boot_statistics'] if x['mode']==mode]
        if xs:rows.append(f'| {mode} | {sum(x["overlap_pairs"] for x in xs)} | {sum(x["next_start_overlap"] for x in xs)} | {sum(x["next_complete_overlap"] for x in xs)} |')
    rows += ['', '生产开始指下一帧 stage0 GraphExecutor.run 进入；生产完成指全部成员 producer_published；旧消费者结束指 stage1 GraphExecutor.run 返回，排除之后协议清理。强制交错控制独立保存在 runs/control_before 与 runs/control_after，不纳入自然重叠统计。',
        '', '### Q2 提前提交', '']
    q2=[d for d in [master,*((master/'boots').glob('boot*') if (master/'boots').exists() else [])] if (d/'q2_complete.json').exists()]
    if not q2:rows.append('尚未完成。按冻结计划仅在三次主实验完成后评估由本程序完整 instruction 峰值推导的 75%、50%、25% 阈值；不得把历史 ResNet 阈值当作本程序证据。')
    else:
        directory=q2[-1];values=read(directory/'q2_complete.json')
        rows+=['仅在第三次 boot 进行探索性 ABBA 配对；以下 Q2 性能差异不具有三 boot 稳定性结论。Q2 沿用 M2/M3 的双槽和输入池复用方式，不与 E1 叠加。各档独立重测完整收尾峰值、重新按相同余量定容，再核对全部输出。',
            '', '| 阈值比例 | 阈值 B | 完整峰值 I/U B | 新 backing B | 相对 Q1 再省 B | 提交/帧 | ABBA Q2/基线 II ms |', '|---|---:|---|---:|---:|---:|---|']
        for ratio,v in values.items():
            if v.get('status')!='passed':rows.append(f'| {ratio}% | {v.get("threshold", "—")} | 未资格化 | — | — | — | {v.get("error",v.get("not_applicable","失败"))} |');continue
            modes={m:[read(r/'verified.json')['ii_ms'] for r in (directory/'runs').glob(f'Q2_{ratio}_paired_*_{m}')] for m in ('M3','Q2')}
            rows.append(f'| {ratio}% | {v["threshold"]:,} | {v["complete_tail_peaks"]} | {v["insn_capacity"]+v["uop_capacity"]:,} | {v["saving_vs_Q1"]:,} | {v["submissions"]/64:g} | {statistics.mean(modes["Q2"]):.5f} / {statistics.mean(modes["M3"]):.5f} |')
        rows+=['',f'完整 Q2 证据：[候选结果]({directory.relative_to(master)}/q2_complete.json)。额外提交、MMIO 和同步等待见 [q2_tradeoff.csv](evidence/q2_tradeoff.csv)；失败候选保留原始记录，仅在确认原进程退出后恢复同一 bitstream、运行健康新执行器，不重试失败配置。']
        rows += ['', 'device_wait 计时包含硬件执行和轮询，不能当作纯同步开销；MMIO 提交控制时间另列。finalize/pack/copy/cache 细分仅来自独立诊断运行，正式整体代价使用关闭详细日志的 ABBA 完成窗口。']
        if (master/'q2_full_ledger_reconciliation.json').exists():
            rows += ['', '另外独立重读六条完整 Q2 profile，核对每个实际队列的提交编号、逐次峰值、全局提交/驱动调用总数、阈值、写入容量、相同提交工作及定容推导。[独立账本](q2_full_ledger_reconciliation.json) 与 [复核脚本](audit_q2_complete_ledger.py)。']
    interpretation=[]
    main=[x for x in data['contrasts'] if (x['old'],x['new']) in CONTRASTS[:3]]
    if len(main)==3 and all(x['independent_boots']==3 and x['ii_reduction_percent_CI95'][0]<=0<=
                           x['ii_reduction_percent_CI95'][1] for x in main):
        interpretation += ['三个主相邻对照的完整流水 II 差异 95% 区间均跨过零，未得到稳定吞吐提升或下降的明确证据；这不构成性能等效证明。此部署已验证的主收益是消除边界框架复制、减少新增成员分配，以及在提交工作不变时缩小外部命令 backing。', '']
    early=next((x for x in data['contrasts'] if (x['old'],x['new'])==('L1','E1')),None)
    dual=next((x for x in data['contrasts'] if (x['old'],x['new'])==('L2','E1')),None)
    if early and dual and early['independent_boots']==dual['independent_boots']==3:
        interval=early['ii_reduction_percent_CI95'];other=dual['ii_reduction_percent_CI95']
        interpretation += [f'单槽最后读者归还 E1 相对单槽阶段结束归还 L1，逐 boot 配对 II 平均降低 {early["ii_reduction_percent_mean"]:.5f}%（95% 区间 [{interval[0]:.5f}, {interval[1]:.5f}]%）。相对双槽 L2 的 II 变化为 {dual["ii_reduction_percent_mean"]:.5f}%（95% 区间 [{other[0]:.5f}, {other[1]:.5f}]%）。结合逐帧数值和两种生产重叠事件，说明本图安全的单槽提前归还能恢复生产重叠能力；未设置等效界限，不据小差异宣称与双槽统计等效。', '']
    if q2:
        passed=[(ratio,value) for ratio,value in values.items() if value.get('status')=='passed']
        if passed:
            ratio,value=min(passed,key=lambda x:x[1]['insn_capacity']+x[1]['uop_capacity'])
            interpretation += [f'Q2 已通过档位中最小的外部命令 backing 为 {value["insn_capacity"]+value["uop_capacity"]:,} B（{ratio}% 阈值），相对 Q1 再减少 {value["saving_vs_Q1"]:,} B，提交次数为 {value["submissions"]/64:g} 次/帧。其吞吐及同步代价只具有第三 boot 的 ABBA 探索证据，不扩大为三启动稳定结论。', '']
        else:interpretation += ['本次 Q2 三档检查未取得进一步缩容的合格配置，原始失败与回退健康检查保留，不制造额外收益。', '']
    rows += ['', '## 结论', '', *interpretation,
        '已验证范围：固定 YOLO224 三阶段程序的全部八输出一致性、有界跨执行器共享、双头同帧同代 bundle、一个合格输入池的复用及整边回退、完整提交需求 Q1 和相同数字手工对照、最后读者资格及单槽提前归还的受控交错。',
        '', '不适用范围：下游普通 CPU 输入池不满足当前物理池复用准入；不强行复用。没有将旧 416 检测结果、历史 FPS 或已有 224 整图时延拼接进本实验。',
        '', '仍未完成的任务精度：没有检测标签集，本次不报告 mAP 或任务检测精度。字节一致性证明同程序 C2 路径保持输出，不能替代任务精度评估。当前固定图的 89 次调度查询均使用 fallback，未进行调度搜索；所得 FPS 只描述该冻结部署对象，未声称 YOLO224 的最佳性能。',
        '', '独立启动及稳定性能状态以本报告表格、boot ID 清单和完成标记为准。Q2 若仅有单 boot 探索，不能宣称其性能代价已获得跨启动稳定复核。',
        '', '## 程序、环境和复核入口', '',
        '- [程序 manifest](package/manifest.json)、[实际运行计划](program.json)、[完整部署 SHA256 清单](deployment_inventory.json)、[本机环境](local_environment.json)。',
        '- [编译命令与日志](compile_native_01.json)、[隔离运行库来源](runtime/build_manifest.json)、[运行源码快照](native_source/)。',
        f'- [Q1 证书](q1/command_capacity_certificate.json)、[工作队列完整峰值](worker_queue_profile.json)、[主机协议 {host_checks} 项]({host_label}/result.json)、[证据校验器负例](evidence_verifier_tests.log)、[板端控制证据复核](qualification_reconciliation.json)。',
        '- [原始板端运行](runs/)、[SSH/环境/加载记录](transport/)、[三启动调度记录](orchestration/)、[原始证据索引](evidence/run_inventory.json)。',
        '- [准备失败的保留说明](preparation_lineage.json)：run01 输入名适配、run02 旧总超时修正；未并入正式统计。',
        '', '真实 boot ID：', '']
    rows += [f'- `{boot}`' for boot in data['boot_ids']]
    if data['hardware_environments']:
        env=data['hardware_environments'][0]
        rows += ['',f'全部已纳入正式统计的块，在开始和结束均核对 CPU {env["CPU_kHz"]:,} kHz、governor={env["governor"]}、PL0 {env["PL0_Hz"]:,} Hz 一致。[逐块环境 CSV](evidence/hardware_environments.csv)。本板未提供所查询的 thermal_zone 温度数据，未将温度记为已测稳定项。']
    return '\n'.join(rows)+'\n'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--publish',action='store_true');p.add_argument('--recheck',action='store_true')
    a=p.parse_args();master=a.out.resolve();data=collect(master,a.recheck or a.publish)
    if a.publish:
        if len(data['boot_ids'])!=3 or len(data['formal_blocks'])!=96 or len(data['boot_statistics'])!=24:
            raise ValueError('three complete boot campaigns required before publication')
        if not (master/'orchestration_complete.json').exists():raise ValueError('Q2/orchestration is unfinished')
        if not (master/'allocation_trace_location.json').exists():raise ValueError('actual allocation trace is unfinished')
        trace=read(master/'allocation_trace_location.json')
        from c2_yolo224_allocation import reconcile
        traced=Path(trace['directory'])
        if str(traced.relative_to(master)) not in data['boot_directories']:raise ValueError('allocation trace is outside this campaign')
        ledger=reconcile(traced)
        if sha(traced/'allocation_trace_reconciliation.json')!=trace['trace_sha256'] or ledger['boot_id'] not in data['boot_ids']:
            raise ValueError('actual allocation trace identity differs')
        data['driver_allocation_trace']=str((traced/'allocation_trace_reconciliation.json').relative_to(master))
        target=master/'evidence';target.mkdir(exist_ok=False)
    else:
        stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        target=master/'progress'/stamp;target.mkdir(parents=True,exist_ok=False)
    write(target/'run_inventory.json',data['run_inventory'])
    write(target/'reconciliation.json',data)
    write(target/'command_reconciliation.json',data['command_reconciliation'])
    if a.publish:
        static=read(master/'last_reader_certificate.json')
        manifest=read(master/'package/manifest.json')
        if static['identities']['graph']!=manifest['stages'][1]['artifact_sha256']['graph_sha256']:
            raise ValueError('last-reader static graph identity differs')
        qualifications=[]
        for sub in data['boot_directories']:
            directory=master/sub;dynamic=read(directory/'dynamic_address_certificate.json')
            qualifications.append({'boot_id':read(directory/'board_state.json')['boot_id'],
                'dynamic_address_certificate_sha256':sha(directory/'dynamic_address_certificate.json'),
                'controlled_interleavings_sha256':sha(directory/'qualification_reconciliation.json'),
                'observed_frames':dynamic['frames'],'later_argument_checks_per_frame':sorted({r['later_argument_checks'] for r in dynamic['rows']})})
        write(target/'last_reader_qualification.json',{'status':'passed','static_certificate_sha256':sha(master/'last_reader_certificate.json'),
            'static_certificate_preserved_as_initial_dynamic_pending_record':True,
            'qualifications':qualifications,'all_saved_positive_outputs_rechecked_byte_exact':True,
            'no_underlying_free_or_early_physical_VTA_unlock':True})
    for name in ('formal_blocks','boot_statistics','physical_allocations','complete_submission_ledger','hardware_environments'):
        csv_file(target/(name+'.csv'),data[name])
    contrasts=[{**{k:v for k,v in x.items() if k!='per_boot_ii_reduction_percent'},
        'ii_reduction_percent_CI95':json.dumps(x['ii_reduction_percent_CI95'])} for x in data['contrasts']]
    csv_file(target/'paired_contrasts.csv',contrasts)
    q2rows=[]
    for sub in data['boot_directories']:
        directory=master/sub
        for run in sorted((directory/'runs').glob('Q2*')):
            if (run/'verified.json').exists() and (run/'exitcode').read_text().strip()=='0':
                q2rows.append({'boot_id':read(directory/'board_state.json')['boot_id'],'label':run.name,
                    **profile(run),'ii_ms':read(run/'verified.json')['ii_ms'],
                    'pool_high_water_bytes':read(run/'verified.json')['pool_high_water_bytes']})
    csv_file(target/'q2_tradeoff.csv',q2rows)
    document=report(master,data,a.publish)
    if not a.publish:
        document=re.sub(r'\]\(([^)]+)\)',lambda m:']('+(
            m[1][len('evidence/'):] if m[1].startswith('evidence/') else '../../'+m[1])+')',document)
    path=master/'REPORT_ZH.md' if a.publish else target/'PROGRESS_ZH.md';path.write_text(document)
    write(target/'provenance.json',{'report_generator_sha256':sha(Path(__file__)),
        'report_sha256':sha(path),'raw_outputs_independently_rechecked':data['raw_rechecked'],
        'formal_protocol_sha256':sha(master/'formal_protocol.json'),
        'protected_existing_sources_unchanged':all(sha(ROOT/name)==value for name,value in read(master/'initial_sources.json').items())})
    print(json.dumps({'report':str(path),'boots':len(data['boot_ids']),'formal_blocks':len(data['formal_blocks']),
        'final':a.publish,'raw_rechecked':data['raw_rechecked']},ensure_ascii=False))


if __name__=='__main__':main()
