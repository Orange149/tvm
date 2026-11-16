"""Read-only attribution boundary: VTA host arithmetic versus driver waiting.

Reuses existing isolated driver traces and frozen modules. No new inference,
poll tuning, CPU-pool expansion, or static-score change is performed here.
"""
from pathlib import Path
from collections import Counter
import json
import re
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from cpu_vta_partitioning.common import read,sha256,verify_inventory,write
from scheduler_gap_audit.curves import csv_write

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/vta_host_work_origin_01'
REPORT=ROOT.parents[4]/'paper/my_paper/analysis/20261111_vta_host_work_origin'


def disassemble(module,function):
    r=subprocess.run(['aarch64-linux-gnu-objdump','-d','--disassemble='+function,str(module)],
        capture_output=True,text=True,check=True,timeout=15)
    return r.stdout


def run():
    profile_path=ROOT/'artifacts/polling_profile_02/analysis/RESULTS.json';profiles=read(profile_path)
    sources={str(profile_path):sha256(profile_path)}
    for path,expected in profiles['source_receipts'].items():
        if sha256(path)!=expected:raise ValueError('isolated driver receipt changed')
        r=read(path);verify_inventory(r['raw'],r['files']);sources[path]=expected
    OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    rows=[];nodes=[];representative=None
    for old in profiles['component_table']:
        network=old['network'];program=old['program']
        capsule=ROOT/'artifacts/scheduler_gap_audit_01'/network/'capsule'
        module=capsule/'programs'/program/'module.so';graph_path=module.parent/'graph.json'
        manifest=read(capsule/'manifest.json');programs=read(capsule/'plans/programs.json')
        if sha256(module)!=programs[program]['files']['module.so']['sha256']:
            raise ValueError('program identity differs from frozen capsule')
        if sha256(module)!=profiles['profiles'][network+'/'+program]['module_sha256']:
            raise ValueError('driver profile and graph module differ')
        sources[str(module)]=sha256(module);sources[str(graph_path)]=sha256(graph_path)
        graph=read(graph_path);counts=Counter()
        for index,node in enumerate(graph['nodes']):
            if node['op']!='tvm_op':continue
            function=node['attrs']['func_name'];helper=function+'_compute_';assembly=disassemble(module,helper)
            calls=re.findall(r'\bbl\s+[0-9a-f]+\s+<([^>]+)>',assembly)
            api=sorted({c.split('@')[0] for c in calls if c.startswith('VTA')})
            has_indirect=bool(re.search(r'\bblr\b',assembly))
            if any(c.startswith(('VTALoadBuffer','VTAStoreBuffer','VTAPush','VTAUop')) for c in api):
                category='VTA_command_construction'
            elif 'VTABufferCPUPtr' in api and has_indirect:
                category='CPU_access_with_indirect_compute'
            else:category='unresolved'
            counts[category]+=1
            nodes.append(dict(network=network,program=program,node=index,function=function,
                helper=helper,category=category,direct_VTA_APIs=','.join(api),
                has_indirect_compute=has_indirect,
                helper_disassembly_sha256=__import__('hashlib').sha256(assembly.encode()).hexdigest(),
                individual_node_runtime_not_measured=True))
            if network=='yolov3_tiny416' and program=='vta_03_06' and function=='tvmgen_default_fused_cast_multiply_transpose_reshape_nn_leaky_relu':
                # These addresses are read from this frozen helper, not guessed
                # from a graph function name. Verify the instruction anchors.
                if not ('5f44:' in assembly and '#0xf64' in assembly and '5f50:' in assembly):
                    raise ValueError('representative CPU launch site changed')
                reloc=subprocess.run(['aarch64-linux-gnu-readelf','-rW',str(module)],capture_output=True,text=True,check=True).stdout
                proof=[l.strip() for l in reloc.splitlines() if re.match(r'^000000000003afb0\s',l)]
                if len(proof)!=1 or '__TVMBackendParallelLaunch' not in proof[0]:
                    raise ValueError('indirect CPU launch relocation changed')
                body=subprocess.run(['aarch64-linux-gnu-objdump','-d','--start-address=0x5f64','--stop-address=0x6190',str(module)],
                    capture_output=True,text=True,check=True).stdout
                fp=[l.strip() for l in body.splitlines() if re.search(r'\b(scvtf|fmul|fcmgt)\b',l)]
                if len(fp)<3:raise ValueError('representative CPU floating arithmetic missing')
                (OUT/'representative_helper.asm').write_text(assembly)
                (OUT/'representative_CPU_body.asm').write_text(body)
                representative=dict(module_sha256=sha256(module),function=function,
                    indirect_launch_PC='0x5f50',parallel_callback_PC='0x5f64',
                    num_tasks_argument='w2=0, uses configured runtime.NumThreads',
                    relocation=proof[0],CPU_arithmetic_examples=fp[:8],
                    proves_CPU_arithmetic_and_runtime_parallel_launch=True,
                    does_not_measure_this_node_individual_cost=True)
        outside=old['observed_CPU_outside_driver_ms'];inside=old['observed_CPU_inside_driver_ms'];total=old['observed_run_CPU_ms']
        rows.append(dict(network=network,program=program,run_CPU_ms=total,inside_driver_CPU_ms=inside,
            outside_driver_CPU_ms=outside,outside_driver_fraction_percent=100*outside/total,
            wrapper_and_clock_scope_difference_ms=total-inside-outside,
            driver_CPU_is_not_pure_polling_CPU=True,counts=dict(counts),
            node_costs_not_measured=True,probe_quality_passed=old['probe_quality_passed']))
    if representative is None:raise ValueError('missing source-backed representative')
    # Read only the already accepted recent uninstrumented affinity evidence.
    admissions=read(ROOT/'artifacts/cpu_exclusion_gap_01/RESULTS.json');thread_rows=[]
    for path,expected in admissions['receipts'].items():
        if sha256(path)!=expected:raise ValueError('recent affinity receipt changed')
        r=read(path)
        if r['task']['spec']['phase']!='clean' or r['task']['cpu_admission']!='off':continue
        verify_inventory(r['raw'],r['files']);sources[path]=expected
        aff=read(Path(r['raw'])/'affinity.json')
        for i,stage in enumerate(r['task']['stages']):
            if stage['device']!='vta':continue
            entry=aff['stages'][i]
            thread_rows.append(dict(network=r['task']['spec']['network'],program=stage['program'],
                requested_threads=stage['threads'],actual_configured_threads=entry['active_runtime_threads'],
                allowed_mask=stage['mask'],
                whole_session_worker_CPU_ms=sum(t['cpu_time_delta_ms'] or 0 for t in r['summary']['thread_ledger']
                    if t.get('stage')==stage['program'] and t['role']=='worker'),
                scope='configured parallelism and whole-session counters; idle helpers not active compute slots'))
    result=dict(rows=rows,nodes=nodes,representative=representative,VTA_pool_configuration=thread_rows,sources=sources,
        new_DNN=0,new_native_sessions=0,new_component_profiles=0,
        total_host_CPU_already_accounted_in_old_costs=True,
        most_outside_driver_work_is_not_automatically_all_activation_work=True,
        no_new_static_formula_or_Top20=True)
    write(OUT/'RESULTS.json',result,replace=True);csv_write(OUT/'components.csv',rows);csv_write(OUT/'nodes.csv',nodes)
    text=['VTA子图的CPU工作来源：轮询之外的实际CPU算子','',
        '本轮仅核验已有隔离轨迹、冻结graph.json、AArch64模块与近期线程配置，没有新增板测或profile。',
        'VTA子图不是全部工作都在FPGA上执行。模块中同时包含命令构造、CPU可访问缓冲区以及CPU算子代码。',
        '代表性YOLO激活函数的计算入口，通过__TVMBackendParallelLaunch调用CPU回调；实际AArch64回调中有scvtf、fmul、fcmgt浮点指令。',
        '该并行启动的num_tasks参数为0，因此使用运行时配置的线程数；不是仅凭函数名认定执行位置。', '',
        '网络 / VTA子图 / 运行CPU核时ms / 底层设备调用内ms / 设备调用外ms / 外部占比']
    for row in rows:text.append(f"{row['network']} / {row['program']} / {row['run_CPU_ms']:.3f} / {row['inside_driver_CPU_ms']:.3f} / {row['outside_driver_CPU_ms']:.3f} / {row['outside_driver_fraction_percent']:.1f}%")
    text+=['','设备调用内的CPU时间也包括提交及其他驱动工作，不能全部叫轮询。设备调用外仍包含命令构造、缓冲管理等，不能全部归给激活或布局变换。',
        '两个YOLO子图合计约48.10核毫秒位于设备调用之外，占其约60.83核毫秒run CPU时间的79.1%；单改轮询间隔无法移除这部分工作。',
        '单项clock包络与包装器记账造成的小差额单列，不强行归零。', '',
        '近期正式线程证据：']
    for row in thread_rows:text.append(f"{row['network']} / {row['program']} / 配置与实际线程数{row['requested_threads']}/{row['actual_configured_threads']} / 允许核{row['allowed_mask']} / 辅助worker整段CPU增量{row['whole_session_worker_CPU_ms']:.1f}ms")
    text+=['允许四核不等于使用四个计算线程。当前这些VTA阶段配置为1线程，其CPU回调在这一线程池配置下执行。',
        '原主机CPU总需求已经包含这部分时间，本报告没有证明漏计总量。新的信息是工作类型与可用并行能力，不能将同一工作再加一次。',
        '对ResNet/YOLO差异的支持：当前两个YOLO模块有更重的设备调用外CPU工作；具体归因仍需节点级计时，不能仅凭网络名称或函数名作结论。', '',
        '下一项最小验证：仅在两个YOLO VTA模块上比较1与4个运行时线程，各做资格及2+8帧隔离测量；不改模块、切图或评分。',
        '上限4配置、8会话、44次VTA子图调用、0次完整DNN；若线程数未真正生效或完整输出不符则拒绝。',
        '这只判断隐藏CPU工作是否值得细化，不证明流水会受益；与CPU前段共享后的收益及新的匹配profile需另行验证。',
        '预留旧1线程策略；不自动扩展完整域，不将新线程成本套到其他子图。']
    (REPORT/'FINDINGS_zh.txt').write_text('\n'.join(text)+'\n')
    print(json.dumps(dict(components=rows,configured_threads=thread_rows,new_DNN=0,new_native_sessions=0),ensure_ascii=False,indent=2))
    return result


if __name__=='__main__':run()
