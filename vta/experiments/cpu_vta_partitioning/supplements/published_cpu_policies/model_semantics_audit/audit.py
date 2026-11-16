"""Source/ELF/clock boundaries, explicitly retaining missing vendor identity."""
from pathlib import Path
import subprocess
from cpu_vta_partitioning.common import read,sha256,write
from .common import ROOT,PARENT,STUDY,freeze,protocol,sources


def audit(output):
    output=Path(output);delivery=read(PARENT/'DELIVERY.json');pred=PARENT/'PREDICTIONS_FREEZE.json'
    if sha256(pred)!=delivery['prediction_sha256']:raise ValueError('parent prediction freeze changed')
    runner=PARENT/'build/sources/vta_stage_pipeline_runner_phase.cc';build=read(PARENT/'build/BUILD.json')
    if sha256(runner)!=build['source_files'][str(runner)]:raise ValueError('actual observer source changed')
    s=runner.read_text()
    for anchor in ('deferred_results.push_back(ResultJSON(', 'RawOutputsJSON(frame.stage_outputs.back())',
                   'FNV1a64(bytes, nbytes)', 'sum += value;', 'frame->done_ms = NowMillis();',
                   'collector_phase.End();'):
        if anchor not in s:raise ValueError('measured collector source path changed')
    runtime=PARENT/'resnet18/capsule/runtime/libtvm_runtime.so'
    manifest=read(PARENT/'resnet18/capsule/manifest.json')
    if sha256(runtime)!=manifest['runtime_sha256']:raise ValueError('actual runtime identity differs')
    nm=subprocess.run(['nm','-D',str(runtime)],capture_output=True,text=True,timeout=30)
    if nm.returncode:raise ValueError('local ELF symbol audit failed')
    symbols={n:n in nm.stdout for n in ('TVMBackendParallelLaunch','TVMBackendParallelBarrier','_ZN3tvm7runtime9threading5YieldEv')}
    if not all(symbols.values()):raise ValueError('observed runtime symbols absent')
    output.mkdir(parents=True,exist_ok=True);(output/'runtime_symbols.txt').write_text(nm.stdout)
    ranges=[]
    for work,terms,timing,model,unknown in [
        ('CPU子图计算与交接',['RunStage(','phase_set.End();','phase_get.End();'],'阶段set/run/get墙钟、线程CPU和回调','线程映射、CPU工作和串行段','回调外worker运行时CPU未完整定位'),
        ('VTA主机与物理设备',['std::unique_lock<std::mutex> lock(vta_run_mutex);','timing.vta_run_call_start_ms'],'主机阶段与设备调用分开','主机CPU/设备非运行片段，单VTA累计','共享主机实际迁移过程未知'),
        ('输出核验与记录',['RawOutputsJSON(','FNV1a64(bytes, nbytes)','deferred_results.push_back(ResultJSON('],'write_completed整体CPU，不含主线程全部工作','旧回放省略；新诊断模式才补入','哈希、数值遍历、JSON未单独测量，不能拆分份额'),
        ('输入生产与结果收集',['producer_phase.End();','collector_phase.End();'],'CPU时间与阻塞墙钟分开','独立控制执行单元及深度2队列','析构、回收和控制段外工作保持未知'),
        ('并行启动与完成',['ConfigureThreadPool(','--cpu-worker-pinning'],'回调/启动/尾部同钟观测','同时就绪的理想共享及完成汇合','实际入队、唤醒、CFS历史状态不能由静态源码直接还原')]:
        ranges.append(dict(work=work,source_terms=[dict(text=t,lines=[i for i,l in enumerate(s.splitlines(),1) if t in l]) for t in terms],
            observed_scope=timing,model_scope=model,remaining_unknown=unknown))
    runtime_build=ROOT/'artifacts/bundles/27b583640470adcc2bbd2194/plans/runtime_build.json'
    result=dict(protocol=protocol(),sources=sources(),parent_delivery_sha256=sha256(PARENT/'DELIVERY.json'),
        prediction_sha256=sha256(pred),intrinsics_sha256=sha256(STUDY/'pool_temporal/INTRINSICS.json'),
        runner_source_sha256=sha256(runner),runtime_sha256=sha256(runtime),runtime_symbols=symbols,scope_table=ranges,
        runtime_build_record_sha256=sha256(runtime_build),
        runtime_full_source_identity='UNVERIFIED: historical build record has no complete runtime source hash list; public ABI and interception were verified',
        linux_reference=dict(version='upstream Linux v5.4 reference, not vendor build identity',
            documentation='https://www.kernel.org/doc/html/v5.4/scheduler/sched-design-CFS.html',
            implementation='https://github.com/torvalds/linux/blob/v5.4/kernel/sched/fair.c',
            CFS_uses_vruntime_and_preemption_granularity=True,verified_board_granularity_ns=None,
            vendor_kernel_source_identity=None,default_timeslice_not_invented=True),
        no_new_board_access=True,no_performance_values_decoded=True)
    freeze(output/'AUDIT.json',result);return dict(audited=True,scope_rows=len(ranges),runtime_full_source_identity=result['runtime_full_source_identity'])
