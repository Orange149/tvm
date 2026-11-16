"""Read-only disassembly of the shipped runtime; no hotspot inference."""
import ast
from pathlib import Path
import shutil
import subprocess
from .common import api,REPO

def audit(bundle,output):
    bundle,output=Path(bundle),Path(output);c=api(bundle);output.mkdir(parents=True,exist_ok=True)
    runtime=bundle/'runtime/libtvm_runtime.so';record=c.read(bundle/'plans/runtime_build.json')
    if record['runtime_sha256']!=c.sha256(runtime):raise ValueError('runtime build identity differs')
    nm=shutil.which('aarch64-linux-gnu-nm');objdump=shutil.which('aarch64-linux-gnu-objdump')
    if not nm or not objdump:raise RuntimeError('AArch64 binary inspection tools missing')
    symbols=subprocess.run([nm,'-D','-C',str(runtime)],capture_output=True,text=True,check=True).stdout
    all_disasm=subprocess.run([objdump,'-d','-C',str(runtime)],capture_output=True,text=True,check=True).stdout
    # Keep whole relevant symbol bodies, including inlined WaitForJobs in Launch.
    pieces=[];current=[];keep=False
    for line in all_disasm.splitlines():
        if line.endswith('>:') and '<' in line:
            if keep:pieces.append('\n'.join(current))
            keep=any(name in line for name in ('TVMBackendParallelBarrier>','ThreadPool::Launch(',
                                               'threading::Yield()>','SpscTaskQueue::Pop('))
            current=[line]
        elif keep:current.append(line)
    if keep:pieces.append('\n'.join(current))
    selected='\n\n'.join(pieces)+'\n'
    if 'TVMBackendParallelBarrier>:' not in selected or 'ThreadPool::Launch(' not in selected:
        raise ValueError('required waiting symbols absent')
    if 'threading::Yield()' not in selected:raise ValueError('waiting yield path not found')
    (output/'runtime_waiting_disassembly.txt').write_text(selected)
    (output/'runtime_waiting_symbols.txt').write_text('\n'.join(l for l in symbols.splitlines()
        if any(n in l for n in ('ParallelBarrier','ThreadPool::Launch','threading::Yield')))+'\n')
    origins={}
    for relative in ('src/runtime/thread_pool.cc','src/runtime/threading_backend.cc'):
        source=REPO/relative;target=output/Path(relative).name
        shutil.copyfile(source,target);origins[relative]={'sha256':c.sha256(source),'snapshot':target.name}
    result={'runtime_sha256':c.sha256(runtime),'runtime_build_record_sha256':c.sha256(bundle/'plans/runtime_build.json'),
        'sources':origins,'binary_wait_paths_confirmed':['TVMBackendParallelBarrier','ThreadPool::Launch (WaitForJobs may be inlined)'],
        'binary_yield_path_present':True,'idle_spin_count_is_not_barrier_wait_policy':True,
        'runtime_rebuilt':False,'hotspot_or_causation_proven':False,
        'source_binary_exact_revision_attested':False,
        'limitation':'build record does not bind these C++ source hashes; current source snapshots are explanatory, binary evidence is authoritative',
        'scope':'read-only; OS yield is not a blocking condition-variable wait and is not evidence of current runtime overhead'}
    c.write(output/'runtime_audit.json',result,replace=True)
    (output/'REPORT_zh.md').write_text(
        '# 冻结运行库等待路径只读核验\n\n'
        '核验对象：实际部署的 `libtvm_runtime.so`，SHA-256：`'+result['runtime_sha256']+'`。\n\n'
        '- 二进制中确认了并行屏障、ThreadPool::Launch（可包含内联WaitForJobs）的Yield调用路径。\n'
        '- TVM_THREAD_POOL_SPIN_COUNT控制空闲任务队列的旋转次数，不等于并行屏障或任务完成等待策略。\n'
        '- Yield路径与条件变量阻塞不同；存在该路径不证明它占用了多少CPU或造成当前吞吐误差。\n'
        '- 当前源码快照仅用于解释：原构建记录未绑定这两个C++文件的精确版本。\n\n'
        '本轮没有修改、重编译运行库，没有增加性能探针或干预线程调度。\n\n'
        '[反汇编](runtime_waiting_disassembly.txt) · [符号](runtime_waiting_symbols.txt) · [身份及限制](runtime_audit.json)\n')
    return result

def verify_upstream():
    """Check the vendored function byte-for-byte against its frozen source digest."""
    root=Path(__file__).parent;c=api();origin=c.read(root/'upstream_origin.json')
    text=(root/'upstream.py').read_text()
    node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='submodular_pick')
    function='\n'.join(text.splitlines()[node.lineno-1:node.end_lineno])+'\n'
    from hashlib import sha256
    if sha256(function.encode()).hexdigest()!=origin['function_sha256']:raise ValueError('upstream selector modified')
    return origin
