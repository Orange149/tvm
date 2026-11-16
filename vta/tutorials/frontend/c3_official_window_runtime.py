#!/usr/bin/env python3
"""Independent official-window runtime overlays and collection equivalence gate.

Production and previously sealed sources are read, never changed. B and A use
the same new runtime. Collection flags only gate observation; the new greedy
interval/read guard is a separately disclosed common safety repair.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PYTHON = Path('/home/orange/miniconda3/envs/vta-resnet/bin/python')
COLLECTION = HERE/'c3_official_window_runtime_collection.h'
QUEUE = HERE/'c3_official_window_runtime_uop_queue.h'
HELPER = HERE/'c3_official_window_runtime_helper.cc'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def balanced_end(text, start, left='(', right=')'):
    depth, quote, line, block = 0, None, False, False
    i = start
    while i < len(text):
        c, following = text[i], text[i:i+2]
        if line:
            if c == '\n': line = False
        elif block:
            if following == '*/': block = False; i += 1
        elif quote:
            if c == '\\': i += 1
            elif c == quote: quote = None
        elif following == '//': line = True; i += 1
        elif following == '/*': block = True; i += 1
        elif c in ('"', "'"): quote = c
        elif c == left: depth += 1
        elif c == right:
            depth -= 1
            if depth == 0: return i
        i += 1
    raise ValueError('unbalanced source anchor')


def gate_calls(text, pattern):
    matches = list(re.finditer(pattern, text))
    for match in reversed(matches):
        start = match.start()
        opening = text.index('(', match.end()-1)
        end = balanced_end(text, opening)+1
        if text[end] != ';': raise ValueError('collection call must be a complete statement')
        old = text[start:end+1]
        text = text[:start]+'if (c3_official_window::Detailed()) { '+old+' }'+text[end+1:]
    return text, len(matches)


def gate_clocks(text):
    count = 0
    for match in reversed(list(re.finditer(r'(?:vta::)?NowMicros\(\)', text))):
        prefix = text[text.rfind('\n', 0, match.start())+1:match.start()].strip()
        if prefix in ('double', 'inline double'): continue  # Declaration/definition.
        original = match.group()
        text = text[:match.start()]+'c3_official_window::DetailClock([]() { return '+original+'; })'+text[match.end():]
        count += 1
    return text, count


def gate_methods(text, class_name, method_pattern):
    begin = text.index('class '+class_name)
    opening = text.index('{', begin)
    end = balanced_end(text, opening, '{', '}')
    section = text[opening:end+1]
    matches = list(re.finditer(r'void ('+method_pattern+r')\([^{};]*\)\s*\{', section))
    for match in reversed(matches):
        section = section[:match.end()]+'\n    if (!c3_official_window::Detailed()) return;'+section[match.end():]
    return text[:opening]+section+text[end+1:], [m[1] for m in matches]


def runtime_overlay():
    text = (REPO/'vta/runtime/runtime.cc').read_text()
    start = text.index('template <int kMaxBytes, bool kCoherent, bool kAlwaysCache>\nclass UopQueue')
    end = text.index('// Internal kernel structure\nclass UopKernelMap', start)
    text = text[:start]+QUEUE.read_text()+'\n\n'+text[end:]
    old = '''      VTAMemInsn* insn = insn_queue_.CreateMemInsn(VTA_MEM_ID_UOP);
      insn->opcode = VTA_OPCODE_LOAD;
      uop_queue_.FlushUopLoad(insn);'''
    if text.count(old) != 2: raise ValueError('expected GEMM/ALU load seams changed')
    text = text.replace(old, '      uop_queue_.FlushUopLoad(insn_queue_);')
    text, methods = gate_methods(text, 'RuntimeProfiler', r'Add[A-Z][A-Za-z0-9_]*')
    if len(methods) != 13: raise ValueError('runtime detailed Add census changed')
    text, calls = gate_calls(text, r'(?:vta::)?RuntimeProfiler::Global\(\)\.Add[A-Za-z0-9_]+\(')
    text = text.replace('queue_capacity::DiagnosticEnabled()', '(c3_official_window::Detailed() && queue_capacity::DiagnosticEnabled())')
    text = text.replace('    VTADriverProfilerStatus(&driver_before);', '    if (c3_official_window::Detailed()) VTADriverProfilerStatus(&driver_before);')
    text = text.replace('    VTADriverProfilerStatus(&driver_after);', '    if (c3_official_window::Detailed()) VTADriverProfilerStatus(&driver_after);')
    anchor = '    int timeout = VTADeviceRun(device_, insn_queue_.dram_phy_addr(), insn_count, wait_cycles);'
    if text.count(anchor) != 1: raise ValueError('normal complete submit anchor changed')
    text = text.replace(anchor, anchor+'\n    c3_official_window::RecordSubmit(queue_submit_reason_, insn_count, serialized_uop_bytes, timeout);')
    # Read-only debug attestation; does not affect debug or execution policy.
    anchor = '  void Synchronize(uint32_t wait_cycles) {'
    text = text.replace(anchor, '  uint32_t OfficialDebugFlag() const { return debug_flag_; }\n\n'+anchor, 1)
    text, clock_count = gate_clocks(text)
    text = '#include "c3_official_window_runtime_collection.h"\n'+text
    text += '''
TVM_REGISTER_GLOBAL("vta.runtime.official_submit_ledger_clear").set_body_typed([]() {
  c3_official_window::ClearLedger();
});
TVM_REGISTER_GLOBAL("vta.runtime.official_submit_ledger").set_body_typed([]() {
  return c3_official_window::LedgerJSON();
});
TVM_REGISTER_GLOBAL("vta.runtime.official_collection_status").set_body_typed([]() {
  const auto& state = c3_official_window::State();
  std::ostringstream out;
  out << "{\\"schema\\":\\"c3_official_window_collection_v1\\",\\"mode\\":\\""
      << (state.detailed ? "detailed" : "minimal")
      << "\\",\\"detailed_clock_reads\\":" << state.detailed_clock_reads
      << ",\\"debug_flag\\":" << vta::CommandQueue::ThreadLocal()->OfficialDebugFlag()
      << ",\\"replay\\":" << vta::CommandQueue::ThreadLocal()->TemplateReplayStatusJSON()
      << ",\\"queue\\":" << vta::CommandQueue::ThreadLocal()->CapacityStatusJSON() << '}';
  return out.str();
});
'''
    return text, {'runtime_Add_methods':methods, 'runtime_Add_callers_gated':calls,
                  'runtime_clock_callers_gated':clock_count}


def driver_overlay():
    text = (REPO/'3rdparty/vta-hw/src/axu5evb/axu5evb_driver.cc').read_text()
    text, methods = gate_methods(text, 'DriverProfiler', 'AddRun')
    text, calls = gate_calls(text, r'DriverProfiler::Global\(\)\.AddRun\(')
    text, clocks = gate_clocks(text)
    if methods != ['AddRun'] or calls != 2 or clocks != 7: raise ValueError('driver observation census changed')
    return '#include "c3_official_window_runtime_collection.h"\n'+text, {'driver_AddRun_gated':True,'driver_AddRun_callers_gated':calls,'driver_clock_callers_gated':clocks}


def compiler_command(target):
    if target == 'sim': return 'g++', []
    return 'aarch64-xilinx-linux-g++', ['--sysroot=/home/orange/alinx_plsdk/install/sysroots/aarch64-xilinx-linux', '-march=armv8-a+crc', '-mtune=cortex-a72.cortex-a53']


def run_compiler(command, target):
    if target == 'sim': return subprocess.run(command, capture_output=True, text=True)
    shell = 'unset LD_LIBRARY_PATH; source /home/orange/alinx_plsdk/install/environment-setup-aarch64-xilinx-linux; exec '+shlex.join(command)
    return subprocess.run(['bash','-c',shell], capture_output=True, text=True)


def build_runtime(output, target):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    bound = [REPO/'vta/runtime/runtime.cc', REPO/'vta/runtime/device_api.cc',
             REPO/'3rdparty/vta-hw/src/axu5evb/axu5evb_driver.cc', HERE/'c3_banerjee_uop_queue.h',
             COLLECTION, QUEUE, Path(__file__)]
    bound += [REPO/'3rdparty/vta-hw'/name for name in ('config/vta_config.py','config/vta_config.json',
              'include/vta/hw_spec.h','include/vta/hw_spec_const.h','src/sim/sim_driver.cc',
              'src/sim/sim_tlpp.cc','src/vmem/virtual_memory.cc')]
    source_bindings = {str(path):digest(path) for path in bound}
    runtime, rt_gate = runtime_overlay()
    driver, dr_gate = driver_overlay()
    rt_source, dr_source = [output/name for name in ('c3_official_window_runtime_generated.cc','c3_official_window_runtime_driver_generated.cc')]
    rt_source.write_text(runtime)
    dr_source.write_text(driver)
    (output/COLLECTION.name).write_bytes(COLLECTION.read_bytes())
    (output/QUEUE.name).write_bytes(QUEUE.read_bytes())
    defs = shlex.split(subprocess.check_output([str(PYTHON),str(REPO/'3rdparty/vta-hw/config/vta_config.py'),'--defs'], text=True))
    cxx, options = compiler_command(target)
    includes = [REPO/path for path in ('include','3rdparty/dmlc-core/include','3rdparty/dlpack/include','3rdparty/vta-hw/include','vta/runtime')]
    shared = [REPO/'vta/runtime/device_api.cc']
    if target == 'sim':
        shared += [REPO/'3rdparty/vta-hw'/path for path in ('src/sim/sim_driver.cc','src/sim/sim_tlpp.cc','src/vmem/virtual_memory.cc')]
        defs += ['-DUSE_FSIM_TLPP']
        library = output/'libvta_fsim.so'
    else:
        shared += [dr_source]
        library = output/'libvta.so'
    command = [cxx,'-std=c++17','-O2','-shared','-fPIC','-pthread',str(rt_source),*[str(p) for p in shared],
               *['-I'+str(p) for p in includes],'-I'+str(output),*defs,*options,
               '-DDMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>','-DC3_BANERJEE_VALUE_UOPS=1','-DC3_BANERJEE_GREEDY_UOPS=1','-o',str(library)]
    result = run_compiler(command, target)
    (output/'build.log').write_text(result.stdout+result.stderr)
    for path, expected in source_bindings.items():
        if digest(path) != expected: raise ValueError('source changed during independent build: '+path)
    record = {'schema':'c3_official_window_runtime_build_v1','target':target,'returncode':result.returncode,
              'command':command,'source_bindings':source_bindings,'runtime_overlay_sha256':digest(rt_source),'driver_overlay_sha256':digest(dr_source),
              'library':str(library),'binary_sha256':digest(library) if result.returncode == 0 else None,
              'observation_gate':{**rt_gate,**dr_gate},'collection_env':'C3_OFFICIAL_WINDOW_COLLECTION',
              'collection_modes':['detailed','minimal'],'submit_ledger_limit':16,'greedy_safety_repair':'reject moved new tail intersecting any crossed GEMM/ALU UOP range',
              'execution_cache_barrier_AutoSync_changed_by_collection':False,'sealed_sources_unchanged':True,'board_contacted':False}
    write_new(output/'build.json',record)
    if result.returncode: raise RuntimeError('independent runtime build failed; complete attempt retained')
    return record


def build_helper(output, target):
    output = Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    cxx, options = compiler_command(target)
    binary = output/'c3_official_window_runtime_helper.so'
    runtime_lib = REPO/'build' if target == 'sim' else REPO/'build_axu_aarch64'
    command = [cxx,'-std=c++17','-O2','-shared','-fPIC',str(HELPER),'-I'+str(REPO/'include'),'-I'+str(REPO/'3rdparty/dlpack/include'),
               '-I'+str(REPO/'3rdparty/dmlc-core/include'),'-L'+str(runtime_lib),
               '-ltvm' if target == 'sim' else '-ltvm_runtime','-ldl','-Wl,-rpath,$ORIGIN',*options,'-o',str(binary)]
    result = run_compiler(command,target)
    (output/'build.log').write_text(result.stdout+result.stderr)
    write_new(output/'build.json',{'target':target,'returncode':result.returncode,'command':command,'source':str(HELPER),'source_sha256':digest(HELPER),
              'binary':str(binary),'binary_sha256':digest(binary) if result.returncode==0 else None,'links_VTA_runtime':False,'board_contacted':False})
    if result.returncode: raise RuntimeError('generic once helper build failed')
    return binary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('build','build-helper'))
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--target',choices=('sim','arm'),default='sim')
    args = parser.parse_args()
    if args.phase == 'build': record = build_runtime(args.output_dir,args.target)
    else: record = {'binary':str(build_helper(args.output_dir,args.target))}
    print(json.dumps(record))


if __name__ == '__main__': main()
