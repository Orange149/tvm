"""Generate an independent original-wait observer from the authenticated source."""
from pathlib import Path
from cpu_vta_partitioning.common import sha256
from .common import ROOT


def once(text,a,b):
    if text.count(a)!=1:raise ValueError('scheduler observer patch anchor changed: '+a[:100])
    return text.replace(a,b,1)


def adapter_source():
    path=ROOT/'wait_diagnosis/adapter.cc';original=path.read_text()
    # The archived capsule owns the authoritative old adapter source bytes.
    archive=ROOT/'artifacts/yolo_wait_diagnosis_01/capsule/sources/wait_diagnosis/adapter.cc'
    if not archive.exists() or sha256(path)!=sha256(archive):raise ValueError('original observer source identity changed')
    source=once(original,'#include <atomic>','#include <atomic>\n#include "taskstats_client.h"')
    source=once(source,'  int kind,task,tasks,result,module_index;','  int kind,task,tasks,result,module_index;\n'
        '  uint64_t q0=0,q1=0,qws0=0,qwe0=0,qws1=0,qwe1=0,qcpu0=0,qcpu1=0,tail_queue=0;\n'
        '  uint64_t baseline_tid[4]{},baseline_q[4]{},baseline_start[4]{},baseline_end[4]{};')
    helper='''
bool queue_enabled=true;
std::atomic<uint64_t> queue_errors{0},queue_reads{0};
bool queue_snapshot(uint64_t* out) noexcept {
  if(!queue_enabled){for(int i=0;i<5;++i)out[i]=0;return true;}
  try {
    static thread_local scheduler_gap::Client client;
    out[1]=clock_ns(CLOCK_MONOTONIC);const auto cpu=clock_ns(CLOCK_THREAD_CPUTIME_ID);
    const auto value=client.read(static_cast<uint32_t>(tid()));out[0]=value.delay;
    out[4]=value.runtime;out[3]=clock_ns(CLOCK_THREAD_CPUTIME_ID)-cpu;
    out[2]=clock_ns(CLOCK_MONOTONIC);queue_reads.fetch_add(1);return true;
  }catch(...){queue_errors.fetch_add(1);for(int i=0;i<5;++i)out[i]=0;return false;}
}
void queue_start(Record& r) {uint64_t q[5]{};queue_snapshot(q);r.q0=q[0];r.qws0=q[1];r.qwe0=q[2];r.qcpu0=q[3];}
void queue_end(Record& r) {uint64_t q[5]{};queue_snapshot(q);r.q1=q[0];r.qws1=q[1];r.qwe1=q[2];r.qcpu1=q[3];if(r.q1<r.q0)queue_errors.fetch_add(1);}
thread_local uint64_t known_worker_tids[4]{};
void queue_baselines(Record& r) {
  if(!queue_enabled)return;
  try {
    static thread_local scheduler_gap::Client client;
    for(int i=0;i<4;++i)if(known_worker_tids[i]) {
      r.baseline_tid[i]=known_worker_tids[i];r.baseline_start[i]=clock_ns(CLOCK_MONOTONIC);
      r.baseline_q[i]=client.read(known_worker_tids[i]).delay;
      r.baseline_end[i]=clock_ns(CLOCK_MONOTONIC);queue_reads.fetch_add(1);
    }
  }catch(...){queue_errors.fetch_add(1);}
}
'''
    source=once(source,'void quote(FILE* f,const char* s) {',helper+'\nvoid quote(FILE* f,const char* s) {')
    source=once(source,'  initialized=true;atexit(flush);','  queue_enabled=!(getenv("VTA_QUEUE_COUNTERS") && !strcmp(getenv("VTA_QUEUE_COUNTERS"),"0"));\n  initialized=true;atexit(flush);')
    source=once(source,'  bool caller_done=false;uint64_t tail_wall=0,tail_cpu=0;',
        '  bool caller_done=false;uint64_t tail_wall=0,tail_cpu=0,tail_queue=0;\n  uint64_t worker_tids[4]{};')
    source=once(source,'  r.module_index=c.module_index;\n  r.callback=reinterpret_cast<uintptr_t>(c.function);',
        '  r.module_index=c.module_index;\n  c.worker_tids[task]=r.tid;\n  r.callback=reinterpret_cast<uintptr_t>(c.function);')
    # Three scope entries (callback, launch, barrier), preserving clocks around
    # the original function body. Counter query envelopes are separate fields.
    for prefix in ('  try {r.result=c.function(task,env,c.data);}',
                   '  r.result=original_launch(call_task,&c,tasks);',
                   '  local.barrier=true;'):
        source=once(source,prefix,'  if(probe)queue_start(r);\n'+prefix)
    source=once(source,'  if(probe)queue_start(r);\n  r.result=original_launch(call_task,&c,tasks);',
        '  if(probe){queue_baselines(r);queue_start(r);}\n  r.result=original_launch(call_task,&c,tasks);\n'
        '  if(probe)for(int i=0;i<4;++i)if(c.worker_tids[i])known_worker_tids[i]=c.worker_tids[i];')
    marker='if(probe) {r.ce=clock_ns(CLOCK_THREAD_CPUTIME_ID);r.we=clock_ns(CLOCK_MONOTONIC);}'
    source=once(source,marker,marker+'\n  if(probe)queue_end(r);')
    marker='if(probe){r.ce=clock_ns(CLOCK_THREAD_CPUTIME_ID);r.we=clock_ns(CLOCK_MONOTONIC);}'
    if source.count(marker)!=2:raise ValueError('launch/barrier end anchors changed')
    source=source.replace(marker,marker+'\n  if(probe)queue_end(r);')
    source=once(source,'c.tail_wall=r.we;c.tail_cpu=r.ce;','c.tail_wall=r.we;c.tail_cpu=r.ce;c.tail_queue=r.q1;')
    source=once(source,'  r.last_task_end=c.last_end.load();','  r.last_task_end=c.last_end.load();r.tail_queue=c.tail_queue;')
    marker='        fprintf(f,",\\\"function_offset\\\":%llu}\\n",(unsigned long long)(found?modules[r.module_index].offset:0));'
    replacement='''        fprintf(f,",\\\"queue_before_ns\\\":%llu,\\\"queue_after_ns\\\":%llu,\\\"queue_read_start0_ns\\\":%llu,\\\"queue_read_end0_ns\\\":%llu,\\\"queue_read_start1_ns\\\":%llu,\\\"queue_read_end1_ns\\\":%llu,\\\"queue_read_cpu0_ns\\\":%llu,\\\"queue_read_cpu1_ns\\\":%llu,\\\"caller_tail_queue_start_ns\\\":%llu,\\\"function_offset\\\":%llu",
          (unsigned long long)r.q0,(unsigned long long)r.q1,(unsigned long long)r.qws0,(unsigned long long)r.qwe0,
          (unsigned long long)r.qws1,(unsigned long long)r.qwe1,(unsigned long long)r.qcpu0,(unsigned long long)r.qcpu1,
          (unsigned long long)r.tail_queue,(unsigned long long)(found?modules[r.module_index].offset:0));
        fprintf(f,",\\\"worker_queue_baselines\\\":[");
        for(int j=0;j<4;++j){if(j)fputc(',',f);fprintf(f,"{\\\"tid\\\":%llu,\\\"queue_ns\\\":%llu,\\\"read_start_ns\\\":%llu,\\\"read_end_ns\\\":%llu}",
          (unsigned long long)r.baseline_tid[j],(unsigned long long)r.baseline_q[j],(unsigned long long)r.baseline_start[j],(unsigned long long)r.baseline_end[j]);}
        fprintf(f,"]}\\n");'''
    source=once(source,marker,replacement)
    source=once(source,'  FILE* f=status_path?fopen(status_path,"w"):nullptr;', '''  const char* qp=getenv("VTA_QUEUE_STATUS");
  if(qp){FILE* q=fopen(qp,"w");if(q){fprintf(q,"{\\\"enabled\\\":%s,\\\"errors\\\":%llu,\\\"reads\\\":%llu}\\n",queue_enabled?"true":"false",(unsigned long long)queue_errors.load(),(unsigned long long)queue_reads.load());fclose(q);}}
  FILE* f=status_path?fopen(status_path,"w"):nullptr;''')
    source+='\nextern "C" int SchedulerGapSnapshot(uint64_t* out) {pthread_once(&once,initialize);return queue_snapshot(out)?0:-1;}\n'
    source=source.replace('if(probe)queue_start(r);',
        'if(probe){queue_start(r);r.ws=clock_ns(CLOCK_MONOTONIC);r.cs=clock_ns(CLOCK_THREAD_CPUTIME_ID);}')
    source=source.replace('if(probe){queue_baselines(r);queue_start(r);}',
        'if(probe){queue_baselines(r);queue_start(r);r.ws=clock_ns(CLOCK_MONOTONIC);r.cs=clock_ns(CLOCK_THREAD_CPUTIME_ID);}')
    return source


def phase_header():
    source=(ROOT/'phase_profile/phase_probe.h').read_text()
    source=once(source,'namespace vta_workload_phase {','extern "C" int SchedulerGapSnapshot(unsigned long long*) __attribute__((weak));\nnamespace vta_workload_phase {')
    source=once(source,'  int core_start=-1, core_end=-1;','  int core_start=-1, core_end=-1;\n  unsigned long long q0[5]{},q1[5]{};')
    source=once(source,'    event_.wall_start=Clock(CLOCK_MONOTONIC);',
        '    if(!SchedulerGapSnapshot || SchedulerGapSnapshot(event_.q0))throw std::runtime_error("phase queue counter unavailable");\n    event_.wall_start=Clock(CLOCK_MONOTONIC);')
    source=once(source,'    active_=false;State().Add(std::move(event_));',
        '    if(SchedulerGapSnapshot(event_.q1))throw std::runtime_error("phase queue end failed");\n    active_=false;State().Add(std::move(event_));')
    source=once(source,'    for (const auto& e:events_) {','    for (const auto& e:events_) {')
    source=once(source,'          "\\\"core_start\\\":%d,\\\"core_end\\\":%d}\\n",',
        '          "\\\"core_start\\\":%d,\\\"core_end\\\":%d,\\\"queue_before_ns\\\":%llu,\\\"queue_after_ns\\\":%llu,\\\"read_start0_ns\\\":%llu,\\\"read_end0_ns\\\":%llu,\\\"read_start1_ns\\\":%llu,\\\"read_end1_ns\\\":%llu}\\n",')
    source=once(source,'          e.wall_start,e.wall_end,e.cpu_start,e.cpu_end,e.core_start,e.core_end);',
        '          e.wall_start,e.wall_end,e.cpu_start,e.cpu_end,e.core_start,e.core_end,e.q0[0],e.q1[0],e.q0[1],e.q0[2],e.q1[1],e.q1[2]);')
    # AArch64 uint64_t is unsigned long while the phase clock uses unsigned
    # long long. The C ABI uses a pointer to five 64-bit words in either case.
    return source


def control_patch(source):
    """Observe existing producer/collector work; no new actor or queue change."""
    start=source.index('      std::thread producer_thread([&]() {')
    a,b=source[:start],source[start:]
    b=once(b,'          for (int i = 0; i < args.runs; ++i) {',
        '          for (int i = 0; i < args.runs; ++i) {\n'
        '            vta_workload_phase::Span producer_phase("_producer", i, "feed");')
    b=once(b,'            if (!queues[0]->Push(frame, frame->QueueTimingFor(0))) {',
        '            const bool pushed = queues[0]->Push(frame, frame->QueueTimingFor(0));\n'
        '            producer_phase.End();\n            if (!pushed) {')
    b=once(b,'        write_completed(*frame, "pipeline");',
        '        vta_workload_phase::Span collector_phase("_collector", frame->frame_id, "collect");\n'
        '        write_completed(*frame, "pipeline");\n        collector_phase.End();')
    return a+b
