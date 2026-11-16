#include "frame_stream_flow.h"

namespace cpu_frame_stream {
using namespace shared_buffer;
inline int64_t NanoNow() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}
struct Timing {
  int frame{-1}; int64_t wait_start{0},dequeue{0},start{0},end{0}; StageTiming stage;
};
struct Publication {int64_t before{0},after{0};};

int Run(const Args& args,std::vector<StageExecutor>* stages) {
  const auto spec=Parse(ReadFile(args.cpu_frame_stream_spec,false));
  const auto& actors=As<Array>(Field(spec,"actors"));
  Require(Text(Field(spec,"mode"))=="bounded_common_frame_stream","wrong CPU frame-stream mode");
  Require(Number(Field(spec,"frames"))==14&&Number(Field(spec,"queue_depth"))==2,
          "CPU common-stream frame/depth differs");
  Require(actors.size()==stages->size()&&(actors.size()==2||actors.size()==3),"actor count differs");
  Require(args.process_cpu_affinity.empty()&&args.cpu_worker_pinning&&args.cpu_stage_gate=="off",
          "frame stream requires individual pinning and no CPU gate");
  for(const auto& stage:args.stages) Require(stage.device=="cpu","frame stream cannot execute VTA");
  const size_t count=actors.size();
  Flow<BoundedQueue> flow(count,2,14);
  std::vector<std::vector<NDArray>> inputs(count),last_outputs(count);
  std::vector<std::vector<Timing>> records(count,std::vector<Timing>(14));
  std::vector<std::vector<Publication>> publications(count,std::vector<Publication>(14));
  std::vector<ThreadAffinityRecord> affinities(count);
  std::mutex mutex;
  std::condition_variable ready;
  std::vector<bool> initialized(count,false);
  bool started=false,failed=false;
  std::exception_ptr failure;
  int64_t origin=0;
  std::vector<std::thread> workers;
  auto fail=[&](std::exception_ptr error) {
    {std::lock_guard<std::mutex> lock(mutex);if(!failure) failure=error;failed=true;}
    flow.Cancel();ready.notify_all();
  };
  for(size_t i=0;i<count;++i) {
    const auto& paths=As<Array>(Field(actors[i],"inputs"));
    Require(paths.size()==stages->at(i).input_names.size(),"frontier input arity differs");
    for(size_t j=0;j<paths.size();++j) {
      DLTensor* t=stages->at(i).get_input(stages->at(i).input_names[j]);
      Require(t!=nullptr,"missing CPU frontier input");
      inputs[i].push_back(MakeCPUInputLike(t,ReadFile(Text(paths[j]),true)));
    }
  }
  for(size_t index=0;index<count;++index) {
    workers.emplace_back([&,index] {
      try {
        const auto before=ProcessThreadIds();
        policy_pin_cpu=true;
        ConfigureThreadPool(args.stages[index].runtime_num_threads,StageWorkerCPUs(args.stages[index]),
                            "frame_stream.actor"+std::to_string(index));
        const auto after=ProcessThreadIds();auto& affinity=affinities[index];
        std::set_difference(after.begin(),after.end(),before.begin(),before.end(),
                            std::back_inserter(affinity.thread_ids));
        affinity.caller_tid=static_cast<long>(syscall(SYS_gettid));
        affinity.thread_ids.push_back(affinity.caller_tid);
        std::sort(affinity.thread_ids.begin(),affinity.thread_ids.end());
        const PackedFunc* threads=Registry::Get("runtime.NumThreads");
        affinity.active_runtime_threads=threads?(*threads)().operator int():0;
        ProbePolicySlots(&affinity,args.stages[index].runtime_num_threads);
        CaptureThreadAffinities(&affinity);
        StageTiming qualification;
        auto outputs=RunStage(&stages->at(index),inputs[index],&qualification,false);
        controlled_contention::CheckOutputs(outputs,Field(actors[index],"references"));
        {
          std::unique_lock<std::mutex> lock(mutex); initialized[index]=true;ready.notify_all();
          ready.wait(lock,[&]{return started||failed;});
          if(failed) return;
        }
        flow.Consume(index,NanoNow,[&](int frame,int64_t before,int64_t dequeue) {
          Require(frame>=0&&frame<14&&records[index][frame].frame==-1,"duplicate CPU frame");
          auto& record=records[index][frame];record.frame=frame;
          record.wait_start=before;record.dequeue=dequeue;record.start=NanoNow();
          last_outputs[index]=RunStage(&stages->at(index),inputs[index],&record.stage,false);
          record.end=NanoNow();
        });
        for(int frame=0;frame<14;++frame) Require(records[index][frame].frame==frame,"missing CPU frame");
        controlled_contention::CheckOutputs(last_outputs[index],Field(actors[index],"references"));
        FinishThreadActivity(&affinity);
      } catch(...) {fail(std::current_exception());}
      {std::lock_guard<std::mutex> lock(mutex);initialized[index]=true;}
      ready.notify_all();
    });
    std::unique_lock<std::mutex> lock(mutex);
    ready.wait(lock,[&]{return initialized[index]||failed;});
    if(failed) break;
  }
  try {
    std::lock_guard<std::mutex> lock(mutex);
    if(!failed) {
      for(auto& record:affinities) CaptureThreadAffinities(&record);
      CapturePolicyControl(0,args.control_cpu_affinity);origin=NanoNow();started=true;
    }
  } catch(...) {fail(std::current_exception());}
  ready.notify_all();
  std::thread producer([&] {
    try {
      CapturePolicyControl(1,args.control_cpu_affinity);
      {std::lock_guard<std::mutex> lock(mutex);if(failed)return;}
      flow.Produce(NanoNow,[&](size_t actor,int frame,int64_t before,int64_t after) {
        publications[actor][frame]={before,after};
      });
    } catch(...) {fail(std::current_exception());}
  });
  producer.join();for(auto& worker:workers) worker.join();
  if(failure) std::rethrow_exception(failure);
  WriteThreadAffinityReport(args,affinities);
  std::ofstream out(args.output_jsonl);out<<std::fixed<<std::setprecision(9);
  for(size_t i=0;i<count;++i) for(const auto& r:records[i]) {
    const auto& p=publications[i][r.frame];
    out<<"{\"schema\":\"cpu_frame_stream_v1\",\"actor\":"<<i<<",\"frame_id\":"<<r.frame
       <<",\"input_index\":0,\"program\":\""<<JsonEscape(args.stages[i].name)<<"\""
       <<",\"wait_start_ns\":"<<r.wait_start-origin<<",\"dequeue_ns\":"<<r.dequeue-origin
       <<",\"start_ns\":"<<r.start-origin<<",\"end_ns\":"<<r.end-origin
       <<",\"push_start_ns\":"<<p.before-origin<<",\"push_return_ns\":"<<p.after-origin
       <<",\"set_ms\":"<<r.stage.SetMs()<<",\"run_ms\":"<<r.stage.RunMs()
       <<",\"get_ms\":"<<r.stage.GetMs()<<",\"service_ms\":"<<r.stage.end_ms-r.stage.set_start_ms
       <<",\"qualification_passed\":true,\"final_output_passed\":true,\"calls\":14,\"extra_calls\":0}\n";
  }
  Require(out.good(),"CPU frame-stream output write failed");
  return 0;
}
}
