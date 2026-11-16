// Included in the frozen runner translation unit; no TVM/runtime changes.
#include <atomic>
#include <cmath>
#include "turn_gate.h"

namespace controlled_contention {
using namespace shared_buffer;

inline void CheckOutputs(const std::vector<NDArray>& outputs, const JSON& paths) {
  const auto& names = As<Array>(paths);
  Require(outputs.size()==names.size(), "reference output arity differs");
  for (size_t i=0;i<outputs.size();++i) {
    const auto raw = ReadFile(Text(names[i]), true);
    const auto* tensor = outputs[i].operator->();
    Require(raw.size()==TensorNBytes(tensor), "reference byte size differs");
    const auto* data = TensorDataBytes(tensor);
    if (tensor->dtype.code==kDLFloat && tensor->dtype.bits==32) {
      for (size_t j=0;j<raw.size()/4;++j) {
        float a,b; std::memcpy(&a,data+4*j,4); std::memcpy(&b,raw.data()+4*j,4);
        Require(std::isfinite(a)&&std::isfinite(b)&&std::abs(a-b)<=1e-5f+1e-4f*std::abs(b),
                "floating output does not match common frontier reference");
      }
    } else Require(std::memcmp(data,raw.data(),raw.size())==0, "integer output differs");
  }
}

struct Observation { StageTiming timing; int iteration; };

int Run(const Args& args, std::vector<StageExecutor>* stages) {
  const JSON spec = Parse(ReadFile(args.contention_spec,false));
  const auto& actors = As<Array>(Field(spec,"actors"));
  Require(actors.size()==stages->size()&&!actors.empty(), "actor count differs");
  const int scored=Number(Field(spec,"scored")), warmup=Number(Field(spec,"warmup"));
  Require(scored==8&&warmup==2, "component protocol differs");
  const bool diagnostics = Number(Field(spec,"probe"))==1;
  std::vector<size_t> targets, vtas;
  for (const auto& n:As<Array>(Field(spec,"targets"))) targets.push_back(Number(n));
  Require(!targets.empty(), "no CPU targets");
  for (size_t i=0;i<actors.size();++i) if (args.stages[i].device=="vta") vtas.push_back(i);
  const bool isolated=actors.size()==1;
  std::vector<std::vector<NDArray>> inputs(actors.size()), last_outputs(actors.size());
  for (size_t i=0;i<actors.size();++i) {
    const auto& paths=As<Array>(Field(actors[i],"inputs"));
    Require(paths.size()==stages->at(i).input_names.size(), "frontier input arity differs");
    for (size_t j=0;j<paths.size();++j) {
      DLTensor* tensor=stages->at(i).get_input(stages->at(i).input_names[j]);
      Require(tensor!=nullptr, "missing stage input");
      inputs[i].push_back(MakeCPUInputLike(tensor, ReadFile(Text(paths[j]),true)));
    }
  }
  std::mutex mutex;
  std::condition_variable cv;
  std::atomic<bool> stop{false};
  bool start=false;
  std::vector<bool> initialized(actors.size(),false), done(actors.size(),false);
  std::exception_ptr failure;
  TurnGate vta_gate(vtas);
  double origin=0;
  std::vector<ThreadAffinityRecord> affinities(actors.size());
  std::vector<vta_run_scope::StageProbe> probes(actors.size());
  std::vector<std::vector<Observation>> observations(actors.size());
  std::vector<int> calls(actors.size(),0);
  std::vector<double> active_start(actors.size(),0),active_end(actors.size(),0);
  std::vector<std::thread> threads;
  auto fail=[&](std::exception_ptr error) {
    {std::lock_guard<std::mutex> lock(mutex); if (!failure) failure=error; stop=true;}
    cv.notify_all(); vta_gate.NotifyStop();
  };
  for (size_t index=0;index<actors.size();++index) {
    threads.emplace_back([&,index] {
      try {
        const auto before=ProcessThreadIds();
        policy_pin_cpu=args.stages[index].device=="cpu";
        ConfigureThreadPool(args.stages[index].runtime_num_threads,StageWorkerCPUs(args.stages[index]),
                            "contention.actor"+std::to_string(index));
        auto& affinity=affinities[index];
        const auto after=ProcessThreadIds();
        std::set_difference(after.begin(),after.end(),before.begin(),before.end(),
                            std::back_inserter(affinity.thread_ids));
        affinity.caller_tid=static_cast<long>(syscall(SYS_gettid));
        affinity.thread_ids.push_back(affinity.caller_tid);
        std::sort(affinity.thread_ids.begin(),affinity.thread_ids.end());
        const PackedFunc* count=Registry::Get("runtime.NumThreads");
        affinity.active_runtime_threads=count?(*count)().operator int():0;
        if (args.stages[index].device=="cpu") ProbePolicySlots(&affinity,args.stages[index].runtime_num_threads);
        CaptureThreadAffinities(&affinity);
        StageTiming qualification;
        auto q=RunStage(&stages->at(index),inputs[index],&qualification,false);
        CheckOutputs(q,Field(actors[index],"references"));
        for (int n=0;n<warmup;++n) { StageTiming t; RunStage(&stages->at(index),inputs[index],&t,false); }
        if (diagnostics&&args.stages[index].device=="cpu")
          probes[index].Initialize(index,args.stages[index].name,StageWorkerCPUs(args.stages[index]),
              affinity.thread_ids,affinity.caller_tid,{2,6},true);
        {
          std::unique_lock<std::mutex> lock(mutex);
          initialized[index]=true; cv.notify_all();
          cv.wait(lock,[&]{return start||stop.load();});
        }
        if (!stop) {
          active_start[index]=NowMillis();
          int iteration=0;
          while (!stop.load()) {
            StageTiming timing;
            std::vector<NDArray> output;
            auto run=[&] { output=RunStage(&stages->at(index),inputs[index],&timing,isolated,nullptr,false,
                                          diagnostics?&probes[index]:nullptr,iteration); };
            if (args.stages[index].device=="vta"&&!isolated) {
              if (!vta_gate.Execute(index,stop,run)) break;
            } else run();
            if (iteration<scored) observations[index].push_back({timing,iteration});
            last_outputs[index]=std::move(output);
            ++iteration; calls[index]=iteration;
            if (iteration==scored&&std::find(targets.begin(),targets.end(),index)!=targets.end()) {
              std::lock_guard<std::mutex> lock(mutex); done[index]=true;
              if (std::all_of(targets.begin(),targets.end(),[&](size_t n){return done[n];})) {
                stop=true; cv.notify_all(); vta_gate.NotifyStop();
              }
            }
            if (iteration>=100000) throw std::runtime_error("background safety call limit reached");
          }
          active_end[index]=NowMillis();
        }
        FinishThreadActivity(&affinity);
      } catch (...) {fail(std::current_exception());}
      {std::lock_guard<std::mutex> lock(mutex); initialized[index]=true;}
      cv.notify_all();
    });
    std::unique_lock<std::mutex> lock(mutex);
    cv.wait(lock,[&]{return initialized[index]||stop.load();});
    if (stop) break;
  }
  try {
    std::lock_guard<std::mutex> lock(mutex);
    if (!stop) {
      for (auto& affinity:affinities) CaptureThreadAffinities(&affinity);
      origin=NowMillis(); start=true;
    }
  } catch (...) { fail(std::current_exception()); }
  cv.notify_all();
  for (auto& thread:threads) thread.join();
  if (failure) std::rethrow_exception(failure);
  WriteThreadAffinityReport(args,affinities);
  if (diagnostics) vta_run_scope::WriteTrace(args.run_thread_counters_json,origin,{2,6},probes);
  std::ofstream out(args.output_jsonl);
  out << std::fixed << std::setprecision(9);
  auto core_json=[&](double value) {
    if (!isolated) return std::string("null");
    std::ostringstream text; text << std::fixed << std::setprecision(9) << value; return text.str();
  };
  for (size_t i=0;i<actors.size();++i) {
    CheckOutputs(last_outputs[i],Field(actors[i],"references"));
    for (const auto& observation:observations[i]) {
      const auto& t=observation.timing;
      out << "{\"schema\":\"controlled_contention_v1\",\"actor\":" << i
          << ",\"iteration\":" << observation.iteration << ",\"program\":\"" << JsonEscape(args.stages[i].name)
          << "\",\"start_ms\":" << t.set_start_ms-origin << ",\"end_ms\":" << t.end_ms-origin
          << ",\"set_ms\":" << t.SetMs() << ",\"run_ms\":" << t.RunMs()
          << ",\"get_ms\":" << t.GetMs() << ",\"service_ms\":" << t.end_ms-t.set_start_ms
          << ",\"calls\":" << calls[i] << ",\"extra_calls\":" << std::max(0,calls[i]-scored)
          << ",\"active_start_ms\":" << active_start[i]-origin << ",\"active_end_ms\":" << active_end[i]-origin
          << ",\"qualification_passed\":true,\"final_output_passed\":true"
          << ",\"process_cpu_valid\":" << (isolated?"true":"false")
          << ",\"set_core_ms\":" << core_json(t.process_cpu_after_set_ms-t.process_cpu_start_ms)
          << ",\"run_core_ms\":" << core_json(t.process_cpu_after_run_ms-t.process_cpu_after_set_ms)
          << ",\"get_core_ms\":" << core_json(t.process_cpu_end_ms-t.process_cpu_after_run_ms) << "}\n";
    }
  }
  Require(out.good(),"component result write failed");
  return 0;
}
} // namespace controlled_contention
