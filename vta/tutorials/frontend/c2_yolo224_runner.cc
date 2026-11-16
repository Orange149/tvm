// One native executable for all YOLO C2 transport and lease experiments.
// Reuse the existing allocator, ABI utilities, lease state machine and controls.
#define main C2_UNUSED_HISTORICAL_MAIN
#include "c3_input_early_release_runner.cc"
#undef main

namespace c2_yolo {
using namespace c3_last_read;

struct Options {
  std::string plan, output, mode, control;
  int frames{64}, warmup{0}, settle{0};
  bool serial{false}, detailed{false}, proxy_off{false};
};
struct Record {
  int id{0}, input{0}; bool measured{true}, complete{false};
  std::array<P8SlotToken, 2> tokens{};
  std::array<double, 3> begin{}, end{}, set_ms{}, get_ms{}, waits{};
  std::array<double,2> handoff_ms{};
  std::array<uint64_t, 2> copied{};
  std::vector<NDArray> outputs;
  BufferRange reader_input, reader_output;
  uint64_t checked_arguments{0}; bool reader_done{false}, released{false};
  double completion{0};
};

class Bundle {
 public:
  LeaseBook book;
  bool copy;
  std::vector<std::vector<P8SharedEdge>> shared;
  std::vector<std::vector<NDArray>> cpu;
  std::vector<uint64_t> generation, bound, written;
  size_t tensors;
  Bundle(std::vector<StageExecutor>* stages, int edge, int count, bool copied,
         std::vector<P8SharedEdge> anchors = {})
      : book(count, edge), copy(copied), generation(count), bound(count), written(count),
        tensors(stages->at(edge).NumOutputs()) {
    Require(tensors > 0 && tensors < 64 && tensors == stages->at(edge+1).input_names.size(), "bundle arity mismatch");
    for (int slot=0; slot<count; ++slot) {
      if (copy) {
        std::vector<NDArray> arrays;
        for (size_t t=0;t<tensors;++t) {
          NDArray model=stages->at(edge).get_output(static_cast<int>(t));
          arrays.push_back(NDArray::Empty(model.Shape(),model->dtype,tvm::Device{kDLCPU,0}));
        }
        cpu.push_back(std::move(arrays));
      } else shared.push_back(slot==0 && !anchors.empty() ? std::move(anchors) : AllocateP8HeterogeneousEdgeBundle(stages,edge));
    }
  }
  uint64_t Full() const { return (uint64_t(1)<<tensors)-1; }
  P8SlotToken Acquire(int frame) {
    auto token=book.Acquire(frame); generation[token.slot_id]=token.generation;
    bound[token.slot_id]=written[token.slot_id]=0; return token;
  }
  void Bound(const P8SlotToken& token,size_t t) {
    book.Validate(token,LeaseBook::State::Writing);
    Require(t<tensors && generation[token.slot_id]==token.generation, "invalid bundle binding");
    bound[token.slot_id]|=uint64_t(1)<<t;
  }
  void Written(const P8SlotToken& token,size_t t) {
    book.Validate(token,LeaseBook::State::Writing);
    Require(t<tensors && (bound[token.slot_id]&(uint64_t(1)<<t)), "unbound bundle member");
    written[token.slot_id]|=uint64_t(1)<<t;
  }
  void Publish(const P8SlotToken& token) {
    book.Validate(token,LeaseBook::State::Writing);
    Require(generation[token.slot_id]==token.generation && bound[token.slot_id]==Full() &&
            written[token.slot_id]==Full(), "incomplete same-generation bundle cannot publish");
    book.Publish(token);
  }
  const NDArray& Producer(const P8SlotToken& token,size_t t) {
    book.Validate(token,LeaseBook::State::Writing);
    return copy ? cpu.at(token.slot_id).at(t) : shared.at(token.slot_id).at(t).producer_owner;
  }
  const NDArray& Consumer(const P8SlotToken& token,size_t t) {
    book.Validate(token,LeaseBook::State::Using);
    return copy ? cpu.at(token.slot_id).at(t) : shared.at(token.slot_id).at(t).consumer_owner;
  }
  const NDArray& CPU(const P8SlotToken& token,size_t t) {
    book.Validate(token,LeaseBook::State::Using);
    return copy ? cpu.at(token.slot_id).at(t) : shared.at(token.slot_id).at(t).cpu_owner;
  }
};

// Only the reviewed synchronous reader is wrapped during formal timing.
// Qualification additionally checks every subsequent actual DLTensor ABI range.
class ReaderModule final : public tvm::runtime::ModuleNode {
 public:
  Module inner; std::string reader; bool detailed{false};
  Record* active{nullptr}; void* expected{nullptr};
  std::function<void(Record*)> before, after;
  const char* type_key() const final { return "c2_yolo_reader_proxy"; }
  PackedFunc GetFunction(const tvm::runtime::String& name,const tvm::runtime::ObjectPtr<tvm::runtime::Object>& self) final {
    auto original=inner.GetFunction(name,true);
    if (!original.defined() || (!detailed && std::string(name)!=reader)) return original;
    bool target=std::string(name)==reader;
    return PackedFunc([this,self,original,target](tvm::runtime::TVMArgs args,tvm::runtime::TVMRetValue* rv) {
      Require(active!=nullptr,"unarmed reader invocation");
      if (target) {
        Require(!active->reader_done && args.size()==2,"last reader ABI/call count changed");
        auto* input=args[0].operator DLTensor*();
        Require(input && input->data==expected,"reader did not receive this generation's input");
        active->reader_input=ResolveRange(input); active->reader_output=ResolveRange(args[1].operator DLTensor*());
        Require(Separate(active->reader_input,active->reader_output),"reader output aliases original input");
        if(before) before(active);
        original.CallPacked(args,rv);
        active->reader_done=true;
        if(after) after(active);
      } else {
        Require(active->reader_done,"unexpected computation before last-reader node");
        for(int i=0;i<args.size();++i) {
          auto range=ResolveRange(args[i].operator DLTensor*());
          Require(Separate(active->reader_input,range),"later node ABI still accesses original input range");
          ++active->checked_arguments;
        }
        original.CallPacked(args,rv);
      }
    });
  }
};

class Experiment {
 public:
  Options options; JSON plan; Args args;
  std::vector<StageExecutor> stages;
  tvm::runtime::ObjectPtr<ReaderModule> proxy;
  std::array<std::unique_ptr<Bundle>,2> bundles;
  std::array<std::unique_ptr<BoundedQueue<Record*>>,2> queues;
  std::vector<NDArray> inputs; NDArray private_input;
  std::vector<Record> frames; std::vector<Event> events; std::atomic<size_t> event_count{0};
  std::vector<JSON> snapshots;
  std::array<uint64_t,3> calls{}; std::array<int,3> thread_counts{};
  std::array<ThreadAffinityRecord,3> affinities;
  std::mutex mutex,device_mutex; std::condition_variable condition;
  bool start{false},finish{false},aborted{false},producer_blocked{false},next_published{false},full_blocked{false};
  int init_turn{0},completed{0}; std::exception_ptr failure;
  bool ordinary{false},early_group{false};
  JSON shared_plan; std::vector<bool> anchors;
  explicit Experiment(Options value):options(std::move(value)) {}
  void Emit(const char* name,int stage,int frame,const P8SlotToken* token=nullptr) {
    auto i=event_count.fetch_add(1); Require(i<events.size(),"event buffer exhausted");
    auto& e=events[i]; e.name=name;e.stage=stage;e.frame=frame;e.time=NowMillis();
    if(token){e.edge=token->edge_index;e.slot=token->slot_id;e.generation=token->generation;}
  }
  void Wait(const std::function<bool()>& predicate,const char* reason,int seconds=60) {
    std::unique_lock<std::mutex> lock(mutex);
    Require(condition.wait_for(lock,std::chrono::seconds(seconds),[&]{return aborted||predicate();}),reason);
    Require(!aborted,"experiment aborted");
  }
  void Fail(std::exception_ptr error) {
    {std::lock_guard<std::mutex> lock(mutex);if(!failure)failure=error;aborted=true;condition.notify_all();}
    for(auto& b:bundles)if(b)b->book.Abort();
    for(auto& q:queues)if(q)q->Close();
  }
  void Snapshot(const char* phase) {
    auto* function=Registry::Get("vta.runtime.udmabuf_allocation_snapshot");
    Require(function!=nullptr,"allocation snapshot ABI missing");
    JSON value=shared_buffer::Parse((*function)().operator std::string());
    Object object=As<Object>(value);object["phase"]=JSON(phase); snapshots.emplace_back(object);
  }
  void Prepare() {
    plan=shared_buffer::Parse(ReadFile(options.plan));
    Require(S(plan,"schema")=="c2-yolo224-program-v1" ||
            S(plan,"schema")=="c2-yolo416-program-v1","unsupported program contract");
    Require(TrimLine(ReadFile("/proc/sys/kernel/random/boot_id"))==S(plan,"boot_id"),"boot changed");
    for(auto item:std::vector<std::pair<const char*,const char*>>{{"TVMFuncGetGlobal","runtime_sha256"},{"VTAMemAlloc","driver_sha256"}})
      shared_buffer::CheckHash(LoadedObjectPath(item.first),Field(plan,item.second));
    shared_buffer::CheckHash(ExecutablePath(),Field(plan,"runner_sha256"));
    shared_buffer::CheckHash(S(plan,"bitstream"),Field(plan,"bitstream_sha256"));
    ordinary=options.mode=="M0";
    early_group=options.mode=="L1"||options.mode=="L2"||options.mode=="CP"||options.mode=="E1";
    events.resize((options.frames+options.warmup+options.settle)*64+256);
    const auto& specifications=As<Array>(Field(plan,"stages"));
    Require(specifications.size()==3,"three stages required");
    for(size_t i=0;i<3;++i) {
      auto& spec=specifications[i]; StageArgs a;
      a.input_names.clear();
      a.name=S(spec,"name");a.device=S(spec,"device");
      a.graph=S(spec,"graph");a.lib=S(spec,"lib");a.params=S(spec,"params");
      a.runtime_num_threads=Number(Field(spec,"runtime_num_threads"));a.cpu_affinity={0,1,2,3};
      for(auto& name:As<Array>(Field(spec,"input_names")))a.input_names.push_back(Text(name));
      for(auto key:{"graph","lib","params"})shared_buffer::CheckHash(S(spec,key),Field(spec,std::string(key)+"_sha256"));
      args.stages.push_back(a);
    }
    args.p8_managed_slots=2;
    args.shared_buffer_plan=S(plan,"shared_buffer_plan");
    args.bitstream_artifact=S(plan,"bitstream");
    if(options.mode=="M3") {
      args.command_capacity_certificate=S(plan,"capacity_certificate");
      (void)ValidateCommandCapacityCertificate(args);
    }
    shared_plan=ValidateSharedBufferPlan(args);
    Snapshot("before_graph_load");
    stages.reserve(3);
    for(int i=0;i<3;++i) {
      const auto& a=args.stages[i];
      if(i!=1 || !early_group || options.proxy_off) {stages.push_back(LoadStage(a.name,a.device,a));continue;}
      auto& proof=Field(plan,"last_reader");
      Require(S(proof,"status")=="qualified","last-reader certificate not qualified");
      shared_buffer::CheckHash(S(proof,"certificate"),Field(proof,"certificate_sha256"));
      auto graph_json=shared_buffer::Parse(ReadFile(a.graph));
      auto input_node=Number(Field(proof,"input_node"));auto reader_node=Number(Field(proof,"node"));
      const auto& nodes=As<Array>(Field(graph_json,"nodes"));
      Require(input_node<nodes.size() && reader_node<nodes.size(),"last reader node out of range");
      auto input_entry=Number(As<Array>(Field(graph_json,"node_row_ptr"))[input_node]);
      auto sid=Number(As<Array>(As<Array>(Field(Field(graph_json,"attrs"),"storage_id"))[1])[input_entry]);
      shared_buffer::CheckExclusiveInput(graph_json,a.input_names[0],input_entry,sid,
                                         shared_buffer::ParameterNames(ReadFile(a.params,true)));
      size_t uses=0;
      for(size_t n=0;n<nodes.size();++n)for(auto& ref:As<Array>(Field(nodes[n],"inputs"))) {
        auto& r=As<Array>(ref);
        if(Number(r[0])==input_node) {++uses;Require(n==reader_node && Number(r[1])==0,"other raw input reader");}
      }
      Require(uses==1,"expected one last direct reader");
      proxy=tvm::runtime::make_object<ReaderModule>();proxy->inner=Module::LoadFromFile(a.lib);
      proxy->reader=S(proof,"function");proxy->detailed=options.detailed;
      Require(S(Field(nodes[reader_node],"attrs"),"func_name")==proxy->reader,"reader symbol changed");
      auto create=Registry::Get("tvm.graph_executor.create");
      Module graph=(*create)(ReadFile(a.graph),Module(proxy),static_cast<int>(kDLExtDev),0,static_cast<int>(kDLCPU),0);
      LoadParamsBlob(graph.GetFunction("load_params"),ReadFile(a.params,true));
      StageExecutor stage;stage.name=a.name;stage.device=a.device;stage.input_names=a.input_names;stage.graph=graph;
      stage.set_input=graph.GetFunction("set_input");stage.set_input_zero_copy=graph.GetFunction("set_input_zero_copy");
      stage.set_output_zero_copy=graph.GetFunction("set_output_zero_copy");stage.run=graph.GetFunction("run");
      stage.get_input=graph.GetFunction("get_input");stage.get_output=graph.GetFunction("get_output");
      stage.get_num_outputs=graph.GetFunction("get_num_outputs");
      stage.output_live_input_indices.assign(stage.NumOutputs(),-1);
      stages.push_back(std::move(stage));
    }
    Require(stages[0].NumOutputs()==1 && stages[1].NumOutputs()==2 && stages[2].NumOutputs()==8,"compiled output arity changed");
    private_input=stages[1].get_input(args.stages[1].input_names[0]).operator NDArray();
    Snapshot("graphs_loaded");
    std::vector<std::pair<uint64_t,uint64_t>> claimed;
    for(int edge=0;edge<2;++edge) {
      int count=(edge==0 && early_group && options.mode!="L2")?1:2;
      std::vector<P8SharedEdge> borrowed;
      if(!ordinary && !early_group && options.mode!="M1")
        borrowed=PrepareSharedBufferAnchor(&stages,args,shared_plan,edge,&claimed);
      anchors.push_back(!borrowed.empty());
      bundles[edge]=std::make_unique<Bundle>(&stages,edge,count,ordinary,std::move(borrowed));
      bundles[edge]->book.observer=[this,edge](const char* name,const P8SlotToken& token) {
        Emit(name,edge,token.frame_id,&token);
        if(edge==0 && token.frame_id==1) {
          std::lock_guard<std::mutex> lock(mutex);
          if(std::strcmp(name,"producer_wait")==0)producer_blocked=true;
          if(std::strcmp(name,"producer_published")==0)next_published=true;
          condition.notify_all();
        }
        if(edge==0 && token.frame_id==2 && std::strcmp(name,"producer_wait")==0) {
          std::lock_guard<std::mutex> lock(mutex);full_blocked=true;condition.notify_all();
        }
      };
      queues[edge]=std::make_unique<BoundedQueue<Record*>>(2);
    }
    // Actual ranges for all slots and every tensor must be disjoint.
    std::vector<std::pair<uint64_t,uint64_t>> actual;
    for(auto& b:bundles)for(auto& slot:b->shared)for(auto& t:slot) {
      auto cpu=reinterpret_cast<uintptr_t>(t.cpu_owner->data);uint64_t pool[8]{};
      auto snap=reinterpret_cast<int(*)(uint64_t*,size_t)>(dlsym(RTLD_DEFAULT,"VTAMemAllocationSnapshotV1"));
      Require(snap && snap(pool,8)==0,"physical pool ABI unavailable");
      auto physical=shared_buffer::CheckPhysicalRange(cpu,t.nbytes,pool);
      Require(physical==t.physical_address,"slot physical translation mismatch");
      for(auto& old:actual)Require(!shared_buffer::Overlaps(physical,t.nbytes,old.first,old.second),"slot/bundle ranges overlap");
      actual.emplace_back(physical,t.nbytes);
    }
    Snapshot("slots_allocated");
    NDArray first=stages[0].get_input(args.stages[0].input_names[0]).operator NDArray();
    for(auto& entry:As<Array>(Field(plan,"inputs"))) {
      shared_buffer::CheckHash(S(entry,"path"),Field(entry,"sha256"));
      inputs.push_back(MakeCPUInputLike(first.operator->(),ReadFile(S(entry,"path"),true)));
    }
    const auto& program_object=As<Object>(plan);
    auto count_entry=program_object.find("input_count");
    size_t expected_inputs=count_entry==program_object.end()?64:Number(count_entry->second);
    Require(expected_inputs>0 && inputs.size()==expected_inputs,"frozen input count mismatch");
    frames.resize(options.frames+options.warmup+options.settle);
    for(size_t i=0;i<frames.size();++i) {
      auto& f=frames[i];f.id=static_cast<int>(i)-options.warmup;
      f.input=(f.id<0?static_cast<int>(i):f.id)%inputs.size();f.measured=f.id>=options.settle;
      for(int o=0;o<stages[2].NumOutputs();++o) {
        NDArray model=stages[2].get_output(o);
        f.outputs.push_back(NDArray::Empty(model.Shape(),model->dtype,tvm::Device{kDLCPU,0}));
      }
    }
  }
  void Return(Record* f,int edge) {
    Require(edge!=0 || !f->released,"input lease returned twice");
    bundles[edge]->book.Release(f->tokens[edge]);if(edge==0)f->released=true;
  }
  void RunOne(int index,Record* f) {
    auto& stage=stages[index];Emit("stage_begin",index,f->id);
    if(index==1 && f->id==0 && options.control=="stale")--f->tokens[0].generation;
    if(index>0)bundles[index-1]->book.Consume(f->tokens[index-1]);
    if(index==1 && f->id==0 && (options.control=="full" || options.control=="abort")) {
      Wait([&]{return full_blocked;},"producer did not block on two full slots");
      bundles[0]->book.Validate(f->tokens[0],LeaseBook::State::Using);
      bundles[0]->book.Validate(frames.at(options.warmup+1).tokens[0],LeaseBook::State::Ready);
      Emit("two_full_slots_and_blocked_producer_verified",1,0,&f->tokens[0]);
      if(options.control=="abort")throw std::runtime_error("controlled abort with blocked producer and waiting output consumer");
    }
    if(index<2)f->tokens[index]=bundles[index]->Acquire(f->id);
    double before=NowMillis();
    if(index==0)stage.set_input(stage.input_names[0],inputs[f->input]);
    double handoff=NowMillis();
    if(index>0)for(size_t t=0;t<stage.input_names.size();++t) {
      auto& b=*bundles[index-1];auto& token=f->tokens[index-1];
      if(ordinary || (index==1 && options.mode=="CP")) {
        // Restore the retained original input if a previous call used a zero-copy view.
        if(index==1 && options.mode=="CP")stage.set_input_zero_copy(stage.input_names[t],private_input);
        const auto& source=b.CPU(token,t);
        if(index==1 && options.mode=="CP")
          Require(Separate(ResolveRange(source.operator->()),ResolveRange(private_input.operator->())),"copy source aliases private input");
        stage.set_input(stage.input_names[t],source);f->copied[index-1]+=TensorNBytes(source.operator->());
      } else stage.set_input_zero_copy(stage.input_names[t],b.Consumer(token,t));
    }
    if(index==1 && options.mode=="CP")Return(f,0);
    if(index>0)f->handoff_ms[index-1]+=NowMillis()-handoff;
    handoff=NowMillis();
    if(index<2)for(size_t t=0;t<bundles[index]->tensors;++t) {
      if(!ordinary)stage.set_output_zero_copy(static_cast<int>(t),bundles[index]->Producer(f->tokens[index],t));
      bundles[index]->Bound(f->tokens[index],t);
    }
    if(index<2)f->handoff_ms[index]+=NowMillis()-handoff;
    f->set_ms[index]=NowMillis()-before;
    if(index==1 && proxy) {
      proxy->active=f;proxy->expected=options.mode=="CP"?private_input->data:bundles[0]->Consumer(f->tokens[0],0)->data;
      proxy->before=[this](Record* frame) {
        Emit("last_reader_start",1,frame->id,&frame->tokens[0]);
        if(options.control=="before" && frame->id==0) {
          Wait([&]{return producer_blocked;},"producer did not block before last reader");
          Emit("before_read_overwrite_blocked",1,frame->id,&frame->tokens[0]);
        }
      };
      proxy->after=[this](Record* frame) {
        Emit("last_reader_return",1,frame->id,&frame->tokens[0]);
        if(options.mode=="E1") {
          Return(frame,0);Emit("validated_early_release",1,frame->id,&frame->tokens[0]);
          if(options.control=="after" && frame->id==0) {
            Wait([&]{return next_published;},"next frame did not finish production after release");
            Emit("next_frame_fully_produced_before_consumer_tail",1,frame->id,&frame->tokens[0]);
          }
        }
      };
    }
    f->begin[index]=NowMillis();
    try {
      if(index==1){std::lock_guard<std::mutex> lock(device_mutex);stage.run();}
      else stage.run();
    } catch(...) {if(index==1 && proxy)proxy->active=nullptr;throw;}
    f->end[index]=NowMillis();++calls[index];
    if(index==1 && proxy){proxy->active=nullptr;Require(f->reader_done,"last reader was not called");}
    before=NowMillis();
    if(index<2) {
      auto& b=*bundles[index];
      for(size_t t=0;t<b.tensors;++t) {
        if(ordinary) {
          stage.get_output(static_cast<int>(t),b.Producer(f->tokens[index],t));
          f->copied[index]+=TensorNBytes(b.Producer(f->tokens[index],t).operator->());
        }
        if(options.control=="incomplete" && index==1 && f->id==0 && t==1)continue;
        b.Written(f->tokens[index],t);
      }
      b.Publish(f->tokens[index]);
    } else for(int o=0;o<stage.NumOutputs();++o)stage.get_output(o,f->outputs[o]);
    f->get_ms[index]=NowMillis()-before;
    if(index<2)f->handoff_ms[index]+=f->get_ms[index];
    if(index>0 && (index!=1 || !f->released)) {
      before=NowMillis();Return(f,index-1);f->handoff_ms[index-1]+=NowMillis()-before;
      if(index==1 && f->id==0 && options.control=="duplicate")Return(f,0);
    }
    Emit("stage_end",index,f->id);
    if(index==2) {
      f->completion=NowMillis();f->complete=true;
      std::lock_guard<std::mutex> lock(mutex);++completed;condition.notify_all();
    }
  }
  void Actor(int index) {
    try {
      Wait([&]{return init_turn==index;},"actor initialization timeout");
      auto old=ProcessThreadIds();SelfAffinity({0,1,2,3});
      ConfigureThreadPool(args.stages[index].runtime_num_threads,{0,1,2,3},"c2.actor"+std::to_string(index));
      thread_counts[index]=(*Registry::Get("runtime.NumThreads"))().operator int();
      Require(thread_counts[index]==args.stages[index].runtime_num_threads,"actor thread count mismatch");
      auto now=ProcessThreadIds();auto& a=affinities[index];a.caller_tid=syscall(SYS_gettid);
      std::set_difference(now.begin(),now.end(),old.begin(),old.end(),std::back_inserter(a.thread_ids));
      a.thread_ids.push_back(a.caller_tid);
      {std::lock_guard<std::mutex> lock(mutex);++init_turn;condition.notify_all();}
      Wait([&]{return start;},"actor start timeout");
      if(index==0) {
        for(size_t i=0;i<frames.size();++i) {
          if(options.serial && i)Wait([&]{return completed==static_cast<int>(i);},"reference drain timeout");
          if(i==static_cast<size_t>(options.warmup) && options.warmup)
            {Wait([&]{return completed==options.warmup;},"warmup drain timeout");Snapshot("after_warmup");}
          RunOne(0,&frames[i]);Require(queues[0]->Push(&frames[i]),"queue aborted");
        }
        queues[0]->Close();
      } else {
        Record* frame=nullptr;
        double request=NowMillis();
        while(queues[index-1]->Pop(&frame)) {
          frame->waits[index]=NowMillis()-request;
          RunOne(index,frame);
          if(index==1)Require(queues[1]->Push(frame),"queue aborted");
          request=NowMillis();
        }
        if(index==1)queues[1]->Close();
      }
      Wait([&]{return finish;},"actor finish timeout");
    } catch(...) {Fail(std::current_exception());}
  }
  void Run() {
    std::array<std::thread,3> workers;
    for(int i=0;i<3;++i)workers[i]=std::thread([&,i]{Actor(i);});
    try {
      Wait([&]{return init_turn==3;},"thread initialization failed");
      for(auto& a:affinities){CaptureThreadAffinities(&a);CheckAffinity(a,{0,1,2,3});}
      {std::lock_guard<std::mutex> lock(mutex);start=true;condition.notify_all();}
      Wait([&]{return completed==static_cast<int>(frames.size());},"pipeline drain timeout",600);
      for(auto& b:bundles)b->book.AllFree();
      Snapshot("after_steady_state");
      {std::lock_guard<std::mutex> lock(mutex);finish=true;condition.notify_all();}
    } catch(...) {Fail(std::current_exception());}
    for(auto& t:workers)if(t.joinable())t.join();
    if(failure)std::rethrow_exception(failure);
    for(auto c:calls)Require(c==frames.size(),"stage call count mismatch");
  }
  void Save(const char* status,const std::string& error="") {
    EnsureDir(options.output);Require(!std::filesystem::exists(options.output+"/result.json"),"result already exists");
    std::ofstream bytes(options.output+"/outputs.bin",std::ios::binary);
    Array rows;uint64_t offset=0,retained=0;
    for(auto& f:frames) {
      Object r;r["frame_id"]=J(f.id);r["input_index"]=J(f.input);r["measured"]=JSON(f.measured);
      r["complete"]=JSON(f.complete);r["completion_ms"]=JSON(f.completion);
      Array outputs,begin,end,set,get,tokens,copied,handoff,waits;
      for(int i=0;i<3;++i){begin.push_back(JSON(f.begin[i]));end.push_back(JSON(f.end[i]));set.push_back(JSON(f.set_ms[i]));get.push_back(JSON(f.get_ms[i]));}
      for(int i=0;i<2;++i){Object t;t["edge"]=J(i);t["slot"]=J(static_cast<uint64_t>(f.tokens[i].slot_id));
        t["generation"]=J(f.tokens[i].generation);t["frame"]=J(f.tokens[i].frame_id);tokens.emplace_back(t);copied.push_back(J(f.copied[i]));handoff.push_back(JSON(f.handoff_ms[i]));}
      for(auto w:f.waits)waits.push_back(JSON(w));
      if(f.complete)for(size_t i=0;i<f.outputs.size();++i) {
        const auto& output=f.outputs[i];size_t n=TensorNBytes(output.operator->());
        bytes.write(reinterpret_cast<const char*>(TensorDataBytes(output.operator->())),n);
        Object o;o["index"]=J(static_cast<uint64_t>(i));o["offset"]=J(offset);o["bytes"]=J(n);
        o["dtype"]=JSON(DTypeString(output->dtype));outputs.emplace_back(o);offset+=n;
      }
      for(auto& o:f.outputs)retained+=TensorNBytes(o.operator->());
      r["outputs"]=JSON(outputs);r["stage_begin_ms"]=JSON(begin);r["stage_end_ms"]=JSON(end);
      r["set_ms"]=JSON(set);r["get_ms"]=JSON(get);r["tokens"]=JSON(tokens);r["boundary_copied_bytes"]=JSON(copied);
      r["boundary_handoff_ms"]=JSON(handoff);r["input_queue_wait_ms"]=JSON(waits);
      r["reader_done"]=JSON(f.reader_done);r["reader_argument_checks"]=J(f.checked_arguments);
      r["early_released"]=JSON(f.released && options.mode=="E1");rows.emplace_back(r);
    }
    bytes.close();Require(static_cast<bool>(bytes),"deferred output write failed");
    Array trace,allocation,affinity;
    for(size_t i=0;i<std::min(event_count.load(),events.size());++i){auto& e=events[i];if(!e.name)continue;
      Object r;r["event"]=JSON(e.name);r["stage"]=J(e.stage);r["frame"]=J(e.frame);r["edge"]=J(e.edge);
      r["slot"]=J(e.slot);r["generation"]=J(e.generation);r["time_ms"]=JSON(e.time);r["sequence"]=J(i);trace.emplace_back(r);}
    for(size_t edge=0;edge<2;++edge)if(bundles[edge]) {
      auto& b=*bundles[edge];
      for(size_t slot=0;slot<b.generation.size();++slot)for(size_t t=0;t<b.tensors;++t) {
        Object r;r["edge"]=J(edge);r["slot"]=J(slot);r["tensor"]=J(t);r["copy_staging"]=JSON(b.copy);
        if(b.copy){auto& x=b.cpu[slot][t];r["bytes"]=J(TensorNBytes(x.operator->()));r["virtual"]=JSON("0x"+UInt64Hex(reinterpret_cast<uintptr_t>(x->data)));}
        else {auto& x=b.shared[slot][t];r["bytes"]=J(x.nbytes);r["physical"]=JSON("0x"+UInt64Hex(x.physical_address));
          r["virtual"]=JSON("0x"+UInt64Hex(reinterpret_cast<uintptr_t>(x.cpu_owner->data)));r["borrowed"]=JSON(!x.allocation_owner);}
        allocation.emplace_back(r);
      }
    }
    for(int i=0;i<3;++i) {Object r;r["stage"]=J(i);r["threads"]=J(thread_counts[i]);r["caller_tid"]=J(static_cast<uint64_t>(affinities[i].caller_tid));
      Array workers;for(size_t j=0;j<affinities[i].thread_ids.size();++j){Object w;w["tid"]=J(static_cast<uint64_t>(affinities[i].thread_ids[j]));
        Array mask;if(j<affinities[i].effective_affinities.size())for(auto cpu:affinities[i].effective_affinities[j])mask.push_back(J(static_cast<int>(cpu)));
        w["cpus"]=JSON(mask);workers.emplace_back(w);}r["workers"]=JSON(workers);affinity.emplace_back(r);}
    Object result;result["schema"]=JSON(S(plan,"schema")=="c2-yolo416-program-v1"?
        "c2-yolo416-result-v1":"c2-yolo224-result-v1");result["status"]=JSON(status);result["error"]=JSON(error);
    result["input_count"]=J(inputs.size());
    Array payloads;
    for(int edge=0;edge<2;++edge) {
      uint64_t payload=0;
      if(bundles[edge])for(size_t t=0;t<bundles[edge]->tensors;++t) {
        const auto& b=*bundles[edge];
        payload+=b.copy?TensorNBytes(b.cpu.at(0).at(t).operator->()):b.shared.at(0).at(t).nbytes;
      }
      payloads.push_back(J(payload));
    }
    result["boundary_payload_bytes"]=JSON(payloads);
    result["mode"]=JSON(options.mode);result["control"]=JSON(options.control);result["frames"]=JSON(rows);result["events"]=JSON(trace);
    result["allocation_snapshots"]=JSON(Array(snapshots.begin(),snapshots.end()));result["slots"]=JSON(allocation);result["affinity"]=JSON(affinity);
    result["retained_output_bytes"]=J(retained);result["deferred_output_bytes"]=J(offset);result["all_free"]=JSON(std::string(status)=="passed");
    result["proxy_enabled"]=JSON(static_cast<bool>(proxy));result["detailed_argument_checks"]=JSON(options.detailed);
    result["boot_id"]=JSON(TrimLine(ReadFile("/proc/sys/kernel/random/boot_id")));
    if(auto* profiler=Registry::Get("vta.runtime.profiler_status"))result["runtime_profile"]=shared_buffer::Parse((*profiler)().operator std::string());
    std::ofstream output(options.output+"/result.json");output<<JSON(result).serialize(true)<<'\n';Require(static_cast<bool>(output),"result write failed");
  }
};

Options Parse(int argc,char** argv) {
  Options out;
  for(int i=1;i<argc;++i) {
    std::string key=argv[i];
    if(key=="--serial"){out.serial=true;continue;}if(key=="--detailed"){out.detailed=true;continue;}
    if(key=="--proxy-off"){out.proxy_off=true;continue;}
    Require(i+1<argc,"missing option value");std::string value=argv[++i];
    if(key=="--plan")out.plan=value;else if(key=="--output-dir")out.output=value;else if(key=="--mode")out.mode=value;
    else if(key=="--frames")out.frames=std::stoi(value);else if(key=="--warmup")out.warmup=std::stoi(value);
    else if(key=="--settle")out.settle=std::stoi(value);else if(key=="--control")out.control=value;
    else throw std::runtime_error("unknown option: "+key);
  }
  Require(!out.plan.empty()&&!out.output.empty(),"plan and output required");
  Require(std::set<std::string>{"M0","M1","M2","M3","manual","Q2","L1","L2","CP","E1"}.count(out.mode),"unknown mode");
  Require(out.frames>0&&out.frames<=10000&&out.warmup>=0&&out.settle>=0,"invalid frame count");
  return out;
}
}  // namespace c2_yolo

int main(int argc,char** argv) {
  std::unique_ptr<c2_yolo::Experiment> experiment;
  try {
    experiment=std::make_unique<c2_yolo::Experiment>(c2_yolo::Parse(argc,argv));
    experiment->Prepare();experiment->Run();experiment->Save("passed");return 0;
  } catch(const std::exception& error) {
    if(experiment)try{experiment->Save("INVALID",error.what());}catch(...){}
    std::cerr<<"c2_yolo224_runner: "<<error.what()<<'\n';return 1;
  }
}
