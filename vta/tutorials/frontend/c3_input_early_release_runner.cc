// Isolated last-reader experiment. No changes to GraphExecutor or the VTA runtime.
#define VTA_STAGE_PIPELINE_NO_MAIN
#include "../../apps/native_deploy/vta_stage_pipeline_runner.cc"

#include <array>
#include <atomic>
#include <optional>

namespace c3_last_read {
using shared_buffer::Array;
using shared_buffer::As;
using shared_buffer::Field;
using shared_buffer::JSON;
using shared_buffer::Number;
using shared_buffer::Require;
using shared_buffer::Text;
using Object = picojson::object;

static JSON J(uint64_t value) { return JSON(static_cast<double>(value)); }
static JSON J(int value) { return JSON(static_cast<double>(value)); }
static bool Has(const JSON& value, const char* key) {
  return As<Object>(value).count(key) != 0;
}
static std::string S(const JSON& value, const char* key) { return Text(Field(value, key)); }

struct Event {
  const char* name{nullptr};
  int stage{-1}, frame{-1}, edge{-1}, slot{-1};
  uint64_t generation{0};
  double time{0};
};

// Both production experiment calls and CPU-only selftests use this state machine.
class LeaseBook {
 public:
  enum class State { Free, Writing, Ready, Using };
  struct Record { State state{State::Free}; int frame{-1}; uint64_t generation{0}; };
  explicit LeaseBook(size_t count, int edge, int timeout_ms = 60000)
      : slots_(count), edge_(edge), timeout_ms_(timeout_ms) {
    Require(count == 1 || count == 2, "only one/two slots are supported");
  }
  std::function<void(const char*, const P8SlotToken&)> observer;
  P8SlotToken Acquire(int frame) {
    std::unique_lock<std::mutex> lock(mutex_);
    auto ready = [&] { return aborted_ || FreeIndex() < slots_.size(); };
    if (!ready()) Emit("producer_wait", P8SlotToken{static_cast<size_t>(edge_), 0, 0, frame});
    Require(condition_.wait_for(lock, std::chrono::milliseconds(timeout_ms_), ready),
            "timed out acquiring a free slot");
    Require(!aborted_, "lease book aborted");
    size_t index = FreeIndex();
    Record& slot = slots_.at(index);
    slot.state = State::Writing; slot.frame = frame; ++slot.generation;
    cursor_ = (index + 1) % slots_.size();
    P8SlotToken token{static_cast<size_t>(edge_), index, slot.generation, frame};
    Emit("producer_acquired", token);
    return token;
  }
  void Publish(const P8SlotToken& token) {
    std::lock_guard<std::mutex> lock(mutex_);
    Check(token, State::Writing).state = State::Ready;
    Emit("producer_published", token); condition_.notify_all();
  }
  void Consume(const P8SlotToken& token) {
    std::lock_guard<std::mutex> lock(mutex_);
    Check(token, State::Ready).state = State::Using;
    Emit("consumer_acquired", token);
  }
  void Release(const P8SlotToken& token) {
    std::lock_guard<std::mutex> lock(mutex_);
    Check(token, State::Using).state = State::Free;
    Emit("input_lease_released", token); condition_.notify_all();
  }
  void Validate(const P8SlotToken& token, State state) {
    std::lock_guard<std::mutex> lock(mutex_); (void)Check(token, state);
  }
  void AllFree() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const Record& slot : slots_) Require(slot.state == State::Free, "lease not returned");
  }
  void Abort() {
    std::lock_guard<std::mutex> lock(mutex_); aborted_ = true; condition_.notify_all();
  }
 private:
  size_t FreeIndex() const {
    for (size_t n = 0; n < slots_.size(); ++n) {
      size_t index = (cursor_ + n) % slots_.size();
      if (slots_[index].state == State::Free) return index;
    }
    return slots_.size();
  }
  Record& Check(const P8SlotToken& token, State state) {
    Require(!aborted_ && token.edge_index == static_cast<size_t>(edge_) &&
            token.slot_id < slots_.size(), "invalid/aborted lease token");
    Record& slot = slots_[token.slot_id];
    Require(slot.frame == token.frame_id && slot.generation == token.generation &&
            slot.state == state, "stale, duplicate or invalid lease transition");
    return slot;
  }
  void Emit(const char* name, const P8SlotToken& token) { if (observer) observer(name, token); }
  std::vector<Record> slots_;
  int edge_, timeout_ms_;
  size_t cursor_{0};
  bool aborted_{false};
  std::mutex mutex_;
  std::condition_variable condition_;
};

struct Invocation {
  P8SlotToken token{};
  bool released{false};
  void* expected_argument{nullptr};
  int target_calls{0}, next_calls{0};
  std::function<void(tvm::runtime::TVMArgs)> inspect_reader, inspect_next;
  std::function<void()> before_reader, after_reader, before_next;
};

struct BufferRange {
  uintptr_t handle{0}, cpu{0};
  uint64_t physical{0};
  size_t bytes{0};
};

static BufferRange ResolveRange(const DLTensor* tensor) {
  Require(tensor != nullptr && tensor->data != nullptr && tensor->strides == nullptr &&
          tensor->byte_offset == 0, "unsupported actual tensor buffer ABI");
  void* cpu = tensor->data;
  if (tensor->device.device_type == kDLExtDev) {
    using Resolve = void* (*)(void*, void*);
    auto resolve = reinterpret_cast<Resolve>(dlsym(RTLD_DEFAULT, "VTABufferCPUPtr"));
    Require(resolve != nullptr, "actual runtime CPU address resolver unavailable");
    // Frozen implementation ignores the command argument and performs no queue
    // work: a real DataBuffer is mapped; an existing raw CPU view is forwarded.
    cpu = resolve(nullptr, tensor->data);
  } else Require(tensor->device.device_type == kDLCPU, "unsupported actual tensor device");
  BufferRange range;
  range.handle = reinterpret_cast<uintptr_t>(tensor->data);
  range.cpu = reinterpret_cast<uintptr_t>(cpu);
  range.bytes = TensorNBytes(tensor);
  range.physical = VTAMemGetPhyAddr(cpu);
  Require(range.cpu != 0 && range.physical != 0 && range.bytes != 0 &&
          range.cpu + range.bytes > range.cpu && range.physical + range.bytes > range.physical,
          "actual buffer range is invalid or outside DMA allocation");
  return range;
}

static bool Separate(const BufferRange& a, const BufferRange& b) {
  return (a.cpu + a.bytes <= b.cpu || b.cpu + b.bytes <= a.cpu) &&
         (a.physical + a.bytes <= b.physical || b.physical + b.bytes <= a.physical);
}

static bool SameRange(const BufferRange& a, const BufferRange& b) {
  return a.cpu == b.cpu && a.physical == b.physical && a.bytes == b.bytes;
}

struct RangeProof {
  BufferRange input, reader_output, next_input, packed_output, copy_source, copy_destination;
  bool verified{false}, copy_separate{false};
};

static void ReturnLease(LeaseBook* book, Invocation* invocation) {
  Require(!invocation->released, "invocation lease was already returned");
  book->Release(invocation->token);
  invocation->released = true;
}

class ReaderProxy final : public tvm::runtime::ModuleNode {
 public:
  Module inner;
  std::string target, next;
  Invocation* active{nullptr};
  uint64_t calls{0}, next_calls{0};
  const char* type_key() const final { return "c3_last_reader_proxy"; }
  PackedFunc GetFunction(const tvm::runtime::String& name,
                         const tvm::runtime::ObjectPtr<tvm::runtime::Object>& self) final {
    PackedFunc original = inner.GetFunction(name, true);
    const bool reader = std::string(name) == target;
    if (!original.defined() || (!reader && std::string(name) != next)) return original;
    return PackedFunc([this, self, original, reader](tvm::runtime::TVMArgs args,
                                            tvm::runtime::TVMRetValue* rv) {
      Require(active != nullptr, "unexpected unarmed graph invocation");
      Require(args.size() == 2, "CPU reader/packing ABI changed");
      if (!reader) {
        Require(active->target_calls == 1 && active->next_calls == 0, "unexpected next-node invocation");
        if (active->inspect_next) active->inspect_next(args);
        // The conservative release occurs only after BOTH ABI ranges are known,
        // after the last reader returned and before node2 executes, not at an
        // invented instantaneous reader-return timestamp.
        if (active->before_next) active->before_next();
        original.CallPacked(args, rv);
        ++active->next_calls; ++next_calls;
        return;
      }
      Require(active->target_calls == 0, "unexpected target invocation");
      DLTensor* input = args[0];
      Require(input != nullptr && input->data == active->expected_argument,
              "target did not receive the frozen input view");
      if (active->inspect_reader) active->inspect_reader(args);
      if (active->before_reader) active->before_reader();
      original.CallPacked(args, rv);  // No new execution, copy, wait, timer or synchronization.
      ++active->target_calls; ++calls;
      if (active->after_reader) active->after_reader();
    });
  }
};

struct FrameRecord {
  int id{0}, input{0};
  bool warmup{false}, complete{false};
  std::array<P8SlotToken, 2> tokens{};
  std::array<double, 3> run_begin{}, run_end{};
  double enqueue{0}, completion{0}, copy_ms{0};
  NDArray output;
  RangeProof ranges;
};

// Bounded, preallocated pointer queues. No frame allocation in the measured phase.
class Queue {
 public:
  void Push(FrameRecord* frame) {
    std::unique_lock<std::mutex> lock(mutex_);
    Require(cv_.wait_for(lock, std::chrono::seconds(60), [&] { return aborted_ || size_ < 2; }),
            "queue push timeout");
    Require(!aborted_, "queue aborted");
    data_[(head_ + size_) % 2] = frame; ++size_; cv_.notify_all();
  }
  FrameRecord* Pop() {
    std::unique_lock<std::mutex> lock(mutex_);
    Require(cv_.wait_for(lock, std::chrono::seconds(60), [&] { return aborted_ || size_ || closed_; }),
            "queue pop timeout");
    Require(!aborted_, "queue aborted");
    if (!size_) return nullptr;
    FrameRecord* frame = data_[head_]; head_ = (head_ + 1) % 2; --size_;
    cv_.notify_all(); return frame;
  }
  void Close() { std::lock_guard<std::mutex> lock(mutex_); closed_ = true; cv_.notify_all(); }
  void Abort() { std::lock_guard<std::mutex> lock(mutex_); aborted_ = true; cv_.notify_all(); }
 private:
  std::array<FrameRecord*, 2> data_{};
  size_t head_{0}, size_{0};
  bool closed_{false}, aborted_{false};
  std::mutex mutex_;
  std::condition_variable cv_;
};

class Manager {
 public:
  LeaseBook book;
  std::vector<P8SharedEdge> views;
  Manager(std::vector<StageExecutor>* stages, int edge, int count, bool anchor)
      : book(count, edge) {
    views.reserve(count);
    for (int slot = 0; slot < count; ++slot)
      views.push_back(AllocateP8HeterogeneousEdgeTensor(stages, edge, 0, anchor && slot == 0));
  }
  const NDArray& Producer(const P8SlotToken& token) {
    book.Validate(token, LeaseBook::State::Writing); return views.at(token.slot_id).producer_owner;
  }
  const NDArray& Consumer(const P8SlotToken& token) {
    book.Validate(token, LeaseBook::State::Using); return views.at(token.slot_id).consumer_owner;
  }
  const NDArray& ConsumerCPU(const P8SlotToken& token) {
    book.Validate(token, LeaseBook::State::Using); return views.at(token.slot_id).cpu_owner;
  }
};

static void SelfAffinity(const std::vector<unsigned int>& cpus) {
  cpu_set_t mask; CPU_ZERO(&mask);
  for (unsigned int cpu : cpus) CPU_SET(cpu, &mask);
  Require(sched_setaffinity(0, sizeof(mask), &mask) == 0, "caller affinity setup failed");
}

static void CheckAffinity(const ThreadAffinityRecord& record,
                          const std::vector<unsigned int>& allowed) {
  Require(!record.thread_ids.empty(), "no actor affinity records");
  for (const auto& mask : record.effective_affinities) {
    Require(!mask.empty(), "empty effective actor affinity");
    for (unsigned int cpu : mask)
      Require(std::find(allowed.begin(), allowed.end(), cpu) != allowed.end(),
              "actor/helper worker escaped its frozen CPUs");
  }
}

struct Options {
  std::string plan_path, output, variant, mode{"score"};
  int frames{100}; bool proxy_off{false};
};
static Options Parse(int argc, char** argv) {
  Options out;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    if (key == "--proxy-off") { out.proxy_off = true; continue; }
    Require(i + 1 < argc, "missing command line option value");
    std::string value = argv[++i];
    if (key == "--plan") out.plan_path = value;
    else if (key == "--output-dir") out.output = value;
    else if (key == "--variant") out.variant = value;
    else if (key == "--mode") out.mode = value;
    else if (key == "--frames") out.frames = std::stoi(value);
    else throw std::runtime_error("unknown option: " + key);
  }
  if (out.mode == "selftest") return out;
  Require(!out.plan_path.empty() && !out.output.empty(), "plan/output-dir required");
  Require(out.variant == "L1" || out.variant == "L2" || out.variant == "C1" || out.variant == "E1",
          "unknown variant");
  Require(out.frames > 0 && out.frames <= 10000, "invalid frame count");
  Require(out.mode == "score" || out.mode == "qualify" || out.mode == "reference" ||
          out.mode == "proxycheck" || out.mode == "event-before" || out.mode == "event-after",
          "unknown mode");
  Require(!out.proxy_off || (out.mode == "proxycheck" && out.variant == "L2"),
          "proxy-off is confined to L2 proxycheck");
  if (out.mode == "event-before" || out.mode == "event-after")
    Require(out.variant == "E1" && out.frames == 3, "event controls require three E1 frames");
  return out;
}

class Experiment {
 public:
  Options options;
  JSON plan;
  std::vector<StageExecutor> stages;
  std::array<StageArgs, 3> stage_args;
  std::vector<NDArray> inputs;
  std::vector<FrameRecord> frames;
  std::array<std::unique_ptr<Manager>, 2> managers;
  tvm::runtime::ObjectPtr<ReaderProxy> proxy;
  NDArray private_input;
  std::array<NDArray, 2> reference_edges;
  std::array<Queue, 2> queues;
  std::vector<Event> events;
  std::atomic<size_t> event_count{0};
  std::array<ThreadAffinityRecord, 3> affinity;
  std::array<std::array<int, 2>, 3> actual_cpu{};
  std::array<int, 3> actual_threads{};
  std::array<uint64_t, 3> stage_calls{};
  std::vector<JSON> snapshots;
  std::mutex control_mutex, device_mutex;
  std::condition_variable control;
  bool aborted{false}, start{false}, score_start{false}, finish{false};
  int arrived{0}, ready{0}, init_turn{0}, completed{0};
  bool producer_blocked{false}, next_produced{false};
  std::exception_ptr error;
  int warmup{0};
  double formal_start{0}, formal_end{0};
  std::string target;
  bool reference{false};
  bool identities_verified{false}, contract_verified{false};
  uint64_t pool_virtual{0}, pool_physical{0}, pool_used{0};
  Array loaded_libraries;

  explicit Experiment(Options value) : options(std::move(value)) {}
  void Emit(const char* name, int stage, int frame, const P8SlotToken* token = nullptr) {
    size_t index = event_count.fetch_add(1);
    Require(index < events.size(), "preallocated event capacity exceeded");
    Event& event = events[index]; event.name = name; event.stage = stage; event.frame = frame;
    event.time = NowMillis();
    if (token) {
      event.edge = static_cast<int>(token->edge_index); event.slot = static_cast<int>(token->slot_id);
      event.generation = token->generation;
    }
  }
  void Fail(std::exception_ptr failure) {
    {
      std::lock_guard<std::mutex> lock(control_mutex);
      if (!error) error = failure;
      aborted = true; control.notify_all();
    }
    for (auto& queue : queues) queue.Abort();
    for (auto& manager : managers) if (manager) manager->book.Abort();
  }
  template <typename Predicate> void Wait(Predicate predicate, const char* reason) {
    std::unique_lock<std::mutex> lock(control_mutex);
    Require(control.wait_for(lock, std::chrono::seconds(60), [&] { return aborted || predicate(); }), reason);
    Require(!aborted, "experiment aborted");
  }
  void Snapshot(const char* phase) {
    using Function = int (*)(uint64_t*, size_t);
    auto function = reinterpret_cast<Function>(dlsym(RTLD_DEFAULT, "VTAMemAllocationSnapshotV1"));
    uint64_t data[8]{};
    Require(function != nullptr && function(data, 8) == 0, "allocation snapshot ABI unavailable");
    Object record; record["phase"] = JSON(phase);
    const char* names[] = {"initialized", "capacity_bytes", "high_water_bytes", "requested_bytes",
                          "padding_bytes", "allocation_count", "virtual_base", "physical_base"};
    for (size_t i = 0; i < 8; ++i) record[names[i]] = J(data[i]);
    snapshots.emplace_back(record);
  }
  void VerifyLibrary(const char* symbol, const char* key) {
    Dl_info info{}; void* address = dlsym(RTLD_DEFAULT, symbol);
    Require(address != nullptr && dladdr(address, &info) != 0 && info.dli_fname != nullptr,
            "cannot identify actual loaded runtime");
    const JSON& specification = Field(Field(plan, "libraries"), key);
    shared_buffer::CheckHash(info.dli_fname, Field(specification, "sha256"));
    shared_buffer::CheckHash(S(specification, "path"), Field(specification, "sha256"));
    Object actual; actual["library"] = JSON(key); actual["actual_path"] = JSON(info.dli_fname);
    actual["sha256"] = Field(specification, "sha256");
    actual["symbol"] = JSON(symbol);
    actual["symbol_address"] = JSON("0x" + UInt64Hex(reinterpret_cast<uintptr_t>(address)));
    loaded_libraries.emplace_back(actual);
  }
  BufferRange CheckedRange(const DLTensor* tensor) {
    BufferRange range = ResolveRange(tensor);
    Require(range.cpu >= pool_virtual && range.cpu - pool_virtual <= pool_used &&
            range.bytes <= pool_used - (range.cpu - pool_virtual) &&
            range.physical == pool_physical + (range.cpu - pool_virtual),
            "actual ABI buffer is outside the startup allocated DMA range");
    return range;
  }
  StageExecutor Load(int index, const JSON& specification) {
    StageArgs& args = stage_args[index];
    args.name = S(specification, "name"); args.device = S(specification, "device");
    args.graph = S(specification, "graph"); args.lib = S(specification, "lib");
    args.params = S(specification, "params");
    args.runtime_num_threads = static_cast<int>(Number(Field(specification, "threads")));
    args.input_names.clear();
    for (const JSON& name : As<Array>(Field(specification, "input_names"))) args.input_names.push_back(Text(name));
    for (const JSON& cpu : As<Array>(Field(specification, "cpus"))) args.cpu_affinity.push_back(Number(cpu));
    Require(args.input_names.size() == 1, "only single-tensor three-stage chain is supported");
    Require(args.device == (index == 1 ? "vta" : "cpu"), "expected CPU/VTA/CPU chain");
    const std::vector<unsigned int> expected = index == 0 ? std::vector<unsigned int>{0, 1} :
        (index == 1 ? std::vector<unsigned int>{3} : std::vector<unsigned int>{2});
    Require(args.cpu_affinity == expected && args.runtime_num_threads == (index == 0 ? 2 : 1),
            "actor CPU/thread contract changed");
    shared_buffer::CheckHash(args.graph, Field(specification, "graph_sha256"));
    shared_buffer::CheckHash(args.lib, Field(specification, "lib_sha256"));
    shared_buffer::CheckHash(args.params, Field(specification, "params_sha256"));
    if (index != 1 || options.proxy_off) return LoadStage(args.name, args.device, args);

    Module library = Module::LoadFromFile(args.lib);
    Require(!library.GetFunction("__tvm_lookup_linked_param", true).defined(), "linked parameter module unsupported");
    const JSON compiled = shared_buffer::Parse(ReadFile(args.graph));
    const auto& nodes = As<Array>(Field(compiled, "nodes"));
    Require(nodes.size() > 2 && S(nodes[2], "op") == "tvm_op", "next packing node missing");
    const std::string next = S(Field(nodes[2], "attrs"), "func_name");
    Require(next != target, "reader and next-node symbols must differ");
    size_t next_occurrences = 0;
    for (const JSON& node : nodes)
      if (S(node, "op") == "tvm_op" && S(Field(node, "attrs"), "func_name") == next) ++next_occurrences;
    const auto& next_inputs = As<Array>(Field(nodes[2], "inputs"));
    Require(next_occurrences == 1 && next_inputs.size() == 1 &&
            Number(As<Array>(next_inputs[0])[0]) == 1 &&
            Number(As<Array>(next_inputs[0])[1]) == 0, "next packing node ABI/order changed");
    proxy = tvm::runtime::make_object<ReaderProxy>(); proxy->inner = library; proxy->target = target; proxy->next = next;
    Require(library.GetFunction(target, true).defined(), "last CPU reader symbol missing");
    Module graph = (*Registry::Get("tvm.graph_executor.create"))(
        ReadFile(args.graph), Module(proxy), static_cast<int>(kDLExtDev), 0, static_cast<int>(kDLCPU), 0);
    LoadParamsBlob(graph.GetFunction("load_params"), ReadFile(args.params, true));
    StageExecutor stage; stage.name = args.name; stage.device = args.device;
    stage.input_names = args.input_names; stage.graph = graph;
    stage.set_input = graph.GetFunction("set_input");
    stage.set_input_zero_copy = graph.GetFunction("set_input_zero_copy");
    stage.set_output_zero_copy = graph.GetFunction("set_output_zero_copy");
    stage.run = graph.GetFunction("run"); stage.get_input = graph.GetFunction("get_input");
    stage.get_output = graph.GetFunction("get_output"); stage.get_num_outputs = graph.GetFunction("get_num_outputs");
    for (const JSON& head : As<Array>(Field(compiled, "heads"))) {
      Require(S(nodes.at(Number(As<Array>(head).at(0))), "op") != "null", "live input/parameter head unsupported");
      stage.output_live_input_indices.push_back(-1);
    }
    return stage;
  }
  void Prepare() {
    plan = shared_buffer::Parse(ReadFile(options.plan_path));
    Require(S(plan, "schema") == "c3-input-last-read-v1", "plan schema mismatch");
    reference = options.mode == "reference";
    target = S(Field(plan, "target"), "function");
    if (Has(plan, "identity") && Has(Field(plan, "identity"), "boot_id"))
      Require(TrimLine(ReadFile("/proc/sys/kernel/random/boot_id")) == S(Field(plan, "identity"), "boot_id"),
              "Linux boot identity changed");
    VerifyLibrary("TVMFuncGetGlobal", "tvm_runtime"); VerifyLibrary("VTAMemAlloc", "vta_runtime");
    VerifyLibrary("VTABufferCPUPtr", "vta_runtime");
    VerifyLibrary("VTAMemAllocationSnapshotV1", "vta_runtime");
    identities_verified = true;
    Require(std::string(std::getenv("AXU5EVB_DRIVER_SAFE_COPY") ? std::getenv("AXU5EVB_DRIVER_SAFE_COPY") : "") == "0",
            "fastest qualified memcpy configuration must be explicit");
    Require(std::string(std::getenv("TVM_THREAD_POOL_SPIN_COUNT") ? std::getenv("TVM_THREAD_POOL_SPIN_COUNT") : "") == "0",
            "thread-pool spin contract must be explicit");
    warmup = (options.mode == "score" || options.mode == "proxycheck") ? 5 : 0;
    if (Has(plan, "warmup_count") && warmup) Require(Number(Field(plan, "warmup_count")) == 5, "warmup contract changed");
    events.resize(static_cast<size_t>(options.frames + warmup) * 64 + 128);
    Snapshot("before_graph_load");
    setenv("TVM_NUM_THREADS", "2", 1);
    const auto& specifications = As<Array>(Field(plan, "stages"));
    Require(specifications.size() == 3, "exactly three stages required"); stages.reserve(3);
    for (int i = 0; i < 3; ++i) stages.push_back(Load(i, specifications[i]));
    for (const auto& stage : stages) Require(stage.NumOutputs() == 1, "single output stage required");
    Snapshot("graphs_loaded");
    const JSON graph = shared_buffer::Parse(ReadFile(stage_args[1].graph));
    const auto& goal = Field(plan, "target");
    shared_buffer::CheckExclusiveInput(graph, stage_args[1].input_names[0], Number(Field(goal, "entry")),
        Number(Field(goal, "storage_id")), shared_buffer::ParameterNames(ReadFile(stage_args[1].params, true)));
    const auto& nodes = As<Array>(Field(graph, "nodes"));
    size_t readers = 0, occurrences = 0;
    for (size_t i = 0; i < nodes.size(); ++i) {
      if (S(nodes[i], "op") == "tvm_op" && S(Field(nodes[i], "attrs"), "func_name") == target) ++occurrences;
      for (const JSON& reference : As<Array>(Field(nodes[i], "inputs"))) {
        const auto& entry = As<Array>(reference);
        if (Number(entry[0]) == 0) {
          ++readers;
          Require(i == Number(Field(goal, "node")) && S(Field(nodes[i], "attrs"), "func_name") == target,
                  "unqualified future raw input reader");
        }
      }
    }
    Require(readers == 1 && occurrences == 1 && Number(Field(goal, "entry")) == 0 &&
            Number(Field(goal, "storage_id")) == 0 && Number(Field(goal, "node")) == 1,
            "fixed last-reader eligibility changed");
    private_input = stages[1].get_input(stage_args[1].input_names[0]).operator NDArray();
    Require(TensorNBytes(private_input.operator->()) == Number(Field(goal, "bytes")), "input byte contract mismatch");
    Require(private_input->dtype.code == kDLFloat && private_input->dtype.bits == 32 &&
            private_input->dtype.lanes == 1, "fixed shared boundary must be float32");
    for (int i = 0; i < 3; ++i)
      Require(std::all_of(stages[i].output_live_input_indices.begin(), stages[i].output_live_input_indices.end(),
                          [](int value) { return value < 0; }), "live input output prevents early release");
    const auto& input_specs = As<Array>(Field(plan, "inputs"));
    Require(!input_specs.empty(), "no frozen inputs");
    NDArray first = stages[0].get_input(stage_args[0].input_names[0]).operator NDArray();
    inputs.reserve(input_specs.size());
    for (const JSON& item : input_specs) {
      shared_buffer::CheckHash(S(item, "path"), Field(item, "sha256"));
      inputs.push_back(MakeCPUInputLike(first.operator->(), ReadFile(S(item, "path"), true)));
    }
    if (reference) {
      for (int i = 0; i < 2; ++i) {
        NDArray output = stages[i].get_output(0).operator NDArray();
        reference_edges[i] = NDArray::Empty(output.Shape(), output->dtype, tvm::Device{kDLCPU, 0});
      }
    } else {
      managers[0] = std::make_unique<Manager>(&stages, 0, options.variant == "L2" ? 2 : 1,
                                             options.variant != "C1");
      managers[1] = std::make_unique<Manager>(&stages, 1, 2, false);
      std::vector<std::pair<uint64_t, uint64_t>> ranges;
      for (auto& manager : managers) {
        for (const auto& view : manager->views) {
          uint64_t begin = view.physical_address, end = begin + view.nbytes;
          Require(begin != 0 && end > begin, "invalid slot physical range");
          for (const auto& range : ranges) Require(end <= range.first || begin >= range.second, "slot ranges alias");
          ranges.emplace_back(begin, end);
        }
        manager->book.observer = [this](const char* event, const P8SlotToken& token) {
          int stage = static_cast<int>(token.edge_index) + (std::strncmp(event, "consumer", 8) == 0 ||
                      std::strcmp(event, "input_lease_released") == 0);
          Emit(event, stage, token.frame_id, &token);
          if (token.edge_index == 0 && token.frame_id == 1) {
            std::lock_guard<std::mutex> lock(control_mutex);
            if (std::strcmp(event, "producer_wait") == 0) producer_blocked = true;
            if (std::strcmp(event, "producer_published") == 0) next_produced = true;
            control.notify_all();
          }
        };
      }
    }
    Snapshot("slots_allocated");
    pool_virtual = Number(Field(snapshots.back(), "virtual_base"));
    pool_physical = Number(Field(snapshots.back(), "physical_base"));
    pool_used = Number(Field(snapshots.back(), "high_water_bytes"));
    Require(pool_virtual && pool_physical && pool_used, "DMA allocation bounds unavailable");
    contract_verified = true;
    NDArray final = stages[2].get_output(0).operator NDArray();
    frames.resize(options.frames + warmup);
    for (size_t i = 0; i < frames.size(); ++i) {
      auto& frame = frames[i]; frame.warmup = i < static_cast<size_t>(warmup);
      frame.id = frame.warmup ? static_cast<int>(i) - warmup : static_cast<int>(i) - warmup;
      frame.input = (frame.warmup ? static_cast<int>(i) : frame.id) % inputs.size();
      frame.output = NDArray::Empty(final.Shape(), final->dtype, tvm::Device{kDLCPU, 0});
    }
  }

  void RunOne(int index, FrameRecord* frame) {
    auto& stage = stages[index];
    Emit("stage_begin", index, frame->id);
    Invocation invocation;
    if (reference) {
      stage.set_input(stage.input_names[0], index == 0 ? inputs[frame->input] : reference_edges[index - 1]);
      if (index == 1) invocation.expected_argument = private_input->data;
    } else if (index == 0) {
      stage.set_input(stage.input_names[0], inputs[frame->input]);
    } else {
      auto& incoming = *managers[index - 1];
      invocation.token = frame->tokens[index - 1]; incoming.book.Consume(invocation.token);
      if (index == 1 && options.variant == "C1") {
        frame->ranges.copy_source = CheckedRange(incoming.ConsumerCPU(invocation.token).operator->());
        frame->ranges.copy_destination = CheckedRange(private_input.operator->());
        Require(Separate(frame->ranges.copy_source, frame->ranges.copy_destination),
                "C1 source aliases private destination");
        frame->ranges.copy_separate = true;
        double before = NowMillis(); Emit("copy_begin", index, frame->id, &invocation.token);
        // ConsumerCPU is a real CPU virtual pointer. Consumer() has an ext_dev tag
        // around a raw pointer and MUST NOT be passed to CopyFrom as an ext buffer.
        stage.set_input(stage.input_names[0], incoming.ConsumerCPU(invocation.token));
        frame->copy_ms = NowMillis() - before; Emit("copy_end", index, frame->id, &invocation.token);
        ReturnLease(&incoming.book, &invocation);
        stage.set_input_zero_copy(stage.input_names[0], private_input);
        invocation.expected_argument = private_input->data;
      } else {
        const NDArray& input = incoming.Consumer(invocation.token);
        stage.set_input_zero_copy(stage.input_names[0], input);
        invocation.expected_argument = input->data;
      }
    }
    if (!reference && index < 2) {
      auto& outgoing = *managers[index];
      frame->tokens[index] = outgoing.book.Acquire(frame->id);
      stage.set_output_zero_copy(0, outgoing.Producer(frame->tokens[index]));
    }
    if (index == 1 && proxy) {
      invocation.inspect_reader = [this, frame](tvm::runtime::TVMArgs args) {
        frame->ranges.input = CheckedRange(args[0].operator DLTensor*());
        frame->ranges.reader_output = CheckedRange(args[1].operator DLTensor*());
        Require(Separate(frame->ranges.input, frame->ranges.reader_output),
                "last reader output overlaps the original input");
        if (!reference && options.variant == "C1") {
          Require(frame->ranges.copy_separate && SameRange(frame->ranges.input, frame->ranges.copy_destination) &&
                  Separate(frame->ranges.copy_source, frame->ranges.reader_output),
                  "C1 actual reader ABI does not match its independent private copy");
        }
        Emit("reader_ranges_verified", 1, frame->id);
      };
      invocation.inspect_next = [this, frame](tvm::runtime::TVMArgs args) {
        frame->ranges.next_input = CheckedRange(args[0].operator DLTensor*());
        frame->ranges.packed_output = CheckedRange(args[1].operator DLTensor*());
        Require(SameRange(frame->ranges.reader_output, frame->ranges.next_input),
                "packing input is not the actual last-reader output");
        Require(Separate(frame->ranges.input, frame->ranges.packed_output),
                "packed output overlaps the original input");
        Require(Separate(frame->ranges.next_input, frame->ranges.packed_output),
                "packing source and destination unexpectedly alias");
        if (!reference && options.variant == "C1")
          Require(Separate(frame->ranges.copy_source, frame->ranges.packed_output),
                  "packed output aliases the already-returned C1 source");
        frame->ranges.verified = true; Emit("dynamic_ranges_verified", 1, frame->id);
      };
      invocation.before_reader = [this, frame, &invocation] {
        Emit("last_reader_start", 1, frame->id, reference ? nullptr : &invocation.token);
        if (options.mode == "event-before" && frame->id == 0) {
          Wait([&] { return producer_blocked; }, "producer did not block before reader");
          Emit("before_reader_producer_blocked", 1, frame->id, &invocation.token);
        }
      };
      invocation.after_reader = [this, frame, &invocation] {
        Emit("last_reader_return", 1, frame->id, reference ? nullptr : &invocation.token);
      };
      invocation.before_next = [this, frame, &invocation] {
        Require(frame->ranges.verified, "cannot release before both CPU ABI ranges are validated");
        if (!reference && options.variant == "E1") {
          Emit("validated_release", 1, frame->id, &invocation.token);
          ReturnLease(&managers[0]->book, &invocation);
          if (options.mode == "event-after" && frame->id == 0) {
            Wait([&] { return next_produced; }, "producer did not overwrite after reader");
            Emit("after_reader_next_frame_published", 1, frame->id, &invocation.token);
          }
        }
        Emit("node2_execute", 1, frame->id);
      };
      Require(proxy->active == nullptr, "overlapping consumer graph calls"); proxy->active = &invocation;
    }
    frame->run_begin[index] = NowMillis();
    try {
      if (index == 1) { std::lock_guard<std::mutex> lock(device_mutex); stage.run(); }
      else stage.run();
    } catch (...) { if (index == 1 && proxy) proxy->active = nullptr; throw; }
    frame->run_end[index] = NowMillis(); ++stage_calls[index];
    if (index == 1 && proxy) {
      proxy->active = nullptr;
      Require(invocation.target_calls == 1 && invocation.next_calls == 1 && frame->ranges.verified,
              "reader/packing call count or actual ABI proof mismatch");
    }
    if (reference && index < 2) stage.get_output(0, reference_edges[index]);
    else if (!reference && index < 2) managers[index]->book.Publish(frame->tokens[index]);
    if (index == 2) stage.get_output(0, frame->output);
    if (!reference && index > 0 && !invocation.released) ReturnLease(&managers[index - 1]->book, &invocation);
    Emit("stage_end", index, frame->id);
    if (index == 2) {
      frame->completion = NowMillis(); frame->complete = true;
      std::lock_guard<std::mutex> lock(control_mutex); ++completed; control.notify_all();
    }
  }

  void Actor(int index) {
    try {
      {
        std::lock_guard<std::mutex> lock(control_mutex); ++arrived; control.notify_all();
      }
      // All caller threads must exist before attributing newly created TLS pool
      // workers to one actor by the /proc task-set difference.
      Wait([&] { return arrived == 3 && init_turn == index; }, "actor initialization order timeout");
      auto before = ProcessThreadIds();
      SelfAffinity(stage_args[index].cpu_affinity);
      ConfigureThreadPool(stage_args[index].runtime_num_threads, stage_args[index].cpu_affinity,
                          "c3.actor" + std::to_string(index));
      SelfAffinity(stage_args[index].cpu_affinity);
      const PackedFunc* num_threads = Registry::Get("runtime.NumThreads");
      Require(num_threads != nullptr, "actual runtime thread count unavailable");
      actual_threads[index] = (*num_threads)().operator int();
      Require(actual_threads[index] == stage_args[index].runtime_num_threads,
              "actual actor TVM thread count differs from the frozen contract");
      auto after = ProcessThreadIds();
      auto& record = affinity[index]; record.caller_tid = syscall(SYS_gettid);
      std::set_difference(after.begin(), after.end(), before.begin(), before.end(), std::back_inserter(record.thread_ids));
      record.thread_ids.push_back(record.caller_tid);
      CaptureThreadAffinities(&record); CheckAffinity(record, stage_args[index].cpu_affinity);
      actual_cpu[index][0] = sched_getcpu();
      {
        std::lock_guard<std::mutex> lock(control_mutex); ++ready; ++init_turn; control.notify_all();
      }
      Wait([&] { return start; }, "actor start timeout");
      if (index == 0) {
        for (size_t i = 0; i < frames.size(); ++i) {
          if (i == static_cast<size_t>(warmup)) Wait([&] { return score_start; }, "score start timeout");
          if (reference && i) Wait([&] { return completed == static_cast<int>(i); }, "reference drain timeout");
          FrameRecord* frame = &frames[i]; frame->enqueue = NowMillis();
          RunOne(index, frame); queues[0].Push(frame);
        }
        queues[0].Close();
      } else {
        while (FrameRecord* frame = queues[index - 1].Pop()) {
          RunOne(index, frame); if (index == 1) queues[1].Push(frame);
        }
        if (index == 1) queues[1].Close();
      }
      actual_cpu[index][1] = sched_getcpu();
      // Keep the actor and its TLS workers alive for the final affinity audit.
      Wait([&] { return finish; }, "final actor audit timeout");
    } catch (...) { Fail(std::current_exception()); }
  }
  void Run() {
    std::array<std::thread, 3> workers;
    try {
      for (int i = 0; i < 3; ++i) workers[i] = std::thread([this, i] { Actor(i); });
      Wait([&] { return ready == 3; }, "actors did not initialize");
      { std::lock_guard<std::mutex> lock(control_mutex); start = true; control.notify_all(); }
      Wait([&] { return completed == warmup; }, "warmup did not drain");
      for (auto& manager : managers) if (manager) manager->book.AllFree();
      for (int i = 0; i < 3; ++i) { CaptureThreadAffinities(&affinity[i]); CheckAffinity(affinity[i], stage_args[i].cpu_affinity); }
      Snapshot("post_warmup");
      formal_start = NowMillis(); Emit("score_begin", -1, -1);
      { std::lock_guard<std::mutex> lock(control_mutex); score_start = true; control.notify_all(); }
      Wait([&] { return completed == static_cast<int>(frames.size()); }, "frames did not drain");
      formal_end = NowMillis(); Emit("score_end", -1, -1);
      for (auto& manager : managers) if (manager) manager->book.AllFree();
      for (int i = 0; i < 3; ++i) { CaptureThreadAffinities(&affinity[i]); CheckAffinity(affinity[i], stage_args[i].cpu_affinity); }
      Snapshot("post_measure");
      { std::lock_guard<std::mutex> lock(control_mutex); finish = true; control.notify_all(); }
    } catch (...) { Fail(std::current_exception()); }
    for (auto& worker : workers) if (worker.joinable()) worker.join();
    if (error) std::rethrow_exception(error);
    for (uint64_t count : stage_calls) Require(count == frames.size(), "unexpected stage invocation count");
    if (proxy) Require(proxy->calls == frames.size() && proxy->next_calls == frames.size(),
                       "unexpected total target/packing count");
  }
  void Save(const std::string& status, const std::string& failure = "") {
    Require(!std::filesystem::exists(options.output + "/result.json"), "result file already exists");
    Require(!std::filesystem::exists(options.output + "/outputs.bin"), "deferred output file already exists");
    EnsureDir(options.output);
    std::ofstream outputs(options.output + "/outputs.bin", std::ios::binary);
    Require(static_cast<bool>(outputs), "cannot write deferred outputs");
    auto range_json = [](const BufferRange& range) {
      Object value;
      value["handle"] = JSON("0x" + UInt64Hex(range.handle));
      value["cpu_address"] = JSON("0x" + UInt64Hex(range.cpu));
      value["physical_address"] = JSON("0x" + UInt64Hex(range.physical));
      value["bytes"] = J(range.bytes);
      return JSON(value);
    };
    Array records, range_evidence; uint64_t offset = 0;
    bool all_ranges_verified = static_cast<bool>(proxy) && !frames.empty();
    for (const auto& frame : frames) {
      Object value; value["frame_id"] = J(frame.id); value["input_index"] = J(frame.input);
      value["warmup"] = JSON(frame.warmup); value["complete"] = JSON(frame.complete);
      value["enqueue_ms"] = JSON(frame.enqueue); value["completion_ms"] = JSON(frame.completion);
      if (frame.complete) {
        size_t bytes = TensorNBytes(frame.output.operator->());
        outputs.write(reinterpret_cast<const char*>(TensorDataBytes(frame.output.operator->())), bytes);
        value["output_file"] = JSON("outputs.bin"); value["output_offset"] = J(offset);
        value["output_bytes"] = J(bytes); value["output_dtype"] = JSON(DTypeString(frame.output->dtype));
        offset += bytes;
      }
      value["copy_ms"] = JSON(frame.copy_ms);
      Array begin, end; for (int i = 0; i < 3; ++i) { begin.push_back(JSON(frame.run_begin[i])); end.push_back(JSON(frame.run_end[i])); }
      value["stage_run_begin_ms"] = JSON(begin); value["stage_run_end_ms"] = JSON(end);
      records.emplace_back(value);
      all_ranges_verified &= frame.complete && frame.ranges.verified &&
          (reference || options.variant != "C1" || frame.ranges.copy_separate);
      if (proxy) {
        Object proof; proof["frame_id"] = J(frame.id); proof["verified"] = JSON(frame.ranges.verified);
        proof["input"] = range_json(frame.ranges.input);
        proof["reader_output"] = range_json(frame.ranges.reader_output);
        proof["next_input"] = range_json(frame.ranges.next_input);
        proof["packed_output"] = range_json(frame.ranges.packed_output);
        proof["copy_source"] = frame.ranges.copy_separate ? range_json(frame.ranges.copy_source) : JSON();
        proof["copy_destination"] = frame.ranges.copy_separate ? range_json(frame.ranges.copy_destination) : JSON();
        proof["copy_source_private_separate"] = frame.ranges.copy_separate ? JSON(true) : JSON();
        range_evidence.emplace_back(proof);
      }
    }
    outputs.close(); Require(static_cast<bool>(outputs), "deferred output write failed");
    Array trace;
    for (size_t i = 0; i < std::min(event_count.load(), events.size()); ++i) {
      const auto& event = events[i]; if (!event.name) continue;
      Object item; item["event"] = JSON(event.name); item["stage"] = J(event.stage); item["frame"] = J(event.frame);
      item["edge"] = J(event.edge); item["slot"] = J(event.slot); item["generation"] = J(event.generation);
      item["time_ms"] = JSON(event.time); item["sequence"] = J(i); trace.emplace_back(item);
    }
    Array masks;
    for (int i = 0; i < 3; ++i) {
      Object item; item["stage"] = J(i); item["caller_tid"] = J(static_cast<uint64_t>(affinity[i].caller_tid));
      Array workers;
      for (size_t k = 0; k < affinity[i].thread_ids.size() && k < affinity[i].effective_affinities.size(); ++k) {
        Object worker; worker["tid"] = J(static_cast<uint64_t>(affinity[i].thread_ids[k])); Array cpus;
        for (unsigned int cpu : affinity[i].effective_affinities[k]) cpus.push_back(J(static_cast<int>(cpu)));
        worker["cpus"] = JSON(cpus); workers.emplace_back(worker);
      }
      item["workers"] = JSON(workers); item["actual_cpu_initial"] = J(actual_cpu[i][0]);
      item["actual_cpu_final"] = J(actual_cpu[i][1]); item["runtime_threads"] = J(actual_threads[i]);
      Array requested; for (unsigned int cpu : stage_args[i].cpu_affinity) requested.push_back(J(static_cast<int>(cpu)));
      item["requested_cpus"] = JSON(requested); masks.emplace_back(item);
    }
    Array counts; for (uint64_t count : stage_calls) counts.push_back(J(count));
    Object result; result["schema"] = JSON("c3-input-last-read-result-v1"); result["status"] = JSON(status);
    result["error"] = failure.empty() ? JSON() : JSON(failure); result["variant"] = JSON(options.variant);
    result["mode"] = JSON(options.mode); result["warmup_count"] = J(warmup); result["frame_count"] = J(options.frames);
    result["frames"] = JSON(records); result["events"] = JSON(trace); result["affinity"] = JSON(masks);
    result["allocation_snapshots"] = JSON(Array(snapshots.begin(), snapshots.end()));
    result["stage_call_counts"] = JSON(counts); result["target_call_count"] = proxy ? J(proxy->calls) : JSON();
    result["packed_call_count"] = proxy ? J(proxy->next_calls) : JSON();
    result["dynamic_ranges_verified"] = proxy ? JSON(all_ranges_verified) : JSON();
    result["dynamic_range_evidence"] = JSON(range_evidence);
    result["static_contract_verified"] = JSON(contract_verified);
    result["runtime_identities_verified"] = JSON(identities_verified);
    result["affinity_verified"] = JSON(status == "passed");
    result["loaded_libraries"] = JSON(loaded_libraries);
    uint64_t external_bytes = 0;
    if (managers[0]) for (const auto& slot : managers[0]->views)
      if (slot.allocation_owner) external_bytes += slot.nbytes;
    result["target_external_input_bytes"] = J(external_bytes);
    Array slot_counts; for (const auto& manager : managers) slot_counts.push_back(J(manager ? manager->views.size() : 0));
    result["edge_slot_counts"] = JSON(slot_counts);
    result["release_boundary"] = JSON("after_last_reader_return_and_next_node_ABI_validation_before_node2_execution");
    result["proxy_enabled"] = JSON(static_cast<bool>(proxy)); result["reference_serial"] = JSON(reference);
    result["formal_start_ms"] = JSON(formal_start); result["formal_end_ms"] = JSON(formal_end);
    result["boot_id"] = JSON(TrimLine(ReadFile("/proc/sys/kernel/random/boot_id")));
    std::ofstream out(options.output + "/result.json"); out << JSON(result).serialize(true) << '\n';
    Require(static_cast<bool>(out), "result write failed");
  }
};

static int SelfTest() {
  int tests = 0;
  auto rejects = [&](const std::function<void()>& action) {
    bool rejected = false; try { action(); } catch (const std::exception&) { rejected = true; }
    Require(rejected, "selftest expected rejection"); ++tests;
  };
  LeaseBook book(1, 0, 20);
  auto a = book.Acquire(0); rejects([&] { book.Release(a); }); book.Publish(a);
  rejects([&] { book.Publish(a); }); book.Consume(a); rejects([&] { book.Consume(a); });
  rejects([&] { book.Acquire(1); }); book.Release(a); rejects([&] { book.Release(a); });
  auto b = book.Acquire(1); Require(b.generation > a.generation, "generation not advanced"); ++tests;
  rejects([&] { book.Validate(a, LeaseBook::State::Writing); });
  auto wrong = b; wrong.edge_index = 1; rejects([&] { book.Publish(wrong); });
  book.Publish(b); book.Consume(b); book.Release(b); book.AllFree(); ++tests;
  LeaseBook pair(2, 0); auto x = pair.Acquire(0); auto y = pair.Acquire(1);
  Require(x.slot_id != y.slot_id, "two-slot allocator reused active slot"); ++tests;
  pair.Abort(); rejects([&] { pair.Publish(x); });
  LeaseBook aborted(1, 0); aborted.Abort(); rejects([&] { aborted.Acquire(0); });
  LeaseBook returns(1, 0);
  Invocation invocation; invocation.token = returns.Acquire(7);
  rejects([&] { returns.AllFree(); });
  returns.Publish(invocation.token); returns.Consume(invocation.token);
  ReturnLease(&returns, &invocation);
  Require(invocation.released, "invocation return state not retained"); ++tests;
  rejects([&] { ReturnLease(&returns, &invocation); }); returns.AllFree();
  // Test the exact CPU/physical overlap predicate used by actual ABI guards.
  BufferRange first{1, 0x1000, 0x6000, 256}, second{2, 0x1100, 0x6100, 256};
  Require(Separate(first, second) && SameRange(first, first), "disjoint adjacent ranges rejected"); ++tests;
  second.cpu = 0x1080; Require(!Separate(first, second), "CPU overlap admitted"); ++tests;
  second.cpu = 0x1100; second.physical = 0x6080;
  Require(!Separate(first, second), "physical alias admitted"); ++tests;
  Require(!SameRange(first, second), "different physical range considered identical"); ++tests;
  // A real blocked Acquire cannot overwrite a live lease; Release wakes it.
  LeaseBook threaded(1, 0, 2000);
  auto original = threaded.Acquire(0); threaded.Publish(original); threaded.Consume(original);
  std::mutex gate_mutex; std::condition_variable gate; bool waiting = false;
  std::atomic<bool> acquired{false}; std::exception_ptr thread_error;
  threaded.observer = [&](const char* name, const P8SlotToken&) {
    if (std::strcmp(name, "producer_wait") == 0) {
      std::lock_guard<std::mutex> lock(gate_mutex); waiting = true; gate.notify_all();
    }
  };
  std::thread writer([&] {
    try {
      auto next = threaded.Acquire(1); acquired = true;
      threaded.Publish(next); threaded.Consume(next); threaded.Release(next);
    } catch (...) { thread_error = std::current_exception(); }
  });
  bool observed_wait;
  {
    std::unique_lock<std::mutex> lock(gate_mutex);
    observed_wait = gate.wait_for(lock, std::chrono::seconds(1), [&] { return waiting; });
  }
  const bool blocked_while_reading = observed_wait && !acquired;
  threaded.Release(original); writer.join();
  if (thread_error) std::rethrow_exception(thread_error);
  Require(blocked_while_reading && acquired, "producer did not respect actual lease boundary"); ++tests;
  threaded.AllFree(); ++tests;
  std::cout << "{\"schema\":\"c3-last-read-cpp-selftest-v1\",\"status\":\"PASS\",\"tests\":" << tests << "}\n";
  return 0;
}
}  // namespace c3_last_read

int main(int argc, char** argv) {
  using namespace c3_last_read;
#ifdef C3_LAST_READ_SELFTEST_ONLY
  return SelfTest();
#else
  std::unique_ptr<Experiment> experiment;
  try {
    auto options = Parse(argc, argv);
    if (options.mode == "selftest") return SelfTest();
    experiment = std::make_unique<Experiment>(std::move(options));
    experiment->Prepare(); experiment->Run(); experiment->Save("passed");
    return 0;
  } catch (const std::exception& error) {
    if (experiment && !experiment->options.output.empty()) {
      try { experiment->Save("INVALID", error.what()); } catch (...) {}
    }
    std::cerr << "c3_input_early_release_runner: " << error.what() << '\n'; return 1;
  }
#endif
}
