#include "vta_shared_buffer_plan.h"
#include <functional>
#include <iostream>

using namespace shared_buffer;
int checks = 0;
void Reject(const std::function<void()>& fn) {
  bool rejected = false;
  try { fn(); } catch (const std::exception&) { rejected = true; }
  Require(rejected, "expected rejection");
  ++checks;
}
void U64(std::string* s, uint64_t n) {
  for (int i = 0; i < 8; ++i) s->push_back(static_cast<char>((n >> (8 * i)) & 255));
}
int main(int argc, char** argv) {
  try {
    Require(argc == 3, "test needs path and reference SHA256");
    CheckHash(argv[1], JSON(std::string(argv[2]))); ++checks;
    Reject([&]() { CheckHash(argv[1], JSON(std::string(64, '0'))); });
    auto graph = Parse(R"({"nodes":[{"name":"data0","op":"null","inputs":[]},
      {"name":"compute","op":"tvm_op","inputs":[[0,0,0]],"attrs":{"func_name":"fused_op"}}],
      "node_row_ptr":[0,1,2],"arg_nodes":[0],"heads":[[1,0,0]],
      "attrs":{"storage_id":["list_int",[0,1]],"device_index":["list_int",[12,12]]}})");
    CheckExclusiveInput(graph, "data0", 0, 0, {}); ++checks;
    Reject([&]() { CheckExclusiveInput(graph, "data0", 0, 0, {"data0"}); });
    Reject([&]() { CheckExclusiveInput(graph, "data0", 0, 1, {}); });
    Reject([&]() { CheckExclusiveInput(graph, "missing", 0, 0, {}); });
    auto alias = graph;
    alias.get<picojson::object>()["attrs"].get<picojson::object>()["storage_id"].get<Array>()[1].get<Array>()[1] = JSON(0.0);
    Reject([&]() { CheckExclusiveInput(alias, "data0", 0, 0, {}); });
    auto exported = graph;
    exported.get<picojson::object>()["heads"] = Parse("[[0,0,0]]");
    Reject([&]() { CheckExclusiveInput(exported, "data0", 0, 0, {}); });
    auto nop = graph;
    nop.get<picojson::object>()["nodes"].get<Array>()[1].get<picojson::object>()["attrs"].get<picojson::object>()["func_name"] = JSON(std::string("__nop"));
    Reject([&]() { CheckExclusiveInput(nop, "data0", 0, 0, {}); });
    uint64_t pool[8] = {1, 4096, 2048, 2048, 0, 2, 0x1000, 0x8000};
    Require(CheckPhysicalRange(0x1100, 256, pool) == 0x8100, "physical translation"); ++checks;
    Reject([&]() { CheckPhysicalRange(0x1101, 256, pool); });
    Reject([&]() { CheckPhysicalRange(0xf00, 256, pool); });
    Reject([&]() { CheckPhysicalRange(0x1800, 256, pool); });
    Reject([&]() { CheckPhysicalRange(0x1700, 512, pool); });
    pool[7] = 0x8001;
    Reject([&]() { CheckPhysicalRange(0x1000, 256, pool); });
    pool[7] = UINT64_MAX - 255;
    Reject([&]() { CheckPhysicalRange(0x1100, 256, pool); });
    Require(Overlaps(100, 20, 110, 30) && !Overlaps(100, 20, 120, 30), "overlap boundary"); ++checks;
    std::string params;
    U64(&params, UINT64_C(0xF7E58D4F05049CB7)); U64(&params, 0); U64(&params, 1);
    U64(&params, 2); params += "p0"; U64(&params, 1);
    Require(ParameterNames(params) == std::set<std::string>{"p0"}, "params names"); ++checks;
    Reject([&]() { ParameterNames(params.substr(0, params.size() - 1)); });
    Reject([&]() { ParameterNames("invalid"); });
    Reject([&]() { Number(JSON(-1.0)); });
    Reject([&]() { Number(JSON(1.5)); });
    Reject([&]() { Parse("not JSON"); });
    std::cout << checks << " shared-buffer C++ checks passed\n";
    return 0;
  } catch (const std::exception& err) { std::cerr << err.what() << "\n"; return 1; }
}
