#include <atomic>
#include <cassert>
#include <future>
#include <thread>
#include "vta_cpu_resource_groups.h"

int main() {
  using namespace std::chrono_literals;
  vta_deploy::CpuResourceGroups groups({{0,1},{0,1},{2,3},{}});
  auto first=groups.Acquire(0);
  std::promise<void> waiting;
  auto second=std::async(std::launch::async,[&]{ waiting.set_value();return groups.Acquire(1); });
  waiting.get_future().wait();
  assert(second.wait_for(20ms)==std::future_status::timeout);
  auto independent=std::async(std::launch::async,[&]{ return groups.Acquire(2); });
  assert(independent.wait_for(1s)==std::future_status::ready);
  auto other=independent.get();
  first.reset();
  assert(second.wait_for(1s)==std::future_status::ready);
  auto same=second.get();
  std::promise<void> entered;
  auto cancelled=std::async(std::launch::async,[&]{ entered.set_value();
    try { auto lease=groups.Acquire(0);return false; } catch(const std::runtime_error&) { return true; } });
  entered.get_future().wait();groups.Cancel();
  assert(cancelled.wait_for(1s)==std::future_status::ready && cancelled.get());
  bool bad=false;
  try { vta_deploy::CpuResourceGroups invalid({{0,1},{1,2}}); } catch(const std::invalid_argument&) { bad=true; }
  assert(bad);
}
