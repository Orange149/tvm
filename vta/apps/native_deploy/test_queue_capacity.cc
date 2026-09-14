#include "../../runtime/queue_capacity.h"
#include <cassert>
#include <iostream>

int main() {
  using vta::queue_capacity::Parse;
  using vta::queue_capacity::RequireFits;
  using vta::queue_capacity::SubmitThresholdFromEnv;
  using vta::queue_capacity::ShouldSubmitForThreshold;
  constexpr size_t limit = 1 << 25;
  assert(Parse(nullptr, limit, limit, 256) == limit);
  assert(Parse("4096", limit, limit, 256) == 4096);
  assert(Parse("33554432", limit, limit, 256) == limit);
  for (const char* value : {"", "0", "-256", "+256", " 256", "256 ", "1", "4097",
                             "33554688", "9999999999999999999999999", "4KiB"}) {
    bool rejected = false;
    try { Parse(value, limit, limit, 256); } catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
  }
  RequireFits(4096, 4096);
  RequireFits(0, 4096);
  for (size_t value : {size_t(4097), size_t(4112), size_t(4096 + 6 * 16)}) {
    bool rejected = false;
    try { RequireFits(value, 4096); } catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);
  }
  std::cout << "19 queue capacity policy checks passed\n";
  bool configured = true;
  unsetenv("VTA_INSN_SUBMIT_THRESHOLD_BYTES");
  assert(SubmitThresholdFromEnv(limit, 16, &configured) == limit && !configured);
  setenv("VTA_INSN_SUBMIT_THRESHOLD_BYTES", "2656", 1);
  assert(SubmitThresholdFromEnv(limit, 16, &configured) == 2656 && configured);
  for (const char* value : {"111", "1120x", "2657", "33554448"}) {
    setenv("VTA_INSN_SUBMIT_THRESHOLD_BYTES", value, 1);
    bool rejected = false;
    try { SubmitThresholdFromEnv(limit, 16, &configured); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
  }
  unsetenv("VTA_INSN_SUBMIT_THRESHOLD_BYTES");
  std::cout << "6 submission-threshold parsing checks passed\n";
  assert(!ShouldSubmitForThreshold(2656, limit, false, limit));
  assert(!ShouldSubmitForThreshold(2656, limit, true, 2656));
  assert(ShouldSubmitForThreshold(2672, limit, true, 2656));
  assert(ShouldSubmitForThreshold(limit + 16, limit, false, limit));
  std::cout << "4 strict-threshold decision checks passed\n";
  using vta::queue_capacity::DependencyBalance;
  DependencyBalance a;
  assert(a.Closed());
  a.Add(1, false, false, false, true);
  assert(!a.Closed());
  a.Add(2, true, false, false, false);
  assert(a.Closed());
  a.Add(2, false, false, false, true);
  a.Add(3, true, false, true, false);
  assert(!a.Closed());
  a.Add(2, false, true, true, false);
  a.Add(1, false, true, false, false);
  assert(a.Closed());
  DependencyBalance external_pop;
  external_pop.Add(2, true, false, false, false);
  assert(!external_pop.Closed());
  DependencyBalance invalid;
  invalid.Add(1, true, false, false, false);
  assert(!invalid.Closed());
  std::cout << "7 necessary dependency-balance checks passed (not a scheduling proof)\n";
}
