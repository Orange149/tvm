#include "vta_driver_snapshot.h"
#include <cassert>
#include <sstream>
#include <thread>
VTADriverProfilerStats fake{};
extern "C" void VTADriverProfilerStatus(VTADriverProfilerStats* out) { *out=fake; }
int main() {
  vta_deploy::DriverRunSnapshot s;
  std::ostringstream off;s.WriteJSON(off);assert(off.str()=="null");
  s.Begin();
  fake.run_calls=15;fake.run_insns=3397;fake.poll_iters=100;
  fake.run_total_us=100;fake.submit_mmio_us=1;fake.post_start_sleep_us=10;fake.poll_wait_us=89;
  std::this_thread::sleep_for(std::chrono::milliseconds(2));
  s.End(vta_deploy::SnapshotWallMillis());
  assert(s.valid && s.delta.run_calls==15 && s.delta.run_insns==3397 && s.tid>0);
  assert(s.thread_cpu_ms>=0 && s.runqueue_ms>=0 && s.bracket_end_ms>s.bracket_start_ms);
  std::ostringstream json;s.WriteJSON(json);assert(json.str().find("\"driver_run_calls\":15")!=std::string::npos);
  bool rejected=false;try { vta_deploy::SnapshotCounterDelta(2,1); } catch (...) {rejected=true;}assert(rejected);
  rejected=false;try { vta_deploy::SnapshotDurationDelta(2,1); } catch (...) {rejected=true;}assert(rejected);
  rejected=false;try { vta_deploy::SnapshotDurationDelta(0,NAN); } catch (...) {rejected=true;}assert(rejected);
}
