// Link-only host qualification guard. This supplies no FPGA memory or simulator.
// Every hardware allocation/address call terminates the host experiment.
#include <vta/driver.h>
#include <cstdio>
#include <cstdlib>

namespace {
[[noreturn]] void Forbidden(const char* function) {
  std::fprintf(stderr, "THESIS_FORBIDDEN_HOST_VTA_CALL:%s\n", function);
  std::_Exit(86);
}
}
extern "C" void* VTAMemAlloc(size_t, int) { Forbidden("VTAMemAlloc"); }
extern "C" void VTAMemFree(void*) { Forbidden("VTAMemFree"); }
extern "C" vta_phy_addr_t VTAMemGetPhyAddr(void*) { Forbidden("VTAMemGetPhyAddr"); }
