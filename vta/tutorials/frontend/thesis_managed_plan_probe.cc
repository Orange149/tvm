// Parse the actual native CLI and lifecycle plan; never load a model or VTA device.
#define VTA_STAGE_PIPELINE_NO_MAIN
#include "vta_stage_pipeline_runner.cc"
int main(int argc, char** argv) {
  try {
    const Args args = ParseArgs(argc, argv);
    const auto options = LoadManagedLifecycleOptions(args);
    std::cout << "THESIS_NATIVE_LIFECYCLE_PLAN_VALID gates=" << options.gates.size()
              << " timeout=" << options.timeout_seconds << "\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "THESIS_NATIVE_LIFECYCLE_PLAN_REJECTED:" << error.what() << "\n";
    return 1;
  }
}
