#include "../../apps/native_deploy/vta_command_capacity_certificate.h"

#include <filesystem>
#include <iostream>
#include <string>
#include <unistd.h>

int main(int argc, char** argv) {
  try {
    if (argc != 3) {
      std::cerr << "usage: thesis_command_capacity_probe CERTIFICATE ACTUAL_PROGRAM_JSON\n";
      return 2;
    }
    auto json = shared_buffer::Parse(shared_buffer::Read(argv[2]));
    command_capacity::ProgramIdentity actual;
    for (const auto& stage : shared_buffer::As<shared_buffer::Array>(
             command_capacity::Field(json, "stages"))) {
      command_capacity::StageIdentity identity;
      identity.name = command_capacity::Text(command_capacity::Field(stage, "name"));
      identity.device = command_capacity::Text(command_capacity::Field(stage, "device"));
      identity.graph = command_capacity::Text(command_capacity::Field(stage, "graph"));
      identity.lib = command_capacity::Text(command_capacity::Field(stage, "lib"));
      identity.params = command_capacity::Text(command_capacity::Field(stage, "params"));
      for (const auto& name : shared_buffer::As<shared_buffer::Array>(
               command_capacity::Field(stage, "input_names"))) {
        identity.input_names.push_back(command_capacity::Text(name));
      }
      actual.stages.push_back(std::move(identity));
    }
    char executable[4096];
    ssize_t length = readlink("/proc/self/exe", executable, sizeof(executable) - 1);
    if (length <= 0 || static_cast<size_t>(length) >= sizeof(executable) - 1) {
      throw std::runtime_error("cannot resolve probe executable");
    }
    executable[length] = '\0';
    actual.runner = executable;
    actual.runtime = command_capacity::Text(command_capacity::Field(json, "runtime"));
    actual.driver = command_capacity::Text(command_capacity::Field(json, "driver"));
    actual.bitstream = command_capacity::Text(command_capacity::Field(json, "bitstream"));
    actual.shared_buffer_plan =
        command_capacity::Text(command_capacity::Field(json, "shared_buffer_plan"));
    actual.vta_max_xfer_bytes =
        command_capacity::Number(command_capacity::Field(json, "vta_max_xfer_bytes"));
    auto result = command_capacity::Validate(argv[1], actual);
    std::cout << "{\"status\":\"accepted\",\"certificate_sha256\":\""
              << result.certificate_sha256 << "\",\"program_id\":\""
              << result.program_id << "\",\"insn_capacity_bytes\":"
              << result.insn_capacity_bytes << ",\"uop_capacity_bytes\":"
              << result.uop_capacity_bytes << "}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 86;
  }
}
