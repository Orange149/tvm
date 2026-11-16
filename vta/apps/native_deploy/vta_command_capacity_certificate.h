// Exact-program command-capacity preflight.  Validation performs no VTA allocation.
#ifndef VTA_APPS_NATIVE_DEPLOY_VTA_COMMAND_CAPACITY_CERTIFICATE_H_
#define VTA_APPS_NATIVE_DEPLOY_VTA_COMMAND_CAPACITY_CERTIFICATE_H_

#include "vta_shared_buffer_plan.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

namespace command_capacity {

struct StageIdentity {
  std::string name;
  std::string device;
  std::string graph;
  std::string lib;
  std::string params;
  std::vector<std::string> input_names;
};

struct ProgramIdentity {
  std::vector<StageIdentity> stages;
  std::string runner;
  std::string runtime;
  std::string driver;
  std::string bitstream;
  std::string shared_buffer_plan;
  uint64_t vta_max_xfer_bytes{0};
};

struct VerifiedCertificate {
  std::string certificate_sha256;
  std::string program_id;
  std::string evidence_scope;
  uint64_t insn_peak_bytes{0};
  uint64_t uop_peak_bytes{0};
  uint64_t insn_capacity_bytes{0};
  uint64_t uop_capacity_bytes{0};
};

inline void Require(bool condition, const std::string& reason) {
  if (!condition) throw std::runtime_error("command-capacity: " + reason);
}

inline const shared_buffer::JSON& Field(const shared_buffer::JSON& value,
                                        const std::string& name) {
  const auto& object = shared_buffer::As<picojson::object>(value);
  auto it = object.find(name);
  Require(it != object.end(), "missing field " + name);
  return it->second;
}

inline bool Bool(const shared_buffer::JSON& value) {
  Require(value.is<bool>(), "invalid boolean field");
  return value.get<bool>();
}

inline std::string Text(const shared_buffer::JSON& value) {
  Require(value.is<std::string>(), "invalid string field");
  return value.get<std::string>();
}

inline uint64_t Number(const shared_buffer::JSON& value) {
  Require(value.is<double>(), "invalid numeric field");
  double number = value.get<double>();
  Require(std::isfinite(number) && number >= 0 && std::floor(number) == number &&
              number <= 9007199254740991.0,
          "invalid nonnegative integer");
  return static_cast<uint64_t>(number);
}

inline std::string RequiredEnv(const char* name) {
  const char* value = std::getenv(name);
  Require(value != nullptr, std::string("missing environment ") + name);
  return value;
}

inline uint64_t CheckedMultiply(uint64_t left, uint64_t right) {
  Require(left == 0 || right <= std::numeric_limits<uint64_t>::max() / left,
          "capacity multiplication overflow");
  return left * right;
}

inline uint64_t AlignUp(uint64_t value, uint64_t alignment) {
  Require(value > 0 && alignment > 0 && value <= UINT64_MAX - (alignment - 1),
          "invalid capacity alignment");
  return ((value + alignment - 1) / alignment) * alignment;
}

inline std::filesystem::path BundleFile(const std::string& certificate_path,
                                        const std::string& name) {
  std::filesystem::path relative(name);
  Require(!name.empty() && !relative.is_absolute() && relative.filename() == relative &&
              name != "." && name != "..",
          "bundle member must be a plain filename");
  return std::filesystem::path(certificate_path).parent_path() / relative;
}

inline std::pair<std::string, std::vector<uint64_t>> GraphInputMetadata(
    const std::string& graph_path, const std::string& input_name) {
  using shared_buffer::Array;
  auto graph = shared_buffer::Parse(shared_buffer::Read(graph_path));
  const auto& nodes = shared_buffer::As<Array>(shared_buffer::Field(graph, "nodes"));
  const auto& rows = shared_buffer::As<Array>(shared_buffer::Field(graph, "node_row_ptr"));
  const auto& attrs = shared_buffer::Field(graph, "attrs");
  size_t node_index = nodes.size();
  size_t matches = 0;
  for (size_t i = 0; i < nodes.size(); ++i) {
    if (shared_buffer::Text(shared_buffer::Field(nodes[i], "name")) == input_name) {
      node_index = i;
      ++matches;
    }
  }
  Require(matches == 1 && rows.size() == nodes.size() + 1,
          "graph input is absent, duplicated, or has an invalid entry table");
  uint64_t entry = shared_buffer::Number(rows[node_index]);
  Require(shared_buffer::Number(rows[node_index + 1]) == entry + 1,
          "graph input must have exactly one entry");
  const auto& dtype_values = shared_buffer::As<Array>(
      shared_buffer::As<Array>(shared_buffer::Field(attrs, "dltype")).at(1));
  const auto& shape_values = shared_buffer::As<Array>(
      shared_buffer::As<Array>(shared_buffer::Field(attrs, "shape")).at(1));
  Require(entry < dtype_values.size() && entry < shape_values.size(),
          "graph input metadata entry is out of range");
  std::vector<uint64_t> shape;
  for (const auto& dimension : shared_buffer::As<Array>(shape_values.at(entry))) {
    shape.push_back(shared_buffer::Number(dimension));
  }
  return {shared_buffer::Text(dtype_values.at(entry)), shape};
}

inline void CheckFileHash(const std::string& path, const shared_buffer::JSON& expected,
                          const std::string& label) {
  Require(!path.empty(), "missing actual " + label + " path");
  Require(shared_buffer::FileSHA256(path) == Text(expected), label + " hash mismatch");
}

inline void CheckEnvironment(const shared_buffer::JSON& runtime,
                             const std::string& certificate_sha256) {
  const auto& environment = Field(runtime, "environment");
  const std::string insn = Text(Field(environment, "VTA_INSN_BUFFER_BYTES"));
  const std::string uop = Text(Field(environment, "VTA_UOP_BUFFER_BYTES"));
  Require(RequiredEnv("VTA_INSN_BUFFER_BYTES") == insn,
          "instruction capacity environment mismatch");
  Require(RequiredEnv("VTA_UOP_BUFFER_BYTES") == uop,
          "UOP capacity environment mismatch");
  Require(RequiredEnv("VTA_COMMAND_MANIFEST_ID") == certificate_sha256,
          "manifest ID must equal the certificate file SHA-256");
  Require(RequiredEnv("VTA_REPLAY_POLICY") == "disabled",
          "replay must be explicitly disabled");
  Require(std::getenv("VTA_INSN_SUBMIT_THRESHOLD_BYTES") == nullptr,
          "early instruction submission must be unset");
}

inline VerifiedCertificate Validate(const std::string& certificate_path,
                                    const ProgramIdentity& actual) {
  using shared_buffer::Array;
  Require(actual.vta_max_xfer_bytes > 0, "actual VTA_MAX_XFER is unavailable");
  const std::string certificate_sha256 = shared_buffer::FileSHA256(certificate_path);
  const auto certificate = shared_buffer::Parse(shared_buffer::Read(certificate_path));
  Require(Text(Field(certificate, "schema")) == "thesis_vta_command_capacity_certificate_v1",
          "unsupported certificate schema");
  const std::string scope = Text(Field(certificate, "evidence_scope"));
  Require(scope == "board_profiled_program" || scope == "synthetic_host_fixture",
          "unsupported evidence scope");
  if (scope == "synthetic_host_fixture") {
    Require(std::getenv("VTA_COMMAND_CERTIFICATE_ALLOW_SYNTHETIC") != nullptr &&
                std::string(std::getenv("VTA_COMMAND_CERTIFICATE_ALLOW_SYNTHETIC")) == "1",
            "synthetic certificate requires an explicit test-only gate");
  }
  Require(Number(Field(certificate, "vta_max_xfer_bytes")) == actual.vta_max_xfer_bytes,
          "VTA_MAX_XFER mismatch");

  const auto& stages = shared_buffer::As<Array>(Field(certificate, "stages"));
  Require(stages.size() == actual.stages.size(), "stage count mismatch");
  for (size_t i = 0; i < stages.size(); ++i) {
    const auto& expected = stages[i];
    const auto& observed = actual.stages[i];
    Require(Number(Field(expected, "index")) == i &&
                Text(Field(expected, "name")) == observed.name &&
                Text(Field(expected, "device")) == observed.device,
            "stage identity mismatch");
    CheckFileHash(observed.graph, Field(expected, "graph_sha256"), "stage graph");
    CheckFileHash(observed.lib, Field(expected, "lib_sha256"), "stage library");
    CheckFileHash(observed.params, Field(expected, "params_sha256"), "stage parameters");
    const auto& inputs = shared_buffer::As<Array>(Field(expected, "inputs"));
    Require(inputs.size() == observed.input_names.size(), "stage input arity mismatch");
    for (size_t n = 0; n < inputs.size(); ++n) {
      const auto& input = inputs[n];
      Require(Text(Field(input, "name")) == observed.input_names[n],
              "stage input name mismatch");
      auto metadata = GraphInputMetadata(observed.graph, observed.input_names[n]);
      Require(Text(Field(input, "dtype")) == metadata.first, "stage input dtype mismatch");
      const auto& shape = shared_buffer::As<Array>(Field(input, "shape"));
      Require(shape.size() == metadata.second.size(), "stage input rank mismatch");
      for (size_t d = 0; d < shape.size(); ++d) {
        Require(Number(shape[d]) == metadata.second[d], "stage input shape mismatch");
      }
    }
  }

  const auto& artifacts = Field(certificate, "artifacts");
  CheckFileHash(actual.runner, Field(artifacts, "runner_sha256"), "runner");
  CheckFileHash(actual.runtime, Field(artifacts, "runtime_sha256"), "runtime");
  CheckFileHash(actual.driver, Field(artifacts, "driver_sha256"), "driver");
  CheckFileHash(actual.bitstream, Field(artifacts, "bitstream_sha256"), "bitstream");
  const std::string shared_plan_hash = Text(Field(artifacts, "shared_buffer_plan_sha256"));
  if (shared_plan_hash == "none") {
    Require(actual.shared_buffer_plan.empty(), "unexpected shared-buffer plan");
  } else {
    Require(!actual.shared_buffer_plan.empty(), "missing shared-buffer plan");
    Require(shared_buffer::FileSHA256(actual.shared_buffer_plan) == shared_plan_hash,
            "shared-buffer plan hash mismatch");
  }

  const auto& profile_binding = Field(certificate, "profile");
  const auto profile_path = BundleFile(certificate_path, Text(Field(profile_binding, "file")));
  CheckFileHash(profile_path.string(), Field(profile_binding, "sha256"), "peak profile");
  const auto profile = shared_buffer::Parse(shared_buffer::Read(profile_path.string()));
  Require(Text(Field(profile, "schema")) == "thesis_vta_default_capacity_profile_v1",
          "unsupported peak profile schema");
  Require(Text(Field(profile, "aggregation_schema")) ==
              "thesis_vta_worker_queue_aggregation_v1" &&
              Number(Field(profile, "queue_instances")) > 0,
          "peak profile does not cover all worker queues");
  Require(Text(Field(profile, "program_binding_sha256")) ==
              Text(Field(certificate, "program_binding_sha256")),
          "peak profile program binding mismatch");
  Require(Bool(Field(profile, "finish_included")), "profile does not include FINISH");
  Require(Text(Field(profile, "replay_policy")) == "disabled",
          "profile replay policy is not disabled");
  Require(!Bool(Field(profile, "submit_threshold_configured")),
          "profile used early instruction submission");
  Require(Number(Field(profile, "insn_capacity_bytes")) == actual.vta_max_xfer_bytes &&
              Number(Field(profile, "uop_capacity_bytes")) == actual.vta_max_xfer_bytes,
          "profile was not collected with default queue capacities");
  Require(Number(Field(profile, "submissions")) > 0, "profile has no submissions");

  const auto& derivation = Field(certificate, "derivation");
  const uint64_t insn_peak = Number(Field(profile, "insn_peak_bytes"));
  const uint64_t uop_peak = Number(Field(profile, "uop_peak_bytes"));
  Require(Number(Field(derivation, "insn_peak_bytes")) == insn_peak &&
              Number(Field(derivation, "uop_peak_bytes")) == uop_peak,
          "certificate peak differs from its source profile");
  const uint64_t alignment = Number(Field(derivation, "alignment_bytes"));
  const uint64_t minimum = Number(Field(derivation, "minimum_bytes"));
  const uint64_t multiplier = Number(Field(derivation, "multiplier"));
  Require(alignment == 256 && minimum == 4096 &&
              (multiplier == 1 || multiplier == 2),
          "unsupported P-CAP parameters");
  const uint64_t insn_capacity =
      AlignUp(std::max(minimum, CheckedMultiply(insn_peak, multiplier)), alignment);
  const uint64_t uop_capacity =
      AlignUp(std::max(minimum, CheckedMultiply(uop_peak, multiplier)), alignment);
  Require(insn_capacity <= actual.vta_max_xfer_bytes &&
              uop_capacity <= actual.vta_max_xfer_bytes,
          "derived capacity exceeds VTA_MAX_XFER");
  Require(Number(Field(derivation, "insn_capacity_bytes")) == insn_capacity &&
              Number(Field(derivation, "uop_capacity_bytes")) == uop_capacity,
          "certificate capacity arithmetic mismatch");
  CheckEnvironment(Field(certificate, "runtime_contract"), certificate_sha256);

  VerifiedCertificate result;
  result.certificate_sha256 = certificate_sha256;
  result.program_id = Text(Field(certificate, "program_id"));
  result.evidence_scope = scope;
  result.insn_peak_bytes = insn_peak;
  result.uop_peak_bytes = uop_peak;
  result.insn_capacity_bytes = insn_capacity;
  result.uop_capacity_bytes = uop_capacity;
  return result;
}

}  // namespace command_capacity
#endif  // VTA_APPS_NATIVE_DEPLOY_VTA_COMMAND_CAPACITY_CERTIFICATE_H_
