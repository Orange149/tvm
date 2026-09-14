// Experimental fixed-K2 input-pool reuse. Not a general memory planner.
#ifndef VTA_APPS_NATIVE_DEPLOY_VTA_SHARED_BUFFER_PLAN_H_
#define VTA_APPS_NATIVE_DEPLOY_VTA_SHARED_BUFFER_PLAN_H_

#include "../../../3rdparty/picojson/picojson.h"
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace shared_buffer {
using JSON = picojson::value;
using Array = picojson::array;

inline void Require(bool condition, const std::string& reason) {
  if (!condition) throw std::runtime_error("shared-buffer: " + reason);
}
template <typename T> const T& As(const JSON& v) {
  Require(v.is<T>(), "invalid JSON field type");
  return v.get<T>();
}
inline const JSON& Field(const JSON& v, const std::string& name) {
  return As<picojson::object>(v).at(name);
}
inline uint64_t Number(const JSON& v) {
  double n = As<double>(v);
  Require(std::isfinite(n) && n >= 0 && n <= 9007199254740991.0 && std::floor(n) == n,
          "invalid nonnegative integer");
  return static_cast<uint64_t>(n);
}
inline std::string Text(const JSON& v) { return As<std::string>(v); }
inline JSON Parse(const std::string& text) {
  JSON v;
  std::string error = picojson::parse(v, text);
  Require(error.empty(), "invalid JSON: " + error);
  return v;
}
inline std::string Read(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  Require(static_cast<bool>(in), "cannot read " + path);
  std::ostringstream out;
  out << in.rdbuf();
  Require(!in.bad(), "read failed " + path);
  return out.str();
}

// Run sha256sum before any worker/thread pool is created. No shell interpolation.
// This checks the actual command-line artifacts, not arbitrary paths from a plan.
inline std::string FileSHA256(const std::string& path) {
  int pipefd[2];
  Require(pipe(pipefd) == 0, "hash pipe failed");
  pid_t child = fork();
  if (child < 0) { close(pipefd[0]); close(pipefd[1]); Require(false, "hash fork failed"); }
  if (child == 0) {
    close(pipefd[0]);
    if (dup2(pipefd[1], STDOUT_FILENO) < 0) _exit(126);
    close(pipefd[1]);
    execlp("sha256sum", "sha256sum", "--", path.c_str(), static_cast<char*>(nullptr));
    _exit(127);
  }
  close(pipefd[1]);
  char buffer[512];
  std::string output;
  bool read_ok = true;
  for (;;) {
    ssize_t n = read(pipefd[0], buffer, sizeof(buffer));
    if (n > 0) { if (output.size() < 128) output.append(buffer, n); }
    else if (n == 0) break;
    else if (errno != EINTR) { read_ok = false; break; }
  }
  close(pipefd[0]);
  int status = 0;
  pid_t waited;
  do { waited = waitpid(child, &status, 0); } while (waited < 0 && errno == EINTR);
  Require(read_ok && waited == child && WIFEXITED(status) && WEXITSTATUS(status) == 0 &&
          output.size() >= 65, "sha256sum failed: " + path);
  std::string hash = output.substr(0, 64);
  Require(hash.find_first_not_of("0123456789abcdef") == std::string::npos,
          "invalid SHA-256 output");
  return hash;
}
inline void CheckHash(const std::string& path, const JSON& expected) {
  Require(FileSHA256(path) == Text(expected), "hash mismatch: " + path);
}

// Decode only the name prefix, not NDArrays: no accelerator allocation to audit params.
inline std::set<std::string> ParameterNames(const std::string& blob) {
  size_t pos = 0;
  auto u64 = [&]() {
    Require(pos <= blob.size() && blob.size() - pos >= 8, "truncated params prefix");
    uint64_t value = 0;
    for (int i = 0; i < 8; ++i) value |= uint64_t(static_cast<unsigned char>(blob[pos++])) << (8 * i);
    return value;
  };
  Require(u64() == UINT64_C(0xF7E58D4F05049CB7), "invalid params magic");
  (void)u64();
  uint64_t count = u64();
  Require(count <= blob.size() / 8, "invalid parameter count");
  std::set<std::string> names;
  for (uint64_t i = 0; i < count; ++i) {
    uint64_t size = u64();
    Require(size <= blob.size() - pos, "invalid parameter name size");
    Require(names.insert(blob.substr(pos, size)).second, "duplicate parameter name");
    pos += size;
  }
  Require(u64() == count, "parameter array/name count mismatch");
  return names;
}

// Recheck graph semantics at startup rather than trusting a serialized eligible flag.
inline void CheckExclusiveInput(const JSON& graph, const std::string& name,
                                uint64_t expected_eid, uint64_t expected_sid,
                                const std::set<std::string>& parameters) {
  Require(parameters.count(name) == 0, "parameter cannot be an anchor");
  const auto& nodes = As<Array>(Field(graph, "nodes"));
  const auto& row = As<Array>(Field(graph, "node_row_ptr"));
  const auto& attrs = Field(graph, "attrs");
  const auto& ids = As<Array>(As<Array>(Field(attrs, "storage_id")).at(1));
  Require(row.size() == nodes.size() + 1 && Number(row.back()) == ids.size(), "entry table mismatch");
  size_t found = nodes.size(), matches = 0;
  for (size_t n = 0; n < nodes.size(); ++n) {
    if (Text(Field(nodes[n], "name")) == name) { found = n; ++matches; }
  }
  Require(matches == 1, "input node is not unique");
  Require(Text(Field(nodes[found], "op")) == "null" &&
          As<Array>(Field(nodes[found], "inputs")).empty(), "anchor is not a plain input");
  Require(Number(row[found]) == expected_eid && Number(row[found + 1]) == expected_eid + 1,
          "input entry mismatch");
  bool is_arg = false;
  for (const auto& arg : As<Array>(Field(graph, "arg_nodes"))) is_arg |= Number(arg) == found;
  Require(is_arg, "anchor is not an argument");
  Require(Number(ids.at(expected_eid)) == expected_sid, "storage-id mismatch");
  size_t alias_count = 0;
  for (const auto& sid : ids) alias_count += Number(sid) == expected_sid;
  Require(alias_count == 1, "internal storage alias: unsafe across frames");
  const auto& attr_obj = As<picojson::object>(attrs);
  if (attr_obj.count("device_index")) {
    Require(Number(As<Array>(As<Array>(Field(attrs, "device_index")).at(1)).at(expected_eid)) == 12,
            "anchor is not ext_dev");
  }
  if (attr_obj.count("storage_scope")) {
    std::string scope = Text(As<Array>(As<Array>(Field(attrs, "storage_scope")).at(1)).at(expected_eid));
    Require(scope.empty() || scope == "global", "unsupported storage scope");
  }
  auto entry = [&](const JSON& reference) {
    const auto& ref = As<Array>(reference);
    Require(ref.size() == 3 && Number(ref[2]) == 0, "unsupported entry reference");
    size_t nid = Number(ref[0]), index = Number(ref[1]);
    Require(nid < nodes.size() && Number(row[nid + 1]) >= Number(row[nid]) &&
            index < Number(row[nid + 1]) - Number(row[nid]), "entry reference out of range");
    return Number(row[nid]) + index;
  };
  for (const auto& head : As<Array>(Field(graph, "heads")))
    Require(entry(head) != expected_eid, "input also exported as output");
  size_t uses = 0;
  for (const auto& node : nodes) {
    for (const auto& ref : As<Array>(Field(node, "inputs"))) {
      if (entry(ref) == expected_eid) {
        ++uses;
        Require(Text(Field(node, "op")) == "tvm_op", "input use not rebound by GraphExecutor");
        auto function = Text(Field(Field(node, "attrs"), "func_name"));
        Require(!function.empty() && function != "__nop", "unproven no-op alias use");
      }
    }
  }
  Require(uses > 0, "unused input");
}

inline uint64_t CheckPhysicalRange(uint64_t address, uint64_t bytes, const uint64_t* snapshot) {
  Require(snapshot[0] == 1 && snapshot[2] <= snapshot[1], "pool not initialized");
  Require(bytes > 0 && address % 256 == 0, "unaligned or empty anchor");
  uint64_t base = snapshot[6], used = snapshot[2], phys = snapshot[7];
  Require(address >= base && address - base <= used && bytes <= used - (address - base),
          "anchor outside allocated pool range");
  uint64_t offset = address - base;
  Require(phys <= UINT64_MAX - offset && phys + offset <= UINT64_MAX - bytes,
          "physical range overflow");
  Require((phys + offset) % 256 == 0, "unaligned physical anchor");
  return phys + offset;
}
inline bool Overlaps(uint64_t a, uint64_t n, uint64_t b, uint64_t m) {
  // Subtraction form avoids endpoint overflow.
  return a <= b ? b - a < n : a - b < m;
}
}  // namespace shared_buffer
#endif
