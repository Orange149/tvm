#ifndef VTA_DEPLOY_HOST_PRIORITY_H_
#define VTA_DEPLOY_HOST_PRIORITY_H_
#include <cerrno>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>
namespace vta_deploy {
inline long HostPriorityTid() { return static_cast<long>(syscall(SYS_gettid)); }
inline int ReadHostNice(long tid) {
  if (tid <= 0) throw std::runtime_error("Invalid owned thread ID");
  errno = 0;
  const int value = getpriority(PRIO_PROCESS, static_cast<id_t>(tid));
  if (errno) throw std::runtime_error("getpriority failed for thread " + std::to_string(tid));
  return value;
}
inline void ApplyOwnedHostNice(int requested) {
  if (requested != 0 && requested != -10)
    throw std::runtime_error("Host priority protocol permits only 0 or -10");
  const long tid = HostPriorityTid();
  if (ReadHostNice(tid) != 0) throw std::runtime_error("Host worker inherited unexpected nice");
  if (setpriority(PRIO_PROCESS, static_cast<id_t>(tid), requested) != 0)
    throw std::runtime_error("Unable to set owned VTA host nice: errno " + std::to_string(errno));
  if (ReadHostNice(tid) != requested) throw std::runtime_error("Actual VTA host nice mismatch");
}
}  // namespace vta_deploy
#endif
