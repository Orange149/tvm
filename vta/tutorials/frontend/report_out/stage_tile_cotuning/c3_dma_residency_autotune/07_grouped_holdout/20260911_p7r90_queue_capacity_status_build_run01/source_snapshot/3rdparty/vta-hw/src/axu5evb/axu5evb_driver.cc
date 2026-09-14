/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0.
 *
 * \file axu5evb_driver.cc
 * \brief VTA driver for AXU5EVB/AXU4EVB-like boards using u-dma-buf.
 */

#include <vta/driver.h>

#include <assert.h>
#include <execinfo.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>

/*! \brief VTA configuration register start value */
#define VTA_START 0x1
/*! \brief VTA configuration register auto-restart value */
#define VTA_AUTORESTART 0x81
/*! \brief VTA configuration register done value */
#define VTA_DONE 0x1

namespace {

static const char* DefaultUdmabufDev() { return "/dev/udmabuf0"; }
static const char* DefaultUdmabufSysfsDir() { return "/sys/class/u-dma-buf/udmabuf0"; }
static const char* DefaultDevMemPath() { return "/dev/mem"; }

const char* GetEnvOrDefault(const char* key, const char* defval) {
  const char* v = getenv(key);
  return (v && v[0] != '\0') ? v : defval;
}

uint64_t GetEnvUInt64OrDefault(const char* key, uint64_t defval) {
  const char* v = getenv(key);
  if (!v || v[0] == '\0') return defval;
  char* end = nullptr;
  errno = 0;
  unsigned long long parsed = strtoull(v, &end, 0);
  if (errno != 0 || end == v || (end && end[0] != '\0')) {
    fprintf(stderr, "[axu5evb_driver][WARN ] invalid %s=%s; using default=%llu\n",
            key, v, static_cast<unsigned long long>(defval));
    return defval;
  }
  return static_cast<uint64_t>(parsed);
}

bool EnvEnabled(const char* key) {
  const char* v = getenv(key);
  if (!v || v[0] == '\0') return false;
  return strcmp(v, "0") != 0 && strcasecmp(v, "false") != 0 && strcasecmp(v, "off") != 0;
}

bool DebugEnabled() { return EnvEnabled("AXU5EVB_DRIVER_DEBUG"); }
bool InfoEnabled() { return DebugEnabled() || EnvEnabled("AXU5EVB_DRIVER_INFO"); }
bool DebugZeroOnAlloc() { return EnvEnabled("AXU5EVB_DRIVER_ZERO_ON_ALLOC"); }
bool DebugSafeCopy() {
  const char* v = getenv("AXU5EVB_DRIVER_SAFE_COPY");
  if (!v || v[0] == '\0') return true;
  return strcmp(v, "0") != 0 && strcasecmp(v, "false") != 0 && strcasecmp(v, "off") != 0;
}

double NowMicros() {
  using Clock = std::chrono::steady_clock;
  using Micros = std::chrono::duration<double, std::micro>;
  return std::chrono::duration_cast<Micros>(Clock::now().time_since_epoch()).count();
}

void SleepForNanos(uint64_t ns) {
  if (ns == 0) {
    std::this_thread::yield();
    return;
  }
  struct timespec ts;
  ts.tv_sec = static_cast<time_t>(ns / 1000000000ULL);
  ts.tv_nsec = static_cast<long>(ns % 1000000000ULL);
  nanosleep(&ts, nullptr);
}

void LogErr(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  fprintf(stderr, "[axu5evb_driver][ERROR] ");
  vfprintf(stderr, fmt, ap);
  fprintf(stderr, "\n");
  va_end(ap);
}

void LogWarn(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  fprintf(stderr, "[axu5evb_driver][WARN ] ");
  vfprintf(stderr, fmt, ap);
  fprintf(stderr, "\n");
  va_end(ap);
}

void LogInfo(const char* fmt, ...) {
  if (!InfoEnabled()) return;
  va_list ap;
  va_start(ap, fmt);
  fprintf(stderr, "[axu5evb_driver][INFO ] ");
  vfprintf(stderr, fmt, ap);
  fprintf(stderr, "\n");
  va_end(ap);
}

void LogDebug(const char* fmt, ...) {
  if (!DebugEnabled()) return;
  va_list ap;
  va_start(ap, fmt);
  fprintf(stderr, "[axu5evb_driver][DEBUG] ");
  vfprintf(stderr, fmt, ap);
  fprintf(stderr, "\n");
  va_end(ap);
}

void LogPreview(const char* prefix, const void* ptr, size_t size) {
  if (!DebugEnabled()) return;
  if (!ptr) {
    LogDebug("%s ptr=null size=%zu", prefix, size);
    return;
  }
  size_t n = size < 16 ? size : 16;
  const uint8_t* p = reinterpret_cast<const uint8_t*>(ptr);
  char buf[16 * 3 + 1];
  size_t pos = 0;
  for (size_t i = 0; i < n; ++i) {
    int written = snprintf(buf + pos, sizeof(buf) - pos, "%02x%s", p[i], (i + 1 < n) ? " " : "");
    if (written <= 0) break;
    pos += static_cast<size_t>(written);
    if (pos >= sizeof(buf)) break;
  }
  buf[sizeof(buf) - 1] = '\0';
  LogDebug("%s ptr=%p size=%zu preview[%zu]=%s", prefix, ptr, size, n, buf);
}

class DriverProfiler {
 public:
  static DriverProfiler& Global() {
    static DriverProfiler inst;
    return inst;
  }

  void Clear() {
    std::lock_guard<std::mutex> lock(mu_);
    stats_ = VTADriverProfilerStats();
  }

  void AddRun(uint32_t insn_count, bool timeout, uint64_t poll_iters, double run_total_us,
              double submit_mmio_us, double post_start_sleep_us, double poll_wait_us) {
    std::lock_guard<std::mutex> lock(mu_);
    stats_.run_calls += 1;
    stats_.run_insns += insn_count;
    stats_.timeout_calls += timeout ? 1 : 0;
    stats_.poll_iters += poll_iters;
    stats_.run_total_us += run_total_us;
    stats_.submit_mmio_us += submit_mmio_us;
    stats_.post_start_sleep_us += post_start_sleep_us;
    stats_.poll_wait_us += poll_wait_us;
  }

  VTADriverProfilerStats Snapshot() {
    std::lock_guard<std::mutex> lock(mu_);
    return stats_;
  }

 private:
  std::mutex mu_;
  VTADriverProfilerStats stats_{};
};

void CrashSignalHandler(int signo, siginfo_t* info, void* uctx) {
  (void)uctx;
  void* bt[64];
  int n = backtrace(bt, 64);
  LogErr("caught fatal signal=%d si_code=%d fault_addr=%p", signo,
         info ? info->si_code : 0, info ? info->si_addr : nullptr);
  backtrace_symbols_fd(bt, n, STDERR_FILENO);
  signal(signo, SIG_DFL);
  raise(signo);
}

void InstallCrashHandlersOnce() {
  static std::once_flag once;
  std::call_once(once, []() {
    struct sigaction sa;
    std::memset(&sa, 0, sizeof(sa));
    sa.sa_sigaction = CrashSignalHandler;
    sa.sa_flags = SA_SIGINFO | SA_RESETHAND;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGBUS, &sa, nullptr);
    sigaction(SIGSEGV, &sa, nullptr);
    sigaction(SIGABRT, &sa, nullptr);
  });
}

bool FileExists(const std::string& path) {
  return access(path.c_str(), F_OK) == 0;
}

bool FileWritable(const std::string& path) {
  return access(path.c_str(), W_OK) == 0;
}

bool WriteTextFile(const std::string& path, const std::string& text) {
  int fd = open(path.c_str(), O_WRONLY);
  if (fd < 0) return false;
  ssize_t n = write(fd, text.c_str(), text.size());
  close(fd);
  return n == static_cast<ssize_t>(text.size());
}

bool WriteTextFileVerbose(const std::string& path, const std::string& text) {
  if (WriteTextFile(path, text)) return true;
  LogWarn("u-dma-buf sync write failed: path=%s value=%s errno=%d (%s)",
          path.c_str(), text.c_str(), errno, strerror(errno));
  return false;
}

void TryWriteTextFileDebug(const std::string& path, const std::string& text) {
  if (WriteTextFile(path, text)) return;
  LogDebug("u-dma-buf optional write failed: path=%s value=%s errno=%d (%s)",
           path.c_str(), text.c_str(), errno, strerror(errno));
}

bool ReadULongFromFile(const std::string& path, unsigned long* out) {
  FILE* fp = fopen(path.c_str(), "r");
  if (!fp) return false;
  unsigned long v = 0;
  int ok = fscanf(fp, "%lu", &v);
  fclose(fp);
  if (ok != 1) return false;
  *out = v;
  return true;
}

bool ReadULLHexFromFile(const std::string& path, unsigned long long* out) {
  FILE* fp = fopen(path.c_str(), "r");
  if (!fp) return false;
  unsigned long long v = 0;
  int ok = fscanf(fp, "%llx", &v);
  fclose(fp);
  if (ok != 1) return false;
  *out = v;
  return true;
}

size_t AlignUp(size_t x, size_t align) {
  return (x + align - 1) & ~(align - 1);
}

class UdmabufPool {
 public:
  static UdmabufPool& Global() {
    static UdmabufPool inst;
    return inst;
  }

  void* Alloc(size_t size, int cached) {
    (void)cached;
    std::lock_guard<std::mutex> lock(mu_);
    EnsureInitLocked();

    assert(size <= VTA_MAX_XFER);

    const size_t kAlign = 256;
    size_t aligned_size = AlignUp(size, kAlign);
    size_t aligned_off = AlignUp(offset_, kAlign);

    if (aligned_off + aligned_size > size_) {
      LogErr("udmabuf pool OOM: request=%zu aligned=%zu used=%zu total=%zu",
             size, aligned_size, aligned_off, size_);
      return nullptr;
    }

    char* ptr = static_cast<char*>(base_vaddr_) + aligned_off;
    vta_phy_addr_t phy = static_cast<vta_phy_addr_t>(base_paddr_ + aligned_off);
    offset_ = aligned_off + aligned_size;
    requested_bytes_ += size;
    allocation_count_ += 1;

    LogInfo("Alloc size=%zu aligned=%zu vaddr=%p paddr=0x%llx used=%zu/%zu",
            size,
            aligned_size,
            ptr,
            static_cast<unsigned long long>(phy),
            offset_,
            size_);

    const char* zero_env = getenv("AXU5EVB_DRIVER_ZERO_ON_ALLOC");
    bool zero_on_alloc = DebugZeroOnAlloc();
    LogInfo("Alloc post-check zero_on_alloc=%d env=%s vaddr=%p paddr=0x%llx size=%zu",
            zero_on_alloc ? 1 : 0,
            zero_env ? zero_env : "<unset>",
            ptr,
            static_cast<unsigned long long>(phy),
            aligned_size);

    if (zero_on_alloc) {
      LogInfo("Alloc zero-probe begin vaddr=%p paddr=0x%llx size=%zu",
              ptr, static_cast<unsigned long long>(phy), aligned_size);

      volatile uint8_t* p = reinterpret_cast<volatile uint8_t*>(ptr);

      LogInfo("Alloc zero-probe step1 write byte @0 begin");
      p[0] = 0;
      LogInfo("Alloc zero-probe step1 write byte @0 ok");

      size_t probe16 = aligned_size < 16 ? aligned_size : 16;
      LogInfo("Alloc zero-probe step2 write first %zu bytes begin", probe16);
      for (size_t i = 0; i < probe16; ++i) {
        p[i] = 0;
      }
      LogInfo("Alloc zero-probe step2 write first %zu bytes ok", probe16);

      size_t probe_limit = aligned_size < (64 * 1024) ? aligned_size : (64 * 1024);
      LogInfo("Alloc zero-probe step3 page-stride begin limit=%zu", probe_limit);
      for (size_t off = 0; off < probe_limit; off += 4096) {
        p[off] = 0;
        LogDebug("Alloc zero-probe page write ok off=0x%zx", off);
      }
      LogInfo("Alloc zero-probe step3 page-stride ok");

      size_t zero64 = aligned_size < 64 ? aligned_size : 64;
      LogInfo("Alloc zero-probe step4 memset %zu begin", zero64);
      std::memset(ptr, 0, zero64);
      LogInfo("Alloc zero-probe step4 memset %zu ok", zero64);

      auto probe_memset = [&](size_t n) {
      if (aligned_size < n) return;
      LogInfo("Alloc zero-probe memset %zu begin", n);
      std::memset(ptr, 0, n);
      LogInfo("Alloc zero-probe memset %zu ok", n);
      };

      auto probe_linear_store = [&](size_t n) {
        if (aligned_size < n) return;
        LogInfo("Alloc zero-probe linear store %zu begin", n);
        volatile uint8_t* q = reinterpret_cast<volatile uint8_t*>(ptr);
        for (size_t i = 0; i < n; ++i) {
          q[i] = 0;
        }
        LogInfo("Alloc zero-probe linear store %zu ok", n);
      };

      probe_memset(128);
      probe_linear_store(256);
      probe_linear_store(4096);
      probe_linear_store(65536);


      LogPreview("Alloc zero-probe final preview", ptr, aligned_size);
      LogInfo("Alloc zero-probe end vaddr=%p paddr=0x%llx size=%zu",
              ptr, static_cast<unsigned long long>(phy), aligned_size);
    } else {
      LogDebug("Alloc returned without memset vaddr=%p paddr=0x%llx size=%zu",
               ptr, static_cast<unsigned long long>(phy), aligned_size);
    }

    return ptr;
  }

  void Free(void* buf) {
    LogDebug("Free buf=%p (bump allocator no-op)", buf);
  }

  // Diagnostic ABI v1. Observing must not initialize/map/allocate the pool.
  // Fields: initialized, capacity, high-water, requested bytes, padding,
  // successful allocation count, virtual base, physical base.
  int Snapshot(uint64_t* fields, size_t count) {
    if (fields == nullptr || count != 8) return -1;
    std::lock_guard<std::mutex> lock(mu_);
    fields[0] = inited_ ? 1 : 0;
    fields[1] = size_;
    fields[2] = offset_;
    fields[3] = requested_bytes_;
    fields[4] = offset_ - requested_bytes_;
    fields[5] = allocation_count_;
    fields[6] = reinterpret_cast<uintptr_t>(base_vaddr_);
    fields[7] = base_paddr_;
    return 0;
  }

  vta_phy_addr_t GetPhyAddr(void* buf) {
    std::lock_guard<std::mutex> lock(mu_);
    EnsureInitLocked();

    if (buf == nullptr) return 0;

    char* p = static_cast<char*>(buf);
    char* base = static_cast<char*>(base_vaddr_);
    if (p < base || p >= base + size_) {
      LogErr("GetPhyAddr on pointer %p not in udmabuf pool [%p, %p)",
             buf, base, base + size_);
      return 0;
    }
    size_t off = static_cast<size_t>(p - base);
    vta_phy_addr_t phy = static_cast<vta_phy_addr_t>(base_paddr_ + off);
    LogDebug("GetPhyAddr buf=%p -> paddr=0x%llx", buf, static_cast<unsigned long long>(phy));
    return phy;
  }

  bool ContainsRange(const void* buf, size_t size) {
    std::lock_guard<std::mutex> lock(mu_);
    EnsureInitLocked();
    if (!buf) return false;
    const char* p = static_cast<const char*>(buf);
    const char* base = static_cast<const char*>(base_vaddr_);
    const char* end = base + size_;
    if (p < base || p > end) return false;
    if (size == 0) return true;
    if (size > size_) return false;
    size_t off = static_cast<size_t>(p - base);
    return off <= size_ - size;
  }

  void SyncForDevice(void* buf, size_t size) { SyncCommon(buf, size, false); }
  void SyncForCpu(void* buf, size_t size) { SyncCommon(buf, size, true); }

 private:
  UdmabufPool() = default;

  ~UdmabufPool() {
    LogDebug("~UdmabufPool enter base_vaddr=%p size=%zu fd=%d", base_vaddr_, size_, fd_);
    if (base_vaddr_ && base_vaddr_ != MAP_FAILED) {
      munmap(base_vaddr_, size_);
      base_vaddr_ = nullptr;
    }
    if (fd_ >= 0) {
      close(fd_);
      fd_ = -1;
    }
    LogDebug("~UdmabufPool done");
  }

  void EnsureInitLocked() {
    if (inited_) return;

    dev_path_ = GetEnvOrDefault("UDMABUF_DEV", DefaultUdmabufDev());
    sysfs_dir_ = GetEnvOrDefault("UDMABUF_SYSFS_DIR", DefaultUdmabufSysfsDir());

    unsigned long sz = 0;
    unsigned long long pa = 0;

    std::string size_path = sysfs_dir_ + "/size";
    std::string phys_path = sysfs_dir_ + "/phys_addr";

    if (!ReadULongFromFile(size_path, &sz)) {
      LogErr("failed to read udmabuf size from %s", size_path.c_str());
      abort();
    }
    if (!ReadULLHexFromFile(phys_path, &pa)) {
      LogErr("failed to read udmabuf phys_addr from %s", phys_path.c_str());
      abort();
    }

    int open_flags = O_RDWR;
    if (EnvEnabled("AXU5EVB_DRIVER_OPEN_SYNC")) {
      open_flags |= O_SYNC;
    }
    fd_ = open(dev_path_.c_str(), open_flags);
    if (fd_ < 0) {
      LogErr("failed to open %s: %s", dev_path_.c_str(), strerror(errno));
      abort();
    }
    LogInfo("open %s flags=0x%x", dev_path_.c_str(), open_flags);

    void* vaddr = mmap(nullptr, sz, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
    if (vaddr == MAP_FAILED) {
      LogErr("failed to mmap %s size=%lu: %s", dev_path_.c_str(), sz, strerror(errno));
      close(fd_);
      fd_ = -1;
      abort();
    }

    size_ = static_cast<size_t>(sz);
    base_paddr_ = static_cast<uint64_t>(pa);
    base_vaddr_ = vaddr;
    offset_ = 0;
    inited_ = true;

    LogInfo("udmabuf initialized: dev=%s sysfs=%s vaddr=%p size=%zu paddr=0x%llx",
            dev_path_.c_str(), sysfs_dir_.c_str(), base_vaddr_, size_,
            static_cast<unsigned long long>(base_paddr_));
  }

  void SyncCommon(void* buf, size_t size, bool for_cpu) {
    std::lock_guard<std::mutex> lock(mu_);
    EnsureInitLocked();

    if (!buf || size == 0) return;

    char* p = static_cast<char*>(buf);
    char* base = static_cast<char*>(base_vaddr_);
    if (p < base || p + size > base + size_) {
      LogErr("sync pointer out of range: buf=%p size=%zu base=%p total=%zu",
             buf, size, base, size_);
      return;
    }
    size_t off = static_cast<size_t>(p - base);
    vta_phy_addr_t phy = static_cast<vta_phy_addr_t>(base_paddr_ + off);
    LogInfo("sync start dir=%s buf=%p paddr=0x%llx off=%zu size=%zu",
            for_cpu ? "for_cpu" : "for_device",
            buf,
            static_cast<unsigned long long>(phy),
            off,
            size);
    LogPreview(for_cpu ? "SyncForCpu before" : "SyncForDevice before", buf, size);

    bool synced = false;

    std::string sync_offset = sysfs_dir_ + "/sync_offset";
    std::string sync_size = sysfs_dir_ + "/sync_size";
    std::string sync_for_cpu = sysfs_dir_ + "/sync_for_cpu";
    std::string sync_for_dev = sysfs_dir_ + "/sync_for_device";
    std::string sync_owner = sysfs_dir_ + "/sync_owner";
    std::string sync_direction = sysfs_dir_ + "/sync_direction";

    if (FileWritable(sync_offset) && FileWritable(sync_size) &&
        (FileWritable(sync_for_cpu) || FileWritable(sync_for_dev))) {
      bool ok = true;
      ok &= WriteTextFileVerbose(sync_offset, std::to_string(off));
      ok &= WriteTextFileVerbose(sync_size, std::to_string(size));
      // sync_owner is informational on some u-dma-buf builds and may reject
      // writes even when access() suggests it is writable. Never let it force
      // a fallback to msync().
      if (FileExists(sync_owner)) {
        TryWriteTextFileDebug(sync_owner, for_cpu ? "cpu" : "device");
      }
      if (FileWritable(sync_direction)) {
        ok &= WriteTextFileVerbose(sync_direction, for_cpu ? "2" : "1");
      }
      if (for_cpu && FileWritable(sync_for_cpu)) {
        ok &= WriteTextFileVerbose(sync_for_cpu, "1");
      } else if (!for_cpu && FileWritable(sync_for_dev)) {
        ok &= WriteTextFileVerbose(sync_for_dev, "1");
      }
      if (ok) synced = true;
    }

    if (!synced) {
      static std::atomic<bool> warned{false};
      if (!warned.exchange(true)) {
        LogWarn("u-dma-buf sync sysfs interface not found or unusable; fallback to msync(). DMA cache coherency may still be insufficient.");
      }
      int flags = for_cpu ? (MS_SYNC | MS_INVALIDATE) : MS_SYNC;
      if (msync(buf, size, flags) != 0) {
        LogWarn("msync failed: %s", strerror(errno));
      }
    }

    LogPreview(for_cpu ? "SyncForCpu after" : "SyncForDevice after", buf, size);
    LogInfo("sync end dir=%s buf=%p size=%zu", for_cpu ? "for_cpu" : "for_device", buf, size);
  }

 private:
  std::mutex mu_;
  bool inited_{false};
  int fd_{-1};
  void* base_vaddr_{nullptr};
  size_t size_{0};
  uint64_t base_paddr_{0};
  size_t offset_{0};
  uint64_t requested_bytes_{0};
  uint64_t allocation_count_{0};
  std::string dev_path_;
  std::string sysfs_dir_;
};

struct RegMapHandle {
  void* map_base;
  void* reg_ptr;
  size_t map_size;
};

}  // namespace

extern "C" {

// Optional AXU5EVB diagnostic extension, resolved by dlsym by experimental runners.
int VTAMemAllocationSnapshotV1(uint64_t* fields, size_t count) {
  return UdmabufPool::Global().Snapshot(fields, count);
}

static inline void CopyBytesSafe(void* dst, const void* src, size_t size) {
  volatile uint8_t* d = reinterpret_cast<volatile uint8_t*>(dst);
  const uint8_t* s = reinterpret_cast<const uint8_t*>(src);
  for (size_t i = 0; i < size; ++i) {
    d[i] = s[i];
  }
}

void* VTAMemAlloc(size_t size, int cached) {
  InstallCrashHandlersOnce();
  LogDebug("VTAMemAlloc enter size=%zu cached=%d", size, cached);
  void* p = UdmabufPool::Global().Alloc(size, cached);
  LogDebug("VTAMemAlloc leave ptr=%p", p);
  return p;
}

void VTAMemFree(void* buf) { UdmabufPool::Global().Free(buf); }

vta_phy_addr_t VTAMemGetPhyAddr(void* buf) { return UdmabufPool::Global().GetPhyAddr(buf); }

void VTAMemCopyFromHost(void* dst, const void* src, size_t size) {
  vta_phy_addr_t phy = UdmabufPool::Global().GetPhyAddr(dst);
  LogInfo("CopyDataFromTo H2D dst=%p dst_phy=0x%llx src=%p size=%zu",
          dst, static_cast<unsigned long long>(phy), src, size);
  if (!UdmabufPool::Global().ContainsRange(dst, size)) {
    LogErr("H2D range check failed dst=%p size=%zu (not fully inside udmabuf)", dst, size);
    abort();
  }
  LogPreview("CopyDataFromTo H2D src preview", src, size);
  if (DebugSafeCopy()) {
    CopyBytesSafe(dst, src, size);
  } else {
    memcpy(dst, src, size);
  }
  LogPreview("CopyDataFromTo H2D dst preview", dst, size);
}

void VTAMemCopyToHost(void* dst, const void* src, size_t size) {
  vta_phy_addr_t phy = UdmabufPool::Global().GetPhyAddr(const_cast<void*>(src));
  LogInfo("CopyDataFromTo D2H dst=%p src=%p src_phy=0x%llx size=%zu",
          dst, src, static_cast<unsigned long long>(phy), size);
  if (!UdmabufPool::Global().ContainsRange(src, size)) {
    LogErr("D2H range check failed src=%p size=%zu (not fully inside udmabuf)", src, size);
    abort();
  }
  LogPreview("CopyDataFromTo D2H src preview", src, size);
  if (DebugSafeCopy()) {
    const volatile uint8_t* s = reinterpret_cast<const volatile uint8_t*>(src);
    uint8_t* d = reinterpret_cast<uint8_t*>(dst);
    for (size_t i = 0; i < size; ++i) {
      d[i] = s[i];
    }
  } else {
    memcpy(dst, src, size);
  }
  LogPreview("CopyDataFromTo D2H dst preview", dst, size);
}

void VTAFlushCache(void* vir_addr, vta_phy_addr_t phy_addr, int size) {
  (void)phy_addr;
  if (size > 0) UdmabufPool::Global().SyncForDevice(vir_addr, static_cast<size_t>(size));
}

void VTAInvalidateCache(void* vir_addr, vta_phy_addr_t phy_addr, int size) {
  (void)phy_addr;
  if (size > 0) UdmabufPool::Global().SyncForCpu(vir_addr, static_cast<size_t>(size));
}

void VTADriverProfilerClear() { DriverProfiler::Global().Clear(); }

void VTADriverProfilerStatus(VTADriverProfilerStats* out) {
  if (out == nullptr) return;
  *out = DriverProfiler::Global().Snapshot();
}

void* VTAMapRegister(uint32_t addr) {
  const char* devmem_path = GetEnvOrDefault("VTA_DEVMEM_PATH", DefaultDevMemPath());

  long pagesz = sysconf(_SC_PAGESIZE);
  if (pagesz <= 0) {
    LogErr("sysconf(_SC_PAGESIZE) failed");
    return nullptr;
  }

  uint64_t page_size = static_cast<uint64_t>(pagesz);
  uint64_t phys = static_cast<uint64_t>(addr);
  uint64_t page_base = phys & ~(page_size - 1);
  uint64_t page_off = phys - page_base;
  size_t map_size = static_cast<size_t>(page_off + VTA_IP_REG_MAP_RANGE);

  int fd = open(devmem_path, O_RDWR | O_SYNC);
  if (fd < 0) {
    LogErr("open(%s) failed: %s", devmem_path, strerror(errno));
    return nullptr;
  }

  void* map_base = mmap(nullptr, map_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, page_base);
  close(fd);

  if (map_base == MAP_FAILED) {
    LogErr("mmap register addr=0x%08x failed: %s", addr, strerror(errno));
    return nullptr;
  }

  RegMapHandle* h = new RegMapHandle;
  h->map_base = map_base;
  h->reg_ptr = static_cast<void*>(static_cast<char*>(map_base) + page_off);
  h->map_size = map_size;
  LogInfo("map reg phys=0x%08x map_base=%p reg_ptr=%p map_size=%zu", addr, map_base, h->reg_ptr,
          map_size);
  return h;
}

void VTAUnmapRegister(void* vta) {
  if (!vta) return;
  RegMapHandle* h = static_cast<RegMapHandle*>(vta);
  LogDebug("unmap reg map_base=%p reg_ptr=%p map_size=%zu", h->map_base, h->reg_ptr, h->map_size);
  if (h->map_base && h->map_base != MAP_FAILED) {
    munmap(h->map_base, h->map_size);
  }
  delete h;
}

void VTAWriteMappedReg(void* base_addr, uint32_t offset, uint32_t val) {
  RegMapHandle* h = static_cast<RegMapHandle*>(base_addr);
  *((volatile uint32_t*)(static_cast<char*>(h->reg_ptr) + offset)) = val;
  LogDebug("reg write reg_ptr=%p offset=0x%x val=0x%x", h->reg_ptr, offset, val);
}

uint32_t VTAReadMappedReg(void* base_addr, uint32_t offset) {
  RegMapHandle* h = static_cast<RegMapHandle*>(base_addr);
  uint32_t v = *((volatile uint32_t*)(static_cast<char*>(h->reg_ptr) + offset));
  LogDebug("reg read reg_ptr=%p offset=0x%x -> 0x%x", h->reg_ptr, offset, v);
  return v;
}

}  // extern "C"

class VTADevice {
 public:
  VTADevice() {
    LogInfo("VTADevice ctor enter");
    vta_fetch_handle_ = VTAMapRegister(VTA_FETCH_ADDR);
    vta_load_handle_ = VTAMapRegister(VTA_LOAD_ADDR);
    vta_compute_handle_ = VTAMapRegister(VTA_COMPUTE_ADDR);
    vta_store_handle_ = VTAMapRegister(VTA_STORE_ADDR);

    assert(vta_fetch_handle_ != nullptr);
    assert(vta_load_handle_ != nullptr);
    assert(vta_compute_handle_ != nullptr);
    assert(vta_store_handle_ != nullptr);
    LogInfo("VTADevice ctor done fetch=%p load=%p compute=%p store=%p",
            vta_fetch_handle_, vta_load_handle_, vta_compute_handle_, vta_store_handle_);
  }

  ~VTADevice() {
    LogInfo("VTADevice dtor enter");
    VTAUnmapRegister(vta_fetch_handle_);
    VTAUnmapRegister(vta_load_handle_);
    VTAUnmapRegister(vta_compute_handle_);
    VTAUnmapRegister(vta_store_handle_);
    LogInfo("VTADevice dtor done");
  }

  int Run(vta_phy_addr_t insn_phy_addr, uint32_t insn_count, uint32_t wait_cycles) {
    LogInfo("launch begin insn_paddr=0x%llx insn_count=%u wait_cycles=%u",
            static_cast<unsigned long long>(insn_phy_addr), insn_count, wait_cycles);

    double run_t0 = NowMicros();
    double submit_t0 = run_t0;
    VTAWriteMappedReg(vta_fetch_handle_, VTA_FETCH_INSN_COUNT_OFFSET, insn_count);
    VTAWriteMappedReg(vta_fetch_handle_, VTA_FETCH_INSN_ADDR_OFFSET, static_cast<uint32_t>(insn_phy_addr));
    VTAWriteMappedReg(vta_load_handle_, VTA_LOAD_INP_ADDR_OFFSET, 0);
    VTAWriteMappedReg(vta_load_handle_, VTA_LOAD_WGT_ADDR_OFFSET, 0);
    VTAWriteMappedReg(vta_compute_handle_, VTA_COMPUTE_UOP_ADDR_OFFSET, 0);
    VTAWriteMappedReg(vta_compute_handle_, VTA_COMPUTE_BIAS_ADDR_OFFSET, 0);
    VTAWriteMappedReg(vta_store_handle_, VTA_STORE_OUT_ADDR_OFFSET, 0);

    VTAWriteMappedReg(vta_fetch_handle_, 0x0, VTA_START);
    VTAWriteMappedReg(vta_load_handle_, 0x0, VTA_AUTORESTART);
    VTAWriteMappedReg(vta_compute_handle_, 0x0, VTA_AUTORESTART);
    VTAWriteMappedReg(vta_store_handle_, 0x0, VTA_AUTORESTART);
    LogInfo("launch written fetch_ctrl=0x%x load_ctrl=0x%x compute_ctrl=0x%x store_ctrl=0x%x",
            VTAReadMappedReg(vta_fetch_handle_, 0x0),
            VTAReadMappedReg(vta_load_handle_, 0x0),
            VTAReadMappedReg(vta_compute_handle_, 0x0),
            VTAReadMappedReg(vta_store_handle_, 0x0));
    double submit_mmio_us = NowMicros() - submit_t0;

    uint64_t post_start_sleep_ns = GetEnvUInt64OrDefault("AXU5EVB_DRIVER_POST_START_SLEEP_NS", 1000);
    uint64_t poll_sleep_ns = GetEnvUInt64OrDefault("AXU5EVB_DRIVER_POLL_SLEEP_NS", 1000);
    double sleep_t0 = NowMicros();
    SleepForNanos(post_start_sleep_ns);
    double post_start_sleep_us = NowMicros() - sleep_t0;

    LogInfo("wait polling begin");
    double poll_t0 = NowMicros();
    unsigned t = 0;
    uint32_t flag = 0;
    for (t = 0; t < wait_cycles; ++t) {
      flag = VTAReadMappedReg(vta_compute_handle_, VTA_COMPUTE_DONE_RD_OFFSET);
      if (flag == VTA_DONE) break;
      SleepForNanos(poll_sleep_ns);
    }
    double poll_wait_us = NowMicros() - poll_t0;
    double run_total_us = NowMicros() - run_t0;

    if (t < wait_cycles) {
      DriverProfiler::Global().AddRun(insn_count, false, t + 1, run_total_us, submit_mmio_us,
                                      post_start_sleep_us, poll_wait_us);
      LogInfo("wait done cycles=%u flag=0x%x", t, flag);
      return 0;
    }

    DriverProfiler::Global().AddRun(insn_count, true, t, run_total_us, submit_mmio_us,
                                    post_start_sleep_us, poll_wait_us);
    LogWarn("wait timeout flag=0x%x fetch_ctrl=0x%x load_ctrl=0x%x compute_ctrl=0x%x store_ctrl=0x%x",
            flag,
            VTAReadMappedReg(vta_fetch_handle_, 0x0),
            VTAReadMappedReg(vta_load_handle_, 0x0),
            VTAReadMappedReg(vta_compute_handle_, 0x0),
            VTAReadMappedReg(vta_store_handle_, 0x0));
    return 1;
  }

 private:
  void* vta_fetch_handle_{nullptr};
  void* vta_load_handle_{nullptr};
  void* vta_compute_handle_{nullptr};
  void* vta_store_handle_{nullptr};
};

extern "C" {

VTADeviceHandle VTADeviceAlloc() {
  LogInfo("VTADeviceAlloc enter");
  VTADeviceHandle h = new VTADevice();
  LogInfo("VTADeviceAlloc leave handle=%p", h);
  return h;
}

void VTADeviceFree(VTADeviceHandle handle) {
  LogInfo("VTADeviceFree handle=%p", handle);
  delete static_cast<VTADevice*>(handle);
}

int VTADeviceRun(VTADeviceHandle handle, vta_phy_addr_t insn_phy_addr, uint32_t insn_count,
                 uint32_t wait_cycles) {
  LogInfo("VTADeviceRun handle=%p insn_paddr=0x%llx insn_count=%u wait_cycles=%u",
          handle, static_cast<unsigned long long>(insn_phy_addr), insn_count, wait_cycles);
  int rc = static_cast<VTADevice*>(handle)->Run(insn_phy_addr, insn_count, wait_cycles);
  LogInfo("VTADeviceRun return rc=%d", rc);
  return rc;
}

}  // extern "C"
