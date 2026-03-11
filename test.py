#!/usr/bin/env python3
import mmap
import os
import struct
import sys
import time

UDMABUF_DEV = "/dev/udmabuf0"
UDMABUF_SYSFS = "/sys/class/u-dma-buf/udmabuf0"


def read_first_existing(paths, default=None):
    for p in paths:
        if os.path.exists(p):
            with open(p, "r") as f:
                return f.read().strip()
    return default


def read_udmabuf_info():
    size_str = read_first_existing([
        os.path.join(UDMABUF_SYSFS, "size"),
    ])
    phys_str = read_first_existing([
        os.path.join(UDMABUF_SYSFS, "phys_addr"),
        os.path.join(UDMABUF_SYSFS, "phys_address"),
    ], default="unknown")

    if size_str is None:
        raise RuntimeError("cannot read udmabuf size from sysfs")

    size = int(size_str, 0)
    return size, phys_str


def log(msg):
    print(msg, flush=True)


def main():
    size, phys = read_udmabuf_info()
    log(f"[INFO] dev={UDMABUF_DEV}")
    log(f"[INFO] size={size}")
    log(f"[INFO] phys={phys}")

    fd = os.open(UDMABUF_DEV, os.O_RDWR | os.O_SYNC)
    try:
        mm = mmap.mmap(fd, size, flags=mmap.MAP_SHARED,
                       prot=mmap.PROT_READ | mmap.PROT_WRITE)
    finally:
        os.close(fd)

    log("[STEP] mmap ok")

    # 先读几个字节
    log(f"[STEP] first16 before = {mm[:16].hex(' ')}")

    # 1) 写第 1 个字节
    log("[STEP] write byte @0 begin")
    mm[0] = 0
    log("[STEP] write byte @0 ok")

    # 2) 写第 1 个 32-bit word
    log("[STEP] write u32 @0 begin")
    mm[0:4] = struct.pack("<I", 0x12345678)
    log("[STEP] write u32 @0 ok")

    # 3) 写前 16 字节
    log("[STEP] write first16 begin")
    mm[0:16] = bytes([0xAA] * 16)
    log("[STEP] write first16 ok")

    # 4) 每 4KB 写 1 个字节，先测前 64KB
    log("[STEP] page-stride write 64KB begin")
    for off in range(0, 64 * 1024, 4096):
        mm[off] = (off // 4096) & 0xFF
        log(f"[STEP] page write ok off=0x{off:x}")
    log("[STEP] page-stride write 64KB ok")

    # 5) 小块清零
    log("[STEP] memset-like zero 64B begin")
    mm[0:64] = b"\x00" * 64
    log("[STEP] memset-like zero 64B ok")

    log("[STEP] memset-like zero 4KB begin")
    mm[0:4096] = b"\x00" * 4096
    log("[STEP] memset-like zero 4KB ok")

    log("[STEP] memset-like zero 64KB begin")
    mm[0:65536] = b"\x00" * 65536
    log("[STEP] memset-like zero 64KB ok")

    log(f"[STEP] first16 after = {mm[:16].hex(' ')}")

    mm.flush()
    log("[STEP] flush ok")

    mm.close()
    log("[DONE] all tests passed")


if __name__ == "__main__":
    main()