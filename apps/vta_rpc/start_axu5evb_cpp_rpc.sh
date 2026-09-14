#!/bin/sh
# Start AXU5EVB C++ RPC server from an isolated runtime directory.

set -eu

if [ $# -lt 1 ]; then
  echo "Usage: $0 <runtime-dir> [port]"
  echo "Example: $0 /mnt/sd/tvm_deploy/hp 9090"
  exit 1
fi

RUNTIME_DIR="$1"
PORT="${2:-9090}"
UDMABUF_BYTES="${AXU5EVB_UDMABUF_BYTES:-201326592}"

if [ ! -d "$RUNTIME_DIR" ]; then
  echo "Runtime directory not found: $RUNTIME_DIR" >&2
  exit 2
fi

if [ ! -x "$RUNTIME_DIR/tvm_rpc" ]; then
  echo "Missing executable: $RUNTIME_DIR/tvm_rpc" >&2
  exit 3
fi

if [ ! -f "$RUNTIME_DIR/libtvm_runtime.so" ]; then
  echo "Missing library: $RUNTIME_DIR/libtvm_runtime.so" >&2
  exit 4
fi

if [ ! -f "$RUNTIME_DIR/libvta.so" ]; then
  echo "Missing library: $RUNTIME_DIR/libvta.so" >&2
  exit 5
fi

# A board reboot removes the u-dma-buf module while the SD-card runtime remains.
# Native VTA allocation needs this device even when the RPC process itself starts.
if [ ! -r /sys/class/u-dma-buf/udmabuf0/size ]; then
  UDMABUF_MODULE="$RUNTIME_DIR/u-dma-buf.ko"
  if [ ! -f "$UDMABUF_MODULE" ]; then
    UDMABUF_MODULE=/mnt/sd/tvm_deploy/hp/u-dma-buf.ko
  fi
  if [ ! -f "$UDMABUF_MODULE" ]; then
    echo "Missing u-dma-buf.ko for native VTA allocation" >&2
    exit 6
  fi
  /sbin/insmod "$UDMABUF_MODULE" "udmabuf0=$UDMABUF_BYTES"
fi

OBSERVED_UDMABUF_BYTES=$(cat /sys/class/u-dma-buf/udmabuf0/size 2>/dev/null || true)
if [ "$OBSERVED_UDMABUF_BYTES" != "$UDMABUF_BYTES" ]; then
  echo "Unexpected udmabuf0 size: expected=$UDMABUF_BYTES observed=$OBSERVED_UDMABUF_BYTES" >&2
  exit 7
fi

cd "$RUNTIME_DIR"
export LD_LIBRARY_PATH="$RUNTIME_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_PRELOAD="$RUNTIME_DIR/libtvm_runtime.so:$RUNTIME_DIR/libvta.so"
: "${AXU5EVB_DRIVER_POST_START_SLEEP_NS:=1000}"
: "${AXU5EVB_DRIVER_POLL_SLEEP_NS:=1000}"
export AXU5EVB_DRIVER_POST_START_SLEEP_NS
export AXU5EVB_DRIVER_POLL_SLEEP_NS

# This deployed C++ RPC binary uses dmlc-style ``--key=value`` parsing.  With
# split arguments it silently kept the default port (9090), which made recovery
# after a lingering TCP close look like a failed restart.
exec ./tvm_rpc server --host=0.0.0.0 --port="$PORT"
