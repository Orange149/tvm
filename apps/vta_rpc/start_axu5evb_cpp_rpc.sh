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

cd "$RUNTIME_DIR"
export LD_LIBRARY_PATH="$RUNTIME_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_PRELOAD="$RUNTIME_DIR/libtvm_runtime.so:$RUNTIME_DIR/libvta.so"

exec ./tvm_rpc server --host 0.0.0.0 --port "$PORT"
