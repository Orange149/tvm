#!/bin/sh
set -eu

if command -v pkill >/dev/null 2>&1; then
  pkill -f tvm_rpc || true
else
  killall tvm_rpc 2>/dev/null || true
fi

BITSTREAM=vta_hp.bit
SD_FIRMWARE_DIR=/mnt/sd/tvm_deploy/firmware

mkdir -p /lib/firmware
if [ ! -f /lib/firmware/${BITSTREAM} ] && [ -f "${SD_FIRMWARE_DIR}/${BITSTREAM}" ]; then
  cp "${SD_FIRMWARE_DIR}/${BITSTREAM}" /lib/firmware/${BITSTREAM}
fi

if [ ! -f /lib/firmware/${BITSTREAM} ]; then
  echo "missing /lib/firmware/${BITSTREAM} or ${SD_FIRMWARE_DIR}/${BITSTREAM}" >&2
  exit 1
fi

echo ${BITSTREAM} > /sys/class/fpga_manager/fpga0/firmware
exec /mnt/sd/tvm_deploy/start_axu5evb_cpp_rpc.sh /mnt/sd/tvm_deploy/hp "${1:-9090}"
