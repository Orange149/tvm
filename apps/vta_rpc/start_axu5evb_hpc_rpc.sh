#!/bin/sh
set -eu
pkill -f tvm_rpc || true
echo vta_hpc.bit > /sys/class/fpga_manager/fpga0/firmware
exec /mnt/sd/tvm_deploy/start_axu5evb_cpp_rpc.sh /mnt/sd/tvm_deploy/hpc "${1:-9090}"
