# AXU5EVB Board Runtime Layout

目标：把板端运行时整理成两个完全独立的目录，避免 `HP-only` 和 `HPC/coherent` 的 `libvta.so`、`libtvm_runtime.so`、`tvm_rpc` 混用。

建议板端目录：

```text
/mnt/sd/tvm_deploy/
  hp/
    libtvm_runtime.so
    libvta.so
    tvm_rpc
    vta_hp.bit
    u-dma-buf.ko
  hpc/
    libtvm_runtime.so
    libvta.so
    tvm_rpc
    vta_hpc.bit
    u-dma-buf.ko
  current -> hp   # 可选软链接
```

约定：

- `hp/`
  - 对应 `HP-only + non-coherent` bitstream/runtime
- `hpc/`
  - 对应 `HPC/coherent` bitstream/runtime
- 两个目录都放完整的一套：
  - `libtvm_runtime.so`
  - `libvta.so`
  - `tvm_rpc`
  - 对应 bitstream：`vta_hp.bit` / `vta_hpc.bit`
- 不建议继续把两套 `.so` 混放在同一个目录里靠手工覆盖切换

## 推荐启动方式

使用仓库里的这个脚本：

- [start_axu5evb_cpp_rpc.sh](/home/orange/code/tvm/apps/vta_rpc/start_axu5evb_cpp_rpc.sh)

如果你把脚本复制到板端 `/mnt/sd/tvm_deploy/`，推荐这样运行：

```bash
cd /mnt/sd/tvm_deploy
./start_axu5evb_hp_rpc.sh
```

或者：

```bash
cd /mnt/sd/tvm_deploy
./start_axu5evb_hpc_rpc.sh
```

这两个 wrapper 会自动执行：

- `pkill -f tvm_rpc || true`
- `echo vta_hp.bit > /sys/class/fpga_manager/fpga0/firmware`
  或
- `echo vta_hpc.bit > /sys/class/fpga_manager/fpga0/firmware`
- 再启动对应目录下的 `tvm_rpc`

## 手工启动命令

### HP-only

```bash
echo vta_hp.bit > /sys/class/fpga_manager/fpga0/firmware
cd /mnt/sd/tvm_deploy/hp
export LD_LIBRARY_PATH=/mnt/sd/tvm_deploy/hp:${LD_LIBRARY_PATH}
LD_PRELOAD=/mnt/sd/tvm_deploy/hp/libtvm_runtime.so:/mnt/sd/tvm_deploy/hp/libvta.so \
./tvm_rpc server --host 0.0.0.0 --port 9090
```

### HPC/coherent

```bash
echo vta_hpc.bit > /sys/class/fpga_manager/fpga0/firmware
cd /mnt/sd/tvm_deploy/hpc
export LD_LIBRARY_PATH=/mnt/sd/tvm_deploy/hpc:${LD_LIBRARY_PATH}
LD_PRELOAD=/mnt/sd/tvm_deploy/hpc/libtvm_runtime.so:/mnt/sd/tvm_deploy/hpc/libvta.so \
./tvm_rpc server --host 0.0.0.0 --port 9090
```

## 可选软链接切换

如果你希望统一只用一个路径，也可以在板端维护：

```bash
ln -sfn /mnt/sd/tvm_deploy/hp /mnt/sd/tvm_deploy/current
```

然后统一从 `current/` 启动：

```bash
cd /mnt/sd/tvm_deploy/current
export LD_LIBRARY_PATH=/mnt/sd/tvm_deploy/current:${LD_LIBRARY_PATH}
LD_PRELOAD=/mnt/sd/tvm_deploy/current/libtvm_runtime.so:/mnt/sd/tvm_deploy/current/libvta.so \
./tvm_rpc server --host 0.0.0.0 --port 9090
```

切换到 `hpc`：

```bash
ln -sfn /mnt/sd/tvm_deploy/hpc /mnt/sd/tvm_deploy/current
```

## 最小运维建议

- 切换 `hp`/`hpc` 前先停止旧 RPC：

```bash
pkill -f tvm_rpc || true
```

- 每次切换配置时同时切这两样：
  - bitstream
  - 对应目录下的 runtime

- 不要出现：
  - `HP` bitstream + `HPC` runtime
  - `HPC` bitstream + `HP` runtime
```
