# AXU5EVB RPC Startup

These scripts load an AXU5EVB bitstream, prepare `/lib/firmware`, and start
the TVM C++ RPC runtime from `/mnt/sd/tvm_deploy`.

## Scripts

- `start_axu5evb_hp_rpc.sh`
  - Loads `vta_hp.bit`
  - Starts runtime from `/mnt/sd/tvm_deploy/hp`
- `start_axu5evb_hpc_rpc.sh`
  - Loads `vta_hpc.bit`
  - Starts runtime from `/mnt/sd/tvm_deploy/hpc`
- `start_axu5evb_hp4_rpc.sh`
  - Loads `vta_hp4.bit`
  - Currently reuses `/mnt/sd/tvm_deploy/hp`
- `start_axu5evb_cpp_rpc.sh`
  - Shared launcher for `tvm_rpc`, `libtvm_runtime.so`, and `libvta.so`

## Directory Layout on Board

Recommended layout:

```text
/mnt/sd/tvm_deploy/
  firmware/
    vta_hp.bit
    vta_hpc.bit
    vta_hp4.bit
  hp/
    tvm_rpc
    libtvm_runtime.so
    libvta.so
  hpc/
    tvm_rpc
    libtvm_runtime.so
    libvta.so
```

At startup, the script checks `/lib/firmware/<bitstream>` first. If missing, it
copies the matching file from `/mnt/sd/tvm_deploy/firmware/`.

## Commands

On the board:

```sh
/mnt/sd/tvm_deploy/start_axu5evb_hp_rpc.sh 9090
```

```sh
/mnt/sd/tvm_deploy/start_axu5evb_hpc_rpc.sh 9090
```

```sh
/mnt/sd/tvm_deploy/start_axu5evb_hp4_rpc.sh 9090
```

## Sleep-Polling Runtime Knobs

`start_axu5evb_cpp_rpc.sh` exports these defaults before launching `tvm_rpc`:

- `AXU5EVB_DRIVER_POST_START_SLEEP_NS=1000`
- `AXU5EVB_DRIVER_POLL_SLEEP_NS=1000`

Override them before invoking the script if needed:

```sh
AXU5EVB_DRIVER_POLL_SLEEP_NS=5000 /mnt/sd/tvm_deploy/start_axu5evb_hpc_rpc.sh 9090
```

## Notes

- If `pkill` is unavailable on the board image, the scripts fall back to `killall`.
- `hp4` currently reuses the `hp` runtime directory. If the 4HP runtime diverges
  later, point `start_axu5evb_hp4_rpc.sh` at a dedicated directory.
