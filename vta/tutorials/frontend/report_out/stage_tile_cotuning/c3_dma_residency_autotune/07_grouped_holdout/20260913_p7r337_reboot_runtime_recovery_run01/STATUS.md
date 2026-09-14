# P7R337 reboot recovery

The new boot `737f64bf-de64-4daa-9fa8-a216db9042eb` is qualified for continued VTA
experiments. The frozen 192 MiB u-dma-buf was restored at physical address
`0x68500000`, and the exact frozen RPC/runtime files were copied from the host into
`/var/volatile/vta_c3_ram/runtime` and verified by SHA-256 before start.

The ext4 SD partition mounted and remained readable/writable without a new EXT4 or
MMC error in the kernel log. The FAT boot partition reported an unclean previous
unmount; no experiment artifact or runtime file is written there. This record does
not claim any performance result.
