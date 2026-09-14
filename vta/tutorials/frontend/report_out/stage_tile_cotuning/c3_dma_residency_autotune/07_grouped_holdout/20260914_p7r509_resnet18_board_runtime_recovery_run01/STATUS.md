# P7R509 board runtime recovery

The serial Dropbear host key and the network key matched exactly. The frozen bitstream,
u-dma-buf module, C++ RPC server and runtime libraries were restored only through tmpfs-backed
paths and verified by SHA-256. The board reports FPGA `operating`, u-dma-buf 192 MiB at
`0x68500000`, and the RPC working directory `/var/volatile/vta_c3_ram/runtime`.

No R18 candidate was dispatched and no target correctness or latency label was read. The FAT
boot partition retained a prior unclean-unmount warning, but no new EXT4/MMC error was present;
the experiment does not write artifacts to the SD card.
