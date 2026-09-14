# P7R344 frozen-bitstream recovery

P7R343 stopped fail-closed before candidate 0 because reboot had removed
`/lib/firmware/vta_hpc.bit`. No candidate was loaded and no R50E correctness or
latency label was collected.

The exact frozen bitstream was restored to the RAM rootfs, verified by SHA-256,
loaded successfully, and followed by restoration of the default tmpfs RPC. The
boot ID did not change, the FPGA is `operating`, u-dma-buf remains 192 MiB, and
no new ext4/MMC error was observed.
