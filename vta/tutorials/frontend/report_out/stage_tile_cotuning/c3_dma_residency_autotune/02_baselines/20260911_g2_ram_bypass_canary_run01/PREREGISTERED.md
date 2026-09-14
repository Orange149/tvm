# G2 RAM-bypass recovery canary

Status before execution: `preregistered_recovery_pilot`.

- Board: `root@192.168.1.247`, boot ID `a68a7983-719f-47bf-94d7-41f974c5342c`.
- `/dev/mmcblk1p2` has ext4 metadata errors and is 100% full; it was remounted read-only before deployment.
- Runtime, RPC work directory, bitstream, u-dma-buf module, uploaded modules and temporary outputs must reside under the RAM-backed `/var/volatile/vta_c3_ram` or on the host.
- Frozen bitstream SHA-256: `7bf1ac95b1182c670cd25241111f33725b4ea37ed6d66f2df37b0b2e7df528d6`.
- Frozen u-dma-buf module SHA-256: `4c0aa5713225cd9b29548d9c1cd6e6127aeaafcc1f7d8650d68e3a0a527042c0`; requested size: 201326592 bytes.
- Run the ten frozen ResNet18 convolution workloads with the frozen TopHub log, exact correctness enabled, `warmup=0`, `number=1`.
- This run can establish RAM-only path viability and a current-boot correctness canary. It cannot establish stable performance, complete G2, G5/G6, or thesis FPS claims.
- Any new MMC/ext4/I/O error after the read-only remount invalidates the run.
