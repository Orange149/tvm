# G2 post-SD-repair recovery canary

Status before execution: `preregistered_current_boot_recovery_check`.

- Board: `root@192.168.1.247`; required boot ID: `a68a7983-719f-47bf-94d7-41f974c5342c`.
- Required FPGA state: `operating`; required u-dma-buf size: 201326592 bytes.
- RPC process and its working directory must remain on `/var/volatile` tmpfs.
- The repaired ext4 partition must have at least 20% free space and no new ext4/mmc/I/O error after dmesg second 4058.
- Run all ten frozen ResNet-18 convolution workloads with the frozen TopHub records, exact output comparison, `warmup=0`, and `number=1`.
- This batch is a correctness/recovery gate, not a stable performance comparison. Its one-shot latency values cannot be used as P7 labels.
- Any wrong answer, RPC/device failure, boot change, or new storage error stops the batch before P7 holdout measurements.

