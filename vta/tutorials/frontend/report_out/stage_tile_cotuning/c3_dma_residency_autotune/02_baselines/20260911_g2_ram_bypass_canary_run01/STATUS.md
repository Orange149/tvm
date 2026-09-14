# G2 RAM-only recovery canary

Status: **passed for RAM-only board correctness; not a performance baseline**.

The current board boot successfully loaded the FPGA bitstream, `u-dma-buf.ko`, TVM runtime, VTA runtime, RPC server, and RPC uploads without using the damaged ext4 partition for the experiment working set. `/dev/mmcblk1p2` remained mounted read-only.

All 10 frozen ResNet-18 VTA convolution workloads returned `status=ok` and passed exact output comparison. No new storage error appeared after the read-only remount. The SD card is still corrupt and 100% full, so this result does not mean the card is healthy or repaired.

The recorded `kernel_ms` values are diagnostic only: each workload used `number=1`, `warmup=0`. They must not be cited as stable performance results or as evidence that a residency mechanism improves latency.

One host-side benchmark defect was exposed and fixed before the accepted run: the CSV rows already contained `memory_bytes_est` and `effective_memory_bandwidth_GBps`, but the writer field list omitted both columns. The accepted CSV was produced after adding those fields.
