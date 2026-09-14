# G2 post-SD-repair baseline gate

Status: **passed on the current boot**.

The ten frozen VTA convolution workloads all passed exact correctness. The preserved topology-B package was then replayed three times with 22 frames per repeat and the first two frames discarded. FPS was 10.1072, 10.7772, and 10.4511; the three-repeat median was **10.4511 FPS**, 2.54% below the frozen 10.724 FPS reference and above the 10.1878 FPS acceptance floor.

The first repeat alone was 5.75% below the reference, so it was not selectively accepted. The gate uses the median of all three retained repeats. All 66 topology-B outputs had top-1 285 and FNV-1a `a8c613584e081e5f`.

Postflight retained the same boot, FPGA `operating`, 192 MiB u-dma-buf, 381.2 MiB free on the repaired ext4 partition, and zero new storage errors after dmesg second 4058. This accepts current-boot G2 and permits a separately frozen P7 correctness batch. It is not a new end-to-end speedup claim.
