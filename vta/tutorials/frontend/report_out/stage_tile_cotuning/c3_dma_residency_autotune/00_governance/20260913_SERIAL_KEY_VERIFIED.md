# 串口公钥核验与连接恢复

用户从串口运行dropbearkey -y提供RSA公钥。主机ssh-keygen计算SHA256为
`t9DRcG2GjHEachlJi8Mncib+pPEt5r5+cmGhNBZsAjg`，与网络握手一致。
公钥保存在独立 `/tmp/vta_c3_known_hosts_serial_verified`，旧记录未覆盖。
随后StrictHostKeyChecking=yes登录成功。

当前boot：4d232b6f-6e2a-4395-aafe-853a021403e8。
FPGA manager状态operating；/var/volatile为tmpfs；SD p2 ext4已rw挂载。
本次读取的dmesg末50行显示FAT p1未正常卸载警告，未见ext4或mmc I/O错误。
这不是全盘健康或离线fsck证明，不能写SD完全正常。

/sys/class/u-dma-buf与/var/volatile/vta_c3_ram均不存在，未见tvm进程。
断电丢失了RAM实验环境，尚未恢复runtime/module或启动新实验。下一步使用
历史核验产物恢复RAM环境，再做当前boot健康门；原Y08候选合同保持不变。
板端时钟显示3月，与主机实验编号日期不同，boot身份比板端日期更可靠。
