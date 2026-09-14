# P7R378：Y08 冻结全池上板完成

状态：`COMPLETE`。先前 P7R376 的网络/主机身份阻塞已经解除；串口导出的 Dropbear RSA 公钥与
网络握手的 `SHA256:t9DRcG2GjHEachlJi8Mncib+pPEt5r5+cmGhNBZsAjg` 一致。当前 boot 为
`4d232b6f-6e2a-4395-aafe-853a021403e8`，实验前后 FPGA 均为 `operating`、u-dma-buf 均为
192 MiB，RPC 工作目录为 `/var/volatile/vta_c3_ram/runtime`，最新内核日志没有 EXT4/mmc 错误。

## 冻结关系

- P7R354：Y08 YOLOv3-tiny conv12 候选集合；
- P7R355：预算与成本协议；
- P7R356：24 个身份中 6 个通过真实 lowering 与三 seed FSim；
- P7R357：在板端正确性和 latency 未见时冻结五种顺序；
- P7R377：严格校验上述哈希后完成全池 FPGA 正确性与计时；
- P7R378：只对冻结顺序做事后前缀回放，不重新排序。

## 结果

- 健康 canary 3/3 正确；Y08 六候选 18/18 correctness invocation 正确；
- 六候选七轮交错计时共 42/42 样本正确；每个 time evaluator 样本实际执行两次 kernel；
- 完整池 oracle 为 Y08F06 input-stationary，`132.920140 ms`；
- 三个 same-tile input-stationary 配对全部加速，分别为 27.71%、36.22% 和 3.70%；
- bytes、calls 与 Pareto 冻结顺序均在第一测量点得到精确 oracle；单个预注册 Random 顺序第 1 点
  已进入 oracle+2%，第 4 点才得到精确 oracle；
- 完整上板采集进程墙钟 135.309 s，包含交叉编译、clean start、健康门、上传、分配、正确性与计时。

## 新得到的边界

Y08F01 只减少 1.30% 逻辑 DMA bytes，但请求数减少 92.31%，latency 改善 27.71%；Y08F02 与
Y08F06 的 input-stationary 总字节完全相同，延迟仍相差约 0.24%。这再次说明 DMA bytes 是有效的
冷启动先验而不是 latency 定理，请求数/请求粒度与计算结构仍需保留。另一方面，本池只有一个
冻结 Random seed，而且它在 trial 1 已进入 2% 等价带，因此不能把该池写成 success@2% 优于随机；
它主要补上第三个网络的机制证据与 exact-oracle time-to-target 证据。

原 P7R376 中“Y08 未采集 FPGA/latency 标签”的断点已经关闭。下一步不应重新调 Y08 规则，而应做
跨 boot 复验或在新网络上冻结并执行完整的 one-at-a-time 外层调优成本协议。

原始证据：

- `07_grouped_holdout/20260913_p7r377_y08_frozen_board_pool_run01/`
- `07_grouped_holdout/20260913_p7r378_y08_frozen_pool_analysis_run01/`

