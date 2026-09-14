# P7R351：真实单候选编译与 FPGA 预算执行链

本轮完成 P7R350 本地执行器到真实模型编译与开发板的接入。固定 300 s 预算，使用已暴露的
R50EF03 input-stationary 身份 `3d57d3562324529ce9a615afe20aace280bcbcc79f5e707efc81807b7937fb56`。
该身份只用于执行链集成测试，不增加未见 workload、独立 holdout 或搜索性能结果。

## 实际动作

`run_vta_single_candidate_process.py` 在 build 子进程中读取并核对既有目标合同、三 seed FSim
资格，重新准备预训练 ResNet50 模型，完整构建 stock 和单候选，核对图结构与参数语义相等，
提取最终 fused TIR 的逻辑 DMA，并生成不可变 bundle。

measure 子进程验证 bundle，使用既有 clean-start 上板流程：核对 runtime 哈希、固定 bitstream
重载、tmpfs RPC，加载 stock/候选两个 executor，完成三 seed correctness 与七轮交错配对计时，
核对 correctness/timing 的逻辑 DMA 差与最终 fused TIR。新鲜完成凭据必须通过身份与状态检查，
外层预算执行器才能接受本次完成。

## 结果

- wrapper elapsed：144.168501 s，预算 300 s，无 overrun；
- 构建子进程外层：114.233822 s；
- 板端测量子进程外层：29.887541 s；
- 三 seed 共 6 条 stock/候选正确性记录全部通过；七轮共 14 条计时记录正确；
- 候选/stock 配对 latency ratio：1.129532。候选仍比 stock 慢，此结果不主张部署加速；
- boot 保持 `737f64bf-de64-4daa-9fa8-a216db9042eb`，FPGA operating、u-dma-buf 201326592 B、
  RPC cwd 位于 `/var/volatile/vta_c3_ram/runtime`，终态未发现新存储错误。

## 从真实外层计时发现的缺口

| 范围 | 内部计时和 | 外层实测 |
|---|---:|---:|
| stock + 单候选构建 | 59.003536 s | 114.233822 s |
| correctness + timing 调用 | 11.470711 s | 29.887541 s |
| 整个执行链 | 70.474247 s | 144.168501 s |

原来的构建加调用子计时少覆盖 73.694254 s。差额包括模型准备、导入、语义/哈希校验、产物归档、
RPC 启停/重载/上传/分配和其余编排。当前没有进一步拆分每项差额，不能把它全部归因于模型准备
或线程数。该差额属于本次运行，不能直接回填旧 workload，也不能由此修改 P7R348 的原始计时。

它确认 P7R349 的口径修正具有实际意义：仅相加内部阶段会明显低估真实使用成本。后续预算比较
必须直接用 wrapper elapsed，同时保留阶段和以解释构建/访问成本变化。

## 边界与下一步

既有算子资格在本次预算外：本轮测的是单候选编译—上板执行集成，并未宣称完整在线调优墙钟。
wrapper 计时仍不包括解释器启动和最终报告序列化。当前适配器每次重建 stock、重新准备模型；
多候选实验需要显式决定哪些可共享、哪些是候选增量成本，并对所有基线统一收费。

下一轮应把共同准备作为一次性 prepare 阶段，再逐候选构建和测量；冻结未见 workload 与预算后
比较前沿、bytes、calls、Random。不能把已知 R50E oracle 当作在线停止判据。

`execution`、`bundle`、`board` 三个 manifest 共 288 条文件哈希重算通过。执行器源码哈希保持
P7R350 版本；实际编译与三 seed/七轮检查构成适配器的集成验证。原始日志、结果、凭据和预算
事件位于 `07_grouped_holdout/20260913_p7r351_single_candidate_budget_board_run01/`。
