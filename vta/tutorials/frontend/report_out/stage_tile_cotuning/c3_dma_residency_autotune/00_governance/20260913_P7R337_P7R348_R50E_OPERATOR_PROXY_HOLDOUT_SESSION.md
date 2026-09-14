# P7R337--P7R348：R50E 廉价算子代理前沿独立前瞻留出

> P7R349 后续审计：本文“端到端成本”仅覆盖已计时阶段，尚缺外层进程墙钟；73.60% 为
> static/FSim/build 三阶段和。原先前沿内部哈希顺序首点命中也不构成排序能力证明；calls 按需
> 构建基线在同目标回放中更便宜。原始数字不变，解释范围见 `20260913_P7R349_EQUAL_TARGET_COST_AUDIT.md`。

日期：2026-09-13  
开发板 boot：`737f64bf-de64-4daa-9fa8-a216db9042eb`

## 研究问题

R50D 已证明最终融合程序的 `(bytes,calls,extra submissions)` Pareto 前沿可以安全减少 FPGA
候选，但为了得到该前沿，仍先对全部候选支付了 656.866 s 的预训练 ResNet50 完整图构建成本。
P7R336 在标签已暴露的 R50D 上开发了一个便宜代理：直接使用算子 lowered-TIR 的 expanded
DMA bytes、expanded DMA calls 和 weight-barrier 标志形成 Pareto 前沿。本轮在一个新的真实
ResNet50 workload 上严格验证：

> 能否在任何目标完整图、FPGA correctness 和 latency 标签出现之前冻结算子代理前沿，只构建并
> 上板该前沿，同时在补齐完整池后证明它保留 FPGA-correct oracle，并真正减少端到端构建成本？

目标层为 ResNet50-v2 `stage3_unit2_conv3`：`CI=256,CO=1024,H=W=14,1x1,stride=1,pad=0`，
编号 R50E。主性能指标仍为每轮 `candidate_latency/stock_latency` 的中位数；stock 是同轮漂移
参照，不参加候选 Pareto。

## P7R337：重启恢复

重启后 FPGA 仍为 `operating`，但 u-dma-buf 模块、tmpfs runtime、RPC 和 `/lib/firmware/vta_hpc.bit`
均不再存在。先将精确 `u-dma-buf.ko`、`tvm_rpc`、`libtvm_runtime.so`、`libvta.so` 和启动脚本只
上传到 `/var/volatile/vta_c3_ram`，加载 `udmabuf0=201326592` 并恢复 RPC。第一次前沿上板尝试
P7R343 又在候选 0 前发现缺少重启后的 bitstream 并 fail closed；它没有执行候选、没有产生
correctness 或 latency 标签。P7R344 随后从本地精确副本恢复
`vta_hpc.bit`（SHA-256 `7bf1ac95b1182c670cd25241111f33725b4ea37ed6d66f2df37b0b2e7df528d6`），
重新配置 FPGA 并重启 tmpfs RPC。失败尝试被原样保留，不计作候选派发。

## P7R338--P7R341：无标签冻结与算子资格

P7R338 在目标 lowering、FSim、完整图、FPGA 和 latency 之前冻结 2240 个 ConfigEntity 的完整域；
手写硬件谓词保留 355 点，再按既定哈希规则冻结 8 个 family、32 个 legacy 四模式身份。P7R339
绑定 P7R336 的开发规则和信息可见边界。P7R340 对 24 个正式三模式身份执行真实 lowering 和三 seed
FSim，结果为 12/24 static-pass、12/24 FSim-pass：original 5/8、input-stationary 5/8、
weight-resident-barrier 2/8。真实 compact-buffer、allocation 和 2D DMA 约束继续证明手写容量公式
不能替代编译器资格。

P7R341 仅在这 12 个合法身份的算子 lowered-TIR 上计算三轴 Pareto，冻结 2/12 个候选：

| family/mode | operator expanded bytes | operator expanded calls | barrier | 是否入前沿 |
|---|---:|---:|---:|---|
| R50EF03 input-stationary | 2,085,888 | 476 | 0 | 是 |
| R50EF02 weight-resident-barrier | 864,256 | 13,336 | 1 | 是 |

此时没有构建任何目标完整图，也没有接触板端或性能标签。

## P7R342--P7R347：真正先筛后构建、前沿上板与 oracle 补齐

P7R342 只构建 stock 和已冻结的两个代理前沿候选，实际模型构建总耗时 100.120 s。最终融合 TIR
表明两个候选分别为 `12,515,328 B/2,856 calls/0 extra submissions` 和
`5,185,536 B/80,016 calls/6 extra submissions`；这些高保真特征只用于审计，没有反向改变前沿。

P7R345 在恢复后的同一 boot 上只执行两个前沿候选，2/2 通过三 seed 完整图 correctness，并完成
28 次交错计时。前沿最好点是 R50EF03 input-stationary，配对 stock ratio 为 1.128623。

只有在前沿结果已经封存后，P7R346 才复制 immutable stock/front 产物并构建剩余 10 个候选；新增
候选模型构建耗时 342.688 s。P7R347 在同一 boot 上完成剩余池：10/10 FPGA-correct、140 次计时
调用正确。合并得到 12/12 的完整 FPGA-correct 池，oracle 仍是前瞻前沿中的 R50EF03
input-stationary，前沿 exact-oracle regret 为 0。

全部 12 个搜索候选都比 stock 慢，oracle ratio 为 1.128623。因此本轮证明的是搜索成本与可靠性，
不是完整 ResNet50 加速，也不是超过 TopHub。

## P7R348：端到端成本结果

| 成本 | 前瞻代理前沿 | 完整 12 点反事实 | 减少 |
|---|---:|---:|---:|
| candidate full-graph builds | 2 | 12 | 83.33% |
| stock + candidate build count | 3 | 13 | 76.92% |
| model-build wall | 100.120 s | 442.808 s | 77.39% |
| static + FSim + model-build wall | 122.950 s | 465.638 s | 73.60% |
| candidate dispatch | 2 | 12 | 83.33% |
| graph API invocations | 40 | 240 | 83.33% |
| FPGA kernel invocations | 4,352 | 28,560 | 84.76% |
| host call wall | 28,062.3 ms | 242,989.9 ms | 88.45% |
| logical DMA bytes | 5,755,572,224 | 61,372,495,872 | 90.62% |
| logical DMA calls | 2,109,224 | 13,660,248 | 84.56% |

20.567 s static diagnostic 和 2.263 s FSim diagnostic 是选择前已经实际支付的公共算子资格成本，
没有被藏入“节省”；加回这 22.830 s 后，选前本地资格与构建墙钟仍减少 73.60%。补齐 oracle 的
342.688 s 构建和后续上板只为研究审计发生，不属于前瞻策略
达到目标所需成本。冻结前沿的第一个候选即为 exact oracle；若只按 operator DMA bytes 顺序且
fail-fast，则需要两个派发，因此代理 Pareto 的首派发次序在本池优于简单 bytes，但单池不足以声称
稳定优势。

## 机制结果与反例

R50E 形成 7 个 same-tile 配对：5 个 input-stationary 全部 FPGA-correct 且全部更快，相对各自
original 的 stock-normalized 改善为 4.86%--53.89%；两个 weight barrier 一快一慢。R50EF03
weight barrier 的最终融合 DMA bytes/calls 分别下降 29.98%/37.14%，但 latency 反而恶化 1.09%。
因此算子 DMA 是有效的无标签 proposal/筛选先验，却不是 latency 定理；多目标冲突和真实 FPGA
correctness/latency gate 仍必须保留。

## 结论与边界

本轮是 P7R336 规则的第一个独立前瞻留出，真正按“便宜算子代理筛选 → 只构建前沿 → 只上板前沿
→ 标签暴露后补齐 oracle”的时间顺序执行。它关闭了 R50D 中“必须先构建所有完整图，故只能节省
最后一级 FPGA 成本”的主要缺口：在 R50E 上同时取得 77.39% 的实际模型构建墙钟节省；连同未隐藏
的 static/FSim 公共资格成本，选前本地墙钟仍节省 73.60%；候选派发节省 83.33%，
候选上板节省，并保留完整正确池 oracle。

仍不能外推为通用最优：这是一个 ResNet50 stage3 1x1 workload、一个 boot、12 点合法池；全部候选
都慢于 stock；logical DMA 不是物理 AXI；代理的 barrier indicator 只是同步类别；尚未跨模型或
跨 boot 独立复现，也不是持续学习的 ML tuner。论文可将其作为“硬件知识驱动、按需付费的分层搜索”
强正证据，但不能写成代理永不漏 oracle、Pareto dominance 蕴含更低 latency 或取得整网性能提升。

## 完整性

- P7R337--P7R348 共 12 个 manifest 目录、2,295 个条目全部重算 SHA-256 通过；
- 新增/修改的冻结器、构建器、分析器和上板执行器均通过 `py_compile`，相关定向测试通过；
- 实验结束时 boot 未变化，FPGA=`operating`，u-dma-buf=`201326592`，RPC cwd 位于
  `/var/volatile/vta_c3_ram/runtime`，未发现新的 EXT4/I/O/mmc 错误。

## 关键产物

- `20260913_p7r337_reboot_runtime_recovery_run01`
- `20260913_p7r338_r50e_bottleneck_adaptive_holdout_contract_run01`
- `20260913_p7r339_r50e_operator_proxy_pareto_holdout_contract_run01`
- `20260913_p7r340_r50e_full_local_qualification_run01`
- `20260913_p7r341_r50e_operator_proxy_front_contract_run01`
- `20260913_p7r342_r50e_operator_proxy_front_build_run01`
- `20260913_p7r343_r50e_operator_proxy_front_board_run01`（候选 0 前失败）
- `20260913_p7r344_reboot_bitstream_recovery_run01`
- `20260913_p7r345_r50e_operator_proxy_front_board_run01`
- `20260913_p7r346_r50e_operator_proxy_oracle_pool_run01`
- `20260913_p7r347_r50e_operator_proxy_oracle_completion_board_run01`
- `20260913_p7r348_r50e_operator_proxy_pareto_holdout_analysis_run01`
