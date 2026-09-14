# P7R329--P7R336：R50D 最终融合程序 Pareto 前瞻留出

日期：2026-09-13  
开发板 boot：`aa7a3e5c-021d-4d6b-88ae-7f696faa567c`

## 研究问题

P7R326 已经前瞻否定固定 `B+65536N` 标量代理，P7R328 随后只在标签已暴露的重叠组上说明
`(bytes,calls,submissions)` Pareto dominance 可以保留 oracle。本轮选择此前没有性能标签的
ResNet50-v2 `stage4_unit2_conv1`，即 `CI=2048,CO=512,H=W=7,1x1`，验证：

> 在完整 tile×residency 候选经真实 lowering/FSim 后，按最终预训练 ResNet50 融合程序的逻辑
> DMA 字节、请求数和额外 submission 构造 Pareto 前沿，是否能在显著减少真实 FPGA 测量的同时
> 保留完整 FPGA-correct 池 oracle？

主 latency 指标在接触目标板前冻结为每轮 `candidate_latency/stock_latency` 的中位数，绝对中位
latency 只作次指标。这样每个候选可以使用独立 clean start，避免非释放 u-dma-buf 分配累积，又用
同轮 stock 抵消不同 RPC 启动之间的漂移。

## P7R329--P7R332：无标签冻结、资格与最终程序前沿

P7R329 在目标 lowering/FSim/FPGA/latency 前冻结完整 original ConfigEntity 域 768 点；纯硬件谓词
保留 78 点，再按既定规则冻结 8 family、32 个 legacy 四模式身份。P7R330 在任何目标资格标签前
冻结三轴规则：只有另一程序在 LOAD+STORE bytes、LOAD+STORE calls、extra submissions 三轴均不
差且至少一轴严格更好时，候选才被淘汰；所有非支配程序必须进入真实 FPGA correctness 和 latency。

P7R331 将 24 个正式三模式身份全部执行真实 lowering：19/24 static-pass，18/24 三 seed FSim-pass。
按模式为 original 8/8、input-stationary 5/8、weight-resident-barrier 5/8 FSim-pass。P7R332 对 stock
和 18 个合法程序各构建一次完整预训练 ResNet50，捕获最终 CPUAccessRewrite TIR，并对目标层在图中
的两个真实出现点聚合十项 LOAD/STORE。19 份整图构建耗时 656.866 s，最终三轴前沿为 4/18：

| family/mode | bytes | calls | extra submissions | 前瞻 FPGA 结果 | 配对比/候选中位 |
|---|---:|---:|---:|---|---:|
| R50DF03 weight barrier | 3,781,632 | 464 | 2 | seed 0 输出 1000/1000 不同，拒绝 | n/a |
| R50DF05 original | 15,160,320 | 168 | 0 | 3/3 正确、7 轮可计时 | 1.103902 / 354.923 ms |
| R50DF07 input stationary | 14,959,616 | 518 | 0 | 3/3 正确、7 轮可计时 | 1.096984 / 352.567 ms |
| R50DF07 weight barrier | 2,978,816 | 1,016 | 2 | seed 0 输出 1000/1000 不同，拒绝 | n/a |

stock 只作相同完整图的正确性、profile 和漂移参考，不参加候选 Pareto 或搜索排序。

## P7R333--P7R335：前瞻波次与完整池 oracle

P7R333 只执行已冻结的 4 点前沿。每个候选均停止 RPC、重载固定 bitstream、启动默认 tmpfs RPC，
并且只同时分配 stock 与一个候选。两个 weight barrier 在首 seed 因整图 logits 不同而 fail closed；
两个正确点的 correctness/timing profile 都与最终 fused-TIR 十字段差值精确相等。前沿最好点为
R50DF07 input-stationary，配对比 1.096984。

P7R334 在标签已经暴露后才补测另外 14 个被支配点，用于构造无偏完整池 oracle，不作为前瞻排序
证据。13/14 正确，R50DF01 weight barrier 同样在首 seed 被拒绝。于是 18 个 FSim-pass 程序中
15 个 FPGA-correct：original 8/8、input 5/5、weight barrier 2/5。完整正确池最好点仍是前瞻前沿
中的 R50DF07 input-stationary，配对比 1.096984，Pareto 前沿 exact-oracle regret 为 0。

| 真实高保真成本 | 前瞻 4 点 | 穷举 18 点 | 减少 |
|---|---:|---:|---:|
| candidate dispatch | 4 | 18 | 77.78% |
| graph API invocations | 44 | 306 | 85.62% |
| VTA driver invocations | 3,768 | 28,096 | 86.59% |
| 被调用区间 host wall | 24,546.1 ms | 214,045.8 ms | 88.53% |
| logical DMA bytes | 6,475,800,576 | 53,914,372,096 | 87.99% |
| logical DMA calls | 810,902 | 10,521,762 | 92.29% |

这些节省只属于 FPGA 高保真阶段。static diagnostic 42.924 s、FSim diagnostic 3.509 s 和最终融合
构建 656.866 s 是两种策略都已支付的公共成本，不能藏在搜索器外，也不能据此宣称端到端墙钟已经
下降。当前 exact fused-Pareto 是“用昂贵编译换更少板端测量”的动作；下一步需要用便宜特征预测
前沿或按需构建，才可能同时降低端到端成本。

## 与简单 DMA-bytes 的关系

按 DMA bytes 顺序并对板端错误首错即停，在本池第 2 次派发就进入 oracle+2%/5% 带；冻结 Pareto
顺序同样需要 2 次。因此本轮不能声称 Pareto 改善 2% time-to-target。若要求 exact oracle，bytes
顺序需 6 次，Pareto 顺序需 3 次。准确口径是：

- Pareto 在一个真正的未见 workload 上安全保留 oracle，并大幅减少“完整测完候选集合”的板端成本；
- 对常用 2% 目标，简单 bytes+fail-fast 已经同样有效，三轴方法没有新增优势；
- Pareto 的公共 all-candidate fused-build 成本很高，尚不是完成的端到端在线调优器。

## 新的机制边界

R50D 形成 7 个可比较的 same-tile residence 对。5 个 input-stationary 均 FPGA-correct，相对 original
的 stock-normalized latency 改善 3.94%--41.34%；两个 FPGA-correct weight barrier 改善 5.51% 和
27.04%，另外三个 weight barrier 虽通过 lowering/FSim，却在完整图上数值错误。说明：

1. input 生命周期扩展在本几何方向稳定，但收益仍随 tile 从 3.94% 到 41.34% 变化；
2. weight barrier 不能按 mode 整体判定安全，必须绑定 tile、最终融合程序和真实 FPGA seed；
3. 最小 bytes 的 R50DF07 weight barrier 恰是错误点，正确性资格必须位于性能标签之前；
4. 全部 15 个正确候选仍比 stock 慢，本文结果是搜索成本/可靠性证据，不是超过 TopHub 的结果。

## P7R336：廉价算子代理的开发分析

为直接处理 656.866 s 公共构建瓶颈，P7R336 在标签已经暴露的 R50D 上测试只使用算子级最终 TIR
`expanded bytes / expanded calls / weight-barrier indicator` 的三轴前沿。代理同样保留 4/18 点，
与 exact fused 前沿重合 3 点；它漏掉的 exact-only 点是 FPGA-invalid R50DF07 weight barrier，换入
的 proxy-only 点是 FPGA-correct R50DF04 weight barrier，仍保留完整池 oracle。按已经暴露的实际
构建耗时回放，stock+候选构建由 656.866 s 降至 169.592 s（-74.18%），候选整图构建数由 18 降至
4（-77.78%）。

这只是 post-hoc 规则开发，不能计为第二次前瞻成功或真实构建节省。下一未见 workload 必须先按
该代理冻结前沿，只构建代理前沿并上板；标签暴露后再构建/补测其余点，才能验证它是否真正同时
减少 full-graph build 和 FPGA 测量成本。

## 完整性与边界

- P7R329--P7R336 共 8 个成功 manifest 目录、2,709 个条目全部重算哈希通过；
- 新执行器/构建器/分析器及相关辅助函数 6 项定向测试通过；
- 实验后 boot 未变化，FPGA=`operating`，u-dma-buf=`201326592`，默认 RPC 存活，未发现新的
  EXT4/I/O/mmc 错误；
- 结论只覆盖一个 ResNet50 workload、一个 boot、固定图和 schedule-regression logits 等价；逻辑
  DMA 不是物理 AXI，未做 ImageNet accuracy，也没有证明 Pareto dominance 是 latency 定理。

## 关键产物

- `20260913_p7r329_r50d_bottleneck_adaptive_holdout_contract_run01`
- `20260913_p7r330_r50d_final_fused_pareto_holdout_contract_run01`
- `20260913_p7r331_r50d_full_local_qualification_run01`
- `20260913_p7r332_r50d_resnet50_fused_program_pool_run01`
- `20260913_p7r333_r50d_final_fused_pareto_front_board_run01`
- `20260913_p7r334_r50d_final_fused_oracle_completion_board_run01`
- `20260913_p7r335_r50d_final_fused_pareto_holdout_analysis_run01`
- `20260913_p7r336_r50d_operator_proxy_pareto_development_run01`
