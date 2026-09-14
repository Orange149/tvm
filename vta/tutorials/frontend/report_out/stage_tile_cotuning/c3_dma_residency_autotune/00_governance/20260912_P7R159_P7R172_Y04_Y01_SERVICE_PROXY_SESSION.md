# P7R159--P7R172：Y04 反例、服务代价修正与 Y01 latency-unseen 验证

## 结论

本轮没有延续“DMA 总字节越少，候选就一定越快”的过强结论。新增 1×1 YOLO 几何 Y04 后，
bytes-only 首派发比池最优慢 75.35%，说明请求启动次数和额外 submission 必须进入共享内存时间模型。
随后冻结的服务代价代理为：

```math
S(x)=B_{dma}(x)+65536N_{dma}(x)+131072\max(N_{submit}(x)-1,0).
```

两个系数是搜索用的等效字节惩罚，不是物理 AXI 流量或直接测得的硬件常数。该公式在已经暴露的
七个 workload 上完成开发冻结后，保持不变地进入从未取得候选 latency 的 Y01 六点恢复池。最终
service proxy 用 2 次 gross dispatch 命中正确池 oracle，bytes-only/bytes-first 用 3 次。

## Y04：第三个标签隔离完整池

- 几何：YOLOv3-tiny conv18，1×1，CI=256、CO=128、H=W=13。
- P7R159 在任何目标 lowering/FSim/FPGA/latency 标签前冻结 8 个 family × 3 模式。
- P7R161：24/24 static-pass，24/24 三 seed FSim-pass。
- P7R163：复用过的 RPC/bitstream 状态下 0/24 正确；该结果后来被判为执行状态污染，保留但不计为
  候选无效。
- P7R164：重载相同 bitstream 后，按冻结顺序选出的 canary 由错误恢复为 3/3 正确。
- P7R165：把停止默认 RPC、重载固定 bitstream、启动全新 RPC 放在任何健康/目标 tensor 分配之前；
  24/24 身份三 seed 正确，168/168 计时调用正确，pool oracle 为 3.137311 ms。

Y04 的排序结果：

| 方法 | budget=1 pool-oracle+2% | 首派发 regret |
|---|---:|---:|
| Random | 0% | 中位 75.35% |
| stock knob-XGB | 10% | 232.37% |
| paper minimum-access | 0% | 222.07% |
| bytes-only | 0% | 75.35% |
| bytes-first lexicographic | 0% | 37.96% |

Y04 候选内，latency 与 DMA calls 的 Spearman 约 0.9487，而与 DMA bytes 约 0.3745。最低字节的
weight barrier 只有 140,928 B，却有 54 次 LOAD、3 次提交，latency 4.328113 ms；pool oracle 是
input-stationary，490,880 B、52 次 LOAD、1 次提交，latency 3.137311 ms。字节少约 71%，反而慢
37.96%，这就是加入请求与提交代价的直接动机。

## P7R166：开发集参数冻结

输入为 P7Q 的 W01/W04/W07/W08 以及 Y00/Y03/Y04，共 7 个已暴露 workload。网格分别取
`0, 4, 8, 16, 32, 64, 128, 256, 512 KiB`，共 81 个系数组合；其中 24 个使七个 workload 的
首派发全部进入 pool oracle+2%。按“先最小化两系数之和，再最小化最大 regret、平均 regret 和固定
字典序”的预定规则，选择 64 KiB/request 与 128 KiB/extra-submission。开发集最大 regret 为
0.385478%，平均 0.066166%。Y04 已用于选参数，所以 P7R166 不是确认实验。

## Y01：严格的 latency-unseen recovery pool

审计发现旧 P7R126 的 17 个 correctness 结果带有 runtime profile 时间代理，因此没有拿完整 24 点
冒充严格未见集。P7R169 只冻结 P7R125 早先预注册的结构 rank-0 六点池：Y01F02/Y01F07 ×
original/input-stationary/weight-resident-barrier。P7R125 对这六点的 `candidate_timing_samples=0`，没有
`timing.jsonl`，候选 latency 从未存在；历史 correctness 已暴露这一事实则在合同中明确声明。
P7R167 是审计前生成的 24 点草案，P7R168 已缩到六点但尚未区分 recovery claim status；两者均未
接触开发板、未产生性能标签并被 P7R169 取代，不能作为正式合同入口。

P7R170 在 boot `aa7a3e5c-021d-4d6b-88ae-7f696faa567c` 上先执行 clean start，再上板：

- W05 健康门 3/3 通过；
- 六个身份中 3 个通过三 seed，3 个在首 seed 错误；
- fail-fast 实际 correctness 检查 12 次，完整策略本需 18 次，减少 33.33%；
- 三个正确候选各做 7 轮，21/21 timed call 逐元素正确；
- oracle 为 Y01F02 input-stationary，13.675876 ms；同 tile original 为 19.806108 ms，改善 30.95%；
- 对应每次推理 LOAD 3,753,984→3,580,928 B（-4.61%），LOAD calls 52→26（-50%）。

等 gross-budget 结果：

| 方法 | 达到 oracle+2% 的中位 gross dispatch |
|---|---:|
| service proxy（冻结） | 2 |
| bytes-only | 3 |
| bytes-first lexicographic | 3 |
| Random | 3 |
| stock knob-XGB | 3.5 |
| paper minimum-access | 4.5 |

service proxy 第一次仍选择了一个 FPGA-invalid barrier，第二次才命中 oracle。相对 bytes-first，
time-to-target 的已知五阶段成本为：gross dispatch 2 对 3、逻辑 LOAD 18,512,896 对 21,543,936 B、
逻辑 DMA calls 314 对 24,666、FPGA kernel invocation 8 对 25、墙钟 1.083922 对 6.863881 s。
墙钟差异大部分来自 bytes-first 第二个候选自身昂贵的 lowering/FSim/FPGA 路径，不能简化为纯硬件
执行加速。

## 主张边界

- Y04 是标签隔离的完整池；Y01 是 latency-unseen recovery confirmation，不是 correctness-label
  隔离池。
- 服务代价代理只对搜索顺序负责；首派发仍错误，说明它不能替代 validity/correctness 模型。
- Y01 只有 6 个候选，不能据此宣称跨网络通用；下一步应在更完整候选流上做在线多保真调度。
- runtime 的逻辑 LOAD/calls 不是物理 AXI burst。
- clean start 是当前板端实验正确性合同的一部分；P7R163 不能被解释为 1×1 VTA 不支持。

## 证据入口

- P7R166 开发冻结：`07_grouped_holdout/20260912_p7r166_shared_memory_service_proxy_development_run01`
- P7R169 Y01 合同：`07_grouped_holdout/20260912_p7r169_y01_six_point_latency_unseen_contract_run01`
- P7R170 板端结果：`07_grouped_holdout/20260912_p7r170_y01_six_point_clean_start_board_run01`
- P7R172 六点等预算重放：`07_grouped_holdout/20260912_p7r172_y01_service_proxy_recovery_replay_run01`

上述四项 ledger SHA-256 分别为：

- P7R166：`362e9edfbbbe6a0879c75132e175cb75a635971f39ee2c5ec2597a36b42b47d5`；
- P7R169：`8120c9921b91282dabe7c5bdbaf2c0fb8a24bce8443e8ec28090568af44329c1`；
- P7R170：`7e81792e755ffd46d23c9f153ebccd3fa81dbc24e9566a31692fda144c9cca7f`；
- P7R172：`eebc266be8100ead7b8c044939e738c796df695008f145e0dbf8719837976c0d`。
