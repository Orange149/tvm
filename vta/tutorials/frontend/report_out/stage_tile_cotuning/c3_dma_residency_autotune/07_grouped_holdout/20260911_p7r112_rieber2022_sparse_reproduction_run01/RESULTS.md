# P7R112 Rieber 2022 严格方法审计与稀疏池降级复现

> 结论边界：`degraded_sparse_pool_replay`。这不是完整 AutoTVM ConfigSpace 上的论文原样复现，也不是板端实验。

## 为什么必须降级

- 310 行由 10 个 workload、5 个独立 residency task 组成；每个 workload 只有 31 行和 7 个独特 ConfigEntity。
- 冻结文件没有完整 ConfigSpace/所有 Manhattan-1 邻居；同 mode 内只能复现 observed-knob-rank Manhattan-1。
- 论文的 E0=50、presample=min(1000,|S|)、750 次硬件测量无法用于每个仅 31 行的稀疏池，本实验缩放为 presample=16、parallel=4、E0=8。
- 310 行没有候选 latency，无法训练论文中的 AutoTVM performance model 或忠实运行 SA；第三级仅对确定性 [-7,7] 基础分数施加论文的 valid +1 / invalid -1e6 bias。

## 等 gross compiler-call 结果

所有方法每个 workload/seed 都只查询 16 个不重复 frozen lowering labels；R2/R3 只重排 R1 已缓存结果，因此 compiler discovery 指标与 R1 相同。

| 方法 | valid@16 median [IQR] | invalid ratio@16 median [IQR] | 首4个valid calls median [IQR] | 达成率 |
|---|---:|---:|---:|---:|
| Random | 10.00 [2.00] | 0.375 [0.125] | 6.00 [2.25] | 100.0% |
| 冻结静态生成顺序 | 11.50 [5.00] | 0.281 [0.312] | 5.00 [4.00] | 100.0% |
| R1 locality presample | 10.00 [3.00] | 0.375 [0.188] | 6.00 [2.00] | 100.0% |
| R2（compiler顺序同R1） | 10.00 [3.00] | 0.375 [0.188] | 6.00 [2.00] | 100.0% |
| R3（compiler顺序同R1） | 10.00 [3.00] | 0.375 [0.188] | 6.00 [2.00] | 100.0% |

## E0 与 validity-bias dispatch 代理

| 方法 | valid@8 median | valid@12 median | valid@16 median |
|---|---:|---:|---:|
| Random | 5.00 | 7.50 | 10.00 |
| 冻结静态生成顺序 | 5.50 | 8.50 | 11.50 |
| R1 locality presample | 5.00 | 8.00 | 10.00 |
| R2（compiler顺序同R1） | 4.00 | 7.00 | 10.00 |
| R3（compiler顺序同R1） | 4.00 | 8.00 | 10.00 |

E0 valid/invalid median 为 4.0/4.0，size shortfall median 为 0.0。

第三级只能说明已知 legality label 如何影响派发优先级；不能说明性能收敛、FPGA correctness 或真实 SA 收敛。
