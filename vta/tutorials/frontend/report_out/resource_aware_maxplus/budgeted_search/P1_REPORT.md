# RAMPS P1 来源与低预算审计

更新日期：2026-09-01

## 结论

- 200 个候选只能定义 `measured-pool oracle`，尚未证明覆盖完整合法切图空间。
- M0/M1 的排序输入来自编译期 workload 和默认静态常数；候选实测值只作为回顾性标签。
- direct runtime profile 读取每个候选的 stage/copy 时间，只能作为 mechanism upper bound。
- 200 个候选均缺少可用的 measured CPU core demand，因此 M2 结果不具 publication 资格。
- 历史产物没有可靠的编译和上板 wall-clock 字段，当前只能报告 evaluation count。

## 低预算结果

| 模型 | regret@1 | regret@5 | regret@10 | 到 95% oracle 次数 | 资格 |
|---|---:|---:|---:|---:|---|
| `legacy_static_score` | 0.305 | 0.266 | 0.265 | 47 | 可用 |
| `m0_single_vta_compute` | 0.481 | 0.337 | 0.318 | 60 | 可用 |
| `m1_static_communication` | 0.481 | 0.337 | 0.323 | 67 | 可用 |
| `m2_invalid_core_proxy_diagnostic` | 0.150 | 0.072 | 0.067 | 11 | 仅诊断 |

随机基线的 95% oracle 次数中位数为 33，P10--P90 为 6--88。

M1 没有改善 M0，说明‘加上默认 DMA 常数’不足以形成有效排序。M2 的表面改善来自
已否决的 core-demand 代理，不能据此增加模型复杂度。下一步只应测能修复该排序缺口的
最小参数：代表性 CPU stage、VTA island 端到端 service，以及边界 copy/sync。

## 审计摘要

```text
records: 200
M0/M1 zero-feedback eligible: 200
M2 measured core-demand eligible: 0
coverage: measured_pool_only
cost ledger: count_only_wall_clock_unavailable
```

## 决策

暂停 H2 20-template 和完整 HardwareProfile。先设计一个不超过约 12--20 个独立 case 的
P3-min profile，并要求它在同一 measured pool 上显著降低低 K regret；否则采用更简单的
stage lookup + batch shortlist，而不继续扩展 Max-Plus/FIFO 模型。
