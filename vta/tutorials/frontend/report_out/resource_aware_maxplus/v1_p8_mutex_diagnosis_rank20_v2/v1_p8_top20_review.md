# P8 Top-20 Zero-Copy Board Comparison

> **已废止：** 本文件使用逐帧完成间隔中位数估计吞吐，会漏掉突发流水线中的长空洞。
> 请使用同目录 `v1_p8_top20_review_reanalysis.md` 的完整 block 窗口结果。

更新日期：2026-09-06

同一 boot 内对冻结 Top-20 逐项执行 B0 普通拷贝与 B2 双 slot zero-copy。每个模式两个 block，奇偶 rank 采用相反顺序。

- 功能 gate：`1/1`。
- FPS 提升为正：`1/1`。
- 两次配对效应同方向：`1/1`；B0/B2 两种模式的 block spread 均不超过 5%：`1/1`。
- FPS 相对增量：均值 `+2.31%`，中位数 `+2.31%`，范围 `+2.31%--+2.31%`。
- 边界 API 服务时间减少：均值 `94.72%`，中位数 `94.72%`。
- 证据边界：这是单 boot 全候选扫描；候选之间共享同一 boot，不能替代逐候选跨 boot 置信区间。

| Rank | Stages | Threads | B0 FPS | B2 FPS | FPS 增量 | B0 II (ms) | B2 II (ms) | 边界 API 减少 | 消除字节/帧 | Gate |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 20 | 5 | `4,1,2,1,4` | 10.076 | 10.308 | +2.31% | 99.247 | 97.008 | 94.72% | 2609152 | pass |

## Mutex attribution

- B0/B2 stage 1 VTA `run`：`49.634/49.692 ms`。
- B0/B2 stage 3 VTA `run`：`20.111/20.154 ms`。
- B0/B2 stage 3 VTA mutex wait：`25.871/25.553 ms`。
- VTA profiler 在两种模式中记录相同 instruction、LOAD bytes 和 STORE bytes。

因此本轮没有观察到 zero-copy 增加 VTA 设备工作或 VTA 执行时间。旧 runner 曾把 B2 的 mutex
等待混入 stage `set/run`，相关 stage 变慢结论无效。该候选的端到端结果仍只来自一个 boot，需
独立重启复测后才能形成跨 boot 结论。
