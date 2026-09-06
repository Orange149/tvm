# P7C CPU-VTA Shared-DDR Qualification Review

更新日期：2026-09-05

## 结论

P7C 单 boot qualification 已完成；未读取候选吞吐标签，也未生成正式 shared-DDR 参数。
CPU memory worker 固定在核 0--2，VTA host 固定在核 3，降低了用户态 CPU 核竞争混入。

## 四组 Matched Control

- CPU `cache_resident` + VTA `compute_heavy` (`vta:01:03`): CPU slowdown `1.734x`，VTA slowdown `1.035x`，聚合有效带宽 `3.297 GB/s`。
- CPU `cache_resident` + VTA `dma_heavy` (`vta:15:19`): CPU slowdown `1.836x`，VTA slowdown `1.013x`，聚合有效带宽 `4.592 GB/s`。
- CPU `streaming` + VTA `compute_heavy` (`vta:01:03`): CPU slowdown `1.304x`，VTA slowdown `1.117x`，聚合有效带宽 `3.054 GB/s`。
- CPU `streaming` + VTA `dma_heavy` (`vta:15:19`): CPU slowdown `1.419x`，VTA slowdown `1.052x`，聚合有效带宽 `5.205 GB/s`。

## 证据边界

单 boot 只用于 qualification。只有相同子协议在至少三个独立 boot 可重复，且 streaming
相对 cache-resident control 的额外 slowdown 稳定存在，才可归因为共享 DDR 并进入公式。
进程 affinity 不能排除内核中断和驱动线程影响，因此当前结果称为 CPU-VTA interference，
不提前称为纯 DDR bandwidth contention。

## 产物

- qualification summary SHA256: `c58c868234c112d9856f0fe7a5691984ec6de2eae9368cb3a8d83a1910d8ab21`
- source session SHA256: `99c27b545c89f8ad2647643f9a8d08f960186e1cc3b6d6b6afdc92fffc61c493`

本阶段完成后停止。用户审阅前不进入 P7D，也不重跑额外 boot。
