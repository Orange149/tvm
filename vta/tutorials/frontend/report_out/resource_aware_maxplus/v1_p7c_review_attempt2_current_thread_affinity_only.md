# P7C CPU-VTA Shared-DDR Qualification Review

更新日期：2026-09-05

## 结论

P7C 单 boot qualification 已完成；未读取候选吞吐标签，也未生成正式 shared-DDR 参数。
CPU memory worker 固定在核 0--2，VTA host 固定在核 3，降低了用户态 CPU 核竞争混入。

## 四组 Matched Control

- CPU `cache_resident` + VTA `compute_heavy` (`vta:01:03`): CPU slowdown `1.048x`，VTA slowdown `1.008x`，聚合有效带宽 `3.380 GB/s`。
- CPU `cache_resident` + VTA `dma_heavy` (`vta:15:19`): CPU slowdown `1.045x`，VTA slowdown `1.017x`，聚合有效带宽 `6.868 GB/s`。
- CPU `streaming` + VTA `compute_heavy` (`vta:01:03`): CPU slowdown `0.971x`，VTA slowdown `1.042x`，聚合有效带宽 `3.265 GB/s`。
- CPU `streaming` + VTA `dma_heavy` (`vta:15:19`): CPU slowdown `1.038x`，VTA slowdown `1.067x`，聚合有效带宽 `6.544 GB/s`。

唯一超过 5% 且 95% CI 不含 1 的 VTA 信号来自 `streaming + dma_heavy`。该组 VTA
stage 增量约 `1.378 ms`，其中 profiler 的 driver run 增量约 `0.242 ms`，
其余主要落在 host runtime 的 run-minus-driver 部分。因此当前只能称为轻度、非对称的
共享内存/host-runtime 耦合，不能直接拟合成纯 DDR 带宽常数。

## 证据边界

单 boot 只用于 qualification。只有相同子协议在至少三个独立 boot 可重复，且 streaming
相对 cache-resident control 的额外 slowdown 稳定存在，才可归因为共享 DDR 并进入公式。
进程 affinity 不能排除内核中断和驱动线程影响，因此当前结果称为 CPU-VTA interference，
不提前称为纯 DDR bandwidth contention。

## 产物

- qualification summary SHA256: `c4407035f4ba21d1cc444c3b78d7b2139e8a4ff64a9ac1f7c1229d95cbc5cb72`
- source session SHA256: `bc73d6fc20a78d0c3a649d73933e380917e00b7d0de9b5800b285afa9ddd4588`

本阶段完成后停止。用户审阅前不进入 P7D，也不重跑额外 boot。
