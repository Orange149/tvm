# P7C CPU-VTA Shared-DDR Qualification Review

更新日期：2026-09-05

## 结论

P7C 单 boot qualification 已完成；未读取候选吞吐标签，也未生成正式 shared-DDR 参数。
CPU memory worker 固定在核 0--2；VTA host 的全部现有线程固定在核 3，后续线程继承该
affinity，从而降低用户态 CPU 核竞争混入。

## 四组 Matched Control

- CPU `cache_resident` + VTA `compute_heavy` (`vta:01:03`): CPU slowdown `1.065x`，VTA slowdown `1.010x`，声明流量/最长 wall 诊断比率 `3.373 GB/s`。
- CPU `cache_resident` + VTA `dma_heavy` (`vta:15:19`): CPU slowdown `1.020x`，VTA slowdown `1.014x`，声明流量/最长 wall 诊断比率 `6.874 GB/s`。
- CPU `streaming` + VTA `compute_heavy` (`vta:01:03`): CPU slowdown `1.032x`，VTA slowdown `1.052x`，声明流量/最长 wall 诊断比率 `3.241 GB/s`。
- CPU `streaming` + VTA `dma_heavy` (`vta:15:19`): CPU slowdown `0.995x`，VTA slowdown `1.066x`，声明流量/最长 wall 诊断比率 `6.535 GB/s`。

按单组 `slowdown > 1.05` 且 95% CI 下界大于 1 的 qualification gate，CPU 有 1 组、VTA 有 2 组信号。
- CPU signal: `cache_resident` + `compute_heavy` 为 `1.065x`；若 CPU 压力是 cache-resident，该信号不能归因为 DDR。
- VTA signal: `streaming` + `compute_heavy` 为 `1.052x`；stage 增量 `2.110 ms`，driver run 增量 `0.023 ms`，run-minus-driver 增量 `2.041 ms`。
- VTA signal: `streaming` + `dma_heavy` 为 `1.066x`；stage 增量 `1.359 ms`，driver run 增量 `0.266 ms`，run-minus-driver 增量 `1.036 ms`。

streaming 相对 cache-resident control 的 VTA slowdown 比率在 compute-heavy/DMA-heavy
下分别为 `1.042x/1.052x`。它提示 CPU streaming 压力会增加 VTA 延迟，但显著增量
主要落在 host runtime 的 run-minus-driver 部分。因此当前只能称为轻度、非对称的
共享内存/host-runtime 耦合，不能直接拟合成纯 DDR 带宽常数。
上述 GB/s 是声明流量除以最长组件 wall time；cache-resident 重复访问不等于 DDR 流量，
该值不是实测 DDR bandwidth，也不进入参数拟合。

## 证据边界

单 boot 只用于 qualification。只有相同子协议在至少三个独立 boot 可重复，且 streaming
相对 cache-resident control 的额外 slowdown 稳定存在，才可归因为共享 DDR 并进入公式。
进程 affinity 不能排除内核中断和驱动线程影响，因此当前结果称为 CPU-VTA interference，
不提前称为纯 DDR bandwidth contention。

## 产物

- qualification summary SHA256: `cf4f37d151573c9111f5c5444f8b4cdaa1626ba285331d0dc53c924e4b3b7885`
- source session SHA256: `e3e9cc12a4d5bb18272e42cdeace308067a5188e1e0f97eead22bdec6eb0cbb7`

本阶段完成后停止。用户审阅前不进入 P7D，也不重跑额外 boot。
