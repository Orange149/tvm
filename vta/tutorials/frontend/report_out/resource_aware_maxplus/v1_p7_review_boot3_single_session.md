# P7B-1 Single-Boot Qualification Review

更新日期：2026-09-04

## 结论

P7A instrumentation 与 P7B-1 单 boot qualification 已完成。36/36 组 CPU memory case 和
16/16 组真实 CPU stage pair 均通过 correctness、determinism、PMU、同步和统计 gate。
本阶段没有读取候选吞吐拟合参数，也没有把 `+34 ms` 写回公式。

## 为什么需要这一步

旧模型把多个 CPU stage 的孤立服务时间直接放入 pipeline 最大值，但实际 TVM worker 会在
四核 CPU 上竞争。P7B-1 使用两个独立 native 进程，并在每个计分样本前通过 barrier 同时
启动，直接测量这种竞争，而不是使用 `sum(threads) <= 4` 之类的搜索限制代替物理测量。

## 实测结果

- 256 KiB cache-resident copy：t1/t4 为 `9.177/18.813 GB/s`。
- 64 MiB streaming copy：t1/t4 为 `4.335/7.416 GB/s`，3--4 线程已接近饱和。
- 64 MiB streaming write：t3/t4 为 `7.935/7.781 GB/s`，增加线程没有继续提高带宽。
- 16 组 CPU pair 的 A 侧 slowdown 范围为 `1.013--1.683x`，B 侧为 `1.020--2.124x`。
- 单 boot 中 A/B 分别有 `14/14` 组满足 slowdown >5% 且 95% CI 不含 1。
- 并发/孤立 PMU 指令数中位比范围为 `1.000--1.007`，而 cache-miss 比值中位数为 `1.012`；
  slowdown 来自相同计算工作受到资源竞争，而不是并发时额外执行了算子。

## Review 后的修正

最初把并发 wall-time CV <=15% 作为硬 gate，导致 2 组虽有稳定 PMU 工作量和显著 slowdown
却被拒绝。并发 elapsed time 的多峰波动本身来自被测的 Linux/TVM 调度竞争，因此最终冻结为：
孤立基线 CV <=10%，slowdown 中位数 bootstrap 95% CI 相对半宽 <=25%；并发 CV 继续报告为
不确定性。修改后统一重跑全部 16 组，而非选择性重测失败 case。

runner 同时补上 `--input-files`，因为 `cpu:16:20` 与 `cpu:17:20` 需要 main/residual 两个
输入张量。忽略第二个输入会使所谓 stage profile 不对应真实切图。

## 证据边界

这些结果来自同一个 boot，只证明实验可运行并发现明显 CPU 并发耦合。正式 slowdown surface
至少还需两个独立 boot 复现；在此之前 `v1_p7_physical_profile.json` 不生成，静态公式不更新。
P7B-2 固定运行时开销与 P7C CPU-VTA 共享 DDR 竞争尚未执行。

## 产物

- plan SHA256: `598e63a93fc77737be68f32f5d574031e4e1ba74778837a15a6648a5085f61ec`
- memory session SHA256: `b63c77dea5db34693027f6a772655ec3e664605ed671dd81d78de9fd55e1a1e4`
- CPU-pair session SHA256: `b610d6e7c1d118c15ba33d842d7976748000b36071a00f96c94d730903b00a5d`
- combined session SHA256: `02e4af42f9b896c4119c81d41e9ac2bec4f3b940f33d822d0a865050eaf58d09`
- local regression: `24 passed`

当前阶段停在 P7B-1 review。用户确认前不进入 P7B-2 或 P7C。
