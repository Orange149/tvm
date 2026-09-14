# 历史 Top-20 四种切图的 FPS 与 DMA 对照

固定板卡、HPC bitstream、旧高性能 stage 二进制和当前 instrumented runner。每个代表方案先串行再并行运行 22 帧；吞吐丢弃前 2 帧，profile 计数按全部 22 帧归一化。四组串行/并行输出均各自唯一且逐字节相同。

| 拓扑 | VTA island | 实测 FPS | VTA run 中位和 | VTA mutex wait 中位和 | DDR→片上 LOAD 次数/帧 | LOAD payload/帧 | 权重 LOAD/帧 | 框架 H→V 字节/帧 | 框架 V→H 字节/帧 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A：03..17 | 1 | 10.872 | 70.042 ms | 0.001 ms | 1598 | 12.026 MB | 7.528 MB | 0.803 MB | 0.100 MB |
| B：03..15 | 1 | 10.310 | 68.788 ms | 0.001 ms | 1532 | 11.806 MB | 7.397 MB | 0.803 MB | 0.301 MB |
| C：03..16 | 1 | 10.994 | 70.329 ms | 0.001 ms | 1598 | 12.026 MB | 7.528 MB | 0.803 MB | 0.201 MB |
| D：03..12 + 15..19 | 2 | 10.501 | 69.811 ms | 20.337 ms | 1594 | 15.266 MB | 11.067 MB | 1.004 MB | 0.301 MB |

这里能提炼出四条可用于切图搜索的规律：

1. **切点变化不一定改变 VTA 内部 DMA。** A 与 C 的 LOAD/STORE 次数和 payload 完全相同，说明 units 16..17 之间的切点主要改变 GraphExecutor 边界张量，而没有改变实际 VTA 卷积搬运。此类候选不必重复做 tile 搜索，可按相同 VTA workload signature 合并。
2. **边界张量数量与片上 DMA 必须分账。** B 的 VTA 内部 LOAD 略少，但残差边界返回两个张量，V→H 为 0.301 MB/帧，是 A 的 3 倍。只看 VTA tile 会漏掉跨 Executor 物化，只看边界字节又会漏掉片上重复载入。
3. **增加 VTA island 会增加回落和互斥等待。** D 相对 B 多一次 CPU→VTA 进入，H→V 字节增加 25%；第二个 VTA stage 在流水中还出现约 20.337 ms 的 mutex wait，因为两个逻辑 island 仍串行复用一个物理 VTA。
4. **请求次数不能单独代表 DDR 压力。** D 与 B 的 LOAD 次数只差 4.05%，但总 LOAD payload 高 29.30%，其中权重 payload 高 49.61%。因此剪枝至少同时看 `calls`、`bytes`、`weight/input bytes` 和 single-VTA scheduled service。

据此可以把第一创新点的内存感知规则写成：候选先按 VTA workload signature 去重；对新增 island 计算额外 H→V/V→H 边界和单物理 VTA mutex demand；对 tile 记录 LOAD/STORE 请求粒度与输入/权重重复 payload；只有当 VTA 服务下降足以覆盖新增 DDR/边界代价时才保留该切图。最终仍以实际并行 FPS 决胜，静态规则只负责剪枝和排序。

注意：这些计数是 runtime API 口径，不是 AXI burst 或 compute-stall cycle。四组单次重放的 FPS 与历史值接近，但尚未做跨 boot 置信区间；当前用途是识别机制和构造搜索特征，不是声称微小 FPS 差异显著。
