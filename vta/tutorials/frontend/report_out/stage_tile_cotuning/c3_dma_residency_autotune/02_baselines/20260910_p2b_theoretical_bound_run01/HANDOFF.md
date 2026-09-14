# Handoff

主要数据：

- `theoretical_bounds.csv/json`：10 类 workload 的 MAC、compute/DMA 下界、full-duplex 与 shared-serial 两种情景。
- `historical_correct_candidates.csv/json`：32 个历史正确候选的 latency、效率、差距和测后 runtime traffic dominance。
- `summary.json`：总体计数和聚合结果。
- `FORMULAS.md`：公式、假设和适用边界。

后续引用时必须同时带上三项限定：AXI 带宽是 optimistic 而非实测；perfect-overlap/no-overlap 是两条分析参考线；单算子 operator rate 不是整网 FPS。

若要验证请求启动延迟，需要另做板端计数或受控微基准。本次结果不能估计每个 DMA request 的固定延迟，也不能用历史 runtime counter 为未测候选做无泄漏排序。
