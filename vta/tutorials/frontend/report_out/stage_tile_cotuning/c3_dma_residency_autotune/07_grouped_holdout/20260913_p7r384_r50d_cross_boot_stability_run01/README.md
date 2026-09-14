# P7R384：R50D Pareto 跨启动稳定性审计

同一冻结 18 点最终融合程序池在两个独立 boot 上均为 15/18 FPGA-correct，说明三项 weight-barrier
失败分类稳定。旧 boot 的四点静态前沿包含精确 pool oracle；当前 boot 的精确 oracle 改为前沿外
R50DF04 weight-barrier (`043443c785cd`)，因此固定前沿只在 1/2 个 boot 保留精确 oracle。

当前 boot 的前沿最佳仍是 `58c047f4e5c7`，绝对中位 latency 仅比
新 oracle 慢 0.0327%，按预注册主指标
candidate/stock paired ratio 的 regret 为 0.1706%，
所以两个 boot 都保留 oracle+2% 质量带。结论应从“跨启动保留精确 oracle”降为“跨启动保持 2%
等价质量”，不能隐藏这次 exact-oracle miss。

根因不是随机哈希：当前 oracle 只被前沿中的 FPGA-invalid 候选 `9a84abfc7730` 在
`(bytes,calls,extra submissions)` 上支配。若在第一波 correctness 后移除两个 invalid front
候选并重新计算前沿，唯一新增点就是当前 oracle；这提示“invalid-dominator peeling”作为下一版
多保真动作。但该规则是看到本次标签后得到的开发修正，必须在新 workload 上先冻结再验证。
