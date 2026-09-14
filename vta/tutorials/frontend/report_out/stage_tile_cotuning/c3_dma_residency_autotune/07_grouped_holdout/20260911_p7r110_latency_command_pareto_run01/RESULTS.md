# P7Q latency–command backing Pareto

性质：冻结 P7Q/P4J 标签的离线开发分析，不是新的 FPGA 实验。

将 79 个真实 FPGA latency 标签与相同 candidate identity 的 FSim command peak 连接。容量模型为
一条 live instruction queue 和一条 live UOP queue，两者各自至少 4 KiB 并按 4 KiB 对齐。

| Workload | 合法候选 | latency oracle | oracle backing | Pareto 点数 | 2% 等价带最小 backing |
|---|---:|---:|---:|---:|---:|
| W01 | 19 | 3.048571 ms | 8 KiB | 1 | 8 KiB |
| W04 | 18 | 2.772418 ms | 8 KiB | 1 | 8 KiB |
| W07 | 21 | 2.653857 ms | 8 KiB | 1 | 8 KiB |
| W08 | 21 | 5.027001 ms | 8 KiB | 1 | 8 KiB |

四个 latency oracle 都已经处在这个容量模型的最小 backing 上；0%、2%、5%、10% 等价带均没有
得到比 oracle 更小的容量。因此当前候选池不支持“command-aware 联合目标改善配置选择”，只能
支持 exact allowlist 在部署时安全定容。

边界：这里分析的是 host command backing，不是 tensor SRAM、物理碎片、并发队列、replay、FPS
或 FPGA 面积。完整逐候选 Pareto 记录见 `analysis.json`。
