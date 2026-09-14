# 旧 Top-20 高性能基线恢复与 DMA 对照

日期：2026-09-07。板卡：`root@192.168.1.247`。拓扑 B：CPU 00..02 / VTA 03..15 / CPU 16..20，线程为 4/1/2，queue depth 为 2。

## 结论

旧 Top-20 的 10--11 FPS 是真实的原生并行流水线吞吐，不是静态估计。重新加载冻结的 HPC bitstream 后，复用旧 VTA stage 二进制并改用当前 runner，22 帧丢弃前 2 帧得到 **10.724 FPS**；22 帧串行与流水输出均只有一个哈希且逐字节一致。历史 rank05 为 10.656 FPS，复测相差约 0.63%。

当前有界 AutoTVM 的 topology B 虽然比它在本轮新建的“固定 tile”对照快，但只达到 **3.482 FPS**，仅为旧基线的 32.47%。因此，`pipeline_validation` 的 2×2 结果只能说明那四个新编译包内部的 B/D 顺序发生变化，不能证明当前调优超过原 Top-20，也不能替代旧 Top-20 的实际 FPS 排名。

换用当前 runner 后旧二进制仍达到 10.724 FPS；关闭/开启 runtime profile 时分别为 10.724/10.860 FPS，差异处于板端噪声范围。性能缺口由 VTA 编译调度造成，不是“流水线没有并行”、runner 版本或 profile 开销造成。

## DMA 粒度与重复搬运

下面均按每帧归一化。旧基线累计 22 帧，当前 bounded-tuned B 累计 20 帧。

| 指标 | 旧高性能 incumbent | 当前 bounded-tuned B | 新/旧 |
|---|---:|---:|---:|
| VTA stage 中位 run | 73.291 ms | 272.365 ms | 3.72× |
| LOAD 请求数 | 1,532 | 22,840 | 14.91× |
| LOAD payload | 11.806 MB | 44.656 MB | 3.78× |
| 小 LOAD 请求数（payload < 4096 B） | 444 | 22,040 | 49.64× |
| 带 stride 的 LOAD 请求数 | 784 | 9,344 | 11.92× |
| 输入 LOAD payload | 4.388 MB | 20.306 MB | 4.63× |
| 权重 LOAD payload | 7.397 MB | 24.281 MB | 3.28× |
| STORE 请求数 | 92 | 1,080 | 11.74× |
| STORE payload | 1.204 MB | 1.204 MB | 1.00× |
| synchronize 次数 | 14 | 14 | 1.00× |
| 每次运行提交的 VTA 指令数 | 3,270 | 42,630 | 13.04× |
| host 等待设备完成 | 62.799 ms | 212.626 ms | 3.39× |

规律不是“输出数据变少”：两者每帧 STORE payload 完全相同，synchronize 次数也相同。差异是当前候选把同样的输出拆成更多 STORE，并因过细空间/通道分块反复加载输入和权重。它同时增加请求数量、累计 LOAD payload 和设备执行等待，最终拖慢 VTA stage 与流水吞吐。

这些数字是 VTA runtime 的 LOAD/STORE API 请求、payload 和 host wait，不是 AXI performance monitor 测到的物理 DDR burst，也不能把全部 `device_run_wait_us` 解释为 compute stall。但它们已经足以建立可复现的代理规律：

1. 相同 segment、相同输出字节下，优先排除 LOAD/STORE 请求数和小请求数异常放大的 tile；
2. 对输入/权重 payload 分别归一化，识别因空间 tile 过小导致的重复载入；
3. 保留旧高性能二进制作为 incumbent，候选必须先过正确性，再同时比较 stage 时延、DMA 特征和实际流水 FPS；
4. 静态重排只能生成待测顺序，不能称为新的 Top-20 FPS。

## 正确性与 FPGA 状态

第一次直接重放旧二进制时输出不稳定。固定协议重新写入 SHA256 为 `7bf1ac...28d6` 的 `vta_hpc.bit` 并重启 HPC RPC 后，串行和流水各 22 帧都稳定为 top-1 285、FNV-1a `a8c613584e081e5f`。因此后续实验必须把“重载固定 bitstream + RPC/runtime 指纹检查”放在测量前，不能只根据文件存在推断 PL 当前状态。

## 证据文件

- `summary.json`：归一化对照、哈希和方法结论；
- `after_reprogram_serial_t2.jsonl`：最终串行正确性记录；
- `after_reprogram_pipeline_t2.jsonl`：最终无 profile 流水吞吐记录；
- `after_reprogram_pipeline_t2_profiled.jsonl`：最终带 profile 流水记录；
- `profile_old_b_after_reprogram/benchmark_totals_status.json`：22 帧累计 VTA runtime 计数；
- `../pipeline_validation/tuned_b/`：当前 bounded-tuned B 的 20 帧对照。

旧 VTA `graphlib.so` 哈希为 `6db838...c7ea`，当前无 tuning history 的重编译哈希为 `bef5c9...afe8`，bounded-tuned B 为 `57bf9e...6cbc`；三者的 topology B stage1 `graph.json` 哈希相同。这证明网络 segment 相同而 lowering/schedule 二进制不同。
