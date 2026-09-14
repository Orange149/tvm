# G0 扩展 discovery：冻结 A/B/C/D 自然可达方向普查

日期：2026-09-08。**预注册的 60 个条件已全部完成并独立审计，完整 G0 尚未通过。** 本报告对应 [预注册](sweep_a5220e22_natural60/preregistered.json)，SHA-256=`e5c7a204f6d2330b07feee2e41ac207b220ed0f4b902334a377340dbabbb9ec6`。本轮不是新的 AutoTVM、生产流水门控、物理 DDR 计数器实验或正式五 boot confirmation。

## 预先固定的范围与方法

- 从旧 shared K2 readiness trace 枚举 A/B/C/D 的全部 CPU–VTA 有向 encounter；先排除 newly-ready VTA 当时已有其他 VTA owner、无法立即启动的事件。
- 得到 22 个有观测支持的方向。每方向自然样本不少于 10 时，取 active-age 的 q25/q50/q75；少于 10 时只取中位数，并保留 sparse 标签。合计 A/B/C/D=`12/8/12/28` 个时龄条件，共 60 个。
- D 的 `VTA3 active→CPU4 ready` 与反方向在旧 trace 中均无样本，单列为未观察到，不声称永久不可达，也不混入 measured negative。
- 选择覆盖全集，不依据此前 pair slowdown 选取条件。A 的 CPU0/VTA1 中位 pilot 已看过，因此这是探索性普查，不是前瞻 held-out。按 seed `908` 打乱条件顺序，每个条件内部固定 10 个 `allow/wait/wait/allow` ABBA block，之前各策略 5 个 warmup。
- 执行前保存各 stage 的编译哈希、unit 集合、线程、contract 输入/输出字节及历史 VTA 逻辑 DMA 描述。CPU contract 字节不是 CPU 总访存或内存强度标签；历史 DMA descriptor 不是本轮重新测得的物理 DDR traffic。此自然域普查不替代原计划的 isolated low/mid/high × compute/mixed/DMA-heavy 3×3 因果矩阵。

每条件 50 个 pair 样本，预期共 3000 个 pair 样本、6000 次被测 stage invocation，外加每条件一次完整串行 reference。输入、stage binary、tile 和 bitstream 固定。CPU 保留各配置的名义线程/affinity，VTA host 固定 CPU3；它与 CPU worker 可能共核，不能把效应自动归因为 DDR。

使用前一轮已资格化的同进程双 worker harness，SHA-256=`e0e36ad3bbf7a9f0cc9680673aec9752e73331bfba060ab68e23d53663055f01`。双方 binding 完成后，先放行 active，再按冻结偏移将 pending 置为 eligible；allow 当即放行，wait 等 active graph-run 结束。两侧结束后才 get_output 并校验，整条件结束才写 JSON。没有在测量期间部署新硬件、切换 tile、调整线程或添加生产控制器。

两项任务共用固定输入生成的独立 stage 输入，模拟不同帧的就绪任务，不执行同一帧的实时依赖交接。CPU/VTA 使用各自 GraphExecutor 普通 storage 路径，不是完整 K2 外部 slot 绑定；准备/绑定在 active 运行前完成，不重放自然流水的多方 CPU 竞争、slot 压力和 binding 并发。这些差异必须作为外部有效性限制保留。

## 结果口径和状态资格化

主测量值为 pair graph-run makespan：`max(active_end,pending_end)-active_start`。每个 ABBA block 先求 allow 两次与 wait 两次的均值，再算 `1-wait/allow`，最后跨 10 个 block 取中位；正值才表示等待更快。不能将 makespan 倒数当作流水 FPS，也不能与前轮完整流水 P95 互换。

预注册状态条件为：至少 36/40 计分样本在 pending eligible 时 active 仍未结束。所有不匹配条件完整保留，不删除后重新估计或悄悄缩短 offset。分析另报真实 allow graph-run overlap、wait 零 overlap、实际 active-age 与自然观测范围，以免把“发出了 allow”误认为“真的发生了重叠”。

只有状态满足、配对中位改善 `>=5%` 且 `>=8/10` block 同向，才进入“值得复核的探索性候选”；这不是正式 protect 标签，不能取代重复噪声门槛、五 boot、compute-only 负对照、memory-path 归因与机会收益上限。未达到该条件不自动赋予 allow 标签。

## 实测结果与 go/no-go

60/60 条件全部完成，共 3000 个 pair 样本（600 warmup、2400 计分），6000 次被测 stage invocation 的输出均与各自串行 reference 一致；60 次完整 reference 的最终输出均与对应冻结 topology 的 FNV 一致。无测量失败或丢弃重跑。每条件 50 行，样本序号、ABBA 顺序、时间戳、wait 不早于 active done、makespan 公式和原始日志 SHA-256 全部通过审计。

| 配置 | 条件数 | active 状态满足 | 状态不匹配 | 满足条件中的等待退化范围（block 配对中位） | 探索性保护候选 |
|---|---:|---:|---:|---:|---:|
| A | 12 | 12 | 0 | +27.90%～+94.36% | 0 |
| B | 8 | 8 | 0 | +3.65%～+60.58% | 0 |
| C | 12 | 12 | 0 | +27.76%～+94.52% | 0 |
| D | 28 | 25 | 3 | +0.96%～+66.72% | 0 |

表中正值表示 `wait/allow-1`，与 CSV 的 `paired_wait_gain_percent` 符号相反。范围是多个条件的描述性范围，不能视为置信区间、跨 boot 稳定效果或某 topology 的流水性能。

57 个状态满足条件均有 40/40 样本在 eligible 时 active 未结束；每条件 20/20 个 allow 计分样本确实发生 graph-run overlap，所有 wait 样本均无 graph-run overlap。**这 57 个条件分别在 10/10 ABBA block 中都是 wait 更慢**，并非仅因 5% 候选门槛而漏掉小幅正收益。它支持“本轮已复现的 pair 状态没有保护收益候选”，不证明每种小差异都超过噪声，更不证明完整自然流水永远不能获益。最小退化约 0.96% 的 D CPU4→VTA1 只有 2 次历史自然 encounter 支持，也不能正式标 allow。

三个不匹配条件均为 `D CPU0 active→VTA3 ready` 的 q25/q50/q75，offset 为 `57.550/68.147/75.269 ms`。它们各自 0/40 样本在 eligible 时仍有 active，allow 的实际重叠为 0/20。自然流水里 CPU0 同时受到其他任务、共享交接等因素影响，单独配对并未复现该状态；本轮不能具体断言是哪一种因素所致。三者近零 allow/wait 差异**不是负收益证据，也不是已证明的 allow cell**。后续若研究这些状态，必须先复现相应并发背景并重新预注册；不得事后缩短 offset 后声称原条件已经覆盖。

同步、环境及测量边界：各条件 release→start P95 的最大值为 `0.3923 ms`，需在后续小效应确认时计入扰动审计，不能将其当作已通过的 policy-off 开销证明。条件前后 IIO 快照中 PS 温度范围约 `31.99–35.75 °C`，PL 约 `31.04–32.64 °C`；这不是逐 block 温度控制。boot 始终为 `a5220e22-f562-40be-bbe9-058b6fbeed50`，bitstream/runner/阶段产物已核对，结束后 FPGA operating、缓冲 192 MiB、无残留测量进程。

### 本阶段决策

1. **本轮已复现的 pair 域：未发现保护候选，当前不进入 G1/G2/G3。** 没有理由仅因“有重叠、单 stage 有 slowdown”就开发复杂控制器；暂停控制器投入是范围受限的工程止损，不是普遍不可能性证明。
2. **完整 G0：未通过，不冒充全面完成或全面否证。** 3 个原自然时龄条件未复现，另有 2 个方向无自然观测；isolated 内存强度分层、compute-only 负对照、正式噪声门槛、五 boot、memory-path 归因和 opportunity ceiling 尚未完成。`formal_protect_allow_labels` 与 `opportunity_ceiling` 保持 null，不能将“没有候选”替换为“收益上限=0”。
3. 若继续 C3，先解决未复现状态或提出事前有依据的新机制对照，再申请相应预算；不为了补第三点盲目扩充随机组合/调偏移。新机制也需与本轮负结果一起报告。
4. C1 的静态 DMA/共享边界分析、编译上下文审计与 C2 零拷贝证据不被本结果否定。当前优先收口 C1 的 E0/E1/E2/E5a 和 C2 正确性/high-water；本轮不会自动完成这些任务，也不会把 pair 描述性 slowdown 加成新的物理 DDR 成本系数。

## 可复现产物

- [完整预注册、执行顺序与 stage 内存描述](sweep_a5220e22_natural60/preregistered.json)
- [逐条件汇总与全部 ABBA block](sweep_a5220e22_natural60/summary.json)
- [60 个条件完整 CSV](sweep_a5220e22_natural60/all_cells.csv)、[独立审计及逐条件真实重叠/温度](sweep_a5220e22_natural60/census_audit.json)。汇总 SHA-256=`cffa2a57f2724f75b8b70c7cbd6e56f59c52a549f964ff18846b28f3aa9d1b78`。
- 每条件目录：实际 command、stdout、50 个原始 pair 样本、前后 PS/PL IIO raw/offset/scale 温度。温度是条件级前后快照，不是每 block 温度 gate。
- 脚本 `run_stage_memory_g0_sweep.py` 保存 runner、source/helper、trace、历史 DMA、stage/runtime 和原始结果哈希；`analyze_stage_memory_g0_sweep.py` 在完成后独立重算并输出 `all_cells.csv` 与 `census_audit.json`。
- 主机侧 11 项 G0 回归测试覆盖原有协议、makespan/ABBA、真实 treatment exposure，以及枚举完整性、确定顺序和 VTA owner 过滤。
