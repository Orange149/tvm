# P7R 硬件规律迁移确认结果

状态：`ONE_PROSPECTIVE_MECHANISM_SUCCESS; PROTECTED_TOPHUB_RETAINED`  
日期：2026-09-11  
范围：第三创新点；新工作负载的 FPGA 正确性、算子配对计时与硬件证书派发，不包含 stage/FPS。

## 1. 这不是旧实验复跑

本轮使用此前没有做过 C3 驻留板端标签的 ResNet 工作负载 W03、W05、W06。候选在读取目标
工作负载的 FSim、FPGA 正确性和 latency 前冻结，只允许从开发工作负载迁移完全相同的映射结构：

| 目标 | 候选 | 映射结构来源 | 目标静态 DMA 预测 |
|---|---:|---|---|
| W03 | config248 | W01 config248（开发集同 tile +9.86%） | 输入 -50.00%，总 bytes -34.31%，总 calls -44.44% |
| W05 | config174 | W04 config142（开发集同 tile +18.61%） | 输入 -75.00%，总 bytes -11.79%，总 calls -72.73% |
| W06 | config142 | W04 config142（开发集同 tile +18.61%） | 输入 -75.00%，总 bytes -56.73%，总 calls -70.59% |

三个目标的 original 与 input-stationary 均先通过三 seed FSim。板端合同固定为
`original-before -> residency -> original-after`，每段三个 seed；任一段错误即停止且禁止计时。

## 2. FPGA 正确性门

| 工作负载 | 实际执行到的门 | 结果 | 后续动作 |
|---|---|---|---|
| W03 | original-before config248，3 seeds | 0/3；错误元素 57727/58019/58252 | 停止，未执行 residency，未计时 |
| W05 | original-before、residency、original-after config174 | 9/9 | 允许配对计时 |
| W06 | original-before config142，3 seeds | 0/3；错误元素 28054/28029/28062 | 停止，未执行 residency，未计时 |

W03/W06 失败的是原始 AutoTVM 配置，不是驻留改写。两者在 FSim 正确而 FPGA 错误，说明
“只在 Runner 返回后把错误 cost 设为 0”太晚：错误配置已经消耗板端派发，并可能污染代价模型。
这组结果直接支持把真实硬件正确性证书放到性能搜索之前。

## 3. W05 前瞻配对计时

W05 在同一 boot 内执行 7 个随机完整区组，每个样本 `number=5`，TopHub config575 在每个区组
前后作为漂移哨兵。计时前后再次逐元素核对输出。

| 对象 | 中位延迟 |
|---|---:|
| 同 tile original config174 | 7.891081 ms |
| input-stationary config174 | 6.892653 ms |
| 受保护 TopHub config575 | 5.138112 ms |

- input-stationary 相对同 tile original 的成对中位加速为 **12.59%**，7/7 区组获胜；
- 这验证了“已知正收益映射结构 + 目标本地 DMA-Pareto”能够迁移到一个新工作负载；
- 但 residency 仍比 TopHub 慢 **34.15%**，因此最终安全选择是 TopHub，而不是实验候选。

该结果同时给出正证据和边界：硬件规律能够减少搜索并找到局部更优 schedule，但强 incumbent
保护仍不可删除。不能把 12.59% 写成相对 TopHub、stage 或整网 FPS 提升。

## 4. 对调优器的实际落地

新增/验证后的派发顺序为：

1. 静态容量、lowering 与 DMA 规则先过滤；
2. 只对静态合法且 DMA-Pareto 的 exact `(hardware, workload, mode, config, TIR)` 查询硬件证书；
3. 未知 identity 只允许做正确性 canary；失败 identity 在计时前拒绝；通过 identity 才允许计时；
4. 即使候选计时成功，最终仍与 TopHub 比较，失败时零回归回退。

扩充后的账本含 10 个 exact 证书：7 个允许计时、3 个拒绝。对 W03/W05/W06 的完整本地空间
重新派发后：

| 工作负载 | 静态拒绝 | 未知、仅 canary | exact allow-timing | 受保护 incumbent |
|---|---:|---:|---:|---:|
| W03 | 337 | 95 | 0 | 1 |
| W05 | 358 | 41 | 1 | 1 |
| W06 | 230 | 90 | 0 | 1 |

派发器同时修复了一个顺序缺陷：静态淘汰项没有 DMA 记录时，旧实现会在静态判断前访问空值；
现在严格按“静态 -> DMA -> 硬件证书”执行，并由回归测试覆盖。

## 5. 当前论文口径

当前可以主张：固定 FPGA 信息可在板端测量前大幅排除不合法/不值得测的候选；exact 硬件证书
防止 FSim 不可见错误进入 latency cost model；结构迁移在一个新的可评价工作负载上取得稳定
12.59% 同 tile 收益；TopHub 保护使最终派发不发生性能回退。

当前不能主张：新候选击败 TopHub；已在 stage/FPS 上加速；三目标都验证了性能预测；现有静态
规则已经完全预测真实 FPGA 正确性。

## 6. 证据目录

- 候选冻结：`20260911_p7r53_support_transfer_contract_run01`
- 双路径 FSim：`20260911_p7r54_support_transfer_residency_fsim_run01`、
  `20260911_p7r56_support_transfer_original_fsim_run01`
- FPGA 正确性：`20260911_p7r58_w03_support_transfer_correctness_run01`、
  `20260911_p7r59_w05_support_transfer_correctness_run01`、
  `20260911_p7r60_w06_support_transfer_correctness_run01`
- W05 配对计时：`20260911_p7r61_w05_support_transfer_timing_run01`
- 扩充证书账本：`20260911_p7r62_hardware_certificate_ledger_run01`
- 新派发计划：`20260911_p7r63_w03_certified_dispatch_run02`、
  `20260911_p7r64_w05_certified_dispatch_run02`、
  `20260911_p7r65_w06_certified_dispatch_run02`

## 7. TopHub 邻域的 bounded-hybrid + virtual-thread 扩展

W05 config174 的正收益没有打赢 TopHub，主要因为它关闭了 `oc_nthread=2`。因此后续没有重测
config174，而是枚举全新的 W05 `oc_nthread=2` 空间：

1. 纯 input-stationary：400 个配置中 27 个双路径 lower 且减少输入流量，但 27/27 都使总 DMA
   bytes 增加 33.17%--78.08%，DMA-Pareto 全部 abstain；
2. bounded hybrid 第一版将虚线程绑定在组内轴，两个静态 Pareto 候选均被 FSim runtime 的
   两周期 UOP `dst_idx` 依赖规则拒绝，未上板；
3. 修正为“虚线程跨组分配、组内 tile 同一 context 顺序执行”后，400 个配置中 19 个同时
   通过双路径 lowering、输入减少和全 DMA Pareto；
4. 只用与 TopHub config575 的完整 tile 对数距离和静态 DMA 冻结 config574/494，未使用动态
   正确性或时间标签；两候选及同 tile original 共 12/12 FSim seed 正确、4/4 交叉编译通过；
5. FPGA 合同含 TopHub 前后哨兵与两个 original/hybrid 配对，共 18/18 seed 正确。

7 个随机完整区组结果：

| 配置 | original/ms | bounded hybrid/ms | 同 tile 成对加速 | 获胜区组 | 相对 TopHub |
|---|---:|---:|---:|---:|---:|
| 574 | 5.414274 | 5.148322 | **+4.96%** | 7/7 | -0.31% |
| 494 | 6.023886 | 5.430942 | **+9.86%** | 7/7 | -5.81% |
| TopHub 575 | — | 5.132598 | — | — | 0 |

预冻结的“离 TopHub 最近的 config574 具有最佳最终延迟”预测成立。config574 与 TopHub 的
差距只有 0.31%，处于预注册 2% 等价带内，也小于本轮 TopHub 哨兵 range/median 1.03%；因此
只能称“接近/等价带内”，不能声称击败。严格最小中位数派发仍选择 TopHub。

从 400 个 t2 配置到 2 个板端性能候选，候选派发数按空间大小计算减少 99.5%，并在极小预算
下获得距强基线 0.31% 的结果。这里 398 个未测点没有 latency，故这是候选空间压缩与
strong-incumbent regret 上界，不是完整 pool-oracle discovery 证明。结构化分析见
`20260911_p7r84_w05_t2_hybrid_budget_analysis_run01`。

## 8. 与共享内存主题直接对应的命令资源证书

在不重复板端正确性或性能计时的前提下，对最终派发路径重新做了无 timing 的本地 FSim
命令结构采集。比较对象在采集前固定为 TopHub config575、同 tile original config574 和
bounded-hybrid config574。峰值均已包含 runtime 在提交前补入的 FINISH：

| 角色 | instruction 峰值 | UOP 峰值 |
|---|---:|---:|
| TopHub config575 | 3744 B | 1480 B |
| same-tile original config574 | 7200 B | 752 B |
| bounded-hybrid config574 | 3808 B | 1424 B |

原 runtime 的 `VTA_MAX_XFER=2^25`，为 instruction/UOP 各请求 32 MiB，即每个命令队列共
64 MiB 连续 FPGA 可访问 backing。对最终允许派发的 `{TopHub575, hybrid574}` 取各字段最大值
并按 4 KiB 页对齐，冻结容量为 instruction 4096 B + UOP 4096 B，共 8192 B。随后不是只做
算术推断，而是在该缩小容量下再次运行两个部署身份；same-tile 对照则使用 8192 B + 4096 B。
三者均保持 seed-0 逐元素正确，runtime 日志报告的实际容量与冻结计划一致，峰值与首次采集一致。

因此，对这个精确 W05 静态形状和源码指纹，命令 backing 的请求量从 64 MiB 降到 8 KiB，
减少 **99.9878%**，且不改变提交次数和执行顺序。这是“编译/离线 dry-run 生成资源证书，runtime
在提交前检查完整放入”的实例证据。它不能外推为完整 ResNet-18 或任意动态形状的容量；完整
stage 应对其所有可能提交取峰值并重新验证。证据目录为
`20260911_p7r85_w05_command_resource_signatures_run02` 和
`20260911_p7r86_w05_reduced_command_capacity_run02`。run01 仅保留为证据卫生审计记录：其结构化
结果已去除 timing，但原始 stderr 尚含 FSim 诊断耗时，故不作为正式无标签产物引用。

随后将集合范围扩到 W00--W09 的 197 个冻结身份。按全池最大峰值定出的 24 KiB instruction +
12 KiB UOP 在 197/197 次缩容复验中通过，相对 64 MiB requested backing 减少 99.9451%；逐身份
容量中位数仅 8 KiB、P90 为 12 KiB。留一 workload 固定容量覆盖 9/10，W08 需要额外三个
instruction 页，证明容量必须随编译 allowlist 重算，不能把 36 KiB 当成另一个平台常数。

W05 的 4096+4096 B 计划进一步在真实 FPGA/u-dma-buf 路径验证：同一 RPC 会话直接报告峰值
3808/1480 B、提交 6 次，TopHub575 与 hybrid574 共 6/6 seed 正确，实验后默认 RPC 已恢复。
完整公式、失败记录和边界见 `P7R_COMMAND_RESOURCE_CERTIFICATE_RESULTS.md`。
