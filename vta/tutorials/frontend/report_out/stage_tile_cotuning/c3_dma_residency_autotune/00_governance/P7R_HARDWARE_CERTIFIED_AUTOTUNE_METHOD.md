# 第三创新点修正版：固定 FPGA 的硬件证书引导安全 AutoTune

状态：`RESOURCE_SUBSYSTEM_ATTESTED; SEARCH_GATE_G7_NO_GO; LITERATURE_GUIDED_ABLATIONS_REOPENED`  
日期：2026-09-11

## 1. 不是重复旧实验，而是改变候选进入 AutoTune 的流程

旧 AutoTune 的基本对象是 schedule knob，构建或运行失败通常也会消耗一次试验。本文新增的对象是
“带数据驻留变换的候选”，其正确性和收益同时受 SRAM、DMA、UOP 依赖、虚拟线程以及真实 FPGA
实现约束影响。因此第三创新点不是再跑一遍原始配置，而是在测量器前增加可审计的分层证书：

```text
候选生成
  -> 解析容量/复用证书
  -> original 与 residency 双路径 lowering/命令证书
  -> 双路径三 seed FSim
  -> 精确硬件指纹下的 FPGA correctness canary
  -> 只有正确候选才允许计时并进入 cost model
  -> 与 original/TopHub 比较，不能获益则安全回退
```

前四层回答“能不能成为合法训练样本”，后两层才回答“是否更快”。正确性检查不是性能重复实验。

## 2. 理论模型与 DMA Pareto 证书

候选时间写为：

```math
T(x)\approx \max\{T_{load}(x),T_{compute}(x),T_{store}(x)\}
       +T_{sync}(x)+T_{launch}(x)+T_{tail}(x),
```

```math
T_{dma}(x)=N_{dma}(x)L_{req}+\frac{B_{dma}(x)}{BW_{eff}(shape)}.
```

这说明只减少 input bytes 不够：驻留改变 loop order 或 virtual-thread context 后，可能增加 weight
bytes、总传输量、UOP、同步或降低计算 tile 效率。对 exact-same-tile 原方案 `x_0` 和驻留方案
`x_r`，本文先使用不拟合板卡常数的充分优先条件：

```math
B_{inp}(x_r)<B_{inp}(x_0),
\quad B_{total}(x_r)\le B_{total}(x_0),
\quad N_{total}(x_r)\le N_{total}(x_0).
```

满足条件称为 DMA Pareto-certified，只代表“值得优先”，不代表理论保证加速；不满足则 abstain 并
保留 original。这样避免把开发板特定的经验阈值伪装成通用规律。

## 3. 硬件正确性证书

FSim 无法覆盖全部 HLS/RTL 行为，所以真实 FPGA canary 的缓存键必须足够严格：

```text
certificate_key = SHA256({
  boot/bitstream/u-dma-buf/source hardware fingerprint,
  workload,
  residence_mode,
  complete ConfigEntity,
  lowered TIR hash
})
```

路由只有三种：

| 精确证书状态 | 调优器动作 |
|---|---|
| passed | 允许计时并把 latency 写入 cost model |
| failed | 拒绝候选，不计时，不训练 |
| unknown | 只执行多 seed correctness canary |

硬件/源码/启动指纹改变后，旧证书不会自动外推。工具
`build_vta_hardware_certificate_ledger.py` 已实现该账本；当前 E02/E03 输入形成 6 个唯一证书，5 个
允许计时、1 个明确拒绝。

`plan_vta_hardware_certified_dispatch.py` 已将静态证书、DMA Pareto 与上述账本组合为派发文件；
`tune_resnet18_vta.py --certificate-dispatch` 在 AutoTVM 生成 `MeasureInput` 前只保留
`allow_timing` ConfigEntity，并校验 workload、residence mode 和派发文件哈希。E03 控制流测试中，
一个候选被硬件证书拒绝、一个仍为 unknown，调优入口按预期拒绝启动，未创建 tuning log，也没有
RPC/FPGA 执行。

这一顺序是必要的：当前 AutoTVM `ModelBasedTuner.update` 会把失败测量以零吞吐加入训练数据；若只
在 Runner 内返回 `WRONG_ANSWER`，错误候选仍会影响模型。本文的证书层因此必须位于 tuner 前，而
不只是 measurement 后处理。

## 4. 已完成的消融

### 4.1 搜索前过滤

| 候选集 | 原联合空间 | 本地证书后 | 板前过滤率 |
|---|---:|---:|---:|
| E00/E01/E02，`oc_nthread=2` | 2368 | 209 | 91.17% |
| 新 E03，`oc_nthread=1` | 864 | 74 | 91.44% |

这里减少的是进入 FSim/板卡资格阶段的数量，不等于已经证明相同预算达到全空间 oracle。

### 4.2 为什么从 input-only 改为 total-DMA Pareto

E02 两个候选的 input bytes 都减少 50%，但总 DMA bytes 分别增加约 42.65%/37.32%，板端同 tile
延迟分别退化 14.80%/13.18%。旧 input/request-shape 规则失败。

DMA Pareto 规则回看 P7Q 的 14 个 input-stationary 配对，选择 11 个，11/11 实测更快，中位
+8.31%，零退化；对 E02 两个退化点均 abstain。该结果是回顾性消融，不能写成前瞻成功。

### 4.3 E03 前瞻正确性结果

E03 的选择规则和两组候选在任何 E03 FSim/FPGA/latency 标签前冻结。两条 original 和两条
residency 均通过三 seed FSim。真实 FPGA 上：

- original config17：3/3 seed 逐元素正确；
- Pareto-priority residency config17：0/3 正确；
- 合同立即停止，boundary config576 未执行，latency 样本为 0。

因此 E03 没有支持 Pareto 裕量的性能排序，却前瞻支持了分层 canary 的必要性：若直接把 FSim
通过或 DMA 占优当作可计时候选，AutoTune 会接收错误标签；本文流程在 cost model 前将其隔离。

### 4.4 W05 TopHub 邻域的固定预算结果

在保留硬件 `oc_nthread=2` 约束后，400 个 bounded-hybrid 配置先由容量和全张量 DMA Pareto
缩到 19 个，再按与 TopHub tile 的语义距离在任何目标 latency 前冻结 2 个。两候选及同 tile
original 通过 12/12 FSim、4/4 ARM 交叉编译和 18/18 FPGA seed。config574/494 相对各自同
tile original 分别快 4.96%/9.86%，冻结首选 config574 距最强 TopHub575 仅 0.31%；严格策略
仍保留 TopHub 为最终 primary。这里成立的是 400→2 的派发预算缩减和 2% strongest-incumbent
等价带，不把未测的 398 个点伪称为已知 oracle。

### 4.5 allowlist 命令资源与 runtime attestation

对 exact allowlist `S`，本文按下式确定每个运行时队列容量：

```math
C_q(S)=align_A\left(\max_{c\in S,b\in submits(c)}B_q(c,b)\right),
\qquad
M=\sum_{j\in live\ queues}(C_{insn}(S_j)+C_{uop}(S_j)).
```

因此通用规律不是“参数取 1”或“8 KiB 最好”，而是：先由固定 FPGA、完整 ConfigEntity、lowered
TIR 和每次提交的真实序列化峰值得到单队列需求，再按 allocator 对齐，并仅对同时存活的队列求和。
W05 三身份调优集合推导 12 KiB，最终两身份集合推导 8 KiB；W00--W09 的 197 身份联合集合则为
36 KiB，留一 workload 仅 9/10 可迁移。这些不同结果正是公式普适、常数不普适的证据。

最终两身份在当前 ARM runtime 上完成 prospective manifest attestation：pending manifest 先冻结，
板端 6/6 seed 后同会话回报相同 manifest ID、3808/1480 B 峰值、4096/4096 B 容量、6 次提交和
`replay=disabled`，ready manifest 保持原 ID。FSim capture/replay 负向控制均被 runtime 拒绝。
相对旧 runtime 两队列各预留 32 MiB，该精确静态部署的 requested command backing 减少
99.9878%；这是空间结果，不解释为 FPS 收益。

## 5. 调优伪代码

```text
best = measured_or_archived_TopHub
for candidate in certificate_ranked_candidates:
    if not analytic_and_both_path_lowering(candidate):
        continue
    if not both_path_fsim(candidate):
        continue
    state = exact_hardware_ledger.lookup(candidate)
    if state == FAILED:
        continue
    if state == UNKNOWN and not fpga_correctness_canary(candidate):
        ledger.record_failed(candidate)
        continue
    ledger.record_passed(candidate)
    latency = paired_measure(candidate)
    cost_model.update(candidate, latency)
    best = min(best, candidate)
return best
```

## 6. 当前论文口径

可以主张：固定 FPGA 参数已进入驻留候选生成、91% 级板前过滤、双路径命令/FSim、硬件指纹证书
和安全回退；W05 用 2 次板端候选计时达到 strongest incumbent 的 0.31% 等价带；最终 exact
allowlist 的命令 backing 已形成内容寻址且 runtime-attested 的 8 KiB 部署合同。E02 证明局部
DMA 指标会误导，E03 证明真实 FPGA canary 不能被模拟器替代。

不能主张：新候选击败 TopHub；398 个未测配置的真实 oracle 已知；W05 规律已跨模型、跨 boot
稳定；8 KiB 适用于完整 stage/整网；stage 或整网 FPS 已提升。

W05 满足的是一个单 workload 的局部资源/安全子目标：400 个候选静态缩到 2 个派发，最终由
strongest incumbent 回退保持在 0.31% 等价带，同时 exact deployment allowlist 得到 8 KiB
命令 backing。它不能升级为整个第三创新点的 B 级闭环，因为主计划 Gate G7 仍要求跨 workload
的等预算搜索收益、full request signature 的独立增量，以及至少两种几何的不同驻留赢家；P7Q
对这些要求的判定仍为 NO-GO。

2026-09-11 重新打开论文方法消融后，冻结的 310 个本地候选显示：budget=16 时 Random、
Rieber-inspired 邻域、ML²-inspired 基础 validity 模型和加入硬件 working-set 特征的模型平均分别
得到 10.18、11.80、14.20、14.50 个 lowering 合法样本。56 个 same-tile FPGA 开发配对上，
DMA bytes 特征已足以把平均 regret@1 从 11.38 个百分点降至 0.02；同一 Ridge replay 中完整请求
形态和命令/硬件特征没有继续改善。随后 P7R113 用非负物理系数标定 `ΔT_ms`：bytes/calls 获得
最低 MAE 和最高 Spearman；request-shape 虽使四个开发折的 regret@1 变为 0，MAE/Spearman 却
下降；command 仍无稳定增量。因此 request 结论是混合的，必须前瞻确认。以上 replay 只支持继续
研究独立 validity 与 same-tile 增量模型，不构成未见 workload 确认。

因此当前只冻结“分层正确性证书 + exact allowlist 命令定容”子模块；性能搜索方法继续处于开发态。
下一次板端实验必须使用预注册的新 workload/几何和完整候选池，不能重复使用 W01/W04/W07/W08
或 W05 标签充当确认集。

## 7. P7R115/P7R116：搜索问题已按 no-leak 重新打开

P7R115 已在任何新目标 latency 前冻结三类 YOLOv3-tiny 真层：conv2
`16->32,208x208,3x3`、conv13 `1024->256,13x13,1x1`、conv21
`384->256,26x26,3x3`。三类完整 ConfigSpace 共 3920 个实体；纯硬件/结构条件保留
78/56/343。36 点 pilot 和同一 SHA 前缀扩展的 96 点正式确认池都已不可变冻结，四种 mode 平衡。
候选池承诺后才审计到三个 workload 各有一条精确 TopHub 记录；其 ConfigEntity、index 和 cost
仍未向生成器或选择器披露。

P7R116 已实现逐次 reveal 的等 gross-budget 搜索器。TopHub/其他 sealed reference 不进入候选、
首派发或训练；lower/compile/FPGA/measure 任一失败都消耗预算。它统一比较 Random、stock-knob
XGB、rules-only、validity V，以及 bytes/calls、+request、+command 三档 `V+DeltaT`，并报告
20 seeds 的 regret、success、trials 和阶段墙钟。

旧 P7Q 只作 development smoke：76 个非 reference 候选、7 策略、20 seeds、4 workload 共
560 runs。rules-only 在非 TopHub 小池上的 median regret@4 为 0%，而 validity V 和三档复杂模型
没有稳定优于 stock-XGB；所有策略对 TopHub +2%/+5% 的 success 都为 0。这个负结果阻止我们把
复杂特征数量包装成贡献。主创新能否升级，只由 P7R115b 的 96 点未见确认决定。
