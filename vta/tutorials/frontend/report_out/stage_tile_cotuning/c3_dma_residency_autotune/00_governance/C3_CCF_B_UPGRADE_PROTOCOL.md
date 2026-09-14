# 第三创新点的 CCF-B 级升级协议

状态：`PREREGISTERED; BOARD_EXECUTION_NOT_YET_AUTHORIZED_BY_CONTRACT`  
日期：2026-09-11

> 2026-09-11 范围更新：用户将目标调整为 CCF-C，并要求共享内存保持主线。本文保留为 stretch
> protocol；当前执行与论文定稿以 `C3_CCF_C_SHARED_MEMORY_INNOVATION.md` 为准。

## 1. 最终研究问题

第三创新点不以超过 TopHub 为目标，也不把命令缓冲缩容当成主创新。研究问题冻结为：

> 对硬件已经定型、由编译器显式生成 DMA 和微指令的 FPGA 张量加速器，能否把片上容量、数据
> 驻留/复用规律、编译合法性和真实硬件正确性作为搜索先验，在不预先泄露 TopHub 或目标设备
> latency 的条件下，用更少的编译与 FPGA 测量成本找到与强参考实现等价的配置？

目标函数不是单纯的最低 latency，而是达到质量门槛的搜索成本：

```math
\min_\pi C_\pi(\epsilon),\qquad
C_\pi(\epsilon)=\min\{C(b):T_{best}^{\pi}(b)\le(1+\epsilon)T_{ref}\}.
```

其中 `pi` 是搜索策略，`b` 是已派发候选数，`C(b)` 同时报告 gross dispatch 数和真实墙钟成本；
`T_ref` 在有精确 TopHub 命中时取冻结 TopHub，在没有命中时取完整确认池 oracle，并另列长预算
原始 AutoTVM 的最好值。`epsilon` 冻结为 2% 和 5%。

## 2. 一条完整的方法链

```text
固定 FPGA 描述 H
  -> 生成 original / input-prioritized / weight-resident / hybrid 映射
  -> V：预测并验证 tensorize、SRAM、UOP、依赖与 lowering 合法性
  -> A：从 lowered TIR 提取 input/weight/output bytes、DMA calls 和复用强度
  -> PΔ：预测同一 tile 下驻留变换相对 original 的增量时间
  -> 按固定预算选择下一个候选，unknown 先过 FPGA correctness canary
  -> 找到 2%/5% 等价配置后停止；失败则返回最优已测配置
  -> 对最终 allowlist 生成 instruction/UOP 安全部署容量
```

这条链复现 2026 minimum-access 工作的三条核心规律：输入优先、权重片上复用、按最小外存访问
联合选择；本文在 VTA 上将它们变成可搜索的 schedule mode。新增部分是把这些模式与独立合法性
模型、same-tile 增量模型、逐层硬件证书和显式停止准则组合为一个低测量成本搜索过程。

命令缓冲定容只位于最后一层，回答“最终被允许部署的配置需要多少共享命令内存”，不再单独承担
第三创新点的创新性。

## 3. 公平基线和禁止泄漏规则

所有方法看到相同候选身份、初始空标签、gross budget 和设备。必须比较：

| 编号 | 方法 | 作用 |
|---|---|---|
| B0 | Random | 无先验下界 |
| B1 | 原始 AutoTVM XGBoost | 官方学习型调优基线 |
| B2 | 2026-rules-only | 只按 input/weight/minimum-access 规律排序 |
| B3 | V + XGBoost | 只验证独立合法性模型是否节省无效编译 |
| B4 | V + rules + `P_delta(bytes,calls)` | 本文完整方法 |
| B5 | B4 + request-shape | 检验请求形态是否有独立增量 |
| B6 | B5 + command footprint | 检验命令特征是否有独立增量 |
| O | 完整池 oracle | 只供事后计算 regret，绝不参与选择 |

冻结以下规则：

1. TopHub/长预算参考的 latency、排名和配置位置不得用于初始化、首派发或训练；若其实体在统一池
   内，必须与其他候选一样由策略选中后才暴露标签。
2. 开发集为既有 ResNet 数据；确认集使用此前没有 C3 per-config latency 标签的 YOLOv3-tiny
   卷积几何。禁止根据确认集结果修改特征、权重、候选或阈值。
3. 相同完整 `ConfigEntity`、workload、mode 和 lowered-TIR hash 才是同一候选；`config_index`
   只用于审计，不能作为模型特征。
4. 编译失败、数值失败和超时都消耗一次 gross dispatch；只报告 successful trials 会偏袒过滤器。
5. correctness canary 不作为性能标签；错误候选不得进入性能模型。

## 4. 确认工作负载与候选池

确认集至少包含三类此前未用于 C3 性能训练的 YOLO 几何：

- 大空间输入复用型：YOLO conv2，`CI=16, CO=32, H=W=208, K=3`；
- `1x1` 高通道型：YOLO conv13，`CI=1024, CO=256, H=W=13, K=1`；
- route 拼接后的非规则通道型：YOLO conv21，`CI=384, CO=256, H=W=26, K=3`。

选择 route-concat 的 384 通道边界，不是根据板端 latency 换点。候选池在看到任何目标 latency
前固定，包含相同 tile 下的 original、
input-prioritized、weight-resident 和 hybrid。P7R115 先以每类 3 个 ConfigEntity、共 36 点做机制和
接口 pilot；最终确认必须按同一 SHA 排名的稳定前缀扩成每类至少 8 个 ConfigEntity、32 点，共
96 点。首先完成本地 lowering、双路径 FSim、ARM 交叉编译与 semantic-v2 身份冻结。进入确认后，
所有通过 correctness 的候选都必须计时，形成真实 pool oracle；不能只测算法喜欢的点后再宣称
regret。

若精确 TopHub 日志不覆盖某个 YOLO workload，报告中必须写“完整池 oracle”，不能称为 TopHub。
可另跑一条不参与候选选择的长预算 stock-XGB reference，作为池外强参考。

## 5. 测量协议

### 5.1 本地阶段

1. 对全池做静态容量、tensorize、依赖与 TIR 检查，保留拒绝理由。
2. original 与机制路径各做三 seed FSim；任何错误 fail closed。
3. 对通过项做 ARM 交叉编译并记录编译时间、峰值 instruction/UOP、DMA bytes/calls。
4. 在不读取确认 latency 的前提下生成 B0--B6 的完整派发序列及 SHA-256。

### 5.2 板端阶段

1. 只写 RAM 目录，先核对 boot、bitstream、u-dma-buf、runtime 和源码哈希；不自动重启开发板。
2. 每个 unknown 候选先做三个固定 seed 的逐元素 correctness；失败即停止该候选且记入成本。
3. 对正确候选做五轮平衡随机顺序计时，报告中位数和 IQR；同一候选的重复计时只形成一个标签。
4. 完整池测完并冻结后才离线重放 20 个搜索 seed。搜索器不能看到未来标签，只能按派发顺序逐点
   更新。
5. 如果板端环境 canary 相对历史参考偏移超过 5%，整批数据标为环境失败，不进入结论。

### 5.3 同成本口径

至少同时给出两种横轴：

```math
C_{count}=N_{compile\ attempt}+N_{fpga\ correctness}+N_{fpga\ timing},
```

```math
C_{wall}=\sum t_{compile}+\sum t_{upload}+\sum t_{correctness}+\sum t_{timing}.
```

预算点固定为 4、8、12、24、36；若池更小则取池大小。主要指标为：

- success@budget：20 个 seed 中进入 2%/5% 等价带的比例；
- trials-to-2%/5%：未命中按删失样本报告，不用池大小伪装命中；
- regret@budget：`(T_best-T_oracle)/T_oracle`；
- invalid compile、wrong-answer FPGA 和有效性能标签的数量；
- 搜索墙钟分解以及最终候选的 DMA、SRAM、instruction/UOP 资源。

## 6. 论文级成功门槛

### G1：机制复现

至少两类确认 workload 上，input/weight/hybrid 中至少一种相对 same-tile original 显著减少目标
数据访问，并在正确性通过后获得方向一致的 latency 改善。若只减某一 tensor 却增加总 DMA，作为
反例而不是删除。

### G2：搜索效率（主门槛）

在三个确认 workload、20 个搜索 seed 上，B4 相对 B0 和 B1：

- `success@12` 的中位/总体成功率更高；并且
- 达到 5% oracle 等价带的 gross dispatch 中位数至少减少 30%，或墙钟成本至少减少 25%；并且
- 最终 `regret@36` 不劣于 B1 超过 2 个百分点。

统计以 workload 为组报告 bootstrap 95% 区间，并同时列出每个 workload，禁止只报合并均值。
上述 30%/25% 是“方法存在最低独立收益”的门槛；面向 CCF-B 的目标值仍冻结为 gross dispatch
至少减少 50%，且最终进入 strongest sealed reference 的 2% 等价带。

### G3：消融和可解释性

B3 必须说明合法性模型的独立收益；B4 必须说明复用/增量模型在合法候选中的独立收益。B5/B6 若
无增量则如实作为否定结果，从最终方法删除，不再把特征数量当贡献。

### G4：系统可信性

完整确认池必须具备 semantic-v2 身份、FSim、交叉编译、FPGA correctness 和 latency 账本。最终
方法在未知候选上 fail closed，最终 allowlist 的命令容量可由 runtime manifest 验证。

只有 G1--G4 都通过，才可把第三点表述为“硬件与数据复用规律引导的低成本 AutoTune 方法”。
这仍不能保证某个 CCF-B venue 录用，但证据结构达到可投稿系统/编译论文的基本形态。若 G2 未过，
应降级为硕士论文中的 VTA case study，不能用 W05 单点或 91% 静态过滤率替代搜索效率证据。

## 7. 当前水平与执行状态

当前已经具备方法原型和较强工程证据：四类驻留 mode、91% 级板前过滤、same-tile 正收益样本、
W05 两次候选测量达到 TopHub 0.31% 等价带、分层硬件正确性证书和 8 KiB exact-allowlist 部署。
但尚缺无泄漏的跨网络完整池结果，因此当前水平是“硕士论文创新点可成立，CCF-B 潜力尚未被实验
证明”。

本协议之后只允许新增两类证据：全新 YOLO 完整候选池，或另一网络的独立复现。继续重测
W01/W04/W07/W08/W05、继续手选赢点、或者继续缩小命令缓冲，都不能提升主创新的证据等级。
