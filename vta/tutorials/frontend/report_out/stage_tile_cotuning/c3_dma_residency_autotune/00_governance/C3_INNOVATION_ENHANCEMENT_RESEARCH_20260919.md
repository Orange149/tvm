# C3 创新性增强调研与推荐路线

更新日期：2026-09-19。

本文只提出研究定位和后续验证方案，不改写既有实验结果，也不把尚未执行的算法或实验写成已经完成的贡献。现有 C3 的详细机制、结果和边界仍以 [C3 输入与权重复用补充计划](C3_INPUT_WEIGHT_REUSE_SUPPLEMENT_PLAN.md)、[论文主叙事第六章](../../thesis_story_20260915/01_THESIS_STORY_AND_INNOVATIONS.md)以及各封存运行的原始记录为准。

## 1. 结论先行

第三个创新点继续增加驻留模式、DMA 图表或同 Tile 个案，能够增强机制证据，却很难显著提高方法创新性。输入优先、权重驻留、最小访存、数据流映射、硬件合法性过滤和多保真调优都已有直接先例。

当前最值得投入的升级方向是：

> **面向显式 DMA FPGA 的驻留语义驱动、成本感知多保真调优方法。** 将同 Tile 下驻留方式引起的输入、权重、请求粒度、同步和命令需求变化表示为可逐阶段解锁的语义增量；调优器每一步联合选择“哪个候选”与“支付哪个下一资格阶段”，在真实 FPGA 数值正确性和完整图部署门的约束下，最小化首次取得合格完整图程序的全部实付成本。

建议把这个方法简称为 **RS-CMF**（Residency-Semantic Cost-Aware Multi-Fidelity Tuning）。正式中文名称可以使用“驻留语义驱动的成本感知分层调优”，避免把一般的 multi-fidelity 或 Bayesian optimization 本身写成原创。

该方向比“再设计一种驻留模板”更适合当前论文，原因有三点：

1. 它直接回答当前最薄弱的证据问题：共享内存语义是否在同一候选空间、同一初始信息和相同付费规则下真正缩短调优时间。
2. 它可以复用已有 ConfigEntity×mode、lowering/FSim/FPGA/完整图流水线、same-tile 四元组、阶段成本账本、invalid peeling 和阶段 05 在线协议。
3. 它能与创新点二形成真实接口：共享对象的身份、布局、独占性、生命周期和完成合同进入调优状态；C3 输出 exact 程序身份、route 与命令需求合同，而不是只输出一个孤立算子延迟。

## 2. 当前 C3 已经具备什么

当前 C3 已经不是简单的“跑 AutoTVM”：

- 搜索单位包含完整 `ConfigEntity × {O,I,W,IW}` 程序身份。
- 候选信息按静态约束、真实 lowering、FSim/命令生成、FPGA 正确性、计时和完整图准入逐阶段付费获得。
- unknown 不当作零，数值失败不产生性能标签，失败成本被保留。
- 支持稀疏 family-wave、密集 4→2 前沿，以及无效支配点删除后的前沿重算。
- 选择结果需要通过真实 FPGA 输出和完整图 route，并保留强 incumbent 与 fallback。
- 已有两个不同空间的系统流程正例：YOLO 和 ResNet18 的成功会话 T0→T1 中位时间分别下降 64.84% 和 53.67%，最终完整图质量接近基线。
- 已有 same-tile 机制、双启动稳定性、完整图拒绝、TopHub 反例和全部失败成本等较完整的系统证据。

但决定 C3 创新强度的关键缺口仍未闭合：

1. **不同空间流程比较不能归因于选择算法。** AutoTVM-XGB 与本文流程的候选空间、测量粒度和资格路径不同。
2. **同空间在线证据仍局限于一个 12 点池。** Y10 中 exact TTQ 提前，但两种方法首点都进入 +2% 带；执行满预算后总墙钟并未下降。
3. **跨几何结果不稳定。** R18 三几何中本文仅 H2 明显占优，H1/H3 不优于 stock-XGB 或 ML² 类基线。
4. **最新机制池带有主动筛选。** 11 组四模式是为确认双复用而选择的机制适用池，不能作为自然配置分布的平均收益。
5. **真实强基线仍可能更快。** 真实 R18-H1 中组合方案相对同 Tile 原程序加速约 12.56%，但仍比重新资格的 TopHub 慢约 5.43%；LOAD 少 61.40% 也没有转化为最优时延。
6. **阶段 05 尚未取得真实在线结果。** 公共协议已经完成 144 身份、27 attempt、paid known-mask、deadline 和失败保留的主机预资格，但真实任务选择、LiveExecutor 与 27 次板端运行仍未执行。

因此，C3 当前最稳妥的定位是“可审计的分层资格与部署系统”。要进一步成为较强的方法创新，必须补一个清楚的新决策问题和前瞻的公平实验。

## 3. 相关工作对创新边界的约束

### 3.1 不能再作为核心原创的内容

| 已有方向 | 代表工作 | 对本文表述的约束 |
|---|---|---|
| 模板搜索和学习代价模型 | [AutoTVM / Learning to Optimize Tensor Programs](https://proceedings.neurips.cc/paper/2018/hash/8b5700012be65c9da25f49408d959ca0-Abstract.html) | 不能把学习排序或硬件测量式调优作为原创。 |
| 自动搜索空间和任务调度 | [Ansor](https://www.usenix.org/conference/osdi20/presentation/zheng) | 不能把层次搜索、进化搜索或跨任务预算分配作为原创。 |
| 模块化概率搜索空间 | [MetaSchedule](https://arxiv.org/abs/2205.13603) | 不能把将硬件知识组合进搜索空间本身作为原创。 |
| 数据流、复用和容量映射 | [Timeloop](https://ieeexplore.ieee.org/document/8695666)、[MAESTRO](https://doi.org/10.1145/3352460.3358252)、[CoSA](https://arxiv.org/abs/2105.01898) | input/weight stationary、traffic/capacity 约束和映射搜索已有系统研究。 |
| Tile 级跨层访存优化 | [Welder](https://www.usenix.org/conference/osdi23/presentation/shi) | 不能声称首次把 Tile、数据复用或访存流量作为一等优化对象。 |
| 编译静态分析指导流水与搜索 | [ALCOP](https://proceedings.mlsys.org/paper_files/paper/2023/hash/d6cde2c1b161daa31be560d062cf2251-Abstract-mlsys2023.html) | “用 buffer/流水语义减少搜索”这一宽泛说法不够新。 |
| VTA 合法性驱动初始化 | [Rieber 等](https://arxiv.org/abs/2205.15568) | 不能把减少 invalid 配置或硬件感知初始化作为原创。 |
| 有效性预测和编译隐藏特征 | [ML²Tuner](https://pldi25.sigplan.org/details/LCTES-2025-main/18/Multi-level-Machine-Learning-Guided-Autotuning-for-Efficient-Code-Generation-on-a-Dee) | validity model、compiler feature 与 performance model 的一般组合已有直接 VTA 近邻。 |
| 自适应测量和调优成本降低 | [AdaTune](https://proceedings.neurips.cc/paper/2020/hash/a9b7ba70783b617e9998dc4dd82eb3c5-Abstract.html) | 不能把早停或减少硬件测量时间单独作为原创。 |
| 分组感知的编译器调优 | [GroupTuner](https://arxiv.org/abs/2505.08598) | 不能把 family/group-aware 搜索本身称为首次提出。 |
| VTA 输入优先、权重复用与最小访存 | [Cheng 等](https://www.sciencedirect.com/science/article/abs/pii/S0167739X25004595) | O/I/W/IW 及 minimum-access 是本文基础机制，不能再承担原创主张。 |
| 编译调优中的 unknown constraint、多保真和多目标 | [CATBench](https://proceedings.mlr.press/v293/torring25a.html) | 不能声称首次将 multi-fidelity、unknown constraint 或 multi-objective 引入 compiler autotuning。 |

### 3.2 可以防守的研究空白

在本次检索范围内，尚未发现已有系统同时处理下面这一组约束：

1. 目标是固定 bitstream、显式 DMA 与软件管理 scratchpad 的 FPGA 加速器；
2. 驻留变换不仅改变 Tile 参数，还改变输入/权重的加载作用域、请求粒度、同步和命令需求；
3. 候选的精确语义只能在依次支付 lowering、FSim、FPGA 和完整图成本后揭示；
4. FPGA 数值正确性是硬门，仿真合法和局部低流量不能替代；
5. 调优动作同时包含候选和下一资格阶段，目标是首次取得通过完整图质量门的可部署程序所需的实际成本；
6. 最终输出是绑定共享内存合同、程序身份、route、容量和 fallback 的部署记录。

这一空白必须按“特定组合及其 VTA 实现与实证”来主张，不能写成首次提出驻留、合法性过滤、多保真优化或快速 AutoTune。

## 4. 推荐的新问题定义

令候选为

\[
x=(w,\theta,m,r),\qquad m\in\{O,I,W,IW\},
\]

其中 \(w\) 是真实 workload，\(\theta\) 是完整 ConfigEntity，\(m\) 是驻留方式，\(r\) 是共享边界与完整图 route 合同。资格阶段为

\[
j\in\{S,L,F,H,G\},
\]

分别表示静态合同、真实 lowering、FSim/命令、FPGA 正确性与计时、完整图 route 和质量门。候选在阶段 \(j\) 支付实际成本 \(c_j(x)\) 后，才可获得特征 \(\phi_j(x)\) 和资格结果 \(v_j(x)\)。

候选可部署当且仅当

\[
D(x)=\bigwedge_j v_j(x)=1
\]

且完整图质量满足

\[
L_G(x)\le(1+\epsilon)L_{\mathrm{ref}}.
\]

在线策略不能读取事后池 oracle。实验结束后才用冻结全池给出 \(L^*_{\mathrm{pool}}\)，复算 TTQ@+2%/+5%。研究目标不是简单最小化某个候选的算子时延，而是最小化

\[
T_\epsilon^\pi=
\inf\left\{t\mid\exists x\in\mathcal H_t, D(x)=1,
L_G(x)\le(1+\epsilon)L_{\mathrm{ref}}\right\},
\]

并完整报告到达这一时刻前的 lowering、FSim、交叉编译、FPGA、完整图、失败和恢复成本。

这个定义把“最快 kernel 搜索”改成“最早取得合格部署程序”。它与论文现有 time-to-quality 证据一致，也能容纳局部更快但完整图失败、访存更少但时延更慢，以及程序正确但不胜 incumbent 的真实反例。

## 5. 推荐算法：同 Tile 语义增量与候选×阶段决策

### 5.1 同 Tile 反事实族

以相同 \(\theta\) 下的 O/I/W/IW 为一个 family。模型不直接跨 Tile 拟合绝对时延，而分解为

\[
\log L(\theta,m)=\log L(\theta,O)+
\Delta_m\bigl(\Delta\mathbf S(\theta,m)\bigr)+\varepsilon.
\]

\(\Delta\mathbf S\) 是驻留模式相对同 Tile O 的语义增量，包括：

- input/weight/output 的 unique bytes、实际 LOAD/STORE bytes 与 calls；
- reload ratio、请求大小分布、短请求和 stride/padding；
- 分 SRAM 域的 span/lifetime，而非虚构的总 SRAM；
- submission、barrier/drain；
- instruction/UOP 峰值；
- 共享边界的 storage identity、布局、独占性、owner/generation 和完成条件。

使用同 Tile 增量能够直接利用现有配对四元组，降低不同 Tile 计算结构带来的方差；它也符合用户希望“用共享内存语义缩短 AutoTune”的原始目标。GroupTuner 已有一般分组搜索，因此论文增量必须明确落在上述 typed residency delta 和资格链，而不是“首次分组”。

### 5.2 分阶段存活与剩余成本

分别估计：

\[
P_{\mathrm{survive}}(x,j)=P(v_j=\cdots=v_G=1\mid\mathcal H),
\]

\[
\widehat C_{\mathrm{remain}}(x,j)=
E\left[\sum_{k=j}^{G}c_k(x)\mid\mathcal H\right].
\]

资格不是一个混合二值标签。必须保留 `not-applicable`、lower failure、FSim/UOP failure、FPGA numeric failure、infrastructure unknown、full-graph rejection 等 typed gate；unknown 不得训练成 invalid。

### 5.3 候选与下一阶段的联合动作

每一步选择动作 \((x,j_{\mathrm{next}})\)，而不是一旦选中候选就无条件跑完所有阶段。一个可实现的采集分数为

\[
A(x,j)=
\frac{P_{\mathrm{survive}}(x,j)
\left(EI_\epsilon(x)+\beta U(x,j)\right)}
{\widehat C_j(x)},
\]

其中 \(EI_\epsilon\) 是进入冻结 incumbent 质量带的预期改进，\(U\) 是探索不确定性，分母是下一阶段的预测实付成本。正确性始终是硬门，分数不能让一个数值失败候选继续计时。若当前数据不足以稳定训练概率模型，可以使用经过校准的 rank/Borda 与 cost bucket，但仍须保持“candidate×stage”动作、阶段特征可见性和成本归一化。

最低探索配额应覆盖 mode 和几何类别，防止早期语义先验永久屏蔽某类候选。invalid dominator 删除后，只用当前阶段已经支付并完整获得的字段重算前沿。最少 bytes 不能成为单一排序，因为现有 M3 和 TopHub 反例已经否定这一规则。

### 5.4 分阶段程序等价类去重

现有 broker 已在 lowering 后保存 effective program hash，历史结果也存在“多个 requested 身份落到同一有效程序”的情况。可以增加分阶段等价类：首次 requested 身份仍支付 proposal 和 lowering 成本；只有取得可核验的等价证书后，后续昂贵阶段才允许复用已有结果。

- lowering/FSim 等价至少绑定规范化 TIR、参数、布局、ABI、mode 实际效果和命令生成环境；
- FPGA/timing 等价还要绑定最终二进制、参数、bitstream、runtime、输入和 route；
- 任何字段 unknown、哈希不同或合同漂移都不得合并；
- 审计中仍保留原 requested 身份、首次发现时间和已付 lowering 成本。

这项机制可避免 reverted mode 或不同 knob 生成同一程序后重复支付 FSim/FPGA/timing，但结构等价和数据库复用已有广泛先例，不能单独声称首次提出。它应作为 RS-CMF 的成本动作与组件消融，而不是第四个独立创新点。

### 5.5 合同携带的输出

最终结果不是一个配置编号，而是

\[
\Gamma(x)=
(h_w,h_\theta,h_m,h_{\mathrm{TIR}},h_{\mathrm{route}},
q_{\mathrm{insn}},q_{\mathrm{uop}},\rho,A,C,D),
\]

包括 workload、配置、驻留程序、TIR、route 哈希，命令峰值，物理区间/anchor 关系，所有权与完成条件，资格证据和 fallback。运行时身份或完成合同变化时证书失效。这是经验部署资格，不称形式正确性证明。

## 6. 决定性实验：在现有阶段 05 上升级

现有 [05 C3 三个未暴露真实任务计划](../../thesis_supplement_experiments/05_C3_UNSEEN_ONLINE_PLAN.md)是最高性价比的执行骨架。它已经定义真实 source 菜单、历史 signature 排除、12 Tile×4 mode、24 请求/600 秒、27 次串行 attempt、paid known-mask、late-label、后置 oracle 和完整图部署。建议保留这些治理规则，修改策略和消融如下。

### 6.1 主在线比较

最小规模保持 `3 真实未暴露任务 × 3 策略 × 3 配对 seed = 27`：

| 策略 | 作用 |
|---|---|
| mode-aware stock XGB | 同空间的传统性能回归基线 |
| ML²-style P/V/A 功能级实现 | 最接近的 validity＋compiler-feature 强基线；必须明确不是原作者源码复现 |
| RS-CMF | same-tile 语义增量＋typed survival＋cost-normalized candidate×stage 决策 |

Random、minimum-access/DMA-only 与当前 fixed adaptive 策略可以在主 27 次之后，基于封存全池做严格 no-leak 多 seed replay；若板端预算允许，再把 Random 作为第四个 live 策略。只比较 Random/XGB 容易被质疑避开 ML²Tuner 这一最接近工作。

所有 live 策略必须使用相同的 48 requested 身份、相同初始静态信息、相同硬截止和相同完整输出规则。策略不能复用其他策略已经支付的 target label。顺序使用 Latin 轮换，至少两个真实 kernel boot；硬件或时间允许时，把 seed 从 3 提高到 5，但不能用同一任务的多个 seed 冒充多个新任务。

### 6.2 后置完整池与计分

27 条轨迹封存后，补齐每任务 48 身份的资格和测量，形成后置 finite-pool oracle。覆盖不足时只能称 best-known。TopHub 是重新资格后的 fallback/质量参考，不进入在线选择，也不要求本文最终程序一定击败它。

主指标：

1. TTQ@pool+2%/+5% 及实际完整墙钟；
2. 预算内取得 FPGA-correct、full-graph-qualified 程序的成功率；
3. gross request、lower、FSim、cross、FPGA correctness/timing、full-graph 次数；
4. 各阶段及失败、超时、恢复的实付成本；
5. 最终完整图质量、route 接受/拒绝与 fallback；
6. 搜索期间逻辑 DMA bytes/calls，明确不等同物理 AXI 流量。

未达标和基础设施失败按观察终点右删失。逐任务给原始曲线；3×3 配对只能做精确配对排列或 bootstrap 区间，不能用帧数或候选数扩大统计样本。

### 6.3 必需消融

主 live 实验后，利用封存完整池做以下 no-leak replay；特征只能在其原完成时间后出现：

1. 跨 Tile 绝对时延模型 vs same-tile \(\Delta\) 模型；
2. 去掉共享/驻留语义增量，仅保留 raw knob+mode；
3. 去掉 typed survival；
4. 去掉 cost normalization；
5. candidate×stage vs 候选一旦选中即跑到底；
6. 去掉 invalid peeling；
7. 去掉完整图 gate/certificate，仅以孤立算子选点。

这些消融比再增加同 Tile 加速图更重要，因为它们回答每一项方法设计是否真正缩短 TTQ。

### 6.4 预注册成功标准

建议在读取三个新任务的 target label 前冻结：

- 最终完整图正确性和部署成功率不低于最强基线；
- 至少 2/3 未暴露任务的 TTQ 或完整实付成本优于最强基线；
- 跨任务配对中位改善建议达到 20% 以上；
- same-tile delta、typed survival、cost normalization/candidate×stage 三类消融至少表现出可解释退化；
- 最终程序进入冻结 +2% 或 +5% 质量带，不要求超过 TopHub。

若仅一个任务获胜、优势加入失败/恢复成本后消失，或 ML²-style 基线更好，应把结论降为“可靠资格协议与条件性局部收益”，不能宣称更高效的通用调优器。

## 7. 风险分级与优先顺序

### 7.1 低风险：只执行现有阶段 05

不引入新算法，完成 3 任务×3 策略×3 seed 的同空间前瞻实验，把 C3 收缩为“可审计的分层资格与部署协议”。这会显著增强论文可信度；结果好时足以构成系统型硕士创新，但算法新意仍有限。

### 7.2 中风险：实现 RS-CMF，推荐

在现有阶段 05 的 broker 和 LiveExecutor 上加入 same-tile delta、typed survival、阶段成本和 candidate×stage selector。已有 same-tile 配对、214 点池和阶段日志可用于开发，但三个新任务必须留作确认集。实现量和风险相对可控，且与 Rieber、ML²Tuner、Cheng 和 Welder 的区别最清楚。

### 7.3 高风险 A：分块权重驻留

当前 W/IW 使用“整层权重满足 SRAM guard”的保守实现。可以进一步搜索 reduction/channel 子块的部分权重驻留，打破超 256 KiB 即拒绝的边界。这会扩大适用空间并形成新的代码生成机制，但 Timeloop/MAESTRO/CoSA/Welder 已系统研究分块和数据流，且 Cheng 已直接研究 VTA 权重复用。若没有 RS-CMF 的公平搜索实验，它仍容易被评价为新的调度模板。

### 7.4 高风险 B：闭合 C3→C2→C1 反馈

C3 返回 latency、DMA 服务、命令峰值和 route 资格，C2 重新签发容量/生命周期合同，再由 C1 重新选择划分。它能形成最强的三层统一故事，但需要新的联合目标、收敛规则和端到端消融；Welder、XTAT 等已有一般跨层优化，本文差异只能落在 CPU–VTA 共享对象的所有权与真实资格合同。毕业时间线下不建议优先。

## 8. 不建议投入的方向

1. 不继续以“更多 O/I/W/IW 个案”替代搜索算法验证。
2. 不训练大型神经代价模型；当前独立 workload 数量不足，模型复杂度会超过研究问题。
3. 不只与 Random 和 TopHub 比。TopHub 是缓存 incumbent，不是正在运行的调优策略。
4. 不再用不同空间的流程时间声称选择器优越；系统流程结果与同空间算法消融必须分开。
5. 不将逻辑 DMA 请求写成物理 AXI 流量，也不把最低 bytes 当作时延定理。
6. 不把 manifest、日志、哈希、Q1 定容或更多图表单独列成新的算法贡献；它们是形成可核查系统证据的必要工程。

## 9. 推荐的论文创新表述

若 RS-CMF 和未暴露任务实验成功，第三点可改写为：

> 针对显式 DMA FPGA 加速器中驻留变换会同时改变数据复用、请求粒度、同步、命令需求和硬件正确性，而这些信息只能在不同资格阶段逐步获得的问题，本文提出驻留语义驱动的成本感知分层调优方法。该方法以同 Tile 驻留反事实族建模共享内存语义增量，联合选择候选及其下一资格阶段，并将 FPGA 多输入正确性、完整图质量门和精确部署合同作为硬约束，以降低首次取得合格完整图程序所需的真实成本。在三个预冻结未暴露任务的同空间、同信息在线实验中，本文方法……。

最后一句必须等真实结果产生后填写。若中风险算法未完成，当前更严谨的表述仍应保持为“显式 DMA VTA 程序的成本感知分层资格与部署方法”。

## 10. 建议的下一步

1. 先冻结本文档及阶段 05 的修订版，不运行目标任务。
2. 从现有 same-tile 四元组和阶段日志开发 RS-CMF，完成 host 级 no-leak、known-mask、unknown 和删失测试。
3. 将阶段 05 的三个主策略从 Random/XGB/current heuristic 修订为 XGB、ML²-style、RS-CMF；Random 转后置 replay。
4. 接通真实 source/history 扫描和 LiveExecutor，冻结三个未暴露任务及 48 身份。
5. 再按 Latin 顺序执行 27 次在线运行及后置完整池，不在结果出现后修改 acquisition、阈值或任务。

这条路线直接补齐当前论文最缺的“共享内存语义为什么能缩短调优”的因果证据，同时最大限度复用已经完成的基础设施。
