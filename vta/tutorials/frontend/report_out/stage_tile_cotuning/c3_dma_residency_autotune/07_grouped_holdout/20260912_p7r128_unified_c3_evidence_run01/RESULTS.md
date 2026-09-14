# P7R128 第三创新点统一证据摘要

状态：`LOCAL_SYNTHESIS_COMPLETE; BOARD_EXECUTION_REQUALIFICATION_REQUIRED`

## 结论

第三创新点应定稿为“面向固定 FPGA 的共享内存访问感知分层调优与时空协同优化”，而不是“超越 TopHub”或
“单独缩小命令队列”。方法把四类原本分散的量绑定到同一个候选身份：tile/驻留导致的数据 LOAD、
FPGA 正确性与合法性、显式同步代价、instruction/UOP 单提交峰值。目标是在正确候选集合内，以较少
gross dispatch 找到接近已资格化强 incumbent 的配置，并由最终 allowlist 生成安全命令 backing。

## 跨几何资格汇总

| 几何 | workload | 冻结身份 | static pass | 三 seed FSim pass | FPGA 已完成/通过 | 三模式全正确 family | latency |
|---|---|---:|---:|---:|---:|---:|---|
| Y02，3x3，26x26，384→256 | conv21 | 9 | 8 | 8 | 6/2 | 0/2 已测 family | B00 original/barrier 已测 |
| Y01，1x1，13x13，1024→256 | conv13 | 24 | 24 | 24 | 23/0，1 未分类 | 0/8 | 禁止计时 |
| Y00，3x3，208x208，16→32 | conv2 | 24 | 18 | 11 | 0/0 | 0/8（本地） | 尚未上板 |

这里的“0/8 三模式全正确”不等于 Y01 24/24 身份均已测失败：F06 barrier 因 RPC broken pipe 未分类，
但该 family 的 original/input 已失败，所以它不可能成为三模式全正确 family。

## Y02B00 同 tile 时间—空间结果

| mode | FPGA correctness | latency median | LOAD bytes | weight bytes | LOAD calls | sync | insn peak | UOP peak | 4KiB 对齐命令 backing |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| original | 3/3 | 140.989 ms | 19,488,768 | 11,501,568 | 9,984 | 1 | 259,760 B | 328 B | 266,240 B |
| input-stationary | 0/3 | 禁止 | 12,000,768 | 11,501,568 | 624 | 1 | 19,520 B | 5,108 B | 28,672 B |
| weight-resident-barrier | 3/3 | 105.349 ms | 8,871,936 | 884,736 | 5,008 | 17 | 11,792 B | 7,504 B | 20,480 B |

相对 original，weight-resident-barrier 的 latency 改善 25.28%，7/7 配对轮获胜；总 LOAD 减少
54.48%，weight LOAD 减少 92.31%，对齐命令 backing 减少 92.31%。这是当前最强的联合证据：
同一个驻留/分段机制既减少 u-dma-buf 数据重放，也限制单提交命令峰值；增加的 16 次 residency
drain 没有抵消收益。

观测加速比为 `140.989/105.349=1.338x`。若仅作 Amdahl 式一阶辨识，并假设时间对 LOAD bytes
线性、其他成本不变，则有效共享内存敏感占比约为 `0.2528/0.5448=46.4%`；完全消除这部分 LOAD
时的拟合上限约为 `1/(1-0.464)=1.87x`。这只是由单个配对反推的局部上限，不是 FPGA 理论峰值，
后续必须用更多 tile 和几何拟合 `bytes/calls/sync/compute` 多变量模型。

input-stationary 虽然静态 LOAD calls 更少，但板端 0/3 正确，不能进入性能排序。这说明 tuner 的第一
目标必须是 FPGA-correctness，不是最小化某一个静态计数器。

## TopHub 的正确角色

Y02 exact TopHub 的 source template 与 mode0 adapter 生成相同 TIR 和相同交叉二进制，本地 FSim
均 3/3 正确，但板端均 0/3，而且重复调用输出 hash 不稳定。因此该记录当前不能作为 latency 分母、
部署 fallback 或 `success@2%/5%` 的 reference。这个结果不表示本文“超过 TopHub”，而表示 TopHub
也必须经过目标 bitstream/runtime 的正确性资格。搜索目标应写成“用更少测量逼近 correctness-qualified
incumbent”；只有通过资格的 TopHub 才能充当该 incumbent。

## 当前论文等级

当前已形成硕士论文独立创新点的强核心：完整方法链、真实 FPGA fail-closed、同 tile 因果配对，及
DMA—同步—命令空间—latency 的同身份联合证据。它已明显强于单纯的 u-dma-buf 参数调整或固定队列
缩容。

但它仍是 **CCF-C 候选水平，不是已完成的 CCF-C evaluation**。缺口为：第二个正向可计时几何、
至少一个多候选完整正确池、20-seed 等 gross-budget replay，以及新 allowlist 的实际容量通过和不足
容量拒绝。Y01 的 23 个硬件错误与 Y00 的 compact/UOP 失败是有价值的边界证据，但不能替代第二个
正向性能几何。

## 板端恢复后的冻结顺序

1. 不由脚本重启板卡。执行面由用户恢复后，先运行 W05 config575 三 seed 健康门；失败则停止。
2. 只补 P7R126 唯一未分类的 Y01F06 barrier 三 seed，闭合 54 次审计记录；不计时。
3. 按 P7R127 观测前冻结的结构顺序，先测 Y00F01 original/input 各三 seed；任一失败则停止该 pair。
4. 若两者均正确，做 7 轮 AB/BA same-tile 配对计时，形成第二几何的 input-reuse 正/负结果。
5. 只有得到至少一个新的正确 pool 后，才运行 20-seed、budget 4/8/12/24 的等 gross-budget replay。
6. 对最终 selected + qualified fallback allowlist 推导 instruction/UOP 容量，做精确容量通过和小一页
   的 fail-closed 拒绝测试。

## 主张边界

- lowered-TIR/runtime 记录是逻辑 DMA 请求，不等同于物理 AXI burst。
- 20,480 B 只属于当前 Y02B00 barrier 身份，不是 VTA 通用常数。
- 不把 Y01 broken pipe 计为 candidate-invalid，也不声称板卡永久损坏。
- 不声称 YOLO 平均加速、跨 boot 稳定、整网 FPS、TopHub 等价或搜索预算收益已经成立。
- 2026 工作可作为公开机制的并行/后续文献引用；本文可陈述 2025 年已立项并独立实现，但不能省略
  正式论文引用或声称逐行复现未知源码。
