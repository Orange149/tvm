# E3 pilot：相邻 tail 切点的数值资格化

日期：2026-09-08。**发现跨切点数值不等价，完整 E3 尚未完成。** 这是 CPU-only 数值实验；本轮没有新上板、AutoTVM 或性能数据，没有修改正式量化/运行时代码。

后续已完成[局部一次量化六臂主机资格化](E3_ONCE_QUANTIZED_HOST_QUALIFICATION.md)：保留同一数值语义后切分前后逐元素一致，43 个回绕位置也被原样保留，而非修复。该后续还区分了既有屏障处切分与去屏障导致的融合变化；完整 E3 和板端资格化仍未完成。以下内容保留为原 pilot 的证据与当时结论。

## 实验设计

三对区间为 `05..06 / 05..07`、`10..11 / 10..12`、`15..16 / 15..17`。每对使用相同模型参数、相同输入，比较：

- 外部 tail：独立量化的 main+projection stage 输出两个 float32 张量，再做 float32 add/ReLU。
- 内部 tail：把 add/ReLU 纳入 stage，按原 `global_scale=8.0, skip_conv_layers=[]` 独立量化，输出一个 float32 张量。

六份 LLVM reference 均使用原 E2 的量化构建方式；逐参数内容哈希与 E1 冻结记录一致。输入完整复用 E2 三组 `uniform[-1,1] / normal(0,2) / zeros`，不是看过结果后构造的压力输入；同时检查每对 E2 的两个归档输入内容相同。输出、量化 Relay JSON/文本、参数身份、来源哈希及所有差异均已保存。

脚本：`diagnose_vta_tail_quantization.py`；[事前协议](e3_tail_pilot_run1/preregistered.json)、[全部结果](e3_tail_pilot_run1/summary.json)。外部 float32 add/ReLU 是该局部 tail 的数值表达，不包含完整 CPU suffix 编译或整网精度评估。

## 结果与机制

| 对照 | uniform 不同元素 | normal 不同元素 | zeros 不同元素 | normal 比较元素总数 |
|---|---:|---:|---:|---:|
| layer2：05..06 → 05..07 | 0 | 0 | 0 | 100,352 |
| layer3：10..11 → 10..12 | 0 | 1 | 0 | 50,176 |
| layer4：15..16 → 15..17 | 0 | 42 | 0 | 25,088 |

九组共比较 526,848 个输出元素，43 处不同；不同位置最大绝对误差 16，且内部 tail 对应值均变成负数。不能用总体低差异比例掩盖 ReLU 后负值问题，也不能将合成输入结果当成 ImageNet 精度损失。

`analyze_vta_tail_quantization.py` 对完整量化 Relay AST 检查终端链：`int32 add → ReLU → int8 cast（没有 clip）→ float32 cast → ×1/16`，中间含 stop_fusion 注解。随后用外部结果 `x` 推导

`x_wrapped = (((16*x + 128) mod 256) - 128) / 16`。

全部 526,848 个元素的预测均与内部 tail 输出精确一致，43 个差异恰好落在正值超出 int8 表示范围的位置。layer3 的一个值由 `8.5625` 回绕为 `-7.4375`。这说明本组输入的差异可由 tail 的 int8 窄化解释，不需要诉诸 DMA、线程竞争或 AutoTVM tile。该解释为结果出现后的机制检查，单列于 [wrap_analysis.json](e3_tail_pilot_run1/wrap_analysis.json)，未伪称事前假设。

## 现有证据如何解释

1. E2 并未失效：它检查每个 stage 与**该 stage 自己的量化 LLVM reference**一致。reference 本身保留窄化行为，二者一致不等于跨切图数值等价。
2. E1 的同卷积 TIR 不变仍成立；变化发生在 tail 数值表示，而不是已经发现同 workload 的 tile 最优解变化。
3. 边界 payload/逻辑 DMA 账本仍可描述各自二进制，但公平性能比较必须补共同数值政策/精度资格化。形状、dtype 相同不能作为语义等价证明。
4. 尚未解释旧 Experiment C 的 0.120%：那是另一对完整 Executor 边界，必须用其原输入、输出及量化图单独归因，不能将本轮机制直接套用。

## 后续执行顺序

- 先冻结共同数值参考，检查一次量化后的同图切分是否能保持一致；逐项记录 cut 前后的 scale、round、clip、cast，以及临时张量 dtype/layout。
- 以 layer4 的上述反例作为必须通过的回归输入；保留原三个固定输入组，并加入正式流水真实激活，不仅测全零/同图自参考。
- 如需改量化政策，必须明确选择浮点 tail、较宽整型交接或一致的饱和规则，并分别报告语义/精度影响；不能把“补一个 clip”默认为等价修复。旧基线保留、不覆盖。
- 数值资格化后再执行 E3 单 Executor 融合/stop_fusion、双 Executor shared/copy 四臂及 block-aligned 负对照，测完整边界与配对性能。

本轮不把 E3 打勾，不据此触发 E4 tile 搜索，也不宣称已经找到性能优化。优先价值是阻止用不等价的计算图证明“少搬运必然更优”。
