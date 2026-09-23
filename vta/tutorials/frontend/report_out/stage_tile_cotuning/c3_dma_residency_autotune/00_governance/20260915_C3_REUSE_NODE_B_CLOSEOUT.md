# C3 节点B收尾：四方案真实复用已本地确认，等待FPGA确认

日期：2026-09-15。状态：`NODE_B_COMPLETE_PAUSED; C1_NOT_STARTED; NO_NEW_FPGA_LABELS`。

## 1. 核心结果与边界

本轮不再仅依靠模式名称推断复用。在三个固定机制几何上，选出 **11个same-tile四元组、44个主身份**，
另有 **4个匹配循环与同步的权重reload控制**。48/48身份完成三seed FSim正确性和ARM交叉编译；
11/11四元组均有四种不同的最终TIR和ARM二进制，IW相对O同时减少input和weight逻辑搬运。

这使“组合实现能否产生真实双复用”的**本地证据**从节点A的不足变为成立。没有联系开发板，
FPGA正确性、延迟、driver累计执行时间和跨启动稳定性仍全部待测；不能写组合一定最快、搜索改善
或整网FPS提高。三个几何是synthetic机制测试，不宣称三个未见网络。

| 几何 | 完整ConfigSpace | 机制条件后的family | 冻结最多检查 | 实际检查family | 选出四元组 | 有效双复用 |
|---|---:|---:|---:|---:|---:|---:|
| M1：64→64，28×28，3×3s1p1 | 1296 | 210 | 64 | 44 | 4 | 4 |
| M2：128→128，14×14，3×3s1p1 | 1024 | 180 | 64 | 64 | 3 | 3 |
| M3：128→256，14×14，1×1s2p0 | 320 | 48 | 48 | 5 | 4 | 4 |
| 超容量边界：128→256，14×14，3×3s1p1 | 1280 | 240 | 64 | 6 | 不上板 | 不适用 |

M2耗尽64个family检查上限后只有3组，不补点、不扩几何，也不伪造第四组。主池因此为44而非48身份；
已超过“至少两个几何各两组有效双复用”的积极确认门槛。边界检查在获得两个original合法family
`BNDD0012/BNDD0067`后停止，其余失败仍保留。整层权重294,912B超过256KiB；当前W/IW共用的
full-layer gate拒绝它们。W gate是额外保守限制，不代表其weight tile本身必然超容量。

## 2. 共享内存语义究竟补实了什么

保存并核对以下链条，而不是只输出一个DMA总数：

```text
相同ConfigEntity + 驻留方式
  → 最终LOAD/STORE所在循环与条件路径
  → packed DDR源矩形/区间、SRAM目标偏移、请求尺寸/跨距/padding
  → 实际请求multiplicity与唯一源区域union
  → StorageRewrite后、设备地址lowering前的SRAM arena跨度
  → FSim三seed数值结果、runtime逻辑DMA与指令实际搬运
  → 命令结构、submission、drain、instruction/UOP峰值
  → 相同TIR的ARM二进制
```

新静态分析求值then/else与Let条件，不对未执行LOAD机械乘循环次数；不变循环折成multiplicity。
padding零填充不计DDR字节，实际请求矩形须在冻结张量范围内。reload ratio分母是实际LOAD的唯一源
区域，不是无条件整张量大小，也不是所有乘加真正使用的元素集合：stride2连续请求可能包含数学上
未使用的中间位置。区域union只描述搬运覆盖，不能冒充完整GEMM消费者/异步覆盖冲突证明。

每个正式seed恰执行一个完整算子；静态bytes/calls逐项与runtime匹配，input/weight/output指令字节
再与FSim匹配。UOP的高层外部LOAD为零，不代表没有微指令：runtime生成的UOP和命令峰值单列。
不同张量内存域与命令backing不混为一项空间指标。

## 3. 一个完整四方案例子：M1D0139

固定`tile_h=2, tile_w=28, tile_ci=1, tile_co=2`，两个线程knob均为1。
以下数据均为**本地单完整算子**，不是FPGA性能：

| 方案 | input B / calls | weight B / calls | input arena B | weight arena B | ACC arena B | submission / residency drain |
|---|---:|---:|---:|---:|---:|---:|
| O | 193,536 / 112 | 516,096 / 112 | 1,920 | 4,608 | 7,168 | 1 / 0 |
| I | 96,768 / 56 | 516,096 / 56 | 1,920 | 9,216 | 14,336 | 1 / 0 |
| W | 193,536 / 112 | 36,864 / 2 | 1,920 | 18,432 | 7,168 | 3 / 2 |
| IW | 96,768 / 56 | 36,864 / 1 | 1,920 | 36,864 | 14,336 | 2 / 1 |

这组数据可以直接解释空间换搬运的具体方式：

- I的input bytes减半，但input SRAM跨度没有增大；扩大的是覆盖更多output-channel的ACC区域和
  reduction-scope权重临时区域。weight调用减半、单次请求变大，所以weight总bytes没有下降。
- W将一个output-channel tile的全部reduction权重留在片上，weight arena由4,608B变为18,432B，
  全部weight bytes从516,096B降到36,864B，代价是两次显式residency drain。
- IW同时覆盖所有output-channel和reduction权重，weight arena到36,864B、ACC到14,336B；
  input减50%，weight减92.86%。它是否能用减少的搬运抵消同步和地址/命令变化，要由FPGA时间回答。

对于这个案例，权重arena对应的覆盖范围为：

| 方案 | output-channel blocks | reduction-channel blocks |
|---|---|---|
| O | tile_co | tile_ci |
| I | 全部CO blocks | tile_ci |
| W | tile_co | 全部CI blocks |
| IW | 全部CO blocks | 全部CI blocks |

这解释了与Cheng的关系：功能级重实现其输入优先、权重驻留与组合目标，并用本地lowering核对具体
加载范围和内存代价；不是其未公开“不覆盖地址/runtime”源码的精确复现。USMP可以为既定程序做
offset和liveness packing，却不会自动把上述四种加载/计算作用域互相转换。本轮也未实现USMP的
真实VTA地址重写。不能把USMP概括成必然增加执行时间。

## 4. 跨tile/几何的结构结果

11组IW相对O的input下降50%--93.75%，weight下降50%--97.96%；均保持输出字节不变。
这些是经过机制资格筛选后的池内范围，不是自然ConfigSpace的平均收益。所有原始失败也保留，
不得用11/11成功掩盖筛选成本或宣称任意tile适用。

M2D0140的weight arena从O的9,216B升至IW的147,456B，占固定256KiB容量56.25%；ACC从3,584B
升至7,168B，input仍为768B。M3D0049则从2,048B升至32,768B，input从23,296B降至11,648B。
这些分域数据比“多用一点SRAM”更能体现固定硬件容量下的具体权衡。

arena由真实StorageRewrite后捕获，三种SRAM在本轮候选中各有明确分配。跨arena求和仍只报告跨度
上界，不是已证明的同时存活峰值；本轮硬件SRAM总容量没有变化。FSim命令peak也不是新的FPGA
安全定容结果，不能据此称释放了整个192MiB u-dma-buf。

4个reload控制保留compute作用域、input/output DMA和sync，且恢复后weight搬运确实增加，全部
三seed正确。但SRAM索引/分配仍可能改变，只能称匹配循环与同步的对照，不作完全隔离的物理因果断言。

## 5. 资格成本、协议迭代与真实失败

正式run02包括476个family-mode提案与4个控制，共480条lower阶段记录：114通过、354失败、
12个边界not_applicable；实际compiler attempts为468。58个FSim worker中50通过、8因真实duplicate
`dst_idx` UOP依赖检查失败。通过worker包含一次预热及三seed，合计200次完成调用；另8次预热
尝试未完成，故208次尝试。不能把异常归为错误输出，因为未得到可比较输出。

| 阶段 | 完整阶段墙钟 | 说明 |
|---|---:|---|
| prepare run02 | 9.634s | 生成完整域及确定检查顺序，源码/库/配置绑定 |
| local run02 | 98.083s | 所有lower、FSim、失败；含48次SDK setup失败 |
| cross recovery run01 | 56.826s | 不重跑资格，48/48 ARM编译成功 |
| freeze run02（最终协议包） | 0.645s | 核验门槛、源码、二进制；冻结family/八轮顺序、完整results并封存 |
| independent local analysis run01 | 0.068s | 逐seed/逐指标再验证，生成结构CSV和加载作用域图 |

lower/FSim host墙钟分别26.410/69.167s；成功交叉编译host墙钟合计56.456s。阶段墙钟包含控制及
host bookkeeping，不能机械等于子步骤之和。这些独立已完成阶段的总投入约165.3s，不是一次连续
T0→T1计时，更不是time-to-quality；本轮无板端T0。

另保留prepare run01的9.678s及local run01的104.908s诊断成本。首次调用有初始化相关命令差异，
旧签名门槛将数值正确的点拒绝，不能以run01的零四元组结果证明机制无效。run02显式加入并计账
预热后解决该问题。正式local run02的48次cross失败是SDK拒绝继承的`LD_LIBRARY_PATH`，编译器
尚未启动；恢复脚本只为SDK子进程移除它，保留原错误与TIR身份，不修改用户登录环境或原结果。

## 6. 产物与测试

- [冻结合同与48个ARM二进制](../07_grouped_holdout/20260915_c3_reuse_node_b_freeze_run02/contract.json)。
- [逐模式结构CSV](../07_grouped_holdout/20260915_c3_reuse_node_b_local_analysis_run01/four_mode_structure.csv)。
- [LOAD作用域图](../07_grouped_holdout/20260915_c3_reuse_node_b_local_analysis_run01/load_scopes.svg)。
- [11组实际复用及程序去重分类](../07_grouped_holdout/20260915_c3_reuse_node_b_local_analysis_run01/family_classification.json)。
- [正式lower/FSim尝试和失败](../07_grouped_holdout/20260915_c3_reuse_node_b_local_run02/results.jsonl)。
- [独立交叉编译恢复账本](../07_grouped_holdout/20260915_c3_reuse_node_b_cross_recovery_run01/results.jsonl)。

各阶段保留不可覆盖合同、timeline/summary及artifact hashes。本地阶段保存TIR/storage TIR、
请求区域、三seed输出哈希及命令签名。prepare提案保存在合同内，没有独立results.jsonl；最初freeze
run01也仅有合同/candidates而缺统一results。它们保持只读，最终freeze run02已补齐规范的五类
记录文件，绑定上游账本并保留全部二进制；family顺序seed=20260910、四模式Latin轮转重复两遍
均已冻结，每启动主计时352次。初稿freeze run01的0.416s也保留为额外成本。
bitstream期望哈希来自既有板端协议，**本轮未观测它**。

测试：**56 passed，3 skipped**。三个skip是旧测试要求`TARGET=sim`的FSim分支；新入口四模式的
真实FSim三seed测试均执行通过。覆盖条件LOAD、padding、Let/invariant loop、动态条件拒绝、
源码/产物完整性、未来标签隔离、控制匹配、真实FSim/runtime/静态一致、冷/热签名、无覆盖输出、
SDK环境隔离及缺失指标/错误结果/TIR/seed的拒绝。board阶段尚未实现，显式执行标志也不能绕过。

## 7. 暂停与下一节点

按计划停在节点B。下一步C1必须实现并冻结新的八轮四模式板端协议，不能直接套用硬编码R18/五轮
旧runner。还需SSH/boot/filesystem/bitstream/runtime健康门、合格探针、output-data-weight布局复用、
三seed FPGA correctness fail-fast、逐调用输出检查及独立profile成本。

第一启动完成后再暂停，第二启动须用户实际重启并恢复健康门。**本轮计划尚未整体完成**，
不能用本地11组双复用替代FPGA延迟、跨启动稳定性以及最终四张完整机制/性能图表。
