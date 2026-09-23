# ResNet18 SRAM 统计勘误与可引用口径

日期：2026-09-22。范围：旧 ResNet18 stage-search 的 38.28125% 及其衍生解释；本次只做本地审计与统计代码修正，不上板、不重跑搜索、不改写历史实验原始文件。

## 1. 结论

**撤回“ResNet18 推理 SRAM 利用率为 38.28%”的表述。** 它既不是实际编译分配占比，也不是运行期间有效数据占用的峰值或平均值。原因有两层：分母的字节单位用错；分子是整个计算单元的张量/参数估算，而不是实际 Tile 的 SRAM 分配。

当前可验证的替代证据是 R18-H1 孤立卷积的逐候选静态 arena。不能用它替代为整网利用率。ResNet18 整网静态分配峰值、动态有效占用仍为 **unknown**。

## 2. 38.28125% 的来源

旧 `split_resnet18_stages.py::build_vta_capacity_spec` 使用 `env.INP_BUFF_SIZE * env.INP_ELEM_BYTES` 等表达式作为容量。实际上 `*_BUFF_SIZE` 已经是 byte，不是 vector 数。

本地架构证据：

- `3rdparty/vta-hw/include/vta/hw_spec_const.h`：SIZE 定义为 byte，DEPTH 才是 SIZE / ELEM_BYTES。
- `3rdparty/vta-hw/config/pkg_config.py`：INP/WGT/OUT 的物理容量为 `1 << LOG_*_BUFF_SIZE` byte。
- `build_resnet18_unit_metadata`：输入、输出、ACC 用整单元 shape 估算；权重用单元参数量换算。未读取真实分块分配。

对 [batch004 原始 summary](../../../native_stage_pipeline_searches/batch004/summary.json) 中 best 的旧数值复算如下。各域最大值可能来自不同单元，不是同一时刻的占用快照。

| 域 | 正确容量 B | 旧错误分母 B | 旧分子：整单元估算 B | 旧报告 % | 修正后的整单元/容量 %，非利用率 |
|---|---:|---:|---:|---:|---:|
| INP | 32,768 | 524,288 | 200,704 | 38.28125 | 612.5 |
| WGT | 262,144 | 67,108,864 | 4,722,688 | 7.037353515625 | 1,801.5625 |
| ACC | 131,072 | 8,388,608 | 802,816 | 9.5703125 | 612.5 |
| OUT | 32,768 | 524,288 | 200,704 | 38.28125 | 612.5 |

输入例子：`64 × 56 × 56 × 1 B = 200704 B`；旧公式为 `200704 / (32768 × 16) × 100 = 38.28125%`。

修正后的 612.5% **也不能叫真实 SRAM 利用率或越界率**：整张量可以通过分块合法执行。旧 `onchip_peak_util_pct` 只是上述错误比值的最大值，并没有增加任何实际峰值观测。

因此，基于这些字段作出的“SRAM 尚有 61.72% 空闲”“SRAM 不是约束”“无 spill”结论都不成立。是否发生额外装载/保存，需要实际生成程序与命令证据。

## 3. 现在能可靠引用哪些数值

本次重新反序列化封存的 StorageRewrite 后、LowerDeviceStorageAccessInfo 前的 TIR，逐个统计 `Allocate.extents × dtype 字节数`。循环执行次数不乘入分配容量。然后核对保存的分配记录、后续 TIR、交叉编译程序与二进制 SHA256，以及对应板端成功计时的候选集合。

结果目录：[完整审计 README](../07_grouped_holdout/20260922_resnet18_sram_accounting_audit_run01/README.md)、[逐候选 CSV](../07_grouped_holdout/20260922_resnet18_sram_accounting_audit_run01/candidate_arenas.csv)、[审计 JSON 与 57 个输入文件哈希](../07_grouped_holdout/20260922_resnet18_sram_accounting_audit_run01/audit.json)。21 个请求身份，17 个成功编译身份；其余保留失败状态，不填零。成功身份与历史板端计时集合一致；本次未重新测时。

R18-H1 是输入/输出通道均为 64、空间 56×56、3×3 卷积。以下为同一硬件配置下的部分历史候选；百分比是本存储域 arena 字节数 / 本域容量。

| 配置 | INP arena / 容量 | WGT arena / 容量 | ACC arena / 容量 |
|---|---:|---:|---:|
| TopHub 原方案 | 18,560 / 32,768 B = 56.640625% | 9,216 / 262,144 B = 3.515625% | 114,688 / 131,072 B = 87.5% |
| ALT_251 原方案 O | 8,352 / 32,768 B = 25.48828125% | 4,608 / 262,144 B = 1.7578125% | 50,176 / 131,072 B = 38.28125% |
| ALT_251 双复用 IW | 8,352 / 32,768 B = 25.48828125% | 36,864 / 262,144 B = 14.0625% | 100,352 / 131,072 B = 76.5625% |

注意 ALT_251 的 ACC 恰好也是 38.28125%，但这是另一个有真实分配证据的量，不能据此恢复旧“整网 38.28%”的说法。

口径限制：

- 本表各域恰有一个静态 arena，包含该候选虚拟线程变换后的分配。多个 arena 或多个函数不能未经地址/生命周期分析直接相加后声称同时峰值。
- arena 含分配跨度、复用及可能的对齐空间，不等于每个周期装有多少有效数据，也不是时间平均有效占用。
- INP、WGT、ACC 是不同存储域；各域比例不能直接相加，也不能把 WGT 空余视为 ACC 可用容量。
- OUT/UOP 没有从这组分配产物取得完整峰值证据，保持 unknown。累计 LOAD UOP 字节数及 DDR 命令缓冲区大小不能替代片上 UOP 峰值。
- 此表是 9 月的 R18-H1 算子候选，不是 5 月整网程序；不能外推 ResNet18 整网峰值或平均值，也不推断 YOLOv3-tiny。
- FPGA 实现报告的 BRAM 物理资源占比是另一指标；引用前还需绑定同一个实现/bitstream，不能与 arena 占比混为一谈。

可直接用于报告的表述：

> 旧 ResNet18 stage-search 报告的 38.28% 来自单位错误的整单元张量容量比，已撤回作为 SRAM 利用率。对封存的 R18-H1 TopHub 候选进行编译产物复核后，输入、权重、累加器 SRAM 的静态 arena 分配占比分别为 56.64%、3.52%、87.50%。这些数值不代表整网或运行时平均有效占用。

## 4. 修正代码与历史结果的关系

- `split_resnet18_stages.py`：修正 byte 容量，不再额外乘 vector 宽度；明确整单元比值语义。
- `calibrate_resource_cost_model.py`：OUT 容量按 i8 输出宽度由 ACC 推导，不再误用 a32 ACC 容量。
- `search_resnet18_stage_splits.py`：整单元比值单独放在 `unit_tensor_capacity_ratio_pct`；没有实际分配证据时，`onchip_*_util_pct` 为 null，状态为 `unknown_not_lowered`；新生成 CSV/README 不再把无证据历史数字发布为 SRAM 利用率。
- 取消仅凭上述整张量比值或“需要超过 8 块”得出的容量硬拒绝，以及无证据的 SRAM 风险/实际 spill 罚分；真实 lowering 的容量合法性检查未取消。
- 剩余 tile/DMA/`tile_spill_penalty` 字段仍是旧成本启发式，不是实际 Tile 数、DMA 或 spill 计数。修正分母会改变这些估算，旧校准模型和排名不能直接当作修正后的有效结果；如继续搜索，需要重新校准和验证。

原始 summary、CSV、日志、TIR、库及旧生成 README 均保留；勘误优先于其 SRAM 解释。历史正确性与计时记录未因统计修正被改写，但本次也不重新认证全部旧性能实验。审计在读取后复核 57 个输入文件哈希均未变化。

复现只读审计：在可导入本地 TVM/VTA 的环境中运行 `audit_resnet18_sram_accounting.py --output <新的输出目录>`。脚本拒绝覆盖已有输出目录。

回归测试入口：`python -m pytest -q vta/tutorials/frontend/test_resnet18_sram_accounting.py`。

本次验证：新增统计回归测试 10 项，加上既有成本校准测试 2 项，共 12 项通过；修改的 Python 文件通过语法检查，相关 tracked diff 通过空白检查。未重新搜索、编译整网或访问板卡。

## 5. 整网指标后续如何补齐

若要报告某个确定 ResNet18 整网程序的静态 SRAM 需求，应先绑定该程序、硬件配置、调优记录和实际采用的所有 VTA kernel；逐 kernel 保存相同编译阶段的分配证据，核查每个域的地址范围及并发重叠，再按实际调用/调度关系汇总。不能用未采用的候选最大值代替整网峰值。

即便补齐这些静态数据，也只能报告静态分配/地址高水位。没有时间与有效数据生命周期证据时，整网“平均 SRAM 有效占用率”仍不能给出。本次不以重跑整网或开发硬件计数器作为纠正旧数字的前提。
