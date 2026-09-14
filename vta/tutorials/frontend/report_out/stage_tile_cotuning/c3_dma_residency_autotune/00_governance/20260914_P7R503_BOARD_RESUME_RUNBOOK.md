# P7R503 后续执行清单：ResNet18 文献对齐实验

状态：`P7R524C_FINAL_FIGURES_AND_DOCUMENTATION_COMPLETE`

本文档把 P7R508 以后的输入、输出、顺序、计时边界和暂停点固定下来。P7R508/P7R509 分别追加
SSH 认证冻结和重启恢复证据，P7R510 已完成 214 点板池；它们没有改变 P7R494 的候选或 P7R496
的 ARM 二进制。后续编号顺延，但策略、预算和信息边界不变。

## 1. 已闭合范围

- 节点 A：P7R462--P7R468 已完成 Y10 六次同空间真实运行及归因；Y10 已暴露，只作开发证据。
- 节点 B：六策略接口、HW-Aware 四级消融、ML²Tuner P/V/A 与 A+DMA、Cheng 四模式身份和测试均已
  实现；P7R507 是当前最终无标签修订合同。
- 节点 C 本地侧：三个 ResNet18 几何的完整 original 扫描、扩展选择、四模式 lowering、三 seed
  FSim 和 214/214 交叉编译已经完成。
- 资格追溯审计：P7R494 的 214 个身份在 P7R471/473/475/477/479/481/492 的并集中
  `missing_static=0`、`missing_fsim=0`。因此板后适配不允许重新生成或猜测资格结果。

尚未完成的是板后六策略回放、Cheng 四方案板端比较、策略所选完整图，以及 H1 上官方空历史 XGB
与本文方法各三次 clean-start 的 T0→T1 比较。

## 2. 板端准入门

网络 SSH 主机密钥必须与串口读取的 Dropbear 公钥一致。串口命令为：

```sh
dropbearkey -y -f /etc/dropbear/dropbear_rsa_host_key
```

只有核对完整公钥或 SHA256 指纹后，才能更新本次专用 `known_hosts` 并连接。不得使用
`StrictHostKeyChecking=no`，也不得在密钥未确认时开始 P7R508。默认 runtime 文件哈希、RPC PID、
RPC 工作目录和 bitstream 状态均在 runner 的 W0→T0 preflight 中检查。

2026-09-14 串口输出的完整公钥已与网络侧精确匹配，SHA-256 为
`SHA256:u7LMrzTebC0p8iNcXkRoTweS6DEqa0EYmM0uhFpxLX0`。P7R508 已冻结严格 host-key 与公钥优先认证，
P7R509 已在 tmpfs 恢复冻结 runtime；此门已经通过，未使用 `StrictHostKeyChecking=no`。

## 3. P7R510：214 点完整 FPGA 池（已完成）

输入：

- 候选：`20260914_p7r494_resnet18_merged_literature_board_pool_run01`
- 二进制：`20260914_p7r496_resnet18_merged_cross_compile_run01`
- 最终协议：`20260914_p7r508_resnet18_ssh_auth_amendment_run01`

已执行 `run_vta_c3_resnet18_board_pool.py`，显式使用 `--session-timeout 7200`。输出目录为不可覆盖的
`20260914_p7r510_resnet18_merged_board_pool_run01`。完整会话依次完成：干净板端准备、
214 点三 seed correctness（首错即停）、所有正确点五轮平衡计时。最少 642 次 correctness 调用；
若 214 点全正确，则另有 1070 次 timing 调用。

验收条件：

- `summary.json.status` 为 `complete_non_spliced_board_pool`、`session_spliced=false`，且不存在
  `invalid_session.json`；
- correctness 有且仅有 214 个最终分类；
- 每个 FPGA-correct 候选恰有五轮计时；
- 中断、断电、RPC 重启或身份哈希不一致使整次 P7R510 作废，不得与下一次启动拼接。

验收已通过：207/214 FPGA-correct、7 个 seed0 fail-fast wrong answer、1035 个计时样本；W0→T1
282.612 s、T0→T1 274.737 s，session 未拼接且无板端异常。当前按协议暂停。

## 4. P7R511--P7R513：只读适配、六策略与 Cheng 分析（已完成）

P7R511 用 `prepare_vta_c3_resnet18_literature_replay.py` 生成单向 outcome ledger。必须传入以下七个
`--qualification-dir`，不能少传或用板端结果替代：

```text
P7R471, P7R473, P7R475, P7R477, P7R479, P7R481, P7R492
```

并绑定 P7R494、P7R496 和完整 P7R510。适配器必须拒绝不完整/拼接会话。

P7R512 用 `analyze_vta_c3_resnet18_literature_pool.py` 比较：

```text
random
stock_mode_aware_xgb
rieber_hw_init_xgb
ml2tuner_pva
cheng_minimum_access
ours_dma_multifidelity
```

固定 20 seeds、预算 10/20/50/75/96。输入还包括 P7R470 完整 original contract、P7R484--486 三个
完整扫描、P7R491 HW E0 union、P7R492 E0 本地资格，以及 P7R508 最终修订。报告 oracle+2%/+5%、
trial/time-to-target、median/IQR、gross/compiler/FPGA/kernel/DMA 和五阶段真实墙钟。HW-Aware 的
presampling lowering 必须在第一次性能 dispatch 前计费；ML²Tuner 中未提升的编译成功点保持
censored，不能作为有效正标签。

P7R513 用 `analyze_vta_c3_cheng_four_schemes.py` 对完整池生成 same-tile 四方案结果：original、
input-prioritized、weight-resident-barrier、组合方案。`not_applicable` 必须按 original fallback
单列，最小 DMA 与真实最快不一致也必须保留。

该节点已完成并按协议暂停。主结果为本文仅在 H2 明确领先，H1 失败、H3 墙钟略差；ML² P/A 在
H1/H3 更强。Cheng 的19个完整family中4个最小访存不是最快。所有负结果均保留且规则未修改。

## 5. P7R514--P7R516G：策略所选完整图（安全回退后完成，已暂停）

P7R514 用 `freeze_vta_c3_resnet18_fullgraph_selections.py` 在 P7R512 结果上固定
`seed=57001、budget=50`，绑定 P7R511 outcome ledger 和 P7R498 Relay Testing ResNet18 source
contract。程序包括 stock/官方参考、pool stock-XGB、Cheng、ML²Tuner 和本文方法。

P7R515 用 `build_vta_c3_resnet18_fullgraph_programs.py` 按唯一三 route signature 各构建一次；重复
选择不得重复构建。构建阶段核对 H1/H2/H3 的真实 Graph JSON 节点数为4/3/1；编译期重复调度查询
命中为16/16/7，且 `all_routes_scheduled`。二者不能混称为运行期算子出现次数。

P7R516 用 `run_vta_c3_resnet18_fullgraph_programs.py` 在同一不可拼接 clean session 中，对三个固定
输入检查所有输出与 stock 完全相等，再做七轮平衡交错计时。报告完整图 latency、FPS、相对 stock、
route 命中和运行期逻辑 DMA。随机输入等价不得表述为 ImageNet 准确率。

P7R514/P7R515 已完成。P7R516 在首个 seed 的四个完整图调用后发现输出不一致，整次 session
fail closed，未进入七轮计时。P7R516D 保持冻结二进制不变完成3 seeds×正反顺序诊断：Cheng 的
H3 original 程序6/6与stock相等；共享H3 combined D0098的stock-XGB和ML²/本文程序均0/6正确。
详细证据见 `20260914_P7R514_P7R516D_RESNET18_FULLGRAPH_GATE.md`。

P7R516E 已在没有任何 selected-fullgraph timing 标签的条件下把 H3 combined D0098 标为图上下文
invalid，并统一回退到既有的 H3 D0203 original；冻结池不存在 D0098 same-tile original，不能事后
补造。P7R516F 将四策略去重为两份程序，复用 Cheng 程序并只新增一次构建，耗时11.722 s。

P7R516G 在同 boot、不可拼接 clean-start session 中完成9/9正确性和21/21七轮平衡计时。stock、
stock-XGB fallback、Cheng/ML²/本文共同 fallback 的中位延迟分别为116.464/117.093/117.722 ms；
后两者相对stock慢0.540%/1.080%。四策略均进入2%非劣带，但均未加速。详细证据见
`20260914_P7R516E_P7R516G_RESNET18_FALLBACK_FULLGRAPH.md`。旧 P7R516 仍永久保留为失败成本，
没有与 P7R516G 拼接。

本节点现在暂停。下一步才允许按原冻结顺序进入 P7R517--P7R523；P7R516G 性能标签不能反馈进
H1 两类方法的候选生成、模型或顺序。

## 6. P7R517--P7R523：H1 三对三 clean-start（已完成，节点暂停）

整图错误 route 的无标签回退协议和新一轮整图正确性门已经通过。下一次继续时严格按 P7R499 顺序
运行，不能并行，也不能交换顺序：

```text
P7R517  XGB seed 67001
P7R518  ours run 1
P7R519  ours run 2
P7R520  XGB seed 67002
P7R521  XGB seed 67003
P7R522  ours run 3
```

官方路径使用 `run_vta_from_scratch_autotvm_fullgraph.py`，P7R498 为 target contract，空历史、
`trials=300`、`budget-seconds=600`。本文路径使用
`run_vta_c3_resnet18_from_scratch_ours_fullgraph.py`，P7R498 为 source contract，
`candidate-budget=13`，不复用 P7R510/P7R512 性能标签。两边都必须在 W0 后停止健康的默认 RPC、
重新加载冻结 bitstream、启动同目录新 RPC，T0 从这里开始；失败编译、错误答案、自动恢复和最终
完整图验证全部计入 W0→T1/T0→T1。

任一次断电/RPC 中断使该次 run 作废并使用新目录重做，不能截取幸存片段。P7R523 用
`analyze_vta_c3_resnet18_clean_start_ab.py` 只接受三份完整 XGB 与三份完整 ours session，报告
time-to-quality、最终整图 latency/FPS、搜索成本中位数、范围和 IQR。

实际已按上述串行顺序完成。三次 XGB 与三次本文有效会话的 T0→T1 中位数为668.459/309.699 s，
本文减少53.67%；最终整图中位 latency 为116.912/117.902 ms，本文慢0.847%，但六次均通过三输入
全输出正确性并进入各自stock+2%带。P7R518 run01--run03 因RPC生命周期问题fail closed，合计
T0 548.652 s，保持不可拼接并单列；修正后的run04才是第一个有效样本。完整说明见
`20260914_P7R517_P7R523_RESNET18_H1_CLEAN_START_AB.md`。

## 7. P7R524：最终图表与文档（已完成）

使用 `plot_vta_c3_literature_results.py` 生成且只从不可变 summary/ledger 读取：

1. best-so-far vs FPGA dispatch；
2. best-so-far vs T0 真实墙钟；
3. invalid ratio 与 20-seed 稳定性；
4. DMA 减少—pure-instruction—算子延迟—完整图延迟传递图。

最终更新主创新文档和 claim ledger。逻辑 VTA LOAD/STORE 只称逻辑 DMA，方法称“方法一致复现和
扩展”，不称作者源码级复现。HW-Aware、ML²Tuner、Cheng 若不满足预注册正结论门槛，均保留为负
结果；本文只有在多数新几何以更少真实成本进入 oracle 2%/5% 带且整图不劣于官方 XGB 超过 2%
时，才能写成核心成功。

实际已完成。预注册 `plot_vta_c3_literature_results.py` 先生成 P7R524 原始不可变结果；视觉审计后，
P7R524C 在不改变数据和策略的前提下采用 symlog 改善极端反例可读性，并把第2图明确替换为
P7R523 三对三 clean-start 的真实 T0 轨迹。最终目录为
`20260914_p7r524c_resnet18_literature_publication_figures_run01`，13个产物全部通过账本复核，账本
SHA-256 为 `5750728a7d4c62bc76579a9d89934394d287059cb9f5731efdb91971cd161e59`。完整说明见
`20260914_P7R524_FINAL_LITERATURE_FIGURES.md`。本执行清单至此闭合；后续实验只属于增强外部
有效性，不再属于本计划的未完成节点。
