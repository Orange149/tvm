# P7R524：ResNet18 文献对齐实验终稿图与计划验收

状态：`COMPLETE; BOARD_NOT_CONTACTED; DATA_AND_POLICY_UNCHANGED`

## 1. 最终采用的图

终稿目录为：

```text
07_grouped_holdout/20260914_p7r524c_resnet18_literature_publication_figures_run01
```

其中四张主图为：

1. `figure1_best_vs_trial.pdf`：三个 ResNet18 几何、六种策略的 best-so-far 与 FPGA dispatch；
2. `figure2_best_vs_actual_t0_wall.pdf`：H1 上三次空历史 AutoTVM-XGB 与三次本文流程的真实 T0
   clean-start 轨迹；
3. `figure3_invalid_ratio_and_stability.pdf`：HW-Aware 四级初始化的合法率/IQR，以及六策略在
   budget=50、20 seeds 下的稳定性；
4. `figure4_dma_instruction_operator_fullgraph_transfer.pdf`：Cheng 四方案的逻辑 DMA、pure-
   instruction、算子和安全回退整图结果传递。

PNG 是预览副本，PDF 是论文插图版本。终稿 `artifact_hashes.json` 的 SHA-256 为
`5750728a7d4c62bc76579a9d89934394d287059cb9f5731efdb91971cd161e59`，其中 13 个产物已逐项
重算并全部一致。绘图器哈希为
`7d26e2bdc5a0189a95d1a798bc754d92b3b1d768e8fcb1eb23d7584eb76eb9af`。

## 2. 为什么有 P7R524、A、B、C 四个目录

P7R524 是按 P7R499 预注册哈希运行的原始冻结绘图器输出，完整保留且账本哈希为
`44dae66f06119ca8f45e8fffe92ae8a40d31c961377a1d7da440cc1f06291ec8`。视觉审计发现两个问题：

- Cheng 的大幅负向反例把线性纵轴拉得过宽，其他策略不可读；
- 第 2 张图使用完整池阶段成本构造的反事实 lazy wall，不是用户要求的 clean-start 真实 T0 墙钟。

P7R524A/B 只尝试调整坐标和版面，均作为过程证据保留。P7R524C 最终以对称 symlog 显示极端
regret，并将第 2 张图替换为 P7R523 六个独立有效 session 的实测 T0 轨迹。修订没有改变候选、
策略、预算、随机种子或任何实验标签；`summary.json` 明确记录
`data_or_policy_changed=false`。

## 3. 图中能够支持的结论

- H1 三对三完整系统流程中，本文 T0→T1 中位数为 309.699 s，空历史 AutoTVM-XGB 为
  668.459 s，降低 53.67%；最终 ResNet18 中位 latency 为 117.902/116.912 ms，本文慢 0.847%，
  但六次均通过三输入全部输出正确性并进入各自 stock+2% 带。
- 三几何 214 点完整算子池的结果是混合的：本文只在 H2 明显缩短 time-to-target，在 H1 的算子
  oracle 质量较弱，在 H3 的真实墙钟略差。因此不能写成跨几何稳定优于 XGB 或 ML²Tuner。
- HW-Aware 提高初始化合法率，但固定 presampling 成本在 H2/H3 抵消收益；ML²Tuner 的 P/A 在
  H1/H3 有效，Model V 在本高合法率池无过滤收益，A+DMA 也不稳定。
- Cheng 的 19 个完整 same-tile family 中，15 个 minimum-access 与最快一致，4 个不一致；因此
  共享内存访问量是搜索先验，不是 latency 定理。
- 安全回退后的 ResNet18 完整图达到 2% 非劣但没有加速。逻辑 VTA LOAD/STORE 不等于物理 AXI
  流量，随机输入全输出等价不等于 ImageNet 准确率。

## 4. 原计划逐项验收

| 计划节点 | 最终证据 | 验收 |
|---|---|---|
| A：Y10 六次同空间归因 | P7R462--P7R468 | 完成；标签已暴露，只作算法归因 |
| B：六策略、HW-Aware、ML²Tuner、Cheng 四模式及冻结 | P7R469--P7R507 | 完成；方法一致复现，不称源码级复现 |
| C：三个新 ResNet18 几何完整池 | P7R470--P7R510 | 完成；214 点、207 正确、1035 个计时样本、会话未拼接 |
| D：20-seed 六策略、Cheng、整图和三对三 clean-start | P7R511--P7R523 | 完成；正负结果与失败成本全部保留 |
| 四张论文主图 | P7R524/P7R524C | 完成；最终使用 P7R524C |
| 统一 runner 与五类不可覆盖产物 | `run_vta_c3_literature_baselines.py` | 完成；六个必需 CLI 参数齐全 |
| 泄漏、隔离、失败成本、断电与 oracle 测试 | 文献对齐测试套件 | 完成；最终回归另见本节点收口记录 |

最终收口回归覆盖10个相关测试文件，共 `65 passed`。测试包括未来标签泄漏拒绝、离散邻域与E0
平衡、P/V/A观测隔离、invalid不进入性能回归、四模式及资源回退、失败仍计入gross/墙钟、RPC
中断会话不可拼接、完整池结束前不可连接oracle、输入产物不可覆盖，以及整图协议身份检查。

因此用户给出的文献对齐补充实验计划已经执行完毕。最终研究结论仍遵守预注册门槛：这是较强的
硕士论文系统创新和完整 case-study 证据链，但三几何结果没有通过“多数 workload 的真实墙钟均
优于官方 XGB”这一更强门槛，不能包装成通用 AutoTVM 算法的全面胜利。
