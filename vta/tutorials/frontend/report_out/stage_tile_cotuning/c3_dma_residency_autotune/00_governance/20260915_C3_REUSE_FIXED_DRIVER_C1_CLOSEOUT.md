# 实验版完成握手与四方案 C1 收尾

日期：2026-09-15。状态：`ISOLATED_DRIVER_MICRO_QUALIFIED; C1_COMPLETE_PAUSED; C2_NOT_STARTED`。

## 1. 本次改了什么

用户明确同意修改 driver 后，新增独立构建入口 `build_vta_c3_completion_handshake.py`，从原
`axu5evb_driver.cc` 机械生成实验版本；原文件、原 `build_axu_aarch64/libvta.so`、节点B冻结源码、
48个候选二进制均保持不变。实验库只部署到开发板新 tmpfs 目录，不修改 SD 卡上的驱动或内核模块。

旧完成判断只检查 done 数据是否为1，可能接受上一提交留下的1。本次实验在提交前消费旧的
output-valid 事件，提交后要求同时读到新的 valid=1 和 done=1，才允许 CPU 消费共享输出。
不添加固定完成延时，不增加 MMIO 写操作；每提交1秒诊断超时会拒绝继续执行。

重要限制：`DONE_RD+4`（0x1c）作为 output-valid 的假设已明确记录，并通过当前板端行为验证，
但未取得该 bitstream 的生成寄存器头文件，**不是生成寄存器映射认证，也不是通用生产驱动保证**。
通用 HLS 输出有效协议可参考 [AMD UG1399](https://docs.amd.com/r/2022.2-English/ug1399-vitis-hls/S_AXILITE-and-Port-Level-Protocols)，不能用通用文档替代本 bitstream 的映射。

新 `libvta.so` SHA256：`e6acc970751e55b1350cdd1ed7e0625540a9c773745080a2674886961db47e6e`。
所有四模式统一使用此库，正式计时关闭逐提交诊断日志，完成保护仍生效。

## 2. 完成边界验证

以下目录均位于 `../07_grouped_holdout/`，原失败会话保留，不拼接其计时：

| 实验 | 结果 | 可以说明什么 |
|---|---|---|
| `20260915_c3_completion_trace_board_run01` | 42次立即正确，90提交核对；长提交仍出现 first_done=1、未见0就返回 | 日志扰动可遮蔽错误，不能作为修复 |
| `20260915_c3_completion_fresh_valid_board_run01` | 42/42立即和延后读取均正确，输出不变化；90/90提交在新的 valid=1、done=1上返回 | 新握手通过当前微验证 |
| `20260915_c3_completion_valid_observer_board_run01` | 保留旧完成判断的对照被保护检查拒绝，整次会话无效 | 只增加valid读取不足以保证安全，不计作完整42次成功实验 |

旧判断的对照日志中，长提交出现 done=1、valid=0 就返回，随后尾段等待前段数据执行。
这比“输出延后变正确”更直接地定位到完成事件与旧 done 数据混淆；观察对照受到日志时序扰动，
不能据此计算错误概率或宣称修复加速百分比。新握手的微验证产物为
`20260915_c3_completion_fresh_valid_board_run01_trace_analysis/`。

## 3. 完整 C1 结果

正式准备：`20260915_c3_reuse_fixed_driver_prepare_run01/`。
正式会话：`20260915_c3_reuse_fixed_driver_c1_board_run01/`。
只读分析：`20260915_c3_reuse_fixed_driver_c1_analysis_run01/`。

- boot：`17cec1fe-4bb5-4f9e-90fb-eaf0e5f1b2b8`，从开始到结束未变化。
- 48/48身份通过三seed逐元素正确性；原B池、family顺序和八轮Latin计时均未改变。
- 44主身份×8轮=352次主计时；独立reload对照64次；共416次计时全部正确。
- 连同健康门、正确性、显式warmup和独立profile，共659次完整算子调用，全部有实际dispatch记录。
- 48个独立profile的逻辑DMA与冻结静态/FSim语义全部一致。
- 连续 W0→T1=61.635 s，T0→T1=36.001 s。节点B资格、C0诊断和历史失败开销另列，不能把此数当完整从头调优成本。
- 本次新增完成驱动相关测试10项通过。结束后自有RPC已清理，未发现新EXT4/MMC错误。

| family | O ms | I ms | W ms | IW ms | 最快 | IW相对O改善 |
|---|---:|---:|---:|---:|---|---:|
| M1D0017 | 4.970 | 2.980 | 4.773 | 2.730 | IW | 45.07% |
| M1D0032 | 3.257 | 2.265 | 2.868 | 2.007 | IW | 38.37% |
| M1D0034 | 2.265 | 1.775 | 2.431 | 1.789 | I | 20.99% |
| M1D0139 | 3.447 | 2.890 | 2.666 | 2.182 | IW | 36.71% |
| M2D0071 | 6.476 | 4.163 | 5.118 | 2.893 | IW | 55.33% |
| M2D0078 | 2.581 | 2.012 | 2.394 | 1.850 | IW | 28.32% |
| M2D0140 | 6.395 | 5.480 | 3.237 | 2.527 | IW | 60.49% |
| M3D0002 | 8.590 | 1.911 | 8.048 | 1.204 | IW | 85.99% |
| M3D0046 | 1.082 | 0.711 | 0.933 | 0.544 | IW | 49.71% |
| M3D0049 | 2.758 | 2.040 | 2.094 | 1.350 | IW | 51.06% |
| M3D0056 | 5.523 | 4.561 | 2.499 | 1.737 | IW | 68.55% |

O/I/W/IW分别为原始、输入优先、权重驻留barrier、组合。延迟为八轮中位数，不是整网时间。
11组均为四种不同程序且实际双复用；组合10组最快，但M1D0034的I为1.775 ms、IW为1.789 ms，
差距很小，仅记本启动中位数排序，不声称显著差异。

同一实测tile集合内，扩展四模式空间相对original子空间的最优点改善为M1 21.63%、
M2 28.32%、M3 49.71%。不代表完整ConfigSpace最优，不与TopHub比较，不代表新搜索策略收益。

## 4. 共享内存语义现在更扎实在哪里

以M3D0046为例，O的输入/权重重复读取比为4/7，IW变为1/1；input LOAD从46,592降至11,648 B，
weight LOAD从229,376降至32,768 B，output STORE保持12,544 B。与此同时，weight arena从
8,192增至32,768 B、ACC arena从1,792增至7,168 B，input arena保持1,664 B；submission从1增至2。
延迟从1.082降至0.544 ms。证据链是“加载作用域和生命周期改变→真实重复访问减少→资源/同步代价→
板端正确且较快”，不是只根据总bytes作解释。

这些arena跨度不是统一同时存活峰值；不能简单求和当精确片上峰值。driver累计执行时间不是纯计算时间，
逻辑LOAD/STORE不是物理AXI流量。统一地址—消费者—覆盖冲突图仍未完整实现。

原分析器生成的 `RESULTS.md` 有通用历史句“M2失败身份完整保留”，在本轮应解读为保留旧失败会话；
**本次新driver C1的M2身份全部通过**。该封存分析不覆盖修改，以本收尾明确区分。
独立reload控制的64次计时已收集，原分析表未汇总其延迟；匹配控制的配对效应留到节点D独立分析，
不能把表中unknown说成未执行，也不能拿另一计时块的主身份与其混配。

## 5. 暂停与下一节点

按计划在C1完成处暂停，不自动重启，不在同boot开展C2。用户实际重启后，保持同一候选池和
实验库，使用新输出目录执行：

```sh
/home/orange/miniconda3/envs/vta-resnet/bin/python vta/tutorials/frontend/run_vta_c3_reuse_fixed_driver.py \
  --phase board \
  --contract vta/tutorials/frontend/report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout/20260915_c3_reuse_fixed_driver_prepare_run01/contract.json \
  --previous-boot 17cec1fe-4bb5-4f9e-90fb-eaf0e5f1b2b8 \
  --output-dir vta/tutorials/frontend/report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout/20260915_c3_reuse_fixed_driver_c2_board_run01 \
  --execute-board
```

C2完成后才报告跨启动方向和量级稳定性；节点D再汇总reload因果对照、资源/时间边界和双启动结论。
本次未新增整网结果、在线tuner比较或相对Cheng的原创复用机制主张。
