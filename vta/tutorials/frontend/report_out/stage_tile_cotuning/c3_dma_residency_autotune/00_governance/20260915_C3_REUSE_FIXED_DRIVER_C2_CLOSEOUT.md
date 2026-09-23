# C2第二启动四方案复测收尾

日期：2026-09-15。状态：`C1_C2_COMPLETE; PAUSED_BEFORE_NODE_D`。

## 已完成

实际重启已确认，C1/C2 boot分别为 `17cec1fe-4bb5-4f9e-90fb-eaf0e5f1b2b8` /
`61017e99-8c27-4857-bf50-faadefd96b7c`。冻结候选manifest、dispatch完成协议指纹、
runtime三文件哈希和绑定源码一致；未改tile、模式、顺序、输入seed或计时辅助库。

| 项目 | C1 | C2 |
|---|---:|---:|
| 三seed正确身份 | 48/48 | 48/48 |
| 正确计时调用 | 416/416 | 416/416 |
| 所有完整算子调用 | 659 | 659 |
| 完整四模式family | 11 | 11 |
| 实际双复用family | 11 | 11 |
| 组合中位延迟最快family | 10 | 10 |
| 输入单项中位延迟最快family | 1 | 1 |
| 静态/FSim/FPGA逻辑DMA一致profile | 48 | 48 |
| 连续W0→T1 | 61.635 s | 51.180 s |
| 连续T0→T1 | 36.001 s | 35.623 s |

两个启动都是M1D0034输入单项最快，差距小；其余10组组合最快。此为重复得到的中位数排序，
“稳定加速”须节点D按每启动至少2%改善、6/8轮获胜门槛逐项核验，不能由计时全部正确直接推出。

同一实测tile集合的best-extended相对best-original改善：

| 几何 | C1 | C2 |
|---|---:|---:|
| M1 | 21.63% | 20.20% |
| M2 | 28.32% | 28.25% |
| M3 | 49.71% | 47.79% |

不是完整ConfigSpace、TopHub、在线搜索或整网FPS结果。新完成协议解决正确性边界，
没有旧/新driver合格计时对照，不声称握手修复本身加速百分比。

## 重启准备与失败记录

首次 `20260915_c3_reuse_fixed_driver_c2_board_run01` 在preflight缺失u-dma-buf和bitstream路径时
退出，无候选调用或性能标签。原目录已封存，不改写。冻结runner在preflight后才写contract，因此
此早失败目录缺contract/完整T1时间，完整失败墙钟为unknown而不是0；原artifact账本已逐文件核验，
新closeout合同记录此缺口，不能冒充完整有效session。

随后核验两启动Linux完整版本/构建时间、启动参数及RAM rootfs一致，使用新脚本
`restore_vta_c3_reuse_boot_prerequisites.py` 将同哈希bitstream和匹配内核的host模块部署至tmpfs，
在已验证为RAM的`/lib/firmware`新增指向tmpfs的链接，以`/sbin/insmod`恢复192 MiB u-dma-buf。
模块版本5.4.2、物理起址0x68500000与C1一致；本次host模块二进制哈希记录，C1未记录其二进制哈希，
不宣称两启动ko逐字节身份已认证。准备耗时4.879 s单列，不拼入后续连续会话W0/T0。
没有SD文件/镜像写操作；FAT仍有未正常卸载历史警告，未发现新的EXT4/MMC错误。

成功复测为新目录 `20260915_c3_reuse_fixed_driver_c2_board_run02`。结束时boot保持不变，自有RPC已清理。
完成驱动相关10项测试通过；源码语法和`git diff --check`通过。

## 产物与下一步

以下目录相对 `../07_grouped_holdout/`：

- `20260915_c3_reuse_c2_boot_restore_run01/`：恢复命令、输出、耗时和哈希。
- `20260915_c3_reuse_fixed_driver_c2_board_run02/`：完整不可拼接板端session。
- `20260915_c3_reuse_fixed_driver_c2_analysis_run01/`：48身份CSV、11组分类和四模式DMA/延迟图。
- `20260915_c3_reuse_fixed_driver_c2_closeout_run01/`：核验C1/C2同实现上下文、成功/失败分账和当前状态。

既有单启动分析器通用标题/status写“第一启动/待C2”，是该分析器的历史固定标签；此目录实际输入
是C2，不能据该固定标签判断C2未执行。输入manifest和boot能定位真实身份。本报告明确纠正且不覆盖
已封存产物。分析器通用“M2失败身份保留”指旧失败历史，本次M2全正确。

按原计划在C2完成处暂停，不继续上板或自动扩域。下一节点D为离线分析：独立reload计时块的配对、
双启动稳定门槛、逐张量作用域/驻留跨度、同步与命令backing权衡、四张主图及成本表。
节点D尚未完成，当前不能标记整个补充计划完成；也无需为D再次重启。
