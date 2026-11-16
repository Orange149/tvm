# C3 mode11：带入口排空的驻留完成合同

## 结论

mode10 `input_weight_resident_composable` 不是任意前驱下的组合安全实现。它在独立执行及已测试的精确
完整图中可以正确，但普通 dependency-token 程序执行后再进入 mode10 force-serial 段会出现错误。
mode11 在 debug=0 时增加一次入口 `VTASynchronize`，当前在3个真实 YOLO-576执行任务、9个 Tile
上通过了冻结的跨程序正确性矩阵。

## 根因证据边界

已观察到：

1. mode10 独立重复执行正确；仅改变 DSO 预加载集合或顺序仍正确。
2. input-stationary 或 original 前驱后执行 mode10 会错误；barrier 前驱后可以正确。
3. 只重建 host CommandQueue 无法恢复；在 successor 前执行空同步可恢复3/3转换。

因此当前机制解释为“进入 force-serial 前必须排空硬件侧未闭合状态”。这不是逐周期根因，不把
runtime profile 中的等待时间解释成硬件 stall，也不声称某个具体 token 是唯一原因。

## 实现合同

mode11 的有序边界为：入口同步、debug=32、驻留程序、阻塞完成同步、debug=0、正常函数尾同步。
mode11 与 mode10 的主要计算、Tile、权重驻留和逻辑数据请求保持同类语义；新增入口排空属于真实
命令成本，selector 特征中记录为 `entry_drain_count`。mode10 被保留用于复现实验，但完成策略注册表
将其标记为非生产；mode11 是当前唯一可继续申请驻留组合证书的 force-serial 路径。

## 当前证据

| 层级 | 结果 |
|---|---:|
| 聚焦回归 | 46/46 passed |
| 真实执行任务 | 3 |
| Tile 家族 | 9 |
| 五模式候选 | 45 |
| static/FSim | 45/45 passed |
| mode11 本地证书 | 9/9 |
| AArch64 交叉编译 | 45/45 passed |
| 板端跨程序矩阵 | 33/33 case |
| 算子调用 | 129 |
| 单会话完整池 | 45/45 candidates，135 calls，1 wave |
| mismatch | 0 |
| 开发计时池 | 27/27 programs，711 total calls，0 mismatch |
| 同 Tile Input→mode11 | 9/9获胜，每项6/6轮 |
| 配对改善 | 中位52.951%，范围5.588%--73.790% |
| YOLO-544 strict-unseen boot-1同Tile对照 | 6/6获胜，每项6/6轮 |
| YOLO-544配对改善 | 中位36.981%，范围13.427%--69.433% |
| boot | `c9d73f4c-c0df-474a-9a74-a28a2f8955e2` |

每个任务的11个 case 覆盖3个 Tile 的独立三 seed、同 Tile Input/Original 前驱、跨 Tile 前驱，以及
三个普通程序组成的混合前缀。每个 case 使用 fresh RPC 与 bitstream reload，结果不跨 case 拼接。

## 开发计时结论与边界

三个完整池的oracle均为mode11。九个同Tile对照中，mode11把权重逻辑请求从393,216--21,233,664 B
降至65,536--262,144 B；入口排空和闭合段令提交数由1增至3，部分UOP指标也更高，但在这些真实
YOLO-576 `1×1`任务中传输节省仍占优势。这是程序级请求与板端调用时间证据，不是DRAM stall计数。

当前selector结果为负：冻结短名单在B3只命中2/3 oracle，而bytes-only在B2命中3/3。原因不是
mode11机制失败，而是低传输mode11之间仍有显著Tile差异，旧短名单未保留足够Tile多样性。该结果
禁止被改写成selector v3成功；后续规则只能作为新版本冻结并在新任务验证。

后续YOLO-544在时延前冻结了Tile多样性selector v4。两个完整15程序池中，六个同Tile
Input→mode11对照全部获胜；但孤立算子oracle是非生产mode5。现有生产完成策略门被补到v4排序之前
以后，每任务九个生产合格程序中的oracle均为mode11。由于该生产子池重放发生在boot-1时延暴露后，
它只证明mode11没有因mode5更快而失去部署候选价值，不能称新的strict-unseen selector成功。详细边界
见`20260924_C3_SELECTOR_V4_STRICT_UNSEEN_BOOT1.md`。

开发计时产物为：

- `07_grouped_holdout/20260924_c3_yolo576_transition_safe_timing_freeze_run01/`；
- `07_grouped_holdout/20260924_c3_yolo576_transition_safe_timing_boot1_run01/`；
- `07_grouped_holdout/20260924_c3_yolo576_transition_safe_timing_analysis_run02/`。

## 下一门禁

1. 所有新holdout先应用生产完成策略硬门，再冻结低传输窗口内保留Tile多样性的排序；参考池与部署池分开。
2. 用新的latency-unseen执行任务评价同预算bytes/calls/random基线；YOLO-544已暴露，不能重复当holdout。
3. 至少完成boot ID不同的跨boot确认和精确完整图正确性后，才恢复图级候选集合的部署资格。

当前不能声称 mode11 已跨 boot、跨硬件、跨网络，或selector已经优于简单基线。
