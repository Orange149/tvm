# C3 P7Q grouped-holdout 结果

## 结论

这次是真正的创新点三上板实验，不是旧基线复测。冻结池包含 W01、W04、W07、W08 共 80 个候选；79 个候选通过真实 FPGA 正确性，1 个 W04 input-stationary 候选稳定算错。随后只对 79 个正确候选做了五组随机完整区组计时，共获得 395 个候选延迟样本，另保留每组前后的 TopHub 哨兵。

结果同时给出一条正结论和一条否定结论：

1. 驻留调度本身有效。在相同 workload、相同七个 tile/thread 参数下，56 组可配对比较中有 42 组驻留版本更快，中位改善 3.31%，22 组改善不少于 5%。
2. 当前静态搜索排序没有击败强 TopHub。四个 workload 的测量池最优全部是原始 TopHub；只按 DMA bytes 或完整请求形态排序会优先试到计算结构较差的配置。B6 没有优于 B5，因此不能声称“完整请求特征已经提高搜索效率”。

最准确的故事是：**减少 DMA 确实能加速同一计算配置，但 DMA 收益不能补偿错误 tile/thread 选择造成的计算并行度损失；FPGA 调优必须把驻留、传输、命令和计算结构联合起来，并始终保护已知强配置。**

## 正确性与环境

- 原始冻结候选：80；真实 FPGA 正确候选：79；逐 seed 结果为 237/240 正确。
- W04 `input_stationary` ConfigSpace 139 在三个 seed 上均错误；重复诊断仍错误，而前后原始哨兵均正确，因此不是 SD、RPC 或参考实现偶发故障。
- 失败候选未被替换、未计时、没有进入 XGBoost 训练，但在所有搜索顺序中仍消耗一次 gross dispatch。
- 四个计时批次均处于同一 boot `a68a7983-719f-47bf-94d7-41f974c5342c`；FPGA 为 `operating`，u-dma-buf 为 192 MiB，SD 保持 78% 使用率。
- 计时后只读检查未发现 4057 秒水位之后的新 EXT4/MMC 错误；实验上传和运行继续位于 tmpfs。

## 真实 FPGA 测量池最优

| Workload | 合法候选 | 最优中位延迟 | 最优模式 | 结论 |
|---|---:|---:|---|---|
| W01 | 19 | 3.048571 ms | original / TopHub | 驻留候选未超过强基线 |
| W04 | 18 | 2.772418 ms | original / TopHub | 另有 1 个 FPGA wrong-answer |
| W07 | 21 | 2.653857 ms | original / TopHub | 驻留候选与基线差距较大 |
| W08 | 21 | 5.027001 ms | original / TopHub | 驻留候选与基线差距较大 |

这些是冻结小池的 `pool oracle`，不是完整 AutoTVM 空间的全局最优。

## 相同派发预算下的搜索结果

下表是四个 workload 的平均 regret；0% 表示已经找到该冻结池中的真实最优。B2 为 1000 个冻结随机排列的均值。

| 策略 | budget=4 | budget=8 | budget=16 | budget=24 |
|---|---:|---:|---:|---:|
| B0 TopHub | 0.00% | 0.00% | 0.00% | 0.00% |
| B1 paper-inspired 单点 | 44.09% | 44.09% | 44.09% | 44.09% |
| B2 Random | 39.43% | 25.98% | 9.38% | 0.00% |
| B3 ConfigEntity-only XGB | 33.80% | 33.68% | 0.00% | 0.00% |
| B4 冻结源码枚举 | 0.00% | 0.00% | 0.00% | 0.00% |
| B5 DMA bytes | 40.16% | 40.00% | 0.00% | 0.00% |
| B6 完整请求形态 | 40.00% | 40.00% | 1.33% | 0.00% |
| B7 B6 + 模式轮转 + incumbent 保护 | 0.00% | 0.00% | 0.00% | 0.00% |
| B8 B7 + 命令/计算组秩 | 0.00% | 0.00% | 0.00% | 0.00% |

解释边界：B4 也是 0%，因为冻结枚举顺序恰好把原始 incumbent 放在首位，不能把它解释成聪明的搜索算法。B7/B8 的确定性贡献是“绝不先丢掉已知最快配置”；本轮没有观察到 B8 相对 B7、或 B6 相对 B5 的额外收益。

Random 在 budget 4/8/16/24 下对单个 workload 命中 pool oracle 的平均概率分别为 19.48%、39.10%、79.35%、100%。B3 找到四个 oracle 的 gross dispatch 位置依次为 4、1、13、15；这说明小样本 XGBoost 早期仍可能被 tile 参数误导。

## 驻留机制的同 tile 次分析

这项分析使用同一批前瞻计时标签，但属于 P7Q 主搜索终点之后的次分析，不冒充预注册主终点。

| 模式 | 配对数 | 改善中位数 | ≥5% 更快 | 范围 |
|---|---:|---:|---:|---:|
| input-stationary | 14 | 5.56% | 8 | -0.47% 到 18.61% |
| paper-inspired hybrid | 16 | 6.01% | 10 | -0.39% 到 18.69% |
| weight-stationary（安全旧模式） | 18 | 0.03% | 0 | -0.43% 到 0.49% |
| weight-stationary barrier | 8 | 4.58% | 4 | -9.67% 到 17.35% |

这与 P6e 的因果配对结果一致：input/hybrid/barrier 的 DMA 驻留机制可以产生真实加速；没有真正减少权重 DMA 的旧 `weight_stationary` 基本等价。与此同时，同 tile 的最好候选仍未达到 TopHub，说明决定最终性能的是“计算 tile/thread 基础 + 驻留增益”的组合，而不是驻留增益单项。

## 为什么当前 P7 不能作为最终独立创新点证据

实验生成器在早期机制隔离阶段把所有驻留候选强制为 `oc_nthread=h_nthread=1`，而本轮四个 TopHub oracle 都使用 `oc_nthread=2`。因此候选池中不存在“TopHub 最强 tile/thread + 新驻留模式”的组合。当前 P7 可以评价已有池的安全排序，却不能回答真正的联合搜索能否超过 TopHub。

按照冻结 Gate G7：

- 最终性能保护：通过，B7/B8 在 4/4 workload 保留 pool oracle；
- 相对 Random/XGB 的有限预算保护：表面通过，但收益完全来自 incumbent 首位；
- full request signature 相对 bytes-only 的额外贡献：未通过；
- 至少两种几何出现不同的最终驻留赢家：未通过，四个最终赢家均为 original；
- Gate G7 总判定：**NO-GO**。

因此不能继续把同一批 holdout 反复调规则后宣称成功，也不能从本结果进入 P8 并声称 stage/FPS 提升。

## 下一步应怎样改，而不是重复实验

如果继续第三创新点，应建立一个全新的、有日期隔离的开发/确认协议：

1. 让 input/hybrid/barrier schedule 支持 `oc_nthread=2`，先用 FSim 和 FPGA correctness 验证依赖与地址安全；
2. 以 TopHub 和原始 Top-k 的完整 tile/thread 配置为中心，生成同配置驻留变体，而不是先把 virtual thread 强制删掉；
3. 把选择器改成两阶段：先用硬件容量/依赖证书淘汰非法项，再预测“相对同 tile original 的驻留增益”，最后与 protected incumbent 比较；
4. 当前 W01/W04/W07/W08 已经暴露标签，只能作为 development，不得再次充当确认集；确认必须使用未见标签的 workload/几何或独立新网络；
5. 只有新确认集上 full signature 明确优于 bytes-only，且最终配置达到强基线 ±2% 或更快，才能恢复 B 级独立创新点主张。

## 产物

- `analysis.json`：B0--B8 与 1000 个 Random 排列的预算统计；
- `b3_replay.json`：冻结 ConfigEntity-only XGBoost 的逐次派发；
- `same_tile_effects.json`：56 个相同 ConfigEntity 配对的完整结果；
- P7Q1--P7Q4 的 `timing.jsonl` 与 `summary.json`：真实 FPGA 原始计时。
