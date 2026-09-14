# P7R511--P7R513：ResNet18 文献对齐搜索与 Cheng 四方案分析

日期：2026-09-14  
输入板池：P7R510，214 个候选、207 个 FPGA-correct、1035 个五轮计时样本

## 信息隔离与成本口径

P7R511 只在 P7R510 完整、不可拼接会话封存后生成单向 outcome ledger。七份本地资格目录、P7R494
候选池、P7R496 二进制和 P7R510 板端结果均通过 artifact hash。公共全池真实成本单独保留：本地
资格 1122.046 s、交叉编译 294.496 s、板端 W0→T1 282.612 s。策略回放报告逐候选按需成本，不能
与这些公共穷举成本混成同一次在线执行。

P7R512 对三个 workload、20 seeds 和预算10/20/50/75/96执行冻结策略：Random、stock mode-aware
XGB、Rieber/HW-Aware、ML²Tuner P/V/A、Cheng minimum-access 和本文 DMA multifidelity；另保留
ML² `A+DMA` 消融。分析进程自身墙钟为 468.242 s，不计入被回放搜索器的 target runtime。

## 达到 pool-oracle+2% 的结果

下表使用完整预算内首次命中的中位 trial 和累计按需实测墙钟。`—` 表示在可搜索集合内没有命中。

| 策略 | H1 trial / s | H2 trial / s | H3 trial / s |
|---|---:|---:|---:|
| Random | 14 / 30.529 | 7.5 / 18.703 | 25.5 / 28.871 |
| stock mode-aware XGB | 26.5 / 52.251 | 5 / 12.928 | 11.5 / 9.940 |
| HW-Aware init + XGB | 16 / 48.397 | 15 / 45.830 | 26 / 22.094 |
| ML²Tuner P/V/A | 12 / 32.754 | 6 / 18.165 | 7 / 9.368 |
| ML²Tuner A+DMA | 12 / 32.789 | 3 / 12.152 | 9 / 10.654 |
| Cheng minimum-access | 18 / 41.482 | 10 / 28.988 | — |
| 本文 DMA multifidelity | 37 / 76.055 | 1 / 2.129 | 11 / 10.943 |

本文结果具有明显几何依赖：H2 为强正结果，相对 stock-XGB 将首次命中从5次/12.928 s 降到
1次/2.129 s；H1 则明显更差；H3 trial 中位略少0.5次，但墙钟比 stock 多约1.003 s。budget=10 的
跨60条轨迹 oracle+2% 成功率为 Random 40.0%、stock-XGB 51.67%、HW-Aware 0%、ML²与A+DMA
66.67%、Cheng 33.33%、本文33.33%。budget=20 时依次为66.67%、70.0%、58.33%、100%、100%、
66.67%、66.67%；budget=50 除 Cheng 因 H3 不可达而为66.67%外，其余均达到100%。

这不满足“本文在多数新几何按真实墙钟优于官方 XGB”的预注册强成功门槛。应保留 H2 强正结果，
同时把 H1/H3 作为 DMA 先验不是普适 latency 排序器的反例。

## HW-Aware 对齐结论

此前完整 original 空间的20-seed实验已经证明邻域初始化提高 valid yield：H1/H2/H3 的 E0 中位
合法率由 Random 的12%/10%/26%提高到42%/41%/55%；找到前25个合法点的 compiler calls 从
212/285.5/97.5降至57.5/62/46.5。

但把固定1000/1000/480点 presampling 的真实墙钟预付到第一次性能派发前后，性能搜索只在 H1
相对 stock-XGB 略快，H2/H3 均显著更慢。结论是“合法邻域聚集成立，固定大规模预采样不保证端到端
收益”；实际系统应早停或按需采样，不能把全部 presampling 当作免费初始化。

## ML²Tuner 对齐结论

目标板池已经经过严格 lowering/FSim 资格，最终合法率为100%、97.73%、90.91%。在这一高合法率
条件下，跨几何 Model V 实际退化为全 valid 预测：H2/H3 precision 等于基率，H3 top-20 valid
yield 只有85%，还低于90.91%的池基率。因此本轮不能声称 Model V 减少无效 profiling。

P/A 性能排序仍有价值：ML² P/V/A 在 H1 和 H3 比 stock-XGB 更早命中。加入 DMA 后，A 模型 RMSE
在 H2 从56.369降到31.456 ms、H3从109.133降到29.288 ms，但 H1 从30.490恶化到34.694 ms；
实际 time-to-target 对应地只在 H2 从6次降到3次，H3反而从7次增到9次，H1不变。因而 DMA 是
有条件有用的编译后特征，不是对 ML²Tuner 的稳定改进。

## Cheng 四方案结论

P7R513 将 original、input-prioritized、weight-resident-barrier 和 combined 按相同 ConfigEntity
分组。资源不适用只按原论文原则回退 original；lowering/FSim/FPGA 错误不静默回退。H1/H2/H3
分别有45/29/16个 original family，但受 SRAM 和真实 lowering 限制，四方案语义完整可解析数为
0/18/1。

19个完整 family 中，minimum-access 与真实最快一致15次，不一致4次（21.05%）。四个反例均在
H2：其中两个 input-prioritized 总 DMA 更多却分别把算子 latency 从58.791降至30.377 ms、从
36.664降至28.652 ms；另两个总 DMA 相同而时间有极小差异。H3 唯一完整 family 更直接：weight
和 combined 的 pure-instruction 分别改善39.40%和38.01%，但算子 latency 反而恶化4.08%和6.33%，
说明 barrier/submission/host同步会抵消设备执行段收益。

不要求四模式齐全的 same-tile input 配对中，H1/H2/H3 分别有15/11/14对，驻留更快10/4/9对；
延迟改善中位数为+2.81%/-0.067%/+0.492%，范围很宽。该结果支持 Cheng 的访存机制，但也证明
minimum-access 必须保留真实硬件 latency gate。

## 当前结论和下一暂停点

三个文献基线都得到了可证伪结果，而不是只充当陪跑基线：HW-Aware 提高合法率但固定预采样成本
可能得不偿失；ML² P/A 在两个几何优于 stock，而 V 在高合法率池无效；Cheng 最小访存约五分之一
完整 family 不是最快。本文方法在 H2 强胜、H1 失败、H3近似持平，不能写成全面领先。

下一节点严格使用 `seed=57001、budget=50` 冻结各策略选中的 H1/H2/H3 route，构建唯一完整图并
检查三个确定性输入的全部输出，再做七轮平衡计时。整图若进入2%等价带，只能证明最终质量安全，
不能抹去 H1 的搜索成本负结果。
