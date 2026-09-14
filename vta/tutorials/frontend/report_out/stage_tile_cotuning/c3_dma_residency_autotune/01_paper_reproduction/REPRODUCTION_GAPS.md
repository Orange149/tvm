# Cheng 2026 全文取得后的复现缺口

## 1. 状态变化

用户已提供正式论文全文，9 页，SHA-256：
`9fdbe490db46d319fb992ff016f99c0f7bd64718f079f84749ff21d3c8839a02`。

此前由全文不可访问造成的 loop order、四方案、资源条件和实现层级缺口已解除。当前剩余缺口来自论文
本身未报告或未公开的内容，而不是访问失败。

## 2. 已关闭的缺口

| 原缺口 | 全文答案 |
|---|---|
| input-prioritized loop order | output height → output width → output channel；关闭 `oc_nthread`，启用 `h_nthread` |
| weight reuse 防覆盖机制 | weight SRAM 不同地址顺序存放；StorageFlatten、InjectVirtualThread、SchedulePostprocToPrimFunc、InjectCopyIntrin 和 runtime PushGEMMOp 联合修改 |
| minimum-access 的四方案 | Scheme 1 原始；2 input；3 weight reuse；4 两者组合 |
| 资源适用与回退 | 按 ACC/weight SRAM 和卷积参数/stride 判断；不足时自动回退 naive TVM |
| 硬件与网络 | ZCU104、32×32、200 MHz、`15_16_19_18`；YOLOv3/YOLOv5m/ResNet50 |
| 逐层结果 | Table 2 给出 9 个 YOLOv3 卷积四方案 latency；Scheme 4 并非每层最优 |

## 3. 仍然存在的缺口

| ID | 缺口 | 对复现的影响 | 当前处置 |
|---|---|---|---|
| G-R1 | 作者源码和 patch 未公开 | 不能确认所有条件表达式、地址公式和边界处理 | 只能按 Algorithm/Figure 做功能级独立重实现，保存本地 TIR diff |
| G-R2 | TVM/VTA commit 未报告 | API/Pass 结构和原始模板可能不同 | 在当前冻结 commit 上映射，不声称源码逐行一致 |
| G-R3 | “tuning strategy”没有搜索算法、预算、停止条件或 cost model | 不能构造 exact AutoTVM tuner baseline | 把论文基线限定为四方案/same-tile 机制选择，不虚构搜索流程 |
| G-R4 | 没有 Random/XGB、trial、regret、time-to-target | 不能与本文搜索效率数字直接横比 | 本文另做相同池、相同预算比较；只比较公开机制，不冒充论文搜索结果 |
| G-R5 | 正确性输入、重复次数、方差和错误处理未报告 | 不能复制其统计协议 | 本地使用冻结多 seed correctness 和平衡计时协议 |
| G-R6 | 本地硬件为 AXU5EVB 16×16/100 MHz，与论文 ZCU104 32×32/200 MHz 不同 | 不能比较绝对 latency 或直接复用资源阈值 | 只做本平台内部配对和趋势复现 |

## 4. 复现等级

- `paper_mechanism_functional_reimplementation`：可用；必须实现正文描述的 loop/address/UOP语义并核对。
- `paper_code_exact_reproduction`：不可用；缺作者源码、commit 和完整条件代码。
- `paper_autotuner_exact_reproduction`：不存在可复现规格；全文没有给出 AutoTVM 搜索算法。

当前本地 `input_stationary` 可与 Algorithm 2 做结构映射；`weight_resident_barrier` 使用显式 barrier/drain
实现驻留，与论文修改 StorageFlatten/InjectVirtualThread/PushGEMMOp 的机制不同，只能作为同目标替代
实现。若要声称完整机制复现，应新增论文式不覆盖地址和条件 LOAD WGT 实现，并与 barrier 方案消融。
