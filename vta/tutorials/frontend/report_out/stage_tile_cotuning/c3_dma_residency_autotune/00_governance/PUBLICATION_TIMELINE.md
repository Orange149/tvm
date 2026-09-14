# 最接近工作的公开时间线与使用规则

对象：Cheng 等，*Design of Data Access and Schedule Optimization for VTA Compiled Instruction Streams*。

## 时间线

- 2025：存在公开预印本/元数据记录；DOI 字符串为 `10.1016/j.future.2025.108165`。
- 2026：论文正式收入 *Future Generation Computer Systems* 对应卷期。
- 2026-09-10：本实验治理冻结时，将其定义为 B2026 最近直接基线。
- 2026-09-12：取得用户提供的 9 页正式全文并完成 Algorithm/Figure/Table、实现层级和搜索边界核对。

公开入口：

- DOI：<https://doi.org/10.1016/j.future.2025.108165>
- 出版页：<https://www.sciencedirect.com/science/article/abs/pii/S0167739X25004595>

## 学术使用规则

1. B2026 的输入优先、片上权重复用和 minimum-access 调优均不作为本文原创贡献。
2. 课题开始时间和早期带日期代码只能说明独立形成过程，不能替代引用和同平台比较。
3. P1 已可给复现机制标注正文、图、算法和表；作者源码、TVM commit、重复次数和方差仍明确为未知。
4. 未实现论文式地址不覆盖、条件 LOAD WGT 与 PushGEMMOp/UOP 语义前，本地 barrier 只能称功能级替代；实现后也只能称 method-exact，不称 code-exact。
5. 全文确认论文未研究 AutoTVM ConfigEntity 的候选生成、等预算搜索、time-to-target 或搜索流量；本文核心增量应限定为驻留×tile 搜索效率。资源证书、回退和部署合同只作可信执行支撑，不能代替算法创新。
