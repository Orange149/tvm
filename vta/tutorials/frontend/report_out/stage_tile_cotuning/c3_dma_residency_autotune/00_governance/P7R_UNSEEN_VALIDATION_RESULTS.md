# P7R 未见几何验证与硬件合法性结果

状态：`REQUEST_SHAPE_HYPOTHESIS_REJECTED; HARDWARE_VALIDITY_REDESIGN_REQUIRED`  
日期：2026-09-11  
范围：第三创新点；单算子正确性、局部计时与搜索前证书，不包含 stage/FPS。

## 1. 本轮回答的问题

P7R 不再重复旧 P7Q 候选，而是回答两个新问题：

1. W04 上观察到的二维 DMA request-shape 规则能否前瞻迁移到未见卷积几何；
2. lowering、SRAM 容量和单路径 FSim 是否足以保证候选能在真实 FPGA 正确执行。

未见几何在读取其板端 latency 前冻结：

- E00：`CI=96, CO=160, H=W=42, k3s1p1`；
- E01：`CI=192, CO=320, H=W=21, k1s1p0`；
- E02：`CI=320, CO=384, H=W=11, k3s2p1`。

## 2. 搜索空间压缩

固定 `input_stationary × oc_nthread=2 × h_nthread=1` 后：

| 几何 | 枚举配置 | lowering/容量/DMA 证书后保留 | 保留率 |
|---|---:|---:|---:|
| E00 | 1024 | 45 | 4.39% |
| E01 | 576 | 108 | 18.75% |
| E02 | 768 | 56 | 7.29% |
| 合计 | 2368 | 209 | 8.83% |

因此仅解析证书已拒绝 2159/2368（91.17%）候选。这里的“保留”只表示值得进入下一道正确性门，
不是速度或真实 FPGA 正确性结论。

## 3. 前瞻 request-shape 结论

规则只用 W04 config455/461 开发结果冻结，并在 E00/E01/E02 各选择一个短行请求点和一个连续
对照。六个 residency 模块均通过三个 seed 的 FSim。

真实 FPGA 只有 E02 完成了全部 15 个正确性检查并进入计时。7 个随机完整区组结果为：

| E02 配置 | original 中位/ms | input-stationary 中位/ms | 成对中位变化 | 获胜区组 |
|---|---:|---:|---:|---:|
| request-shape config1163 | 5.260739 | 6.038886 | -14.80% | 0/7 |
| contiguous config1167 | 2.792920 | 3.159890 | -13.18% | 0/7 |

预注册的“request-shape 点优于连续对照”预测失败；两点均退化。因此总 bytes/calls 不充分是成立的
诊断结论，但当前 W04 两点拟合出的 request-shape score **不能**作为通用 autotune 排序器。

## 4. 正确性门暴露的缺口

E00/E01 在任何计时之前即被停止：

- E00 config1061 original：三个 seed 分别有 70396/69960/70419 个错误；
- E01 config951 original：59694/59696/59696 个错误；
- E01 config755 original：66400/66120/66280 个错误；
- E01 config957 original：7908/7894/7933 个错误；
- E01 config877 original：7901/7884/7876 个错误；
- E01 config301：关闭虚线程后仍有 92089/92292/91422 个错误，因此 E01 失败不能归因于
  `oc_nthread=2` 单一因素。

E00 config1081 没有进入设备执行：runtime 的 UOP 依赖检查在命令生成时确定性拒绝连续三条内的
重复 `dst_idx`。补做同 tile original FSim 后，11 个代表 original 中 10 个三 seed 全对，只有
config1081 同样被这一检查拒绝。这证明未来流程必须对 original 与 residency **两条路径都做**
命令生成/FSim，而不能只检查 residency。

其余真实 FPGA 错误点在 original FSim 中仍逐元素正确，说明存在尚未建模的 simulator-to-FPGA
合法性差距。不能根据当前少量失败点臆造一个“已经证明”的静态公式。

## 5. 第三创新点的修正版方法

后续候选流水固定为：

1. **解析证书**：复用次数、三类 SRAM 容量、DMA 2-D 可表示性、padding 和 request 形态；
2. **双路径命令证书**：original 与 residency 均实际生成 UOP/指令，拒绝 UOP 地址依赖冲突，
   并检查 instruction/UOP/FINISH/replay 容量；
3. **双路径 FSim**：相同三个 seed 均逐元素正确；
4. **少量真实 FPGA canary**：每个新几何先验执行正确性，不通过即整组安全回退 TopHub，禁止计时；
5. **受保护性能搜索**：TopHub 永远首派发，只有通过以上证书的候选才消耗性能测量预算。

这一路径的论文价值是“利用固定 FPGA 的 SRAM、指令格式、UOP 依赖和 DMA 几何减少无效板端
搜索，并在未建模约束下保持安全回退”，而不是声称当前 request-shape 分数已经提高性能。

## 6. 当前可主张与不可主张

可以主张：解析证书在三个未见几何上将候选从 2368 缩到 209；补齐双路径 FSim 能在上板前
发现 config1081 的 UOP 冲突；真实 FPGA canary 又发现若干 FSim 不可见的错误，证明层级合法性
门和安全回退是必要的。

不能主张：未见几何上的 request-shape 预测成功；新候选击败 TopHub；已证明通用静态
FPGA-correctness 公式；stage 或整网 FPS 提升。

## 7. DMA Pareto 修正与 E03 前瞻结果

E02 失败的直接静态解释不是“输入没有复用”，而是只优化了局部目标：config1163/1167 的输入
LOAD 字节均减少 50%，但总 LOAD+STORE 字节分别增加约 42.65% 和 37.32%。因此新增无标签
DMA Pareto 证书：exact-same-tile residency 必须同时满足输入字节减少、总 DMA 字节不增加、总
DMA 请求数不增加，才获得优先派发资格；否则保留 original。

回看 P7Q 的 14 个 input-stationary 配对，该证书选择 11 个，11/11 实测更快，中位加速
8.31%，零退化；对 E02 两个退化点均选择 abstain。二者都属于规则形成后的回顾性消融，不能
伪装成前瞻验证。

随后在读取任何 E03 板端标签前冻结新几何 `CI=128, CO=192, H=W=28, k3s1p1`：

- `oc_nthread=1` 空间 864 个配置，经解析/lowering/双路径证书保留 74 个，过滤 91.44%；
- 选定 Pareto 裕量最大 config17 和最小 config576；两条 original 与两条 residency 均通过
  三 seed FSim；
- 真实 FPGA 上 original config17 为 3/3 正确，但 residency config17 为 0/3 正确，错误元素数
  36536/36879/37254；合同立即停止，config576 未执行，未采集任何 latency。

因此 E03 没有证明 Pareto 裕量可预测加速，却前瞻证明了另一件对方法更关键的事：即使解析证书、
双路径 lowering、三 seed FSim 和 DMA Pareto 全部通过，仍可能存在真实 FPGA 不接受的候选。
所以最终算法必须把硬件正确性 canary 当作一等搜索阶段，而不能将它降格为实验后的检查；失败
候选不得进入 cost model，系统回退 original/TopHub。

## 8. 新 ResNet 工作负载上的结构迁移确认

在上述修正后，进一步选择此前没有 C3 驻留板端标签的 W03/W05/W06。选择只使用开发工作负载
上已知正收益的完整映射结构和目标本地 DMA-Pareto 特征，并在目标 FSim/FPGA/latency 标签前冻结。

W03 config248 与 W06 config142 均在第一段 original FPGA 检查中 0/3 正确，按合同停止且未计时；
W05 config174 的 original-before、input-stationary、original-after 共 9/9 正确，因而进入 7 区组
配对计时。W05 input-stationary 相对同 tile original 的成对中位加速为 **12.59%**，7/7 获胜，
验证了一次前瞻结构迁移；但其相对受保护 TopHub config575 仍为 **-34.15%**，最终安全回退
TopHub。完整结果与口径见 `P7R_SUPPORT_TRANSFER_RESULTS.md`。
