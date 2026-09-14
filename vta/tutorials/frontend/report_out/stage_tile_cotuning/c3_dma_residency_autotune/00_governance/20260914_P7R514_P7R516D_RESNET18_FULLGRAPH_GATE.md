# P7R514--P7R516D：ResNet18 策略选择整图门与失败归因

状态：`FULLGRAPH_CORRECTNESS_FAIL_CLOSED; H3_COMBINED_ROUTE_ISOLATED; NO_FULLGRAPH_TIMING_LABEL`

## 1. 本节点回答了什么

本节点把 P7R512 在相同算子池、`seed=57001`、`budget=50` 下得到的策略选择真正放回同一份
Relay Testing ResNet18。P7R514 在整图标签不可见时冻结选择，P7R515 对相同三路由签名只构建一次，
P7R516 在干净 bitstream/RPC 会话中先做三输入全部输出等价检查，再准备做七轮平衡计时。

结果不是性能正结果：P7R516 在第一个输入完成四个程序后发现输出不一致，按照协议使整次 session
失效，并且没有进入计时。随后 P7R516D 保持 P7R515 二进制不变，只做输出哈希和差异计数诊断，
将错误强定位到 H3 的 `input_weight_resident_barrier` 组合驻留路由。

## 2. 冻结选择与构建

| 策略 | H1 | H2 | H3 | 唯一整图程序 |
|---|---|---|---|---|
| stock mode-aware XGB | D0637 original | D0575 original | D0098 combined | `selected_42a339b796f2cb38` |
| ML²Tuner P/V/A | D1213 original | D0575 original | D0098 combined | `selected_918f902567c608ec` |
| 本文 DMA 多保真 | D1213 original | D0575 original | D0098 combined | `selected_918f902567c608ec` |
| Cheng minimum-access | D1213 original | D0575 original | D0203 original | `selected_e53bbdbb71050431` |

四个策略去重为三个 selected 程序；另有 stock reference。P7R515 总构建墙钟为 46.541 s。四份
Graph JSON 完全相同，参数语义哈希相同，三条精确 route 均经过真实调度；不同 DSO 只反映调度与
驻留代码差异。H1/H2/H3 的真实 Graph JSON 节点数为4/3/1，编译期重复调度查询命中为16/16/7；
二者不能混称。P7R515 没有接触板端。

## 3. P7R516 正式整图门

P7R516 在 boot `bf5bc78c-115b-40e0-8853-6684f04ecf3d` 上完成严格 host-key、runtime 哈希、
u-dma-buf、FPGA 状态检查，重新加载冻结 bitstream，并从新的默认 RPC 定义 T0。

- W0→T0：7.828 s；
- 完成的正确性调用：4；
- W0→失败：14.441 s，T0→失败约 6.614 s；
- timing 调用：0；
- 状态：`invalid_entire_session_fail_closed`；
- 失败原因：`selected full-graph output mismatch`；
- session 未拼接，所有已付调用与墙钟均保留。

因此本节点没有产生可用的 selected 整图 latency、FPS 或相对 stock 性能数据。失败程序不能进入
后续性能标签，也不能用另一次运行的计时片段补齐。

## 4. P7R516D 只正确性诊断

原 runner 在比较前已经把 invocation 行写盘，导致失败产物没有保存具体 mismatch 数。新增的诊断器
不修改 P7R515 二进制、不做性能计时，对四个程序使用三个确定性 seed，并分别按正序和逆序执行，
共 24 次调用。

| 程序 | 相对 stock 正确观察 | 每次分类输出 mismatch | 同 seed 两种顺序输出哈希稳定 |
|---|---:|---:|---:|
| stock reference | 6/6 | 0 | 是 |
| Cheng / H3 original | 6/6 | 0 | 是 |
| stock-XGB / H3 combined | 0/6 | 1000 | 否 |
| ML²Tuner 与本文 / H3 combined | 0/6 | 1000 | 否 |

P7R516D 为同 boot、不可拼接的完整诊断 session：W0→T1 18.699 s，T0→T1 10.859 s。它只产生
正确性证据，不产生性能标签。

## 5. 可以归因到什么程度

ML²/本文程序与 Cheng 程序的 H1、H2 route 完全相同；两者唯一的路由差异是：前者在 H3 使用
D0098 combined，后者在 H3 使用 D0203 original。stock-XGB 虽然使用另一 H1 tile，却与 ML²/本文
共享同一个 H3 combined，二者均失败。故可以强归因：当前 H3 combined route 不具备整图部署资格。

该 D0098 combined 身份在孤立算子池中三 seed 正确并有五轮计时，说明本次失败不是“单算子从未
正确”，而是单算子资格没有覆盖完整 Relay 图中的调用上下文或跨调用状态。错误程序在同一 seed、
不同执行顺序下输出哈希变化，也支持存在上下文/状态依赖；但在完成地址级、依赖级诊断前，不能把
底层原因进一步写死为某一种 SRAM 覆盖或 DMA race。

运行期画像证明错误程序确实执行了 combined 路径：两个错误程序每图为 21 次 driver/synchronize，
Cheng original 为20次；错误程序逻辑 LOAD 为19,854,080 B/2,108 calls，Cheng 为19,917,312 B/
2,164 calls。少 63,232 B 和56次 LOAD 并没有换来可部署结果，再次证明减少逻辑 DMA 不是正确性
证书，也不是整图性能定理。

## 6. 对实验协议和论文主张的修正

1. P7R510 的算子池、P7R512 搜索回放和 P7R513 Cheng 结果仍然有效，但只在各自的孤立算子作用域
   内解释。
2. “三个 seed 的孤立算子 FPGA-correct”不能直接升级为整图可部署；驻留候选还需要最终 fused/
   call-context 的整图正确性门。
3. 当前 budget=50 下，stock-XGB、ML²Tuner 和本文选中的整图都因共享 H3 combined route 被拒绝；
   Cheng original 路由通过 6/6 诊断，但本节点没有对它计时，不能据此声称 Cheng 整图更快。
4. 后续若继续，必须先冻结一个不读取整图 latency 的修订协议：将整图错误身份标为 invalid，按各
   策略既有顺序 fail-fast 回退到下一个尚未拒绝的候选，或统一回退到 original；然后重新构建、
   三输入正确性、七轮平衡计时。不能在看过错误输出后调整性能排序规则。
5. 原计划 P7R517--P7R522 的 H1 三对三 clean-start 暂不启动。否则其最终图可能复用同类未经整图
   资格的 combined route，无法公平解释 T0→T1。

## 7. 审计信息

- P7R515 ledger：`d25c837467a975fb31c0a811cd8fefd797568012ca70289565acdf3daf43bc49`
- P7R516 ledger：`872e147b50f5adcc38ec198d4bbbd35f2d6b808baef728b54a1b9c5950aa3edb`
- P7R516D ledger：`6b22ec4a9b1421b0d6873fb66d51b8fb0235eebe20a68e5b16241f3e2572c7fb`
- P7R516D runner SHA-256：`a71df884f65e59076f8e4e763b124715fdaf76af66ab6aa8a8cd255e87ba4238`
- 三份 artifact ledger 均已通过完整校验。
- 诊断后板端仍为同一 boot，FPGA `operating`，u-dma-buf 201,326,592 B，默认 RPC cwd 与四项
  runtime 哈希正确，未观察到新的 EXT4/MMC 错误。
