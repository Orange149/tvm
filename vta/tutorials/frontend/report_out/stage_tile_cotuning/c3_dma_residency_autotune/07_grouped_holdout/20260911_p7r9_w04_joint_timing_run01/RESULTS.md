# P7R W04 虚拟线程 × 输入驻留联合实验结果

## 结论

这轮不是 P7Q 的重复。它首次保留 `oc_nthread=2`，并在真实 FPGA 上比较同一 ConfigEntity 的
`original` 与 `input_stationary`。结果证明“驻留是否有效”不仅由减少的 DMA bytes/calls 决定，
还取决于二维 DMA 请求形状和访问--执行流水重叠。

| config | tile `(h,w,co)` | input bytes | LOAD calls | 同 tile 原始中位数 | 驻留中位数 | 比值中位数 | 成对中位提升 | 区组胜负 | 相对 TopHub |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 455 | `(14,2,4)` | `487424→243712` | `448→224` | 8.445312 ms | 7.693685 ms | +8.90% | **+8.93%** | 7/7 胜 | -176.88% |
| 461 | `(2,14,4)` | `487424→243712` | `448→224` | 5.591966 ms | 6.212186 ms | -11.09% | **-12.64%** | 0/7 胜 | -123.56% |

“比值中位数”是两个中位延迟之比；预注册的成对主指标按每个随机完整区组先计算相对变化、再取
中位数。两种算法结论一致。7 个区组全部同号，在独立同概率符号零假设下，双侧 exact sign-test
为 `p=0.015625`；样本仍然只是单 boot 开发证据，不冒充跨 boot 或未见 workload 确认。

TopHub config463 的 14 个前后哨兵中位数为 2.778751 ms，范围 2.769400--2.888789 ms。
两种驻留候选均未接近受保护 incumbent，因此不能写“获得新的最优性能”。

## 为什么静态总量相同，方向却相反

两组候选的 input/weight bytes、LOAD calls、STORE bytes/calls 相同，但具体 descriptor 不同：

```text
config455 input: x_size=4/5,  y_size=28, x_stride=28  （短行、跨距）
config461 input: x_size=28,   y_size=4/5, x_stride=28  （连续整行）
```

驻留后两者都把 input descriptors 和总 LOAD calls 减半，并把 weight request 从
`(x=9,y=4,stride=72)` 合并为 `(x=9,y=12,stride=72)`。但是：

- 455 原来包含大量短行跨距输入请求；减少这些请求的启动/逐行代价，足以抵消更大 weight request
  和可能的流水重叠损失；
- 461 的输入本来就是连续整行，减少逻辑 bytes/calls 的边际收益较低，而 loop/cache lifetime
  改变造成的流水损失占上风。

因此后续代价不能写成 `score = a*bytes + b*calls`，至少要按 descriptor 求和：

```math
\widehat T_{dma}=\sum_q\left(
L_{req}+y_qL_{row}+\frac{bytes_q}{BW_{contig}\eta(x_q,stride_q,pad_q)}
\right),
```

再与计算、同步和重叠项联合：

```math
\widehat T=\max(\widehat T_{load},\widehat T_{compute},\widehat T_{store})
+T_{sync}+T_{launch}+T_{tail}+T_{overlap\ loss}.
```

其中 `eta`、`L_req`、`L_row` 和 overlap loss 应由少量校准样本或板端标签学习；硬件容量、依赖、
整除和明显的零复用仍作为测量前硬剪枝。

## 完整筛选链

1. W04 完整联合空间中有 320 个 `oc_nthread=2,h_nthread=1` 配置。
2. 275 个同 tile original 在完整 VTA lowering 中已不合法。
3. 9 个驻留变换因 accumulator SRAM 超限被拒绝。
4. 9 个可编译但 input DMA 不减少，被规则拒绝。
5. 27 个本地 eligible，三 seed FSim 为 27/27 正确。
6. 标签无关 shortlist 的 config330 在真实 FPGA 上连 original 都是 0/3 seed 正确，立即停止并排除。
7. 独立恢复合同中的 config455/461 original 与驻留版本、前后 TopHub 哨兵，共 18/18 seed 检查正确。
8. 正确性通过后才执行本计时。

这条链说明 FSim、静态容量和 DMA 签名各自都不是最终硬件合法性的替代品；它们的价值是把 320
个候选压缩到极少数真实板端派发，并把错误隔离在计时之前。

## 当前 Gate

- `G7R-local`：部分通过。原模板回归未变，联合空间证书和 27/27 FSim 已通过；真实 FPGA 仍发现
  一个 FSim 未覆盖的原 schedule 错误。
- `G7R-board`：开发机制通过一半。config455 得到稳定同 tile 收益，config461 是稳定反例。
- `G7R-innovation`：尚未通过。当前没有打赢 protected incumbent，也没有未见 workload 的
  regret@budget/派发数比较。

下一步不是重复 W04，而是冻结 request-shape-aware 规则，在此前未计时的卷积几何上验证：

1. 容量证书能否继续过滤非法候选；
2. 短行跨距强度能否预测驻留收益符号；
3. 与 Random、knob-only XGB 相同预算时，是否减少板端派发并保持 near-oracle recall；
4. 无候选超过 incumbent 时必须回退。

