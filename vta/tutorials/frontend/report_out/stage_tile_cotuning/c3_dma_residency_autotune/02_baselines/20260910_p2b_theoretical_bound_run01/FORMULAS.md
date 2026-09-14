# VTA 单算子理论上限与瓶颈模型

## 1. 算力下界

对 packed convolution：

`MAC = N × OH × OW × Cout × Cin × KH × KW`

冻结配置中 `BLOCK_IN=16`、`BLOCK_OUT=16`，采用 100 MHz：

`Peak_MAC/s = BLOCK_IN × BLOCK_OUT × FREQ = 25.6 GMAC/s`

`Tcompute = MAC / Peak_MAC/s`

该式假定 256 个 MAC lane 每周期全部有效，不包含流水线填充、依赖、指令和运行时开销，因此是理想下界。

## 2. 理想 AXI 带宽

采用 128-bit AXI 和 100 MHz：

`Bdirection = AXI_DATA_BITS × FREQ / 8 = 1.6 GB/s`

这是每个方向每周期都传有效载荷的 optimistic 上限，不是开发板实测 DDR/ACP 带宽。

不可约流量定义为：input 和 weight 各读一次、output 写一次。当前静态流量来自冻结的 lowered-TIR DMA 提取结果。

最乐观 full-duplex 模型分别计算：

- `Tread = read_bytes / Bdirection`
- `Twrite = write_bytes / Bdirection`
- perfect overlap：`Tlower = max(Tcompute, Tread, Twrite)`
- no overlap 参考：`Tsum = Tcompute + Tread + Twrite`

另列共享串行瓶颈情景：

- `Tdma_shared = (read_bytes + write_bytes) / Bshared`
- `Tlower_shared = max(Tcompute, Tdma_shared)`

共享情景是一项显式假设，不能与 full-duplex 最乐观下界混称。

## 3. 与历史测量的关系

历史 80 点中可可靠关联 32 个 `correct=true` 候选。对它们报告：

- `gap = observed_median / ideal_lower_bound`
- `efficiency = ideal_lower_bound / observed_median`
- `effective GMAC/s = MAC / observed_median`

这些 latency 和 runtime DMA counter 是已有板端记录，本次脚本只做离线关联，没有重新上板。runtime traffic dominance 也是测后诊断，不能作为未测候选的先验特征。

## 4. 适用边界

这里的“operator rate”只是同一个卷积单算子的理想重复执行速率，不是 ResNet-18 整网 FPS。整网还包含其他算子、CPU/VTA 边界、调度、内存管理、同步和流水线气泡，不能用单算子 rate 直接相加或等同整网吞吐。
