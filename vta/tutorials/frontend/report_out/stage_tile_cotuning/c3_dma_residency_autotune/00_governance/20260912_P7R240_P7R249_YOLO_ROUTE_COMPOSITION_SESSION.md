# P7R240--P7R252：YOLO 多 workload route 组合、空间边界与部署清单

## 研究问题

P7R235 已证明 Y00 input-stationary 的局部访问差值可以传递到 YOLOv3-tiny 整图；P7R123/P7R145
又证明 Y02 weight-resident-barrier 相对同 tile original 在两个启动中加速约 24%。本轮检验：两个
局部正机制能否直接组合，以及它们相对整图现有 incumbent 是否仍值得部署。

## 实验链

- P7R240/P7R241：Y00-only 对 Y00+Y02。两条 route 均精确命中，输出全等，但加入 Y02 后整网
  264.467→342.878 ms，慢 22.87%。
- P7R242/P7R243：固定 Y00，比较 Y02 stock、same-tile original、same-tile barrier。barrier 相对
  original 377.554→343.089 ms（+10.05%，6/6），但相对 stock 265.090 ms 仍慢 22.73%。
- P7R245/P7R246：尝试在一个 RPC 中同时保留 2×2 四个完整图，第一次 VTA 调用因
  `fpga_buff_ == nullptr` 失败；没有候选输出，故不做候选分类。
- P7R247/P7R248：改为每次只加载两个完整图的 clean-start 因子边，补齐 stock→Y02-only 和
  Y02-only→Y00+Y02。
- P7R249：合并同 boot 的四条配对边。Y00 在 Y02 关闭/开启时分别 -3.413/-4.346 ms；Y02 在
  Y00 关闭/开启时分别 +79.473/+78.355 ms。四条边共 56 次 timed graph call 全部输出相同。
- P7R250/P7R251：冻结并执行证据哈希绑定的 incumbent-protected 规划；Y00 五项门全部通过而
  接受，Y02 因 0/7 和 +78.355 ms 被拒绝，最终 route 集只含 Y00。
- P7R252：将选择结果绑定到 exact Y00 identity、graph/params/AArch64 binary、P7R241 板端运行和
  clean-start bitstream/u-dma-buf 状态；manifest 主动声明没有整图 command-capacity 结论。

## 核心规律

Y00 的 DMA 主效应在有无 Y02 时逐项完全相等；Y02 的 DMA 主效应在有无 Y00 时也完全相等，逻辑
DMA byte/call interaction 为 0。延迟交互的两种配对估计约为 -0.93/-1.12 ms，远小于 Y02 自身
约 +79 ms 的代价。因此组合失败不是两个 residency schedule 互相破坏，而是 Y02 选定 tile 相对
TopHub 的绝对质量差；barrier 只追回其中约 34 ms，仍不足以成为部署候选。

## 方法修正

第三创新点不能把“same-tile 机制正收益”直接等同于“最终 route 可部署”。正式链路改为：

```text
驻留机制扩展 tile×mode 空间
 -> 多保真搜索产生 FPGA-correct 候选
 -> 以当前整图/TopHub 作为 protected incumbent
 -> 全图正确性 + live-memory fit + 配对边际 latency 准入
 -> 只把净改善 route 写入 exact manifest
```

当前证据支持两个 route 的贪心边际门，不证明任意多个 route 子集的全局最优。四图并存失败同时
说明调优测量器也必须限制候选替代版本的驻留数量或流式重启，不能让测量工具自身耗尽 u-dma-buf。
规划器另以单测验证陈旧 baseline 和证据哈希被修改时 fail closed；graph manifest 测试验证目标
二进制被修改时拒绝。下一步若做空间收尾，应采集整图 live allocation 与 instruction/UOP peak，
而不是把既有 W05 单 workload 容量抄入整图清单。
