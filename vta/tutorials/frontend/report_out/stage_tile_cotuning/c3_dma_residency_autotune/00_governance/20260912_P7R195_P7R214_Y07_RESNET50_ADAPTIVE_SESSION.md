# P7R195--P7R215：第二个稀疏留出、ResNet50 dense 分支、三池汇总与机制效应

## 结论

本轮在不改变 P7R185 冻结阈值、service proxy 或排序源码的前提下增加两个严格留出：YOLOv3-tiny
Y07 验证第二个 sparse path，ResNet50 R50A 验证 dense 4→2 path。两者的第一次 FPGA 测量均为
完整正确池 oracle。连同 Y06，三个严格 holdout、72 个预注册身份中，自适应策略以 22 个中位 gross
candidate 达到全部三个 oracle，较全池资格的 72 个减少 69.44%；统一墙钟降低 24.05%，相对固定
4→2 降低 12.00%。

## Y07：YOLOv3-tiny conv14 深层 3×3

- 几何：CI=256、CO=512、H=W=13、K=3，完整 ConfigSpace 480 个实体，容量粗筛 44 个。
- P7R195 run01 曾误写为 1×1；在任何目标结果产生前核对模型表后原样废弃，run02 以正确的 3×3
  描述和新 seed 重冻，未预设 adaptive branch。
- 在线首 family 0/3 lowering 通过，第二 family 2/3 通过，因此走 sparse family-wave；执行 6 次
  lowering、1 次 FSim 后只送一个候选上板。
- 完整池只有 2/24 个 lowering/FSim 合法，均三 seed FPGA-correct，14/14 次计时正确。
- 首波候选为 323.006270 ms；独立完整池 oracle 是同一身份，322.948999 ms；另一点为
  340.015270 ms。
- 一致回放中 adaptive 为 7.362704 s，exhaustive 为 8.454560 s（-12.91%），固定 4→2 为
  8.603170 s（-14.42%）；gross 6/24。

Y07 再次表明 3×3 高通道的名义容量合法空间并不等于真实 lowering 空间。其最终池仍只有两点，
所以它增强稀疏路径重复性，但不能单独解决最终性能池过小的问题。

## R50A：ResNet50 bottleneck 1×1

- 几何来自仓库 `relay.testing.resnet` 的 bottleneck 结构：后续 stride-1 stage2 unit 的 conv1，
  CI=512、CO=128、H=W=28、K=1。完整 ConfigSpace 3456 个实体，容量粗筛 510 个。
- 24 个身份和冻结策略在任何目标结果前注册；首 family 2/3 lowering 通过，按原阈值走 dense 4→2。
- 在线前缀执行 10 次 lowering、2 次 FSim，形成两个预编译候选。
- 完整池为 10/24 lowering/FSim 合法，覆盖五个 same-tile family；10/10 三 seed FPGA-correct，
  70/70 次计时正确。original/input 各五点；当前 barrier 在十六个粗筛身份中均因真实布局/compact
  条件失败，不能据此否定权重驻留机制本身。
- 排序第一候选首波为 5.902469 ms，独立完整池为 5.907180 ms，仍是 oracle；其余候选为
  8.334834--192.916629 ms。
- 一致回放中 adaptive/dense-fixed 同为 8.850494 s；exhaustive 为 12.706880 s，因此墙钟降低
  30.35%，gross 10/24。纯 family-wave 为 9.342097 s 且在达到目标前多执行一次 FPGA 测量，说明
  dense 空间保留批量前沿是必要的。
- Random 前沿中位墙钟为 9.892833 s；hardware-diverse validity 和 validity-only 分别为
  14.274194/16.357030 s，继续支持“当前跨 workload Model V 不进入主策略”的负向结论。

首波收集器实际计时了两个晋级点；time-to-oracle 只计固定 dense 策略要求的两个预编译，以及排序
第一点的 correctness/代表测量。多出的第二点板端收集成本不冒充在线早停节省。

## 三个严格 holdout 汇总

P7R214 对 Y06、Y07、R50A 按 seed 配对汇总，未重新调参：

| 方法 | 三池全部命中率 | 中位 gross | 中位统一墙钟 |
|---|---:|---:|---:|
| 生存率自适应多保真 | 100% | 22 | 19.735 s |
| 固定硬件多样 4→2 | 100% | 58 | 22.426 s |
| 全池资格后 service 排序 | 100% | 72 | 25.983 s |

自适应相对全池资格：gross -69.44%、墙钟 -24.05%；相对固定 4→2：墙钟 -12.00%。三个策略最终
都只需各测一个 oracle，因此 FPGA kernel invocation 和目标候选逻辑 DMA 相同；节省来自避免不必要
的 lowering/FSim/compile，而不是虚构板端 DMA 降幅。

## R50A same-tile 机制效应

P7R215 从已经完整测量且 10/10 FPGA-correct 的 R50A 池中构造五个只改变 residence mode 的
original/input-stationary 配对。输入驻留 5/5 更快，改善范围 29.13%--79.08%、中位 62.80%；输入
逻辑 DMA 字节中位下降 75.00%，总逻辑 DMA 字节下降 40.25%，DMA calls 下降 72.73%。这说明首个
搜索候选成为 oracle 并非只有排序上的偶然命中：其上游机制确实延长输入 tile 生命周期并减少
u-dma-buf 重复 LOAD，且在该固定 FPGA 上转化为时间收益。

统计边界必须保留：五个 family 来自同一几何和同一 boot，不是五个独立 workload；精确双侧 sign
test 为 `p=0.0625`。10,000 次 family-cluster bootstrap 给出的中位改善描述区间为
29.13%--79.08%，只能描述本池，不能称总体置信区间。

## 当前方法边界

- 两个 sparse YOLO 留出最终各只有两个正确候选，但 R50A 提供了 10 点正确池和 dense 分支证据。
- 三个留出均第一次测量命中 oracle，这是明确正结果，但样本仍不足以宣称普适最优。
- R50A 是算子级真实网络几何，不是 ResNet50 整网或 stage FPS。
- service 系数是冻结的等效 acquisition penalty；逻辑 VTA 计数不是物理 AXI burst。
- 下一步应做跨启动稳定性和把已选候选传递回 stage；不应在这些留出标签上再次调阈值。

## 关键 ledger SHA-256

- Y07 合同 P7R195 run02：`41f278b6b6a3634f572722b0f9e89e6f533b8e64dd38cfbedfafeae249ea831d`
- Y07 首波 P7R198：`05e2c1d420269024d00341579273d84d23e94aea3b65276166bfa83bcca04d1c`
- Y07 完整板端 P7R200：`09006be37784153edb8bd77037e3ac5201b3e30a2fde742bd8b01ddd6eff79c9`
- Y07 分析 P7R203：`542af4b87ab382cde5b5754b4e165937b4664387f24941e58f1ed645b43f5dcc`
- R50A 合同 P7R205：`b9b8f9de9ee743811f993732685aeaaab51e65c54d7fcdcc8eaf8c6b2128b1c9`
- R50A 首波 P7R208：`db2126304f0128aa36b9ecbf6c5d23101f827d83cbb3486af921274a18642d96`
- R50A 完整板端 P7R210：`4a5cd67667c43318b748e8386fc85581d02707982ea0a05e1d2b555a7099fc63`
- R50A 分析 P7R213：`301df6becc18f6d2a60b0b73c1c5bf49996e8f36e04dd6a63fb5b6cbc68198f4`
- 三池汇总 P7R214：`cd53aa9abe40d7c646694b6a4f573c72fcbc3060dcf24f3be43600b3caacf6ad`
- R50A same-tile 分析 P7R215：`6c7a1da06d2c50523fb4a46702685baf1836ec3d80a964b2af6583f2a770a587`
