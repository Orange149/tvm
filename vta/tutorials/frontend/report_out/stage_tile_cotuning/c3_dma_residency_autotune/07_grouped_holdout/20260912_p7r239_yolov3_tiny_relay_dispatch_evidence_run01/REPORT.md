# P7R234--P7R238：YOLOv3-tiny 整图驻留传递与 TopHub 对照

状态：`PASS_WITH_SAME_TILE_AND_STOCK_BASELINES`

## 实验对象

选择已经在 P7R144 完整 FPGA-correct 池中测完的 Y00F06 same-tile pair：YOLOv3-tiny `conv2`，
3x3、CI16、CO32、208x208。ConfigEntity 固定为 `tile_h=16, tile_w=52, tile_ci=1,
tile_co=1, oc_nthread=1, h_nthread=1`：

- original：`4a6ad8246d54bafe728949e54e4ca0566cdd136659d39d361d5adff8f10822fc`；
- input-stationary：`7b958e8562ed5e04fd14a52e940aa7aa5518ba600d8284c41d5ecd0d9bf18e3c`。

模型使用 Darknet YOLOv3-tiny COCO 权重。旧官方 URL 返回 HTTP 403，因此从公开 GitHub 镜像取得
同名 35,434,956 B 文件；实验绑定 SHA256
`dccea06f59b781ec1234ddf8d1e94b9519a97f4245748a7d4db75d5b7080a42c`。

## P7R234/P7R235：same-tile 机制进入 YOLO 整图

- 两版 graph JSON 完全相同，参数语义逐字节一致，VTA 二进制不同；exact schedule route 各命中一次。
- clean-start 板端用 person 图片和两组随机输入验证：6/6 graph calls 的全部 8 个输出张量均非零，
  A/B 每一元素相同；14/14 timing calls 也全部配对相同。
- 中位整图 `run`：266.249693 -> 264.246962 ms，input-stationary 提升 0.7579%，7/7 配对轮更快；
  配对时间差中位 2.108240 ms。
- runtime profile 的 time-evaluator 计数包含两次执行。按一次推理归一化，input LOAD 减少
  794,368 B，恰等于 P7R144 同一候选 pair 的差值；其他 tensor bytes 不变。整图总逻辑 DMA
  45,825,408 -> 45,031,040 B（-1.73%）。这证明整图只替换一个 `conv2` 实例，且单算子内存收益
  没有在模型集成时消失。

## P7R237/P7R238：stock TopHub 保护基线

重新构建不安装显式 route 的 stock TopHub YOLO 图，并和同一个 selected input-stationary 图比较：

- 三种输入的全部 8 个非零输出张量仍逐元素相同，14/14 timing calls 配对相同；
- 中位整图 `run`：267.503165 -> 264.219812 ms，selected 提升 1.2427%，7/7 配对轮更快；
  配对时间差中位 3.413404 ms；
- 按一次推理归一化，总逻辑 DMA 为 46,513,152 -> 45,031,040 B（-3.19%），总 DMA calls 为
  21,392 -> 20,300（-5.10%）；submission 均为 12；
- 相对 TopHub 的差值同时包含 tile 和 input-residency 两部分，不能用于宣称 residency 的纯机制
  因果；纯机制效应只采用上一节 same-tile A/B。

## 论文意义与边界

这组实验建立了三层互补证据：

1. P7R144 独立候选完整池给出搜索 oracle 和同 tile 机制结果；
2. P7R235 在 YOLO 整图中隔离 residency 效应；
3. P7R238 说明所选 tile+residency 在当前完整模型图上没有被 stock TopHub 保护线否定。

正确表述是“在一个 YOLOv3-tiny exact workload 上，搜索选出的配置以 3.19% 更少逻辑 DMA 和
1.24% 更低整图 `run` 时间通过 TopHub 参考线”，不是“本文全面超过 TopHub”。TopHub 仍是已有精确
记录时可直接使用的强基线；这只是一个模型、一个 workload、一个 boot。输出等价不是 COCO mAP，
runtime 逻辑 DMA 也不是物理 AXI burst。

## 绑定产物

- `../20260912_p7r234_yolov3_tiny_relay_residency_dispatch_cross_build_run01`
- `../20260912_p7r235_yolov3_tiny_relay_residency_dispatch_board_run01`
- `../20260912_p7r237_yolov3_tiny_stock_selected_cross_build_run01`
- `../20260912_p7r238_yolov3_tiny_stock_selected_board_run01`

