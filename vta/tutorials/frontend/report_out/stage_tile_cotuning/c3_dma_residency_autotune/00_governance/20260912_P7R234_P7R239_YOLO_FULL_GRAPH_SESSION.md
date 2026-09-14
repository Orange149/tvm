# P7R234--P7R239：YOLOv3-tiny 第二网络整图传递

## 选择依据

使用 P7R144 已完整测量的 Y00F06 pair，而非查看整图结果后重新调参。目标是 `conv2` 的
CI16/CO32/208x208/3x3，input-stationary 是 Y00 完整候选池 oracle，同 tile original 是机制对照。

## 资产恢复

仓库下载脚本引用的 `pjreddie.com/media/files/yolov3-tiny.weights` 返回 HTTP 403。改用公开 GitHub
镜像下载到 `/tmp/tvm_test_data`，大小 35,434,956 B，SHA256 为
`dccea06f59b781ec1234ddf8d1e94b9519a97f4245748a7d4db75d5b7080a42c`。cfg、Darknet loader 和 person
图片沿用 TVM test-data cache；活跃大文件不写 SD 卡。

## 结果

- P7R234：original/input 两版整图交叉构建成功，graph 相同、二进制不同、exact route 命中。
- P7R235：person 图片和两组随机输入的 8 个非零输出全等；same-tile 266.250→264.247 ms，
  +0.758%、7/7。整图单次 input LOAD 差值 -794,368 B，与独立 Y00 pair 完全一致。
- P7R237：另行构建未安装 route 的 stock TopHub 整图，并与已经冻结的 selected 图配对。
- P7R238：全部输出继续全等；stock→selected 为 267.503→264.220 ms，+1.243%、7/7；单次总逻辑
  DMA -3.19%、DMA calls -5.10%。

same-tile A/B 隔离 residency 机制；stock/selected A/B 评价部署质量但混合 tile 与 residency。两者
必须在论文中分开报告。当前不主张 COCO mAP、物理 AXI、跨 boot 或“全面超过 TopHub”。

板端 boot `aa7a3e5c-021d-4d6b-88ae-7f696faa567c`，每次运行均在任何 target allocation 前停止 RPC、
重载冻结 bitstream、启动 tmpfs RPC；运行后未见 EXT4/mmc 错误。

