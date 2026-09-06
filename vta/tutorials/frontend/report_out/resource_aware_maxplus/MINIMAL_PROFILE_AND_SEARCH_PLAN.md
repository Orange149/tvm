# V1 局部成本表与 Top-K 验证计划

更新日期：2026-09-02

## 1. 目的

本计划为多来源约束的 V1 提供最小成本输入：用 PartitionTuner-style grouped segment profile
支持 Tarnawski/DNNPipe-style Pipeline DP，再加入当前 SoC 的单 VTA、CPU 核池和共享 DDR 约束。
不再尝试一次性校准完整硬件。成本表只回答：

```text
一个 CPU segment 在指定核心数下需要多久？
一个 VTA segment 在冻结 schedule/tile 下需要多久？
一条 CPU-VTA 边界需要多少 adapter/copy/runtime 时间？
候选每帧对共享 DDR 产生多少唯一 resource demand？
```

完整候选吞吐只用于 Top-K 最终验证，不进入局部成本表拟合。

当前板端连接固定为：

```text
SSH: root@192.168.1.234
TVM RPC: 192.168.1.234:9090
```

生成的新命令必须显式包含 `--host 192.168.1.234 --port 9090`（或等价的显式配置），并在结果
manifest 中记录端点和板端身份。不得因为 IP 改变而改写历史测量文件，也不得把 IP 本身当作稳定
硬件身份。

## 2. 成本表键

### CPU

```text
segment lowering/fusion signature
physical input/output shape
dtype/layout
threads in {1,2,3,4}
compiler and runtime fingerprint
```

输出：`wall_ms`、`process_cpu_ms`、`effective_cores`、correctness 和 uncertainty。

### VTA

```text
segment lowering/fusion signature
physical shape and padding
frozen schedule/tile id
bitstream/compiler/runtime fingerprint
```

输出：VTA mutex 内与 boundary 互斥的 device-run `service_ms`、correctness 和 uncertainty。V1
不拆分 overlap 参数。

### Boundary

```text
direction
source/destination dtype and layout
physical tensor bytes/calls and shape
row/pitch/padding/slice/quantization conversion
cache maintenance observed in the actual runtime path
runtime policy fingerprint
```

输出：host adapter、set/get copy 的互斥计费分量及总 boundary service。submit/sync 归属 VTA
device-run service，不在 boundary 重复计费。

### Shared DDR

```text
owner: CPU segment / VTA LOAD-STORE / boundary adapter-copy
direction and access class
physical bytes and calls
stable transaction ids
sustained bandwidth fingerprint
```

输出：每帧 `ddr_demand_ms=bytes/sustained_bandwidth`。相同 transaction id 在整个候选中只能出现
一次。V1 不根据 concurrent case 临时拟合任意 slowdown。

## 3. Manifest 生成

1. 对所有 DP 可达 segment 做 compile-only lowering，生成 fusion/materialization、boundary 和 DDR
   transaction manifest。
2. 完整键相同的实例合并为一个 profile case，并记录复用次数。
3. 对历史 archive 做 fingerprint、correctness、计时边界和 fallback 审计。
4. 可复用数据直接进入草稿成本表；只为缺失 signature 生成新命令。
5. 在看到去重数量和历史覆盖率后，冻结 `B_profile` 和 wall-clock 预算。

不得按“固定 16 个”或规则网格凑数量，也不得为每个完整切图单独 profile。

## 4. 测量协议

运行任何 qualification 前先执行 SSH/RPC preflight，确认 `192.168.1.234` 对应预期板卡、bitstream、
时钟和 runtime；连接成功但身份不匹配也必须停止。

第一轮只运行一个 qualification session：

```text
固定 affinity/governor/runtime
-> package prewarm
-> 3 warmup
-> 10 scored samples
-> independent reference correctness
-> median + dispersion
```

CPU `threads=1..4` 必须验证实际 affinity 和 `effective_cores`。若 runner 只能设置线程数而不能约束
核心，CPU 核心分配搜索暂停，不能把 requested threads 当作资源分配证据。

VTA 只运行冻结 schedule/tile。Boundary 必须调用与候选 Pipeline 相同的 adapter/set/get path，
并使用 accounting id 防止 VTA LOAD/STORE、submit/sync 与 boundary 重复计费。

共享 DDR qualification 至少包含一组 matched control：

```text
CPU streaming only
VTA DMA/device-run only
CPU streaming + VTA concurrent
```

三者保持 bytes、calls、affinity 和 runtime policy 可比。并发结果只用于检验聚合 DDR 下界是否
可能不足；V1 成本表仍使用可解释的持续带宽和唯一 traffic demand。

## 5. 可加性 Holdout

局部成本表需要预测完整 stage，因此预先冻结少量未参与修正的完整 stages：

```text
至少 2 个 CPU fused stages，覆盖不同 threads
至少 2 个 VTA islands，覆盖不同长度/shape
至少 2 个双向异构 boundaries
至少 1 组 shared-DDR concurrent matched control
```

报告 segment prediction error 和排序是否正确。若 unit 求和存在稳定偏差，只允许增加低维
`stage_launch/intercept` 或对高排名 segment 做独立 correction；不恢复完整 HardwareProfile 网格。

## 6. DP 输入和输出

DP 只读取：

```text
legal unit/cut manifest
CPU segment cost table by threads
VTA fixed-schedule segment cost table
boundary cost table
shared DDR transaction manifest and sustained bandwidth
CPU core capacity and max VTA islands
```

禁止读取：

```text
candidate measured FPS
candidate measured stage time
candidate id/layer name correction
未冻结 tile 或 runtime policy
```

输出必须包含 Top-K 方案、逐 segment 成本、CPU 核心分配、VTA island 列表、boundary 列表、DDR
demand、预测 II、cost provenance 和 compile manifest。

## 7. Top-K 实验

第一轮使用回顾性 measured pool 检查排序。通过后再进行前瞻实验：

```text
K curve: 1, 3, 5, 10, 20
V1 Top-K: DP 预测排名
controls:
  B0 random / stratified random
  B1 type-fixed Synergy/NEURAghe-like mapping
  B2 PartitionTuner-like grouped cost + single-frame latency
  B3 Tarnawski/DNNPipe-like Pipeline max load without core/DDR coupling
  B4 proposed full V1
```

每个候选按以下顺序消耗预算：

```text
compile-only -> reference -> native package -> board Pipeline
```

编译失败计入 `B_build`，上板尝试计入 `B_search`；重试必须记录原因和 wall-clock。

## 8. Gate

启动前瞻上板前：

```text
局部 case correctness 全部通过
CPU affinity/core allocation 真实生效
grouped holdout 没有系统性方向错误
共享 DDR transaction ownership 无缺失或重复
DP 与完整静态枚举在相同成本表上结果一致
回顾性 B4 低 K 排名优于 B0-B3
```

完成 V1：

```text
Top-K 找到接近 measured-pool oracle 的方案
B_profile + B_search 明显小于大规模 candidate measurement
所有输入不存在 candidate throughput 泄漏
结果在冻结候选组或第二个 DNN 上复现
```

若 gate 失败，先修正成本表、runner affinity 或所有权；不增加 tile、FIFO 或 Max-Plus。
