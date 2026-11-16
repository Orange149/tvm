# 原 4/1/4 CPU 阶段重叠因果验证

主图固定为 CPU00:02 → VTA03:17 → CPU18:20，沿用已完成三 boot 残差协议的图、权重、
固定真实图像、边界参考及运行库。新实验独立冻结，不向旧 run 续写。正式评分和候选排名不修改。

`--cpu-stage-gate` 默认 `off`。`observe` 仅记录 CPU 门的 request/acquire/release，不阻止重叠；
`exclusive` 使用 FIFO 门，让所有 CPU 阶段的 set/run/get 与输出快照互斥，VTA 不经过此门。
两个模式保持 4/1/4 线程、全部四核 mask、spin=0、queue depth=2 与同一新 runner。
等待在 stage_start 之前，分别输出 `stage*_cpu_gate_wait_ms`，不归入 stage run 或 busy。
门释放后才提交下游队列；异常用 RAII 释放持有的门，pipeline 取消会唤醒全部等待者。

## 冻结和主机验证

```sh
residual_python=/home/orange/miniconda3/envs/vta-resnet/bin/python
overlap_run=vta/tutorials/frontend/report_out/stage_tile_cotuning/cpu_stage_overlap_validation/NEW_UNIQUE_RUN
$residual_python vta/tutorials/frontend/run_cpu_stage_overlap_validation.py freeze --run-dir "$overlap_run"
$residual_python vta/tutorials/frontend/validate_cpu_stage_gate_host.py --run-dir "$overlap_run"
```

freeze 复制全部输入、程序、运行库、原生源码与控制器依赖，并锁定 SHA。主机编译使用该源码，
检查默认 off、显式 off、observe、exclusive、queue 组合及失败取消，另检查 8000 次并发入门、
异常释放、取消唤醒和取消后拒绝。主机验证未通过时禁止执行板端实验。

## 单 boot 探索

```sh
$residual_python vta/tutorials/frontend/run_cpu_stage_overlap_validation.py pilot --run-dir "$overlap_run" --execute-board
```

先做串行 10 帧、off/observe/exclusive 流水各 10 帧完整输出校验。串行正确性保留原 10 ms
边界稳定性检查；性能会话关闭这些等待与输出 dump。串行 20 帧分别在探索前后采集，丢前 5 帧。
每个流水性能窗口 307 帧，丢前 5 帧预热与随后 2 帧 settling，以 300 次完成的 299 个间隔计 II。
逐帧结果只缓存 JSON 文本，完成后写出；queue 遥测关闭，原生启动 affinity 核查开启。

计时开关的两次配对顺序 off→observe、observe→off。两次 II 扰动都不超过 2% 时，
A 使用 observe；否则 A 使用 off。B 固定 exclusive。随后三个探索 block 的顺序为
A→B、B→A、A→B，不按 II 重新选择干预或删除慢帧。

预声明主指标是首 CPU 的平均 run 时间变化；尾 CPU 与 II 为次指标。若首 CPU run 的三个
block 均值变化中位数减少至少 5%，A CPU run 存在重叠、B 全帧 CPU 门区间不重叠，且匹配
串行首 CPU 前后漂移不超过 5%，触发复现。触发后在当前 boot 另采集 A→B、B→A 两个
平衡 block，不混入探索窗口，写入 paired_boot_1.json。

CPU run 恢复说明减少 CPU 阶段同时执行会改变服务。whole-stage 门也减少 set/get 的并发，
并会改变 CPU/VTA 的 DDR 访问时间线，因此此对照不能进一步分开调度、cache 和物理 DDR。
即使 II 没改善，阶段服务恢复仍是机制证据，不能将首尾节省直接相加成 II。

## 独立 boot 复现和分析

用户再进行两次真实 kernel 重启，各次恢复相同的已验证 u-dma-buf/固件环境及频率后执行：

```sh
$residual_python vta/tutorials/frontend/run_cpu_stage_overlap_validation.py replicate --run-dir "$overlap_run" --execute-board
$residual_python vta/tutorials/frontend/run_cpu_stage_overlap_validation.py analyze --run-dir "$overlap_run"
```

RPC 重启和 bitstream 重载不算新 boot。复制到新的 owned tmpfs，使用已验证完成/context reset
协议，不改驱动、bitstream 实现、tile、驻留或频率。实测 CPU 频率逐次核查，实际 FPGA 频率 unknown。
每 boot 保留完整正确性和两个配对 block；统计单位是三个 boot 的配对均值，Student-t df=2。
方向一致、变化中位数至少 5%、boot级 95% CI 排除零及轨迹一致后才宣称主机制跨 boot 验证。

分析每次写入新的 report 目录，输出中文服务/等待/II 表、可独立导出的 SVG 时间线及机器 JSON。
正确性、超时、身份或互斥合同失败立即停止并保留原始/部分结果，失败 run 不静默重试。
只清理本协议拥有的进程与已归档临时会话，原库、SD 部署和历史数据不覆盖。
