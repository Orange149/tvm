# Board Throughput Results

日期：`2026-04-02`

板端：
- RPC: `192.168.1.247:9090`
- 主测量口径：`time_evaluator("run")`
- 串行解释口径：`total.service avg`

## 统一吞吐口径

只保留当前确认的 `4CPU` 版本：

- CPU 总核数固定为 `4`
- `stage0_cpu` 与 `stage2_cpu` 共享这 `4` 个核
- `stage1_vta` 仍为单 VTA 执行

因此吞吐周期统一定义为：

- `pipeline_cycle_run_ms = max(stage0_run_ms + stage2_run_ms, stage1_vta_run_ms)`

说明：
- 不再保留旧的 `3:1` CPU 口径
- 不再使用 `max(stage0, stage1, stage2)` 作为最终吞吐结论

## `all_vta` baseline

- `all_vta run_mean_ms = 103.456`
- `all_vta total.service avg = 169.763 ms`

这组数作为后续所有吞吐对比的正式 baseline。

## 新切图结果

### `three_stage_d`

切图定义：
- `stage0_cpu = stem + layer1_block0`
- `stage1_vta = layer1_block1 + layer2 + layer3`
- `stage2_cpu = layer4 + head`

线程设置：
- `stage0_cpu = 4` 线程
- `stage2_cpu = 4` 线程

实测：
- `stage0_cpu run_mean_ms = 52.257`
- `stage1_vta run_mean_ms = 56.583`
- `stage2_cpu run_mean_ms = 43.386`
- `total.service avg = 221.255 ms`

按统一吞吐口径：
- `cpu_shared_run_ms = 52.257 + 43.386 = 95.643`
- `pipeline_cycle_run_ms = max(95.643, 56.583) = 95.643 ms`

对比 `all_vta`：
- `95.643 / 103.456 = 0.925`

结论：
- `three_stage_d` 已经优于 `all_vta`

### `three_stage_e`

切图定义：
- `stage0_cpu = stem + layer1_block0`
- `stage1_vta = layer1_block1 + layer2 + layer3 + layer4_block0`
- `stage2_cpu = layer4_block1 + head`

线程设置：
- `stage0_cpu = 4` 线程
- `stage2_cpu = 4` 线程

实测：
- `stage0_cpu run_mean_ms = 51.971`
- `stage1_vta run_mean_ms = 65.160`
- `stage2_cpu run_mean_ms = 23.566`
- `total.service avg = 210.770 ms`

按统一吞吐口径：
- `cpu_shared_run_ms = 51.971 + 23.566 = 75.537`
- `pipeline_cycle_run_ms = max(75.537, 65.160) = 75.537 ms`

对比 `all_vta`：
- `75.537 / 103.456 = 0.730`

结论：
- `three_stage_e` 进一步优于 `three_stage_d`

## 当前 4CPU 版本结论

当前只保留下面这组结论：

- `all_vta = 103.456 ms`
- `three_stage_d = 95.643 ms`
- `three_stage_e = 75.537 ms`

所以在当前确认的 `4CPU` 共享口径下：

1. `three_stage_d` 已经优于 `all_vta`
2. `three_stage_e` 比 `three_stage_d` 更好
3. 当前已测方案里，最好的是：
   - `three_stage_e`

## 结论边界与下一步

当前这轮切图实验**只证明了一件事**：

- 在当前 TVM/VTA 编译栈下，异构切图从**吞吐量**角度可以优于 `all_vta`

但当前实验**还没有证明**：

- 已经找到了全局最优切法
- 当前自动搜索策略已经能稳定找到最优切法

也就是说，当前结果更准确的表述是：

- 已经证明“异构在吞吐上可以更好”
- 但还没有证明“现在的自动切图已经找到了最优异构切法”

### 当前最值得继续验证的切法假设

下一步最值得继续验证的方向是：

- 回到 `3:1` CPU 分工
- 让 `1` 个 CPU 只负责最后一点点后处理
- 让 `3` 个 CPU 负责 `stem + layer1_block0`
- 中间主体部分交给 VTA

也就是一个更接近下面语义的三段式：

- `stage0_cpu(3 cores) = stem + layer1_block0`
- `stage1_vta = middle conv-heavy body`
- `stage2_cpu(1 core) = only a very small tail`

当前判断依据是：

- `three_stage_d` 和 `three_stage_e` 都说明，把 `stem + layer1_block0` 留给 CPU、把中间主干交给 VTA 是有潜力的
- 但当前 `4CPU shared` 版本还没有显式利用“`3` 核前处理 + `1` 核极小后处理”这种更细的 CPU 分工
- 因此，最优切法很可能还需要把 `stage2` 再压到“只剩最后一点点尾巴”，并用 `3:1` 的 CPU 分工单独验证
