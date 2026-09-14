# C3 预注册实验协议

冻结日期：2026-09-10。冻结对象是规则和数据分组，不是尚未测得的结果。

## 1. Workload 冻结与分组

权威源为 `stage_memory_experiments/static_workload_dma.json`（SHA-256 见 manifest）。记 packed input 为 `[N, CI/16, H, W, 1, 16]`，weight 为 `[CO/16, CI/16, KH, KW, 16, 16]`。

| ID | input | weight | stride | padding | 分组 |
|---|---|---|---|---|---|
| W00 | 1×4×56×56×1×16 | 4×4×3×3×16×16 | 1 | 1 | development；P3代表 |
| W01 | 1×4×56×56×1×16 | 8×4×3×3×16×16 | 2 | 1 | grouped holdout |
| W02 | 1×8×28×28×1×16 | 8×8×3×3×16×16 | 1 | 1 | development；P3代表 |
| W03 | 1×4×56×56×1×16 | 8×4×1×1×16×16 | 2 | 0 | development |
| W04 | 1×8×28×28×1×16 | 16×8×3×3×16×16 | 2 | 1 | grouped holdout |
| W05 | 1×16×14×14×1×16 | 16×16×3×3×16×16 | 1 | 1 | development |
| W06 | 1×8×28×28×1×16 | 16×8×1×1×16×16 | 2 | 0 | development |
| W07 | 1×16×14×14×1×16 | 32×16×3×3×16×16 | 2 | 1 | grouped holdout |
| W08 | 1×32×7×7×1×16 | 32×32×3×3×16×16 | 1 | 1 | grouped holdout |
| W09 | 1×16×14×14×1×16 | 32×16×1×1×16×16 | 2 | 0 | development；P3代表 |

按 workload 分组，任何同一 workload 的不同配置不得跨 development/holdout。W01/W04/W07/W08 的新候选标签在规则冻结后才允许读取。

额外几何 holdout 预注册为：

- E00：CI=96、CO=160、H=W=42、3×3、stride1、padding1；
- E01：CI=192、CO=320、H=W=21、1×1、stride1、padding0；
- E02：CI=320、CO=384、H=W=11、3×3、stride2、padding1。

它们仅在 lower/build 证明当前 VTA 真正支持后进入 P7；不支持时保留失败记录，不以看过性能标签后的替代几何补位。

## 2. 候选身份

对规范 JSON（UTF-8、key 字典序、无多余空格）计算：

```text
candidate_id = SHA256({
  hardware_fingerprint,
  template_name,
  schedule_version,
  workload,
  residence_mode,
  complete_config_entity
})
```

`config.index` 只作调试字段。原模板的 TopHub 记录不因新 mode 空间而重新解释。

## 3. 基线 B0--B8

| ID | 定义 |
|---|---|
| B0 | 原始 `conv2d_packed.vta` + TopHub incumbent |
| B1 | B2026 exact；若 P1 证据不足则为 paper-inspired hybrid |
| B2 | Random AutoTVM |
| B3 | XGB AutoTVM |
| B4 | compile-valid + SRAM/UOP 合法性 |
| B5 | B4 + DMA total bytes |
| B6 | B4 + full request-shape Pareto/ranking |
| B7 | B6 + residence mode + TopHub/B2026 incumbent protection |
| B8 | B7 + command tie-break + pipeline criticality；仅 P7/P8 使用 |

所有方法共享同一唯一候选测量池；同一候选被多方法选中只测一次。预算按总板端派发数和墙钟计，不按成功候选数计。

## 4. 冻结预算

- P3：3 个固定代表 workload；只做机制正确性和少量固定配置，不做性能寻优。
- P6 pilot：W00/W02/W09，每个最多 12 个唯一候选，总上限 36；1 boot 只写 pilot。
- P7：10 个 ResNet18 workload + 3 个几何 holdout，每个最多 24 个唯一候选，理论上限 312；策略并集去重，满足辨识要求可提前停止，但不得因结果方向选择性停止。
- 离线随机基线：同一冻结池至少 1000 个随机排列；预算点 `{4,8,16,24}`，P6 另报告 `{4,8,12}`。
- 最终性能主张：至少 3 个独立 boot；同一 boot 内的帧、repeat、ABBA block 不是独立样本。

## 5. 正确性与测量

- correctness seed：`[0, 20250901, 20260910]`，每个候选全部通过后才能计时。
- oracle：与同 workload、同量化语义的 LLVM/NumPy 逐元素参考比较；整数张量要求逐元素相等。若某图级输出为浮点，容差必须在对应 run 的 preregistration 中由旧基线误差先冻结。
- P6 计时：每候选 `repeat=5`，报告 median；候选顺序阻塞随机化，强 incumbent 前后插入。
- 主指标：达到 strongest(B0,B1) `±2%` 所需总派发和墙钟、Regret@budget、near-oracle recall、valid-board-trial ratio。
- 次指标：best-so-far、nDCG、各失败类别、特征/编译/板端/恢复时间、LOAD/STORE 分类和 command footprint。
- 2%：性能等价/保护门槛；5%：强性能提升叙述门槛。差异方向不稳定时不作提升主张。

## 6. Gate

- G1 复现：机制来源可追踪；TIR/cache lifetime 真变化；目标 LOAD/reload 至少下降 20%；正确性通过。
- G2 强基线：TopHub 10/10、canary 正确；topology B 相对历史 10.724 FPS 原则上不差超过 5%。
- G3 驻留：至少一个新模式生成正确且不同 TIR；否则停止驻留主线。
- G5 请求形态：相对 compile-valid 和 bytes-only，至少实现同等 best 的派发减半、固定预算 regret 改善或慢候选识别改善且不误删 near-oracle。
- G7 独立方法：多数 holdout 以至多一半派发达到 B2/B3 同等 best；最终值位于 strongest incumbent 2% 内；full signature 有额外贡献。
- G8 系统：完整 stage 正确；至少两个 stage 或一条流水有可重复收益才声称 FPS 传递。

## 7. 故障恢复

wrong answer、timeout 或 device/RPC error 后：立即停止批次并保存 candidate/log/boot；终止残留 runner/RPC；重载冻结 bitstream；重启 RPC；运行 TopHub canary 并逐元素核对。canary 仍失败则当前 boot 作废，等待用户重启，代理不自行重启开发板。lower、compile、wrong answer、timeout、RPC/SSH 和环境失败分别编码。

每次 run 使用 `YYYYMMDD_<phase>_<purpose>_runNN`，失败目录不删除、不覆盖。板端可用 tmpfs，但同一对照两侧日志介质和模式相同。

