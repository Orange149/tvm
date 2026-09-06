# Stage 4a-R Calibration Design Repair

日期：2026-08-31

## 范围与结论

本阶段只修复 RAMPS 校准实验设计并完成本地验证，没有连接开发板，也没有产生新的带宽、
GOP/s 或正式硬件参数。Stage 4a-R 已完成；后续 Stage 4a-Q 已在板端通过，结果见
`../stage4a_q_qualification/STAGE4A_Q_QUALIFICATION_REPORT.md`。

修复后的设计不再用不同 ALU 运算解释 DMA load 数量，也不再把 padded/strided case 与不等价
的 contiguous case 比较。所有原始 measurement 都携带 `measurement_kind`、`includes` 和
`excludes`，综合 stage GOP/s 明确为黑盒诊断字段。

## Matched Case 对照

| 目标变量 | Control | Treatment | 保持不变 | 唯一目标差异 |
|---|---|---|---|---|
| input load 数 | `rhs=identity(x0); x0+rhs` | `rhs=identity(x1); x0+rhs` | ALU opcode/calls、output、store | 1 个或 2 个外部 input load |
| output store 数 | 1 个 output | 2 个相同 output | input load、ALU、shape、dtype | store bytes/calls |
| strided load | source shape 等于 output 的 contiguous input | 更宽 source 中读取相同 output window | ALU、physical output、store | source row stride |
| padded load | 预先补零的扩大 tensor contiguous load | 小 tensor 经 VTA padding 形成相同扩大 tensor | 数值语义、ALU、physical output、store | padding access mechanism |
| VTA compute | 固定 shape/input/output 和 DMA | ALU repeats 为 `1/2/4/8` | load/store bytes/calls | 递增 ALU work |

单输入的数学语义仍是 `x0+x0`，双输入仍是 `x0+x1`。直接写 `x0+x0` 会被 TVM 简化成
`x*2`。最初尝试用独立 identity 临时 buffer 保留 ADD，但 Stage 4a-Q 首轮 smoke 暴露出 VTA
ALU 是 in-place：该临时 buffer 并不是有效的数据复制。最终实现增加一个严格受限的 TIR pass，
只把 `MUL immediate 2` 改写成 `ADD dst,dst`；双输入仍是普通 `ADD dst,src`。实际 lowering 与
板端 correctness 均确认两组 opcode/calls 相同。

Padded control 的物理输入已经是扩大后的补零 tensor。例如 treatment `8x16, pad=1` 与
control `10x18 pre-padded` 都输出 `10x18`，并执行相同 ALU 和 store。二者的时间差才可用于
识别 padding 访问机制；后续分析还会用 contiguous bytes/calls 模型消除两者 load bytes 差异。

## 实现修改

- `calibrate_vta_dma_native.py`
  - case matrix 从旧 24-case 改为 36-case matched design。
  - 增加固定 DMA 的 ALU compute slope microkernel。
  - case 顺序由固定 seed 随机化，协议保存顺序和 SHA256。
  - 每个 case 独立记录状态；失败不终止其余 case；结果逐 case 拉回并在 `finally` 清理远端。
  - analyzer 可保留部分成功结果，并单独输出 failure rows 与 matched-pair delta。
- `vta_dma_microbench_runner.cc`
  - `BATCH`、`BLOCK_OUT`、logical shape 和 fill policy 从 manifest/命令行读取。
  - 删除 shape/checker 中的 `1x16` 硬编码。
  - 支持 pre-padded contiguous control 和 ALU repeat correctness。
- `calibrate_resource_cost_model_native.py`
  - 将综合吞吐字段改为 `effective_stage_gops`，并标记
    `measurement_kind=effective_stage_black_box`、`diagnostic_only=true`、
    `eligible_for_physical_compute_model=false`。
- `resource_aware_maxplus.py`
  - 新增严格 component-level compute calibration 入口；黑盒 stage GOP/s 会被拒绝，避免再与
    DMA、submit/sync 重复计费。

## 本地验证结果

- `python -m py_compile`：通过。
- 单元测试：`33 passed`。
- C++ native runner：AArch64 交叉编译通过；只有 TVM/dmlc 既有 logging macro warning。
- 36/36 个 VTA module：全部完成本地 build/package。
- 单/双输入 lowering：VTA ALU opcode 序列相同。
- Compute sweep lowering：load/store 指令恒为 `1/1`，ALU uop 为 `1/2/4/8`。
- 动态 geometry：测试覆盖非默认 `BATCH=2, BLOCK_OUT=32`，case elements 不再按 16 写死。
- Partial failure：测试确认缺失 case 生成 `profile_missing` failure，已有结果仍生成 summary。

本地完整 package 位于 `/tmp/ramps_stage4ar_local_build`，仅用于构建验证，不是持久实验数据，
也不包含任何板端性能结论。

## 尚未解决与 Stage 4a-Q 建议

1. 目前 compute sweep 是 ALU slope；正式 Stage 4b 仍需要独立 GEMM slope，以覆盖 conv-heavy
   VTA bucket。不要把 ALU slope 外推成 GEMM GOP/s。
2. `device_run_wait_us` 是否能在 busy-poll 下稳定分离短 kernel，必须由 Stage 4a-Q 的 20--100 ms
   inner-repeat 累计时间验证。
3. Matched design 在结构上成立，但实际 profiler counters 是否保持预期不变量仍需一次上板检查。
4. Padded pair 的直接时间差包含 physical load bytes 差异；正式参数必须报告 byte-adjusted residual，
   不能把直接差值命名为固定 padding penalty。
5. Stage 4a-Q 应只运行一个 session，检查 correctness、TIR/manifest 不变量、设计 rank、condition
   number 和 leave-one-size-out error。失败时只修改设计，不启动 Stage 4b 三 session。

## Gate

Stage 4a-R：**通过（local design/build validation）**。

Stage 4a-Q：**已通过**。本文件保留 Stage 4a-R 的设计修复记录；资格验证结论以
`../stage4a_q_qualification/STAGE4A_Q_QUALIFICATION_REPORT.md` 为准。
