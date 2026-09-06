# Stage 4a-Q Native Qualification

日期：2026-08-31

## 结论

Stage 4a-Q 单 session 资格验证通过。该结论只允许冻结并进入 Stage 4b 三 session 正式校准，
不等于正式 `service_model.json` 已经生成。

- Board：`root@192.168.1.240`
- 性能路径：native static packed function；RPC：`false`
- Cases：`36/36` 成功，`0` failure，全部 correctness 通过
- Protocol SHA256：有效
- Contiguous design：rank `5/5`，condition number `19.584`
- Constrained diagnostic fit：`R2=0.9973`，MAE `0.737 us`
- Leave-one-size-out：median APE `2.964%`，P95 APE `6.678%`
- Sample total wall range：`25.929–89.773 ms`，满足 `20–100 ms`
- Matched access pairs：`8/8`，ALU/output/store 不变量成立
- Compute sweep：DMA 固定，ALU slope `5.752 us/repeat`，`R2=0.9996`

本次资格 session 的拟合带宽为 load `1.414 GB/s`、store `0.954 GB/s`。这些值只是单 session
资格结果，不能直接作为论文正式参数；Stage 4b 仍需三个独立重启 session 和分层不确定性汇总。

## 首轮失败与修复

首轮在第一个 case 即发现 correctness 失败，因此在 9 个 case 后中止，没有把错误数据混入正式
资格结果。根因不是 timeout 或 DMA buffer，而是最初用于保持相同 ALU opcode 的 identity 临时
buffer 与 VTA in-place ALU 语义不一致。

修复方式不是放宽 correctness：对单输入 `x0+x0`，在 VTA `InjectALUIntrin` 后用受限 TIR pass
把编译器生成的 `MUL immediate 2` 改写为等价的 `ADD dst,dst`。该 pass 只匹配 opcode、
`use_imm=1` 和 `imm=2` 的完整模式。修复后单 case smoke 通过，完整 36-case session 全部通过。

失败尝试保存在 `session1_failed_checker_before_uop_rewrite/`，有效资格结果保存在 `session1/`。

## Matched Access 结果

| Access | Shape | Output count | Observed delta us | Byte-adjusted delta us |
|---|---|---:|---:|---:|
| padded | `8x16,pad1` | 1 | 3.062 | 5.415 |
| padded | `8x16,pad1` | 2 | 1.747 | 4.100 |
| padded | `16x16,pad2` | 1 | 4.363 | 10.879 |
| padded | `16x16,pad2` | 2 | 3.992 | 10.508 |
| strided | `8x16,stride20` | 1 | 4.036 | 4.036 |
| strided | `8x16,stride20` | 2 | 3.798 | 3.798 |
| strided | `16x16,stride24` | 1 | 8.091 | 8.091 |
| strided | `16x16,stride24` | 2 | 8.366 | 8.366 |

Padded 的 direct delta 不能直接命名为 padding penalty，因为 control 读取预补零后的更大 tensor；
Stage 4b 必须使用 byte-adjusted residual。Strided pair 的 load/store bytes 相同，direct delta 已经
是严格 matched effect。

## 持久产物

- `session1/measurement_protocol.json`
- `session1/case_status.jsonl`
- `session1/dma_identification_raw.csv`
- `session1/dma_identification_summary.json`
- `session1/DMA_IDENTIFICATION_REPORT.md`
- `session1/package/manifest.json`
- `session1/board_results/*.jsonl`

远端资格运行目录已清理；开发板上没有残留 runner。

## Gate

Stage 4a-Q：**通过**。

Stage 4b：**尚未执行**。正式协议不得复用本 session 作为三个正式 session 之一。
