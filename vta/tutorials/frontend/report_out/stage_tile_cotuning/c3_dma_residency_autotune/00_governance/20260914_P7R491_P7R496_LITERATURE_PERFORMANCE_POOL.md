# P7R491--P7R496：文献性能基线补充池与统一板池

状态：`MERGED_214_POINT_POOL_CROSS_COMPILE_COMPLETE; NO_FPGA_OR_LATENCY_LABELS`

## 1. 为什么必须补池

P7R482 的 50 点池足以执行原计划的标签隔离搜索确认，但不足以证明 HW-Aware 的“初始化后接性能
调优器”：20 个 balanced E0 的合法 original 候选并集在 H1/H2/H3 分别为 96/53/25 个，而与原
50 点池中的 original 身份只重合 1/2/1 个。若不补测，只能报告 valid yield，不能评价 E0 的最终
性能质量。

因此 P7R491 在仍未产生任何 R18 FPGA/latency 标签时，重新确定性生成 20 seeds 的邻域反馈，保存
每个 seed 的 25 个 valid E0 成员，并对 174 个唯一 original ConfigEntity 展开 Cheng 四模式。
选择使用了 P7R484--P7R486 的 original lowering validity，但没有使用 FSim、FPGA 或性能标签。

## 2. 补充池资格结果

| workload | E0 original并集 | 四模式身份 | lowering pass | FSim pass |
|---|---:|---:|---:|---:|
| R18-H1 | 96 | 384 | 176 | 96 |
| R18-H2 | 53 | 212 | 82 | 34 |
| R18-H3 | 25 | 100 | 77 | 52 |
| 合计 | 174 | 696 | 335 | 182 |

另外有 255 个真实 lowering 失败；H2 的 106 个 weight/combined 身份因完整权重 589,824 B 超过
262,144 B weight SRAM，被明确标为 `not_applicable`。335 个 lowering-pass 身份均执行三 seed
FSim，182 通过、153 失败。整个 P7R492 墙钟为 848.410 s，失败成本没有隐藏。

按 same-tile family 统计：

- H1 有 14 个 original/input 对、6 个 weight/combined 对，但没有四模式全合法 family；
- H2 有 9 个 original/input 对，两个权重驻留模式均按资源回退；
- H3 有 11 个 original/input 对、14 个 weight/combined 对，并首次得到 1 个四模式全部 FSim-pass
  的完整 family。

因此 Cheng 表可以忠实报告一个完整四方案对照和更多双模式/失败边界，但不能把所有 family 写成
四方案均可执行，也不能预设 combined 最快。

## 3. 统一板池

P7R493 冻结补充 FSim-pass 池 182 点。它与原 P7R482 的 50 点有 18 个 exact candidate 重合；
P7R494 按候选 SHA-256 去重并核对 ConfigEntity、TIR、DMA、command signature，冻结为 214 个
唯一身份：

| workload | merged candidates |
|---|---:|
| R18-H1 | 104 |
| R18-H2 | 44 |
| R18-H3 | 66 |

P7R495 首次交叉编译未加载 Xilinx SDK，前 35 个均在 worker preflight 报
`SDKTARGETSYSROOT is unset`。运行被主动停止，目录保留 `invalid_session.json`，不计为候选失败，
也不得与后续结果拼接。

P7R496 在 `unset LD_LIBRARY_PATH` 后加载同一 aarch64 SDK，从头执行 214 点交叉编译，结果
214/214 成功、0失败，保留 214 个精确 `.so`，完整墙钟 294.496 s。P7R491--P7R496 的所有有效
artifact hash 均已复核通过。

## 4. 板端执行合同

板端恢复后只执行 P7R494+P7R496 的统一池，不再分别运行 50 点和 182 点池：

1. W0 开始严格 SSH/存储/bitstream/default-runtime preflight；
2. T0 为 clean bitstream 和 fresh RPC 就绪；
3. 214 个身份逐个做三 seed fail-fast correctness；
4. 所有正确身份完成后，按 workload 做五轮平衡交错 timing；
5. 全池结束后才连接各 workload oracle；
6. 任何断电、RPC 或执行中断使整个 session 作废，不拼接；
7. 逻辑 LOAD/STORE 与 driver invocations 全部累计，但不称物理 AXI traffic。

当前唯一外部阻塞仍是 Dropbear host key 已变化且未由串口重新认证。交叉编译成功不等于 FPGA
correctness 或性能；在新公钥完成严格核验前不得启动 P7R497 板端 session。

