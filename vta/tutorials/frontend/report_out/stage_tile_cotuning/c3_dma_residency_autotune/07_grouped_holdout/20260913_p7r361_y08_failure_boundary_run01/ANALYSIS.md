# Y08 驻留容量边界：不能使用普通 tile 容量公式代替驻留区域

依据 P7R356 static_results.jsonl 的全部 24 行及当前
vta/python/vta/top/vta_conv2d.py 的 compute_at 位置核对。
本分析没有重编译或改变冻结池，没有板端性能标签。

## 实际失败分类

| 模式 | 通过 | weight allocation 超限 | innermost padding 不支持 |
|---|---:|---:|---:|
| original | 3 | 0 | 5 |
| input_stationary | 3 | 5 | 0 |
| weight_resident_barrier | 0 | 6 | 2 |

因此不能说“8 个 weight 模式全因 SRAM 不够”：其中 F03/F07 的第一处可见
失败是 padding。通过必要容量谓词的 24 点仍只有 6 点实际 lowering 成功。

## 与实际报错一致的驻留区域公式

在本层和当前无 virtual-thread 复制的实现下，普通 tile 的权重必要界为
tile_ci × tile_co × 3 × 3 × 16 × 16 B。但驻留延长生命周期后，区域更大：

- input-stationary：CO × (tile_ci×16) × 3×3 B = 147,456×tile_ci B。
- weight-resident-barrier：CI × (tile_co×16) × 3×3 B = 73,728×tile_co B。

第二式对应 ckernel.compute_at(output, weight_residency_pt)，不再放在
conv2d 的 reduction tile k_o 内。于是驻留区域涵盖完整 CI，而非 tile_ci。
第一式与 input-prioritized 的输出通道生命周期对应；它必须以实际 lowering
为最终依据，不能对不同布局、virtual thread 或其他版本直接套用。

本地 weight SRAM 为 262,144 B。input 模式五个容量失败的 tile_ci 为
2/16/2/8/32，报错 allocation 恰好为 294,912/2,359,296/294,912/
1,179,648/4,718,592 B。weight 模式六个容量失败的 tile_co 为4/4/4/8/8/32，
报错恰为294,912/294,912/294,912/589,824/589,824/2,359,296 B。
原始错误以 bits 报告，以上统一除以8；不是 u-dma-buf 物理预留。

故本实现下可先推导必要条件 input tile_ci≤1、weight tile_co≤3；ConfigEntity
因子约束下后者候选通常只能取1或2。F03/F07 正好满足后者，却仍被 padding
规则拒绝，说明容量条件不是充分条件。

## 可探索的优化及边界

1. 将“驻留区域”而非普通 GEMM tile 容量用于廉价预筛，可以避免明确超限的
   编译尝试。这是已有编译结果支持的待验证规则；本池上评估属于开发集回放，
   不得当成新前瞻搜索收益。
2. weight 失败的可行修复方向是对 CI 再做驻留分组，使区域不覆盖完整 CI；
   但部分和必须跨组保存，可能引入 ACC/输出重放和同步，需新的模式身份与
   三张量 DMA 评估，不能只计算权重节省。不能改动现有冻结池冒充同一实验。
3. padding 拒绝发生于 transform.py 的二维 DMA 映射校验。需要改变合法的
   copy 分块或布局再做数值验证，不能删掉校验，也不能加大 u-dma-buf 解决。

结论：本层的失败来自当前驻留范围和 DMA lowering 表达约束，不能外推为
YOLO 无法运行或 FPGA 不支持权重复用。下一阶段优先验证现有六个合法候选；
驻留分组作为独立探索，不污染已冻结的搜索效率比较。
