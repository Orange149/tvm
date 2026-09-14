# P7R352：一次性模型准备与逐候选构建的真实验证

实验角色：使用已暴露 R50E 的两个前沿身份验证执行链，不新增未见搜索 holdout。
预算：300 s；boot：`737f64bf-de64-4daa-9fa8-a216db9042eb`。

## 实现与结果

新增 `run_vta_shared_preparation_candidate.py`。prepare 子进程只执行一次预训练 ResNet50
转换、量化/打包与 stock 编译，将 Relay 程序和参数持久化；验证序列化往返后结构相等、参数
键/形状/dtype/字节完全相等。各 build 子进程重新校验共享产物 manifest，恢复程序与参数，
复制已构建 stock，仅编译当前候选。各候选的 Graph JSON、参数语义、最终 fused TIR DMA 和
真实板端输出仍独立检查。主机编译产物共享不是 u-dma-buf 张量/slot 复用的新增机制。

| 外层实测阶段 | 时间 |
|---|---:|
| 一次性模型准备、stock 构建及归档 | 84.519 s |
| R50EF03 input-stationary 构建 | 30.956 s |
| R50EF03 完整上板测量过程 | 29.828 s |
| R50EF02 weight barrier 构建 | 31.857 s |
| R50EF02 完整上板测量过程 | 34.761 s |
| wrapper 总耗时 | 211.937 s |

两个候选均在预算内完成，共 12 条 stock/候选 correctness 调用记录、28 条配对计时调用记录。
各自三 seed 正确且七轮计时输出一致，correctness 与 timed profiler 的 DMA 差均通过最终 TIR
对账；候选/stock 配对比为 1.130833、1.985523，两者仍慢于 stock。

共同阶段内部的模型准备耗时 52.742781 s，stock 编译耗时 28.347412 s。候选阶段的共享产物
恢复与 stock 复制分别只需 0.667291、0.791773 s；候选编译本身分别为 28.367102、29.141809 s。
`stock_rebuilt=false`、`model_reprepared=false`，两个候选均绑定同一共享 manifest。

## 对调优实验的意义

当前成本可以按明确执行事件分解为一次性 prepare 与逐候选 build/measure，避免为每个候选重新
准备模型和构建 stock。但此处尚无对这两个候选按相同顺序、条件运行的重复准备对照；不能拿
P7R351 单候选时间乘二后当作实际测量基线，更不能声称取得某个比例的 tuner 加速。

所有排序策略下一轮都必须支付一次相同准备成本，并以 wrapper 总耗时与预算内完成的真实标签
评价。第一次测量约在 145.319 s 完成，第二次在 211.937 s 完成；这只是本次事件时刻，不是
150/200 s 截断预算的独立新实验。

既有算子资格仍在本集成预算外。该版本适配器明确标为 exposed-label development，尚未接入
目标标签前冻结的完整搜索选择过程。解释器启动与最终报告序列化仍在 wrapper 计时边界外。

## 验证与终态

`shared`、两个 `candidate`、两个 `board` 与 `execution` 六个目录合计 717 个 manifest 条目
全部重算 SHA-256 通过。最终 FPGA operating、u-dma-buf 201326592 B、RPC 位于 tmpfs，
boot 未变化，未发现新的存储错误。本轮代码使用实际序列化往返、真实编译与 FPGA 执行进行验证。

完整产物：`07_grouped_holdout/20260913_p7r352_shared_prepare_two_candidate_board_run01/`。

## 下一未见 workload 的选择前检查

顺带检查发现 `prepare_vta_model_conv_adaptive_holdout.py` 调用的旧 `exposure_audit()` 只覆盖
早期固定来源，返回 14 个几何，并不自动纳入后续所有实验。对已有合同递归读取 workload，已经
能找到 24 个几何。因此旧函数返回“未冲突”不足以单独证明目标未见，下一次选择需要补上后续
合同查重。这个发现不自动否定既有前瞻结果，每个既有声明需按实际历史来源判断。

例如 YOLOv3-tiny 的 `CI1024→CO256,13×13,1×1` 出现在 P7R115、P7R125/126 和 P7R167--169
合同中，属于 Y01，不能重新标为未见几何。共享准备路径已跑通后，下一必要动作是完善查重范围，
选定真实未见目标，再执行含完整成本与 Random/bytes/calls 对照的预算实验。
