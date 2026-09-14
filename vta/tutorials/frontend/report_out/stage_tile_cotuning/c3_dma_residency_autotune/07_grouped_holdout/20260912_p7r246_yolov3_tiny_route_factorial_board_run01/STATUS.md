# P7R246 四执行器并存失败记录

状态：`HARNESS_SHARED_MEMORY_ALLOCATION_FAILURE; NO_CANDIDATE_CLASSIFICATION`

该运行在同一个 clean-start RPC 中创建并加载四个完整 YOLOv3-tiny graph executor：全 stock、仅
Y00、仅 Y02、Y00+Y02。三输入正确性阶段的第一个 `stock_all` VTA 调用尚未完成，runtime 即在
`vta/runtime/runtime.cc:1309` 触发：

```text
Check failed: (fpga_buff_ != nullptr) is false
```

失败发生在四个 executor 已构造之后、任何可用候选输出或计时之前。此前同一 boot 的 P7R243 已能
同时保留三个 executor 并完成 9 次正确性与 18 次计时调用。因此本运行只记录“测试器的四版本并存
超过当前 192 MiB u-dma-buf 分配路径”，不能将它计为 stock、Y00、Y02 或组合候选的 FPGA
correctness 失败，也不能据此诊断为 u-dma-buf 驱动缺陷。

后续 P7R247/P7R248 使用每次只加载两个 executor 的 clean-start factorial edge，避免测试器自身
的空间占用污染候选结论。
