# 额外48候选池的成功lowering点：三seed FSim资格

P7R368/P7R369的8个lowering成功身份全部重新提取静态信息，并逐个核对
TIR哈希与原baseline一致；没有选择性排除候选。

- 6个身份通过seed 0、20250901、20260910，合计18次逐元素正确性检查，
  每个身份的三seed命令证据均为available_three_seed_consistent。
- capacity_check_26的original与weight_resident_barrier在warmup即被
  runtime.cc:1264重复dst_idx检查拒绝，没有完成三seed数值比较。
- 老helper将这些失败归入wrong_answer，但真实观察是UOP依赖检查异常，
  不能解释为已经测到输出误差。原始失败对象和日志完整保留。

因此本池资格链为48个候选→8个lowering成功→6个FSim成功；不是8个全部
正确。容量预筛仍保留全部6个FSim正确身份，但不能替代UOP依赖或数值检查。

这6个是额外容量验证池的身份，不是原P7R357正式搜索池的6个身份；不得
混用、替换原池，也不增加独立几何数量。没有FPGA或性能标签。

本轮SSH复查仍返回No route to host。真实板端资格与完整搜索评价等待连接恢复。
