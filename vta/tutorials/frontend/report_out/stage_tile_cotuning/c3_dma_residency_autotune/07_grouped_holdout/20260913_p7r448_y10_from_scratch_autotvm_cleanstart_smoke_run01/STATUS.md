# P7R448 状态

状态：`CLEAN_START_ISOLATION_SMOKE; NO_VALID_CONFIG_IN_FOUR_TRIALS`

四个空历史 XGB 随机候选均在本地 lowering 失败，没有候选接触 FPGA，也没有生成整图结果。该运行
验证了编译失败计入 gross proposal 且不会破坏 RPC，但 trial 数太小，不能作为性能或搜索成本证据。
