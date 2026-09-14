# P7R451 状态

状态：`PRE_T0_PREFLIGHT_ABORT; NO_SEARCH_OR_LABEL`

上一次人工中断发生在 RPC 重载期间，板端当时没有运行 RPC。执行器的初始 clean-start 仍假设存在
PID，在 T0 和搜索开始之前退出。没有候选测量或性能标签；已修正为空 PID 时直接重载 bitstream 并
启动新的 default RPC。
