# P7R450 状态

状态：`INVALIDATED_BEFORE_COMPLETION; RANDOM_SEED_CONTRACT_INCOMPLETE`

执行中核对发现 AutoTVM ConfigSpace 的初始随机选择使用 Python `random.randrange`，而执行器只设置
了 NumPy seed；命令行声明的 XGB seed 因此没有完全约束候选顺序。该会话被人工停止，不计入正式
重复。执行器已同时固定 Python random 与 NumPy random，后续会话才可按 seed 分组比较。
