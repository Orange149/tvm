# CPU–VTA 核心分离验证协议

复用已封存 nice 协议的原始 runner、运行库、模块、图、权重和真实输入，建立独立协议和板端 tmpfs。CPU00:02 → VTA03:17 → CPU18:20，线程始终 3/1/3，CPU 首尾两段使用核0–2且 CPU-only gate 互斥；S 的 VTA host 使用核0–2，D 使用核3。所有线程 nice=0、spin=0、普通复制、queue depth=2。进程继承四核 mask，不传会覆盖阶段配置的全进程 affinity 参数；初始化后核查实际池线程 mask 和 runtime.NumThreads。

每个 boot 分别完成 S/D 串行及流水各10帧全输出校验，再各20帧匹配串行基线。配对性能两个 block：S off、S on、D off、D on；逆序。每会话307帧，前5预热后再丢2，300个完成时间戳计算299个间隔。性能关闭 sleep、checkpoint、输出dump及逐提交日志，延后写出仅缓存JSON。正式性能使用 snapshot off；on/off任一 II 或阶段run 扰动超过2%，on仅诊断。

配对后 S/D 各20帧串行 checkpoint；任一阶段相对前基线漂移超过5%立即停止。如果 D 首段或尾段 run 均值仍超过匹配 D 串行均值5%，预设追加同 mask、同3/1/3的 all-stage gate 对照 D_all：先10帧输出校验，再 D/D_all 正反两个关闭快照性能配对，最后补 S/D 串行漂移检查。这只检验共活动效应，不能独自证明 DDR。

阶段服务变化至少5%且轨迹符合互斥及共活动关系时考虑补足三个真实 kernel boot。每 boot 正反配对；旧 nice/线程/全阶段协议的 boot 不计入本协议。三 boot 配对95%置信区间用 Student-t df=2，方向一致、中位幅度至少5%且区间排除零才确认对应机制；单 boot 为探索性。

输出原始JSONL、身份哈希、匹配串行服务、均值资源组合 max(CPU首段+尾段,VTA)、实际II、重叠和空洞、队列/快照诊断及中文结论。原代理67.613596ms属于4/1/4，作为历史诊断引用，不能称为本3/1/3配置的精确静态预测。DDR及内部DMA、纯硬件计算时间无法测得时保持 unknown；不改静态公式或候选排名、不拟合34ms常数。

阶段：freeze / pilot / replicate / analyze；板端命令必须显式 --execute-board。正确性、超时、身份或漂移失败立即停止，保留部分数据，不静默重跑。结束停止本协议RPC并核查，仅清理已归档的本协议会话，不动旧目录。未完成未来boot的协议交付采用每boot快照哈希清单，最终完成后封存全协议。
