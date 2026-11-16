# 固定并发的 VTA host 优先级因果验证

控制器：`run_cpu_vta_host_priority_validation.py`；分析器：`analyze_cpu_vta_host_priority_validation.py`。
源证据：完成并冻结的 `cpu_vta_driver_snapshot_validation/20260917T134930Z_driver_snapshot_v3`。
本轮使用新协议、新派生 runner、新 tmpfs 和自有 RPC；旧 runner/header/控制器、驱动库、bitstream、图与候选评分全部保留。

## 预先固定的对照

同图 `CPU00:02 → VTA03:17 → CPU18:20`，CPU-only gate exclusive，两CPU阶段互斥、CPU/VTA仍重叠。
请求线程4/1/4、stage/host四核mask、spin=0、普通复制、queue depth=2、真实输入与完整边界参考。
N0 VTA caller nice=0；N1仅初始化后的自有VTA caller nice=−10。其他所有已创建线程nice=0。
不能把请求服务线程数当作实际已创建线程数：实际线程池4/4/4，初始化进程任务16个；每帧核查3个caller的TID与nice。

## 运行与资格

```sh
python run_cpu_vta_host_priority_validation.py freeze --run-dir <new-directory>
python validate_cpu_vta_host_priority_host.py --run-dir <new-directory>
python run_cpu_vta_host_priority_validation.py pilot --run-dir <new-directory> --execute-board
python run_cpu_vta_host_priority_validation.py analyze --run-dir <directory>
```

每boot先独立串行/N0/N1各10帧完整数值核对，前后各20帧匹配串行检查漂移≤5%。
性能307帧、丢前7帧，300完成时间戳/299间隔，固定顺序：

1. N0快照关、N0快照开、N1快照关、N1快照开。
2. N1快照开、N1快照关、N0快照开、N0快照关。

关闭逐提交日志、dump/checkpoint、性能稳定性睡眠。快照仍只使用已存在driver累计API、CLOCK_THREAD_CPUTIME_ID和schedstat；不改驱动。
各优先级分别检查快照开/关II与VTA run扰动≤2%；正式性能用快照关闭会话，快照解释排队与driver分层。
本boot探索性信号是VTA run至少降低5%、runnable等待下降且轨迹一致；不要求II提高。
运行队列、driver/poll和硬件进度相互重叠，不能相加或直接补到II/历史34ms。

合格机制再补两个真实kernel boot，恢复已验证环境后执行：

```sh
python run_cpu_vta_host_priority_validation.py replicate --run-dir <same-directory> --execute-board
```

RPC重启和bitstream重载不算boot；控制器拒绝相同kernel UUID，以及已失败协议内静默重试。
本协议boot交付采用不可覆盖的独立目录和当时资产清单；根timeline为可追加行动记录，各次交付另存其不可变快照，不提前封存仍需后续boot的整个协议。
三boot报告以每boot正反配对的均值为采样单位，Student-t df=2；因果确认还需方向一致、中位变化≥5%、95%CI排除零以及时间线一致。
旧并发开关的boot不能代替本次优先级对照。优先级也可改变CPU执行/内存访问时序；硬件compute-only、物理DDR/DMA保留unknown。

正确性失败、超时、实际nice/affinity/身份漂移均停止受影响实验并保留部分结果。权限不足不修改系统服务或全局调度。
清理只停止本协议自有RPC并移除已归档的自身session，恢复启动前firmware链接；不清理任何他人进程、旧协议tmpfs、SD部署或实验结果。
