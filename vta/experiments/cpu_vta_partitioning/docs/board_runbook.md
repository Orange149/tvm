# 第二阶段：集中上板手册

## 开始前

电脑安装本实验 Python 依赖并可通过 SSH 公钥登录板子，SSH host key 已配置。板子需要 aarch64 Linux、支持 gzip 的 `tar`、`sh`/`awk`/`sha256sum`/`nohup`/`insmod`/`install`/`renice`、FPGA manager 和匹配内核的 u-dma-buf。不要同时运行旧 RPC 或其他 VTA 实验。电脑保留完整包及全部原始数据；板端只保存执行文件和尚未回收的会话。

v1 部署包包含统一工具链身份和 SDK 依赖来源记录。更换 SDK 后在电脑重新 prepare 并新建 run ID；上板不需要安装该 SDK，包内加载器和共享库已完成离线闭包校验。

`configs/local.json` 的 board 地址、用户名、端口、remote_base 必须对应当前板子。可另加 `identity_file`。凭据不打包，连接配置仅存入本机 `runs/<run-id>/state.json`。

SSH 管理命令显式设置 `PATH=/usr/sbin:/usr/bin:/sbin:/bin`，使非交互登录环境也能找到 `insmod` 等系统工具；原生测量仍使用冻结任务中的独立环境。

完整归档约 1.1 GiB，其中参考结果、源码和分析资料留在电脑。`plans/board_payload.json` 冻结约 850 MiB 的板端执行文件及哈希。上传使用 gzip tar 流直接解压，板端不保存传输 tar；首次检查执行文件大小加 256 MiB 余量，每个会话另检查按帧数、阶段数和张量形状估算的输出空间。256 MiB 是磁盘余量，不表示已验证峰值运行内存。失败会话保留，后续空间不足时停止，不自动删失败证据。

示例 `storage_mount=/media/cpu_vta_data`、`remote_base=/media/cpu_vta_data/cpu_vta_partitioning`。必须对应板子上实际健康可写的持久挂载，部署前检查挂载存在且可写；实验入口不会自动格式化、挂载或退回根文件系统。使用 initramfs 根文件系统时，用户每次重启后由电脑操作人员核验分区身份并重新挂载，再调用 resume。若使用其他 SD/eMMC/SSD，同时修改这两个字段。历史 `/var/volatile` 约 1.9 GiB，改用它会消耗 RAM，意外重启也会丢失尚未回收的失败数据。源端 `legacy_rsa` 和 `scp_legacy_protocol` 对应既有 PetaLinux SSH 兼容需求，可按实际服务关闭。

```bash
CPUVTA_BUNDLE=$(python -c 'import json; print(json.load(open("artifacts/LATEST_BUNDLE.json"))["path"])')
python run.py validate --bundle "$CPUVTA_BUNDLE"
python run.py board run --bundle "$CPUVTA_BUNDLE" --run-id resnet18_v1_main_01 --execute-board
```

启动入口仅复制冻结的板端文件集合，逐文件校验 SHA256。中断后按校验结果补传缺失或变化的文件。环境初始化加载 `udmabuf0=201326592`，检查 `/dev/udmabuf0` 和 sysfs 中实际 192 MiB 大小；模块存在但设备或容量不符时停止，不强行卸载驱动。初始化和测量使用同一个启动范围内的会话锁，重置位流前设置完整配置 flags 并检查 operating 状态。

程序接着做完整输出、逐帧输入序列和实际 affinity/活跃线程数检查。资格检查必须覆盖规定的全部输入，回收的输入清单必须与任务一致。任何失败会保留原始输出并停止；不会仅凭 top-1 一致接受候选。

## 启动顺序与重启

| 有效启动序号 | 工作 |
|---|---|
| 1–2 | 组件标定，各两轮；第 1 次还检查各编译阶段完整输出与预热 |
| 3–4 | 七组 Top-5 并集与分层随机候选筛选，各两轮 |
| 5–10 | 冻结后的预算 1/5 独立确认，各两轮；另有共享/专用核及日志扰动诊断 |

每次程序显示 `WAITING_NEW_BOOT` 后由用户真正重启板子，再执行：

```bash
python run.py board resume --run-dir runs/resnet18_v1_main_01 --execute-board
```

复用 boot ID 会继续等待，不把测试进程重启计作独立启动。主实验最少 10 个有效启动（若第一次已是干净启动，则需额外手动重启 9 次）。进入敏感性后增加 50 个有效启动，最多 60 个。程序不会调用 reboot。

每次新启动都重新验证板端文件集合和环境。组件成本、Top-20、筛选选点和诊断选点由已完成的前置阶段生成并冻结，集中上板过程不调用 TVM 编译。每个确认启动的诊断额外核验共享/专用配置，并在三个诊断块前后各测一次锚点；不会增加有效启动数量。

## 进度和数据

```bash
python run.py status --run-dir runs/resnet18_v1_main_01
python run.py analyze --run-dir runs/resnet18_v1_main_01
```

`state.json` 记录当前环境、阶段、启动索引与已完成 boot ID；`events.jsonl` 追加记录开始、完成和失败。`raw/<boot>/<task>/transfers/<snapshot>/` 保存每次回收的归档、输出和 `capture.json`；不完整快照不参与统计。`accepted/` 是验证通过的窗口凭据，每次使用时重新验证文件哈希。完整结果和凭据写盘后才清理板端输出，板端留下任务身份、退出码和回收标记。清理失败写入 `cleanup_pending/`，后续 resume 重试；不重新执行已接受的窗口。

`envN/` 包含 costs、ranking、selection、diagnostic_selection、warmup 和分阶段轮次。完整确认后自动生成 analysis.json；`analyze` 再生成预算 1/5 CSV、模型工作量 CSV、performance.pdf/png、report.txt 和总成本记账。确认或诊断未满六次启动时只报告 INCOMPLETE。

分析前核验当前控制器源码、Python 和统计依赖版本与 run、部署包的身份一致。身份不符或缺少 `state.json` 时，在写入任何输出前拒绝；未完成实验也须通过此校验。分析结果与成本记账的 `provenance` 保留 bundle ID 和控制器身份。分析已有批次须使用该批次冻结的源码和依赖环境，不要修改 state 中的哈希来绕过检查。

`report.txt` 单列主机工作消融、局部 CPU 容量约束收益、机制证据及通过全部主要比较的预算。“继续方法研究”与敏感性推进采用同一门槛；不能由某个预算胜过 G、另一个预算胜过 D 拼出继续结论。

`cost_accounting.json` 区分已接受帧数 `accepted_native_frames` 和包含失败会话的已观测完成帧数 `observed_completed_frames`。重复回收按启动、任务及帧 ID 去重。缺失或不完整会话使 `actual_native_frames=null`、`native_frames_status=lower_bound`，未知任务列在 `unresolved_sessions`；不能把零观测解释为零成本。`distinct_observed_boots` 包含失败启动，`distinct_accepted_boots` 只包含有接受凭据的启动。

测量汇总通过 `_receipt` 路径及哈希关联接受凭据和原始文件。分析核验每次确认的两轮记录、六个不同 boot ID，以及诊断与确认启动的一一对应；还会核验候选/诊断条件、锚点和任务环境。文件缺失、哈希变化或身份不符会报错停止推断。不要只复制 `envN/` 后分析，必须保留同一 run 下的 `state.json`、`accepted/` 和 `raw/`。

## 恢复规则

SSH 断开或电脑进程退出：保持同一 run 目录，恢复连接后直接 resume。远端采用 nohup 和固定 task hash。恢复入口先轻量查询会话锁，等待会话结束后才执行清理、部署校验及回收；这些重型操作也持有同一个锁，避免检查后的竞争。原生会话超时 600 秒，控制器等待上限 660 秒。

等待超时或传输异常时，程序尝试在空闲状态下回收日志，不要求存在 exit_code；锁仍占用或连接不可用时，电脑端的 `capture.json` 记录回收暂缓/失败，不在板端打包正在运行的窗口。持久目录中的原始数据在恢复连接、会话结束或真实重启后再回收。等待锁超时不会强行清锁或启动重型操作，可等待会话结束后 resume，或检查异常后真实重启。禁止把不完整快照当成完成窗口。

原生执行器非零退出、数值错误、缺帧或 affinity 错误：停止当前批次并检查对应 native.log/result.jsonl。不要删除失败会话以伪装成功。需要重置设备后重启板子再 resume；意外重启将把未完成的旧启动整体归档为 abandoned，新的 boot ID 重做该启动，不混合两次启动的轮次。

锚点漂移 >5%：正式实验作废整轮并最多补两轮；诊断作废对应四会话块并最多补两块。补测使用不同任务身份，失败证据保留。仍失败要求重启板子。确认六个有效启动之外的失败启动另计成本。

哈希不符、程序或控制器变化：不可把更换后的代码继续混入旧确认结果。重新 prepare、新建 run ID；旧数据保留。若仅 SSH 中断且哈希未变，不需要新建批次。

远端 `locks/<boot-id>/session.lock` 记录 boot ID、PID、进程启动时刻和任务身份。同一启动的锁不会仅凭 PID 消失就自动回收，因为子执行器可能还在运行。电脑异常退出时可等待远端完成再 resume；wrapper 被强制杀死导致锁遗留时真实重启后 resume。新启动使用独立锁路径，旧启动锁和失败文件保留为证据，无须清空持久目录。意外重启后控制器也会尝试回收未完成旧启动的文件，并将该启动整体作废。

wrapper 收到 HUP/INT/QUIT/TERM 时保留锁，避免不同 shell 的 EXIT trap 行为导致提前清锁。部分 shell 等子任务结束才处理信号，因此等待期间也持续占锁；信号处理后遗留的锁需要真实重启恢复。正常会话结束和普通命令失败仍会释放锁。

资源敏感性只在主实验达到协议内继续条件时自动推进。`COMPLETE` 但未运行全部环境，可能表示主实验未达到继续门槛，分析文件会列出原因；这不是把非显著结果当等效。
