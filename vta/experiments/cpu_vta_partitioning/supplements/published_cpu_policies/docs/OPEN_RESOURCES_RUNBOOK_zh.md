# 九种资源上下文：准备与上板

入口为 `run_open_resources.py`。原 `run.py`、旧控制器快照、原始数据和已完成批次均保留。当前修复批次默认名为 `resnet18_open_cpu_resource_02`；应按下方示例显式复用第一批的有效标定，避免重新采集261项成本。

## 先完成本地准备

```bash
python run_open_resources.py prepare --profile-source runs/resnet18_published_policy_dp_pilot_01
python run_open_resources.py validate --bundle artifacts/bundles/<bundle-id>
```

若前一批已完成261项补充组件，而执行器参数或静态算法修复需要新控制器，显式指定标定来源：

```bash
python run_open_resources.py prepare --profile-source runs/resnet18_published_policy_dp_pilot_01 --calibration-source runs/resnet18_open_cpu_resource_01
python run_open_resources.py board run --bundle artifacts/bundles/<new-bundle-id> --run-id resnet18_open_cpu_resource_02 --execute-board
```

准备会用来源冻结控制器逐项重放261项组件的测量与正确性凭据，以及已有的0–87项有效流量凭据，复制完整证据闭包。新包中775项组件全部标为引用；流量只引用实际通过验证的条目，缺失项继续采集。来源程序、执行器、任务、boot和bundle身份均保留，不改写成新测量。原批次、原控制器及失败现场保留。来源首个流量任务若执行失败，没有有效凭据时复用数量就是0，不视为已完成87项。

`prepare`和`validate`不连接开发板；`board run`才部署并执行。准备过程核验172个现有模块、运行库和执行器，复制输入0与参考结果，将514项旧组件的原始回收凭据和数据作为来源归档；旧boot ID和task ID不会改写。CPU同核数集合的置换、均分逐线程核时属于静态近似，不能称为所有核心集合均已实测。

`OFFLINE_READINESS.json` 的 `PREPARED_OFFLINE` 只说明本地代码、依赖闭包和测试通过，板端绑核、输出和性能仍待验证。新控制器快照位于包内 `sources/open_resources_controller/`，可从该目录调用同一入口校验和续跑；部署包的板端执行闭包不包含历史大体积来源数据。

## 一次启动运行

```bash
python run_open_resources.py board run --bundle artifacts/bundles/<bundle-id> --run-id resnet18_open_cpu_resource_02 --execute-board
python run_open_resources.py status --run-dir runs/resnet18_open_cpu_resource_02
```

流程依次完成环境与程序校验、引用已有组件或采集缺失项、87项VTA流量探针、电脑端完整DP和独立枚举核验、冻结候选、最多43个流水配置、回收和分析。九种资源上下文共同竞争名额，不分别各测43项。

差异化名单保留静态前10项，再补5种未覆盖切图和5个资源策略代表。资源名额依次考虑固定核组同组串行、分开核组开放并发、重叠核组开放并发、主机单核仍共享、为主机及控制保留核心。只从预测间隔不超过最优值1.15倍的候选中补入，不使用本轮流水FPS；某类没有合格代表会记录原因，再按既定规则补足20项。物理核心编号的镜像不单独算一种资源策略。

全部使用输入0。组件预热2帧、计入8帧；流水预热8帧、保留16个完成时间戳、尾部4帧，平均完成间隔由15个间隔计算。流量探针独立预热1次、计数1次，仅用于LOAD/STORE载荷字节。没有漂移拒收、额外锚点、自动延长窗口或显著性判停。

CPU可用核心池、阶段实际核心集合、VTA主机、输入生产线程、收集与结果处理主线程分别保存。主机单核固定仍共享时不禁止计算使用该核；真正保留一核时计算阶段必须排除它。保留不表示隔离Linux系统任务或中断。

## 断线与恢复

```bash
python run_open_resources.py board resume --run-dir runs/resnet18_open_cpu_resource_02 --execute-board
python run_open_resources.py analyze --run-dir runs/resnet18_open_cpu_resource_02
```

SSH断线标记 `INTERRUPTED`，同boot可续跑。已有有效回收凭据直接恢复，不重复启动或计入已完成窗口。缺帧、重复帧、输出错误、身份错误和实际绑核错误保留现场并停止，不作为有效性能结果。

本批是单次新启动；引用旧组件保留旧启动身份，不计作本轮独立启动。若开发板真实重启，控制器标记 `STOPPED_BOOT_CHANGED`，不能将新boot拼入同一批确认结果。程序不自动重启开发板。

停止状态不自动恢复。代码、程序或环境改变后应重新准备并建立新批次；从包内冻结快照可继续分析旧批次。

## 结果含义

`analysis/` 输出全部配置与线程映射、各上下文入选及实测数量、纯静态和差异化两份20项、相同1/5/20预算比较、组件来源和历史参照。没有进入实测池的上下文仅报告静态覆盖，不判断其实际性能。

控制线程开销未单独标定，不能视为零；共享主机容量、同构核置换和均分核时均记录为模型近似。历史硬件DDR参数附原始来源且标明非本轮正式标定。结果是当前ResNet18、当前软件硬件的单图短窗口观察，不作显著性或跨启动稳定改善声明。
