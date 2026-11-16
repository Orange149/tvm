# ResNet18 成熟资源分配与完整 DP Top-20

流程为：三种切图上的资源策略验证 → 独立核验并冻结规则 → 补齐匹配成本 → 完整切图 DP → Top-20 全部实测 → 三次独立确认。决策不使用贝叶斯优化。

借鉴 Pipe-it 的负载平衡与线程固定、Packrat 的线程预算与固定 CPU 设备、Flexi-BOPI 的阶段核数配置。它们落实为明确的执行设置和资源约束，不宣称完整复现原系统。

## 入口

推荐 Python：`/home/orange/miniconda3/envs/vta-resnet/bin/python`。命令由助手执行，用户按提示手动重启。

```bash
python run.py prepare --config configs/resnet18.json
python run.py validate --bundle artifacts/bundles/<bundle-id>
python run.py board run --bundle artifacts/bundles/<bundle-id> --run-id resnet18_resource_policy_dp_01 --execute-board
python run.py board resume --run-dir runs/resnet18_resource_policy_dp_01 --execute-board
python run.py status --run-dir runs/resnet18_resource_policy_dp_01
python run.py analyze --run-dir runs/resnet18_resource_policy_dp_01
```

`prepare` 可用 `--calibration-source` 显式指定旧组件标定来源。默认核验已保存的34种签名、58个块；凭据和所需原始文件放入新包的主机证据快照。输入、程序、运行库、执行器和条件一致才导入，不更改原启动身份。新绑核或主机设置补测新签名。

`prepare`、`validate`、`status`、`analyze`均不使用SSH。部署包复用172个已编译模块，普通核组执行器保持旧哈希，新增执行器使用TVM现有逐线程绑定并补齐caller固定。没有新增模型编译。

## 实测规模

电脑搜索的数百万配置不会逐一上板。策略探索最多26个流水配置，完整DP筛选最多30个（增强方法20个、两个对照各前五名，重合时去重）。确认只测冻结选点和必要机制对照。

组件标定与完整流水是不同任务。初始74种组件签名可精确复用25种，补49种；若核组镜像检查不支持同质性假设，补实际连续核组。完整切图成本在规则冻结后按缺口补齐，可能按小时计，不重新测已有原子诊断曲线。

正式性能固定一张图，8帧预热＋32个完成时间戳＋4尾帧；预热资格检查必要时统一改24帧。另用4张图核验完整输出。单线程物理映射作稳定锚点，漂移超5%整轮作废，最多替换一次。

## 六次独立启动

1. 组件复用／补测，三种切图的六类策略及匹配干预。
2. 独立策略核验，冻结资源规则，补全成本，生成并核验完整DP排名。
3. 增强方法Top-20全部实测及两个对照前五名，冻结最终选点。
4. 第4–6次共同确认资源策略、匹配干预及DP选点。

预检启动不计入六次。每次完成后回收、核验和同步，随后等待用户手动重启。线程组内迁移、逐线程固定、同组串行、主机／控制放置分别记录，正式性能关闭详细日志。

结果位于新批次的 `analysis/REPORT_zh.txt`、CSV及图表。先平均同启动两轮，再报告三个独立启动的配对差距与范围；不将帧当独立样本，不报告显著性或等效性。

旧批次与旧包保留；使用旧包内 `sources/cpu_strategy/run.py` 校验和分析旧结果。历史11.511FPS程序的边界为浮点、输入和程序不同，本轮只在当前共同量化图上复测其切点，不把原数值作为同程序基准。

协议见 [docs/protocol.md](docs/protocol.md)，运行与恢复见 [docs/board_runbook.md](docs/board_runbook.md)。
