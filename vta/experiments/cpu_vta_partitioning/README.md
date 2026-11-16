# CPU–VTA 划分实验

当前研究代码位于 [published_cpu_policies](supplements/published_cpu_policies/README.md)，已包含 ResNet18 和 YOLOv3-tiny。最新共同实测、消融与误差诊断的入口见该目录说明。

清理范围、原始证据位置及恢复方法见 [工作区清理记录](docs/reproduction_and_cleanup_20261112.txt)。编译产物和原始记录保留在各实验目录的 `artifacts/`、`runs/`；论文目录保留报告、图表和可读结果表。

## 初始 v1 批次的使用说明

以下是最初 ResNet18 两阶段实验的协议和入口，保留用于复现该批次。其“首批不含 YOLO”、十次启动及敏感性实验安排不代表当前研究流程；当前批次以各自冻结协议为准。

本项目将 ResNet18 的实验拆为**本地准备**和**集中上板**两阶段。历史脚本与结果保留原路径；新代码通过适配层读取共同量化图和合法阶段身份，不运行旧实验的 `main()`，不读取旧成本表排名。YOLO 不在首批范围。

第一阶段生成 `PREPARED_OFFLINE` 部署包。这表示本地编译、接口核验及模拟测试通过，真实正确性和性能仍待板端验证。

完整归档留在电脑，板端只接收 `plans/board_payload.json` 中约 850 MiB 的执行文件。压缩流直接解压，成功结果在电脑校验保存后清理板端副本。示例部署到独立实验分区下的 `/media/cpu_vta_data/cpu_vta_partitioning`，上板时检查挂载和剩余空间；本地准备不会访问该板端目录。

当前版本为 **v1**（Python 包版本 `1.0.0`）。阶段模块和原生构建绑定实际 SDK 工具链身份，CMake 缓存按身份隔离；随包共享库和加载器从当前 SDK 解析并记录来源与哈希。分析身份校验、准备锁、测量、恢复和统计约束均包含在本版本中。

当前部署包由 `artifacts/LATEST_BUNDLE.json` 指定；完整冻结包位于 `artifacts/bundles/<bundle-id>/`，板端压缩包位于同 ID 的 `artifacts/deployment/` 子目录。最终审查记录见 `artifacts/reviews/v1/final_review.json`，搬迁和部署验收证据位于 `artifacts/validation/`。

`tests/` 包含全部模型、测量、部署和恢复测试。`artifacts/builds/` 保留当前 172 个阶段及对应工具链的编译缓存，供重新准备时校验复用。共同图、输入、调优记录和切图适配器按配置指定的路径读取。

数值判据使用独立 AArch64 整图和边界探针参考；准备入口会拒绝 x86 参考或缺少来源证明的数据。所有成本与性能均在新批次中重新标定。

## 使用

在本目录运行，建议使用已有的 `vta-resnet` Python 环境。

```bash
cp configs/local.example.json configs/local.json
# 核对 local.json 的 Python、SDK、输入来源及第二阶段的 SSH 地址。
python run.py prepare --config configs/resnet18.json --local-config configs/local.json
CPUVTA_BUNDLE=$(python -c 'import json; print(json.load(open("artifacts/LATEST_BUNDLE.json"))["path"])')
python run.py validate --bundle "$CPUVTA_BUNDLE"

# 以下命令才会连接板子。
python run.py board run --bundle "$CPUVTA_BUNDLE" --run-id resnet18_v1_main_01 --execute-board
python run.py board resume --run-dir runs/resnet18_v1_main_01 --execute-board
python run.py status --run-dir runs/resnet18_v1_main_01
python run.py analyze --run-dir runs/resnet18_v1_main_01
```

主实验需要 **10 个不同 kernel boot ID**：标定 2 次、筛选 2 次、独立确认 6 次。程序不会代替用户重启。每轮启动完成后显示 `WAITING_NEW_BOOT` 和续跑命令；相同 boot ID 不会计入下一次启动。

满足预设投入条件后自动安排资源敏感性实验：六种环境中复用主环境，另外五种各需要 10 次启动，合计最多 60 次有效启动。失败启动的替换另计。

## 目录

| 路径 | 职责 |
|---|---|
| `configs/` | 冻结协议、六种资源环境、本机配置示例 |
| `src/cpu_vta_partitioning/domain` | 21 单元、合法切图、线程及 affinity 身份 |
| `models/`、`search/` | 七组模型、全域评分、Top-20、独立枚举核验 |
| `prepare/`、`adapters/` | 共同图切分、SDK 编译、依赖闭包及部署包 |
| `measurement/` | 完整输出、稳态窗口、核时及线程检查 |
| `campaign/`、`transport/` | 启动批次、原子状态、SSH、结果回收与续跑 |
| `analysis/` | 启动级配对统计、Holm、等效检验、图表及成本 |
| `tests/` | 合成域、完整域、故障注入、模拟十次启动流程 |
| `artifacts/builds/` | 可续用编译缓存和构建日志 |
| `artifacts/bundles/` | 自包含、哈希冻结的板端部署包 |
| `runs/` | 每次实验的状态、原始会话、标定、排名和分析 |

`configs/local.json`、缓存、部署包、结果和 Python 缓存均忽略，不提交大体积数据。

详细规则见 [实验协议](docs/protocol.md)、[本地准备验收](docs/offline_preparation.md)、[集中上板手册](docs/board_runbook.md)。

独立测试不加载仓库根部的 TVM pytest 插件：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -c pyproject.toml --confcutdir=. tests
```

不设置 `CPUVTA_WORKSPACE` 时，真实编译产物核验测试跳过；`prepare` 会设置该变量，完整验收不跳过这项。
