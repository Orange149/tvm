# ResNet18 单任务多线程与共享核心竞争补充实验

本实验在同一批十次独立启动中，比较四核共享和三核计算、一核主机两种环境，补充单线程物理核心映射、固定多核设备组及重叠核心集合的对照。旧主实验的控制器源码、部署包和结果继续封存。本目录的代码和批次具有独立身份，版本从 v1 开始。

六种正式方法均使用同一共同量化图、172 个预编译阶段、普通复制、固定 Tile、64 张真实输入及 3 张诊断输入。组件成本由新批次的两次独立启动重新测量，历史成本和合成测试数据不能进入真实排名。

## 入口

所有命令由电脑执行。推荐解释器为 `/home/orange/miniconda3/envs/vta-resnet/bin/python`。

```bash
python run.py prepare
python run.py validate --bundle artifacts/bundles/<bundle-id>
python run.py board run --bundle artifacts/bundles/<bundle-id> \
  --run-id resnet18_multithread_mapping_01 --execute-board
python run.py board resume --run-dir runs/resnet18_multithread_mapping_01 --execute-board
python run.py status --run-dir runs/resnet18_multithread_mapping_01
python run.py analyze --run-dir runs/resnet18_multithread_mapping_01
```

`prepare`、`validate`、`status` 和 `analyze` 不调用 SSH/SCP。`prepare` 逐文件核验父包，在独立目录重新编译执行器，复用父包运行库和阶段，不重新切图。完整域算法核验、原生组互斥测试和加载器参数检查通过后才生成 `PREPARED_OFFLINE` 部署包。

上板入口显式要求 `--execute-board`。默认读取主项目 `configs/local.json` 的连接和 SDK 配置；可以通过 `--local-config` 指定。本实验沿用已授权的 SD 实验分区，只做身份检查和挂载。程序从不重启或重新分区。

## 流程

第 1–2 次启动完成 1,449 种执行签名的资格检查与标定；第 3–4 次筛选两种环境的候选；第 5–10 次独立确认并开展交叉与机制实验。每次完成后回收数据、校验部署文件并同步，进入 `WAITING_NEW_BOOT`。助手会通知用户手动重启，再负责续跑。

四核搜索完整枚举 182,877,075 个配置，三核搜索枚举 8,857,779 个配置。搜索在电脑端进行，内存中最多保留单个拓扑的 50,625 个映射。每种方法保留前 20，筛选前 5，并加入按岛数量分层的随机配置。相同程序与执行模式共用测量。

详细规则见 [实验协议](docs/protocol.md) 和 [板端操作与恢复](docs/board_runbook.md)。输出包括中文报告、六种方法的预算一次/五次结果、逐启动记录、阶段加速曲线、固定划分四格图、核心放置干预和历史提升复核。

## 本地验证

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -c pyproject.toml --confcutdir=. tests -q
CPUVTA_FULL_AUDIT=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python -m pytest -c pyproject.toml --confcutdir=. tests -q
```

完整验收将分块向量化搜索与独立 C++ 逐配置穷举比较全部方法的前 20 及计数，覆盖两个完整扩展域；小型案例另与独立 Python 标量实现核对。十次启动的模拟流程包括断线、续跑、重复启动、原始数据篡改及启动级统计，不连接真实板子。
