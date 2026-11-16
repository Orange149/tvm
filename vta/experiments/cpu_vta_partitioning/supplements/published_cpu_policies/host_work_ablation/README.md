# ResNet18：VTA主机CPU需求三组消融

三个版本固定开启隔离线程份额和线程组服务约束，区别为：忽略VTA主机CPU需求、计入该需求、计入并补充主机集中约束。

主机需求是所有VTA岛的运行CPU核时与输入输出CPU核时之和。忽略版本只在评分中排除该需求，不修改原始成本、VTA阶段墙钟服务、DDR或真实执行。CPU自身交接始终计费。墙钟等待不直接作为CPU核时。

三组各搜索既有966,240,306项配置，独立C++枚举核验严格Top-20；后两组精确回放既有搜索凭据。名单、执行并集与顺序先冻结，评分版本不进入执行身份，不读取流水FPS选点。

在本目录的父目录运行：

```sh
python run_host_work_ablation.py rank --output artifacts/host_work_ablation_01 --workers 1
python run_host_work_ablation.py prepare --output artifacts/host_work_ablation_01
python run_host_work_ablation.py validate --bundle artifacts/bundles/<bundle-id>
python run_host_work_ablation.py board run --bundle artifacts/bundles/<bundle-id> --run-id resnet18_host_work_ablation_01 --execute-board
python run_host_work_ablation.py board resume --run-dir runs/resnet18_host_work_ablation_01 --execute-board
python run_host_work_ablation.py status --run-dir runs/resnet18_host_work_ablation_01
python run_host_work_ablation.py analyze --run-dir runs/resnet18_host_work_ablation_01
```

准备和排名只在本机运行。部署包包含冻结控制器与搜索凭据，验证、上板和分析会自动转到该副本；不修改旧包与旧控制器身份。

最多60个配置、120会话、1,740次完整推理。每项一帧资格检查，再测输入0的8预热＋16完成时间戳＋4尾帧；16时间戳对应15间隔。顺序种子20261109。不重启、不测锚点、不因性能波动追加或拒收。错误输出、绑核、文件身份和帧完整性仍拒绝；换boot ID停止当前批次，断线续跑不重测已验收任务。

分析分别报告：计入／忽略需求的选点贡献、完整／计入需求的集中约束贡献，以及完整／忽略需求的组合贡献。FPS变化与完成间隔下降使用各自明确分母。相同配置共用一次本批实测；不复用旧FPS，不报告单启动显著性或等效性。

原始线程份额仍为隔离会话份额近似；同核数核心集合使用既有同构核假设。新增约束是规划代理，不是精确调度仿真。未单独标定的生产、收集开销保持为模型近似。
