# ResNet18三因素完整消融

本批以线程负载份额、主机集中约束、线程组服务约束三个独立开关，比较八种组合。
所有版本使用同一775项隔离成本、87项DDR记录、172个阶段模块、九种上下文及完整966,240,306组合域。
每组严格排序20项，按完整执行身份去重后共同上板；不引入历史特殊配置，不使用流水FPS构造评分。

从 `published_cpu_policies` 目录运行，使用 `vta-resnet` Python 环境：

```bash
python run_model_ablation.py rank --output artifacts/model_ablation_01 --workers 2
python run_model_ablation.py prepare --output artifacts/model_ablation_01
python run_model_ablation.py validate --bundle artifacts/bundles/<bundle-id>
python run_model_ablation.py board run --bundle artifacts/bundles/<bundle-id> --run-id resnet18_model_ablation_01 --execute-board
python run_model_ablation.py board resume --run-dir runs/resnet18_model_ablation_01 --execute-board
python run_model_ablation.py status --run-dir runs/resnet18_model_ablation_01
python run_model_ablation.py analyze --run-dir runs/resnet18_model_ablation_01
```

`rank`核验已有四组的原始输入指纹、搜索源码、九个检查点与C++枚举结果，并补齐四组新搜索。
所有组均重建全局Top-20，不在已测小池中重排。同分沿用完整执行配置顺序。
八份名单、每个候选的八种分数、执行顺序与哈希在上板前冻结。`prepare`只复制已核验程序及运行依赖，
运行主机验收和QEMU加载，不编译新执行器或阶段，不连接开发板。

板端输入0，普通复制，队列2，逐线程固定；同组串行由各候选自己的配置决定。
每候选一次单帧完整输出资格检查，再运行8预热＋16完成时间戳＋4尾帧，16时间戳对应15个间隔。
最多160个不同候选、320会话、4,640次完整推理；实际按去重数量执行。
没有锚点、漂移拒收、自动加测或自动重启。错误输出、哈希、帧身份及实际绑核仍拒绝。
续跑回放原始凭据，已验收任务不重复执行；启动身份变化时停止当前批次。

分析分别报告1／5／20次选点预算、完整模型逐项删除的代价、八组合对数FPS主效应／交互、
名单重合、预测同分、共同并集误差及约束主导项，附全部配置、原始哈希、图表及成本。
主效应与交互采用±1编码的标准三因素分解：系数为sum(sign×log FPS)/8，因素效应为两倍系数。
它们是描述性结果，没有独立重复或显著性含义；候选共享测量也不创造独立样本。

线程份额来自整个隔离会话，并非精确计算窗口；三项极短两线程签名保留均分回退。
同核数置换仍是同构核近似；控制线程成本没有独立标定。主机与线程组约束是规划代理，非真实延迟界。
本批是模型选点消融：不同模型允许选出不同切图与配置，不能作为同一配置的操作系统因果分解。
基础容量模型是本项目后续扩展的均分核时模型，不是Tarnawski原作者实现。
