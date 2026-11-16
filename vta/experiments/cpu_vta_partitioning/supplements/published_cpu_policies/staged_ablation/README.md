# ResNet18：Tarnawski基线、规律整理与分阶段消融

先审计历史正反例及基线契约，再排名、冻结、部署和共同实测。只使用ResNet18；不实现2023年TVM方法、YOLO或贝叶斯优化。

```bash
python run_staged_ablation.py audit
python run_staged_ablation.py rank --workers 2
python run_staged_ablation.py prepare
python run_staged_ablation.py validate --bundle artifacts/bundles/<bundle-id>
python run_staged_ablation.py board run --bundle artifacts/bundles/<bundle-id> --execute-board
python run_staged_ablation.py board resume --run-dir runs/resnet18_staged_ablation_01 --execute-board
python run_staged_ablation.py status --run-dir runs/resnet18_staged_ablation_01
python run_staged_ablation.py analyze --run-dir runs/resnet18_staged_ablation_01
```

Tarnawski为明确的平台模型适配：单线程物理CPU、同核心累计隔离服务、单VTA累计、共同DDR项；不继承本文主机容量、线程份额或线程组修正。原作者完整系统未移植。

四个主消融交叉CPU粗粒度／细化及主机需求排除／计入，均生成Top20；线程份额、线程组和集中约束三个补充版本只测前五项。评分版本不进入执行身份。

隔离775项成本、87项DDR记录及172个阶段模块只读复用。Tarnawski单线程域与其余完整配置域分别报告覆盖数；Python评分、向量化资源DP与独立C++枚举核对。

历史CPU重叠证据涉及不同输入或诊断执行器，因此仅补两项当前执行器、输入0的两线程重叠／分开对照。已有组串行及主机热点配对通过原始凭据审计后保留；历史FPS不填入本批性能表。

每配置单帧输出资格检查，再8帧预热＋16完成时间戳＋4尾帧。16时间戳为15间隔；单输入、单启动、短窗口，不自动重启、延长或重复测量。断线仅回收／接续未验收任务；启动ID变化停止当前批次。

部署包携带冻结控制器、搜索凭据、程序、运行库、输入及参考。板端不依赖旧目录；历史证据的原始路径和哈希另有来源索引。Pipe-it和GCAPS为执行组织与资源拆分的依据，不作为独立FPS柱；线程组／集中约束仍为规划代理。
