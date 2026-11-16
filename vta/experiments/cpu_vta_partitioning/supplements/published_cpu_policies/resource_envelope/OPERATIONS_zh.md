# 资源包络与资源行为筛选：复现入口

当前实现复用原21单元、172模块、九种资源上下文、全部非空CPU核心集合及最多三个VTA岛。
阶段调用线程仍采用原域约定的最小编号核心；本轮没有枚举所有调用位置或worker排列。
开放并发与合法的固定核组串行两种模式均保留，完整域不因新评分而缩小。

## 本地准备

在本目录的上级目录执行，使用`vta-resnet`环境：

```bash
python -m resource_envelope.search_adapter --output artifacts/resource_envelope_full_domain_01
python run_envelope_top20.py prepare --search-root artifacts/resource_envelope_full_domain_01
python run_envelope_top20.py validate --bundle artifacts/bundles/<bundle-id>
python run_envelope_top20.py rank --bundle artifacts/bundles/<bundle-id>
```

完整域搜索保留区间路径DP与资源标签，九个上下文分别通过独立C++枚举验证；
新标签增加四维CPU线程组服务累计，主机工作H与逐核需求D沿用原字段。
每条路径最多四个CPU阶段，每阶段四核域15个选项／三核域7个选项；
资源分配规模为m的阶段数次方，向量化半路径标签拼接不使用Top-1支配剪枝丢弃Top-20路径。
此次计数为4623条切图路径、966240306个增强资源分配组合，实测搜索与枚举约690秒。
这不是新的渐近复杂度改进；完整枚举用于核对按所给公式得到的结果，不证明公式准确。

准备过程冻结三份各20项名单：原模型严格排序、新包络严格排序、新包络按资源行为取代表。
行为分类使用隔离profile中阶段的墙钟及核时贡献，阈值为同一切图最大贡献的四分之一；
物理核镜像按角色关系规范化，低贡献阶段资源变体压缩。
该规则损失物理镜像及部分轻阶段变体，且代表来自原有Top-2分层终态池，不能称为全域最优行为族名单。
原评分与新评分均保存在同一配置记录中，配置身份只由实际执行语义决定。

## 集中上板

```bash
python run_envelope_top20.py board run --bundle artifacts/bundles/<bundle-id> \
  --run-id resnet18_resource_envelope_top20_01 --execute-board
python run_envelope_top20.py board resume --run-dir runs/resnet18_resource_envelope_top20_01 --execute-board
python run_envelope_top20.py status --run-dir runs/resnet18_resource_envelope_top20_01
python run_envelope_top20.py analyze --run-dir runs/resnet18_resource_envelope_top20_01
```

统一入口始终转交部署包内冻结控制器。修改工作区源码不会静默改变已经建立的批次。
本轮沿用原程序、运行库和执行器，没有新增组件profile或重新编译。
三份名单共享同一批测量，预算1／5／20按各自名单顺序计算；不以开发回放FPS决定名单。

板端流程不重启、不修改分区或启动文件、不更改CPU频率。使用现有实验分区和会话互斥锁，
只在确认无活动任务时装载已有、哈希匹配的VTA固件。各任务核验启动身份、程序哈希、
完整输出、逐线程实际affinity以及完成帧；断线续跑只接续未验收任务。
输入0，每项单帧资格检查；性能为8预热＋16完成时间戳＋4尾帧，16时间戳对应15间隔。
无锚点、漂移拒收、自动加帧或自动追加测量。同一批次boot ID改变时拒绝混入。

## 分析与证据归档

```bash
python -m resource_envelope.result_audit --run-dir runs/resnet18_resource_envelope_top20_01
python archive_resource_envelope.py --output /home/orange/code/tvm/paper/my_paper/analysis/20261108_resource_envelope
```

归档包含新测量原始文件、验收凭据、冻结配置、源码、公式、搜索检查点及独立枚举证明。
ELF、位流、输入和参考输出继续保存在原自包含部署包，归档记录其完整哈希清单，避免重复大文件。
`cost_audit.py`按原任务ID匹配历史事件：775项隔离签名和87项流量任务共1637个会话、
8699次阶段调用，已验收测量及回收约7971秒。部署、失败尝试与人工等待不包含在该数字中。
DDR带宽参数仍来自历史组件模型，未在本轮重新校准，不能称为当前硬件总线的精确测量。

完整流水阶段服务按帧选取，与隔离profile并列解释；它们不是共同墙钟区间的资源占用账本。
流水整体CPU核时使用进程时钟首尾差，禁止相加并发阶段的进程核时。
新包络是事前规划代理，特别是开放重叠线程组项，不是实际延迟的严格上下界。
开发回放、精确旧配置、物理镜像和新配置分开标记；单启动短窗口只支持本次观察。
