# AutoTVM覆盖筛选与Top-20实测

入口为项目根目录 `run_diverse_top20.py`。本轮保留完整DP与成本口径，
仅替换预算20项的覆盖筛选；没有新组件标定或CPU共同帧流测量。

```bash
python run_diverse_top20.py prepare
python run_diverse_top20.py validate --bundle artifacts/bundles/<bundle-id>
python run_diverse_top20.py rank --bundle artifacts/bundles/<bundle-id>
python run_diverse_top20.py board run --bundle artifacts/bundles/<bundle-id> --execute-board
python run_diverse_top20.py board resume --run-dir runs/resnet18_autotvm_diverse_top20_01 --execute-board
python run_diverse_top20.py status --run-dir runs/resnet18_autotvm_diverse_top20_01
python run_diverse_top20.py analyze --run-dir runs/resnet18_autotvm_diverse_top20_01
python run_diverse_top20.py audit-runtime --bundle artifacts/bundles/<bundle-id> --output artifacts/runtime_readonly_audit
```

在775项隔离成本和87项DDR记录的完整DP结果上，保留各上下文静态前20，
每个切图／执行方式前两名以及每种CPU资源形状／执行方式前两名。
分类内采用1.15倍预测值限制；不是全局阈值，也不缩小DP域。
新名单依次保留原前五名，覆盖九种上下文与三类CPU执行，再用未经修改的
AutoTVM `submodular_pick` 贪心覆盖函数填满20项。

函数在 `upstream.py`，来源与函数哈希在 `upstream_origin.json`。
33维类别特征等权；已有覆盖通过独立编码适配，不改原函数，不训练模型。
不引入历史特殊配置，不读旧FPS，不按测量结果改名单。

三份名单在部署前冻结，原两份名单并集28项；本次新名单并集40项，
每项资格检查1帧，性能窗口28帧，合计80会话、1160次完整推理。
输入0、单启动、一轮；没有锚点、漂移拒收或自动加测。断线续跑复用已验收任务。

结果在 `runs/<run-id>/analysis/`，包括中文报告、预算对照、全部配置表和上下文覆盖。
一次、五次预算与原静态名单相同是保留前五名的设计结果；主要比较二十次预算。
等待机制报告只证明实际二进制存在相应路径，不证明它造成吞吐误差。

旧批次、执行器、运行库及全局配置目录不修改；新增协议放在本包中。
