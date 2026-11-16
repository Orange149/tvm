# CPU计算区间的低扰动线程诊断

独立入口：`run_run_scope.py`。批次：`resnet18_prediction_gap_02`。
本轮只诊断原静态第一名与历史核组／主机固定核心2配置，不改成本、公式、DP或候选名单。
原诊断批次`resnet18_prediction_gap_01`及其冻结部署包保持原身份。

## 固定测量安排

|工作|任务数|推理次数|
|---|---:|---:|
|CPU前段三／四线程、中段四线程／核心3单线程资格检查|4|4|
|上述四项组件分别关闭／开启新探针|8|80|
|两个完整流水资格检查|2|2|
|每个流水：原执行器、新执行器关闭、开启、再次关闭探针|8|224|
|合计|22|310|

只使用输入0，组件2次预热＋8次计入，流水8次预热＋16个完成时间戳＋4次尾部。
流水完成间隔来自C8至C23的15个间隔。六个资格任务开启线程探针，仍各只运行一次DNN。
不设锚点、漂移拒收、自动追加或自动重启。

## 计数作用域

只重新编译独立执行器，旧执行器、TVM/VTA运行库、八个阶段模块及输入均按哈希复用。
新开关默认关闭；开启后，CPU单元0–2及5–7在各自run前后读取所属线程的Linux schedstat。
初始化时打开文件描述符，每次pread从0读取新快照，caller尽量前侧最后读、后侧最先读。
资格任务用额外新开描述符交叉检查计数新鲜性，不增加前向次数。

流水只采第8、12、16、20帧；组件采计入的8帧。数据存入预留的线程私有缓冲，结束后再写
`run_threads.json`。不启用详细队列、全进程session或PMU探针，不新增每帧同步。

原run_ms如实包含前后探针工作，独立诊断run起止时刻仅包围实际run调用。前后读取时间、
各TID的快照时间范围和计数均保留，不能把多个非同时快照称为严格同时采样。
runtime_ns包括计算、同步、yield等在核工作，也包括缓存停顿时仍占用CPU的周期；
它不是有用计算的独立计数。禁止以墙钟减去多线程CPU时间之和作为等待时间。

## 操作

```bash
python run_run_scope.py prepare
python run_run_scope.py validate --bundle artifacts/bundles/<bundle-id>
python run_run_scope.py board run --bundle artifacts/bundles/<bundle-id> --execute-board
python run_run_scope.py board resume --run-dir runs/resnet18_prediction_gap_02 --execute-board
python run_run_scope.py status --run-dir runs/resnet18_prediction_gap_02
python run_run_scope.py analyze --run-dir runs/resnet18_prediction_gap_02
```

准备和校验不访问开发板。任务身份包含执行器哈希、探针开关、采样帧及完整配置。
断线后依据原始凭据恢复；旧执行器、新执行器以及前后两个普通会话不会互相复用。
换boot不混合本批数据；文件损坏、输出错误、错误绑核及重复／缺失帧仍属无效执行。

## 判读和完成标准

先比较新执行器关闭探针与旧执行器，再比较新探针与前后两个普通会话，最后比较同一
子图的隔离和流水内线程核时、等待和工作分布。CPU阶段相互重叠时仍不相加进程CPU时钟。

探针目标为完成间隔变化不超过2%；超出时保留数据并标记定量解释受限，不追加测量。
每阶段四个稀疏点仅逐点报告和给范围，不估总体P95、独立样本显著性或全域修正系数。
无法确定启用的全零等待记为未知，缺失线程不能改配同核上的其他TID。
助手完成回收、完整哈希复核、部署文件复核、同步和报告。结论只用于制定下一步建模建议。
