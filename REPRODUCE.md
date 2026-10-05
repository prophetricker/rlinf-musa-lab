# 路线二复现说明

本导出仓库包含路线二补丁、探针、小型证据，以及从路线一目录原样复制的共用 MuJoCo runner。恢复源码可以在 Mac 完成；MUSA/GPU 探针在已配好匹配 Torch-MUSA 的 Linux S4000 上执行。新增的便携入口经过本地语法/路径检查，首轮 GPU 结果来自详细报告中的原命令，尚未重新运行便携入口。

## 1. 恢复固定源码

在此仓库根目录执行：

```bash
python3 scripts/restore_sources.py --apply-patches
git -C worktrees/rlinf-route2 rev-parse HEAD
git -C worktrees/rlinf-route2 diff --stat
```

`HEAD` 应为 `c70606f08cdca259b8dec03d4430926b5b8fac9d`；适配作为工作区改动应用。脚本保留已有工作树，不覆盖现有内容。源码树恢复后的预期 tree 为 `b78ed3694bcd1d69afd887e6ce11b780d7528ecf`，对应本地适配提交 `3e7329c6`；补丁重建不改变上游依赖声明。

## 2. 选择已有的匹配环境

首轮解释器是 `/root/autodl-tmp/s4000-research/envs/route2/bin/python`，通过 system-site-packages 与 `.pth` 继承系统 Torch-MUSA 及早先的轻量依赖。这不是独立安装的一套新 Toolkit。

```bash
export ROUTE2_PYTHON=/root/autodl-tmp/s4000-research/envs/route2/bin/python
"$ROUTE2_PYTHON" -c 'import torch, torch_musa, ray, gymnasium, mujoco; print(torch.__version__, torch_musa.__version__, ray.__version__, gymnasium.__version__, mujoco.__version__)'
```

预期 runtime 版本依次为 `2.2.0`、`1.3.0+81caf0a`、`2.59.0`、`1.1.1`、`3.3.7`。Torch distribution metadata 是 `2.2.0a0+git8ac9b20`，与 runtime 字符串不同。

新服务器需要先获得匹配的系统 Torch/Torch-MUSA，再准备隔离环境与缺少的轻量依赖，参考 [环境说明](locks/environment.md)。`locks/route2-environment.txt` 是历史环境快照，含本地 wheel 路径，不能直接作为可移植 requirements 安装；其中 `rlinf==0.3.0` 是继承的旧包，探针实际使用固定源树。也不要直接执行上游全项目 `pip install .`：它的 Torch 版本要求会引入另一套运行栈。

## 3. 顺序复测

下面的入口自动从源码锁读取 commit、配置 PYTHONPATH（包括 Ray child 需要的 unit_tests 与共用 runner 路径），并把新输出写到 `results/`，历史证据保留在 `routes/route2/evidence/`。

```bash
bash scripts/run_probes.sh imports
bash scripts/run_probes.sh cpu
bash scripts/run_probes.sh channel
bash scripts/run_probes.sh worker
bash scripts/run_probes.sh fsdp
bash scripts/run_probes.sh loop
```

`imports` 会记录预期的 FSDP/FlashAttention 阻塞；以输出里的每项状态为准，进程退出 0 不能代表所有模块可用。`worker/fsdp/loop` 分配 MUSA 资源，按顺序运行。Channel 模式验证 Ray CPU 路径，fsdp 模式只验证原生 FSDP1 的单进程 `NO_SHARD`。

loop 模式固定首轮配方：seed 7、4 env、32 horizon、2 iterations、2 epochs、minibatch 64、1 eval episode、同批恢复检查。结果应该是 256 transitions、8 次 optimizer step、130 条 Channel 消息；浮点值与耗时可能随系统变化。resume 对照不包含模拟器状态。可通过 `ROUTE2_SOURCE` 指定另一固定源树，通过 `ROUTE2_RESULTS` 指定输出目录。

这不是 GR00T 或完整官方 RLinf benchmark 的启动脚本。扩大学习预算及周期性评估见 [后续节点](ROADMAP.md)。
