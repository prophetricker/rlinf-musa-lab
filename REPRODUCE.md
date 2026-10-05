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

## 4. 学习版

详见 [固定预算协议](routes/route2/learning/README.md)。恢复源码后，仓库根目录配置共用 runner 与学习模块路径，再顺序跑三个 seed：

```bash
export PYTHONPATH="$PWD/worktrees/rlinf-route2:$PWD/probes:$PWD/routes/route2/learning"
export RLINF_MUSA_TORCH_DETECTION_FALLBACK=1
for seed in 7 17 27; do
  "$ROUTE2_PYTHON" routes/route2/learning/ppo_learning.py \
    --source-commit c70606f08cdca259b8dec03d4430926b5b8fac9d \
    --seed "$seed" --resume-check --output "results/learning/pendulum-seed$seed.json"
done
"$ROUTE2_PYTHON" routes/route2/learning/summarize_learning.py \
  results/learning/pendulum-seed7.json results/learning/pendulum-seed17.json \
  results/learning/pendulum-seed27.json --output results/learning/summary.json
```

默认是 Pendulum-v1、每 seed 102,400 transitions、固定 5 validation/20 test episodes。修改预算或测试协议需另记配方，不能沿用历史结论。

HalfCheetah 三种子保持同一预算，增加 `--env-id HalfCheetah-v5 --normalize-observations --batch-evaluation`。这是 MuJoCo 训练；Pendulum 本身不是 MuJoCo 环境。批量 helper 的实机 CPU 对照与 MUSA smoke 已验证，不把串行前测的中断结果计入汇总。

也可运行便携固定配方入口：`bash scripts/run_learning.sh pendulum` 或 `bash scripts/run_learning.sh halfcheetah`，顺序执行。便携入口通过路径与 Bash 语法检查，历史 GPU 结果来自上述同等原命令，未重复跑新入口；默认不画图，安装已有匹配 matplotlib 时可对汇总命令添加 `--plot`。

## 5. 可选 FSDP1 增量

基线补丁保持不变。显式恢复另一份源码，避免把后端实验覆盖到学习基线：

```bash
python3 scripts/restore_sources.py --apply-patches --fsdp1-experimental
"$ROUTE2_PYTHON" routes/route2/fsdp_probes/fsdp1_backend_probe.py \
  --source "$PWD/worktrees/rlinf-route2-fsdp1" --phase imports --enable-torch22
RLINF_MUSA_TORCH_DETECTION_FALLBACK=1 \
  "$ROUTE2_PYTHON" routes/route2/fsdp_probes/fsdp1_backend_probe.py \
  --source "$PWD/worktrees/rlinf-route2-fsdp1" --phase strategy --enable-torch22
```

恢复 tree 为 `1a4eef5dd1f4984652168ebecce1d3cc194ec598`。GPU strategy 顺序运行；单卡 FP32 NO_SHARD RLinf 后端的 MSE 更新、local_shard 与 Torch 2.2 DCP checkpoint 恢复均有独立入口，仍不含 PPO/官方 Actor/多 rank。详细 CPU 测试及失败历史见 [FSDP 说明](routes/route2/fsdp_probes/README.md)。默认不开放旧 Torch，FSDP2 仍拒绝。

## 6. 独立 checkpoint 增量

保持基线源树，另建包含 checkpoint 增量的源树：

```bash
python3 scripts/restore_sources.py --apply-patches --checkpoint-experimental
export ROUTE2_CHECKPOINT_SOURCE="$PWD/worktrees/rlinf-route2-checkpoint"
RLINF_EXPERIMENTAL_FSDP1_TORCH22=1 \
  PYTHONPATH="$ROUTE2_CHECKPOINT_SOURCE" \
  "$ROUTE2_PYTHON" -m pytest -q \
  "$ROUTE2_CHECKPOINT_SOURCE/tests/unit_tests/test_fsdp_checkpoint.py"
for format in local_shard dcp; do
  RLINF_MUSA_TORCH_DETECTION_FALLBACK=1 \
    "$ROUTE2_PYTHON" routes/route2/fsdp_probes/checkpoint_probe.py \
    --source "$ROUTE2_CHECKPOINT_SOURCE" --phase checkpoint \
    --format "$format" --enable-torch22 \
    --checkpoint-dir "$PWD/results/checkpoint-new/$format"
done
```

每次使用从未存在的 checkpoint 目录。重建 tree 为 `a6c18c65206804ce621bf78472c5ed0b3b37387d`；两种格式分别验收新对象恢复、Adam/scheduler、Python/NumPy/Torch CPU/MUSA RNG 和继续更新一致性。结果限于 world size 1、FP32 NO_SHARD、同进程新对象，详见 [checkpoint 记录](routes/route2/fsdp_probes/checkpoint-README.md)。

## 7. 真实 action Attention / 双层 DiT

固定来源代码按 manifest 逐个下载并验证 SHA256；模型依赖放到独立 `route2-models` 环境，仅 no-deps 安装 Diffusers 0.30.2，继承匹配 Torch-MUSA：

```bash
python3 scripts/restore_model_audit.py
bash scripts/prepare_route2_model_env.sh
/root/autodl-tmp/s4000-research/envs/route2-models/bin/python \
  routes/route2/model_probes/gr00t_action_interface_probe.py \
  --device musa --phase all --output results/action-interface.json
```

该脚本使用随机权重，不下载模型 checkpoint。预期原始 `required_interfaces_pass=false`，失败仅在解析零 bias 的 relative-L2；独立 `semantic_interfaces_pass=true` 还要求这些 bias 的 actual/reference 梯度均 finite 且绝对值在原阈值内。退出码仍按原 strict gate 返回1，不把语义结果伪装成原总门槛通过。所有输出、输入/非零参数 VJP 与 mask 合约沿用原数值门槛，见 [action 接口记录](routes/route2/planning/action-interface-probe.md)。
