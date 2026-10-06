# 完整回合与十任务评估入口

2026-10-07：Runner chunk5/240步两轮GPU已通过（192动作块、960有效模拟动作槽、两次更新/rank与超时截断）；十任务pilot-v2已通过（10个唯一task/trial0、4/10 success_once、20项独立核对），不是完整500-state基线。已通过的Runner-v4/v5字节是[冻结脚本](../evidence/multi-gpu/versions/official-runner-v4-probe.py)；当前Runner脚本增加了可配置动作分块、微批次进度和独立终止mask审计，不能把旧结果当新脚本验收。后续阶段协议见[学习计划](../planning/embodied-learning-entry.md)。

## 官方5动作分块与完整回合

`official_runner_probe.py --action-chunks 5 --steps-per-env 240`沿用官方Runner/collector/dispatcher，设置每次决策执行5个动作。2环境每轮有96个策略chunk、480个模拟动作槽，每rank48个micro batch后共同完成一次PPO更新。`T=48`的策略数据保留`T+1=49`的bootstrap；不能把chunk数、denoising步骤或终止后补齐槽当独立有效环境transition。

训练保留`auto_reset=False / ignore_terminations=False`。新增审计独立计算包含首个终止动作的有效前缀，与chunk-level loss mask核对，分别报告有效chunk与有效动作槽；boundary计数是原始true标记数，不是唯一episode数。后续需实际出现成功终止、240步截断及下一轮reset，才能声明对应边界已覆盖。

```bash
cd /root/autodl-tmp/s4000-research
export PYTHONPATH=/root/autodl-tmp/s4000-research/envs/route2-integration/lib/python3.10/site-packages:/root/autodl-tmp/s4000-research/envs/route2-integration
MCCL_P2P_DISABLE=1 MCCL_DEBUG=WARN MUJOCO_GL=egl \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
envs/route2/bin/python route2/model_probes/official_runner_probe.py \
  --rlinf-source route2/RLinf-official-actor \
  --gr00t-source route2/Isaac-GR00T-official-actor \
  --model-path route2/weights/Spatial-73f710e \
  --iterations 2 --steps-per-env 240 --action-chunks 5 \
  --output route2/results/official-two-rank-chunk5-v1.json
```

此处仍为`lr=value_lr=1e-8`、epoch1、microbatch1，无保存/恢复或末尾额外同步；属于工程测试。CPU边界检查只提取固定上游的`compute_loss_mask`，验证chunk1/5、跨chunk终止、终止动作计入及后续标记重复，不启动Cluster/Ray/GPU，不能替代真实GAE/PPO。

## 初始策略十任务pilot

[official_eval_probe.py](official_eval_probe.py)真正调用官方`EmbodiedEvalRunner.init_workers()`和`evaluate()`，只启动Env/Rollout，不创建Actor或更新策略。完整model spec放在`rollout.model`，从固定Spatial权重重新初始化；2环境、5动作chunk、eval auto_reset=True、ignore_terminations=True，按`success_once`记录240步内曾成功。

评估Runner另需[显式Ray数据通道补丁](../patches/eval-ray-transport.patch)，应用在已有[两rank源码锁](two-rank-source-lock.json)之后。补丁只在显式`runner.data_channel_transport=ray`时选CPU/Ray通道，默认仍为collective；CPU构造与精确重建已[通过](../evidence/multi-gpu/eval-ray-transport-cpu-v1.json)，GPU十任务pilot-v2也已通过，指纹见[评估锁](eval-source-lock.json)。

```bash
git -C route2/RLinf-official-actor apply /path/to/rlinf-musa-lab/routes/route2/patches/eval-ray-transport.patch
MCCL_P2P_DISABLE=1 MCCL_DEBUG=WARN MUJOCO_GL=egl \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
envs/route2/bin/python route2/model_probes/official_eval_probe.py \
  --rlinf-source route2/RLinf-official-actor \
  --gr00t-source route2/Isaac-GR00T-official-actor \
  --model-path route2/weights/Spatial-73f710e \
  --trials-per-task 1 --output route2/results/official-spatial-ten-task-pilot-v1.json
```

每lane固定1200模拟步，覆盖5个240步回合；整个pilot预期10个唯一task/trial0、240次两环境策略预测。JSONL记录实际完成pair与episode指标，最终核对集合恰为task0–9的trial0。任务覆盖和动作有限性通过不代表成功率高或学习收益；即使十个任务全部成功，也不是全部init-state的完整benchmark。

第一版只固定整次Rollout的Python/NumPy/Torch/MUSA seed1234；eval仍有初始latent随机采样，没有实现独立per-episode seed hook。前后对照须使用相同batch、顺序和随机数协议。当前原始权重锁及baseline适配仍适用；完整suite须先核对实际每task init-state数量，再设置trials数并核对真实覆盖，不能盲假设50。

首轮pilot-v1执行完10个task/trial0、240次双环境预测、2400个模拟动作槽，官方`success_once=4/10`，但最终读取Rollout报告时把WorkerGroup返回列表当字典，结果标为fail。原始JSON/events和[冻结v1脚本](../evidence/multi-gpu/versions/official-eval-pilot-v1-probe.py)保留；当前v2仅检查单Rollout并取列表首项，已在相同配置下独立重跑为v2并通过，不将v1改写为pass；[20项独立核对](../evidence/multi-gpu/official-spatial-ten-task-pilot-v2.audit.json)关联原始result/events、来源锁和权重哈希。任务曾成功与最后状态成功是不同指标：v1分别4/10和3/10，主口径保留官方`success_once`。

GPU实验顺序执行，CPU的Cluster/Ray配置验证也须等当前作业退出。`runner_progress.py`只读证据且不加载Torch，允许用于正在运行的Runner；partial中初始化的`status=fail`不会被误报为最终失败。
