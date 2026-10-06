# 最小 PPO 适配：下一次单卡批次

本节点把已有真实 LIBERO rollout 从“单步合成 target”推进到真实 GAE 多步更新。它仍然是适配验收，不是成功率实验；8 步零 reward 只能说明当前预训练策略没有在这个固定任务上完成目标。

## 已固化的适配

- `model_probes/gr00t_action_head_fallback.py` 只在当前进程给 action-head 的 Diffusers `Attention` 绑定 `FP32AttentionProcessor`，并把 action head/value head 切到 FP32；Eagle backbone 继续使用 BF16。
- `libero_rlinf_model_episode_probe.py --action-head-fp32` 使用该 helper，不修改 Torch、MUSA、Diffusers 或 GR00T 源码。
- `--ppo-multi-step` 收集每个环境步的 `prev_logprobs`、`prev_values` 和 `forward_inputs`，用实际 `reward`、`terminated/truncated` 与最后 next observation 的 bootstrap value 调用 RLinf `compute_gae_advantages_and_returns`。
- 更新按步调用官方 `compute_ppo_actor_critic_loss` 并累积梯度，完成一次 optimizer step；这样不会为了 8 步把多个完整反向图同时留在 48 GiB 显存中。

## 开机后先跑的批次

先在同一实例会话内完成以下三项，再决定是否继续深挖：

```bash
export R=/root/autodl-tmp/s4000-research
export PYTHONPATH=$R/envs/route2-integration/lib/python3.10/site-packages:$R/envs/route2-integration:$R/route2:$R/rlinf-s4000:$R/route2/RLinf:$R/route2/Isaac-GR00T-spatial-eager
PY=$R/envs/route2-integration/bin/python

# 1. 比较 Torch 2.2 默认门槛和 RLinf 已提供的 FSDP1 opt-in。

Actor 审计使用独立的 `RLinf-fsdp1` 源树（固定 RLinf 源加 FSDP1 实验补丁），并在其上应用 `patches/fsdp1-actor-dtensor.patch`。基础 `RLinf` 源树仍保留不变；该增量只把 Torch 2.2 的 DTensor/Replicate 导入回退到 `_tensor`，不提供缺失的 FSDP2 API。
RLINF_EXPERIMENTAL_FSDP1_TORCH22=1 $PY $R/route2/model_probes/embodied_actor_torch22_probe.py \
  --rlinf-source $R/route2/RLinf-fsdp1 \
  --output $R/route2/results/embodied-actor-torch22-optin-fsdp1-v3.json

# 2. 真实 8 步 episode + 多步真实 GAE PPO 更新。
RLINF_EXPERIMENTAL_FSDP1_TORCH22=1 $PY \
  $R/route2/model_probes/libero_rlinf_model_episode_probe.py \
  --rlinf-source $R/route2/RLinf \
  --gr00t-source $R/route2/Isaac-GR00T-spatial-eager \
  --model-path $R/route2/weights/Spatial-73f710e \
  --env-config $R/route2/RLinf/examples/embodiment/config/env/libero_spatial.yaml \
  --steps 8 --worker-start-method spawn --rollout-mode train \
  --ppo-multi-step --action-head-fp32 \
  --output $R/route2/results/libero-rlinf-ppo-multi-step-v5.json
```

第 2 项的通过条件是：真实 GAE 张量有限；每步 PPO loss、梯度有限；至少一个梯度张量和参数样本发生变化；最终 optimizer step 完成；证据明确记录 rewards/dones/values/advantages/returns。它不要求本轮出现非零 reward。

若 Actor opt-in 导入通过，则下一批才进入官方 `EmbodiedFSDPActor` 构造与 `strategy: fsdp` 的单卡 `NO_SHARD` 初始化；若导入通过但构造失败，保留错误和最小配置，不修改上游版本门槛。只有在同一会话中完成上述证据后才关机。
