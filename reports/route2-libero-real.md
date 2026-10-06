# RLinf 官方 GR00T wrapper 与真实 LIBERO episode

日期：2026-10-06。实机是单卡 MTT S4000 48 GiB，Driver 2.7.0 / MUSA 3.1.0 / Torch 2.2.0 / Torch-MUSA 1.3.0。模型和环境继续使用隔离目录；没有修改宿主驱动、`/usr/local/musa` 或默认 Python。

新增官方生命周期节点：真实 `EmbodiedFSDPActor` 的构造、初始化、6个FSDP1 NO_SHARD模块和322个AdamW状态已通过，优化器预热未改变初始参数与step。独立官方 Rollout 的版本0同步也已通过：907个完整状态张量哈希一致，固定输入 logprob/value 精确相同，按官方denoise索引选择后初始 PPO ratio=1。证据见 [初始化](../routes/route2/evidence/official-actor/official-actor-init-v1.json)、[同步](../routes/route2/evidence/official-actor/official-actor-rollout-sync-v2.json) 和 [复现入口](../routes/route2/model_probes/official-actor-README.md)。下面的8步与单模型更新记录是此前节点；官方持续PPO和新进程同批恢复已通过，见下一节。

## 官方 Actor / Rollout PPO 闭环

2026-10-06，使用真实 `EmbodiedFSDPActor`、独立 `MultiStepRolloutWorker`、RLinf `Trajectory`、官方 GAE 和 `run_training()`，连续完成3轮每轮8步的真实 LIBERO-Spatial 轨迹。每轮8个 reward仍为0；这项验收关注适配执行链，不宣称任务学习。三轮 loss 为 `0.92268`、`0.70369`、`0.68939`，gradient norm 为 `1891.96`、`1778.35`、`1881.32`，每轮309或310个参数哈希改变，Adam step为1、2、3，版本同步为0、1、2、3；907个完整状态张量在每次 Actor→Rollout 同步后精确一致，固定输入输出差异为0。初始 selected PPO ratio=1，更新后固定输入 ratio约0.99962、0.99697、0.99583。

实测 Actor reserved 峰值约 `33.10 GiB`，整卡外部采样峰值约 `35.43 GiB`；48 GiB S4000 尚有余量，但完整任务、更大 batch 或更高 checkpoint 并发仍需单独测量。设备后端实测为 `ProcessGroupMCCL`，但当前 world size=1，不能推广为多卡通过。

训练 checkpoint 的 DCP 文件约16 GB。新进程加载后模型、Adam、scheduler、Python/NumPy/CPU/MUSA RNG、optimizer steps、Adam steps、策略 version 全部精确一致；同一 continuation trajectory 的下一次官方 PPO 更新与不中断路径完全一致。恢复范围是 Actor 训练状态，不含仿真器状态、Rollout RNG 和 Runner 进度。完整原始结果见 [continuous](../routes/route2/evidence/official-actor/official-actor-continuous-v1.json)、[recovery](../routes/route2/evidence/official-actor/official-actor-recovery-v1.json)。


## 完整 episode 初始基线

独立评估初始Spatial模型（没有加载三轮PPO checkpoint），task0 的trial0/trial1分别在79步和69步成功终止，耗时66.78秒和59.51秒；采样seed分别1234/1235，逐次预测平均0.799/0.817秒。两局样本成功率2/2，仅是同一任务两个初始状态，不是完整10任务suite基准，也不是PPO收益对照。所有动作有限，环境success/termination真实触发；episode循环要求成功或240步truncation，未用循环结束代替任务结束。

Rollout进程peak allocated/reserved为8.37/8.48 GiB；与Actor共驻整卡采样峰值35.44 GiB。关闭环境时出现`EGL_NOT_INITIALIZED`析构异常，后续环境重新创建和两局渲染/采样均完成；保留为未修复的非致命清理问题。原始结果见 [eval](../routes/route2/evidence/official-actor/official-actor-libero-eval-v1.json)、[summary](../routes/route2/evidence/official-actor/official-actor-summary-v1.json) 和 [EGL记录](../routes/route2/evidence/official-actor/libero-egl-cleanup-observation-v1.json)。

## 此前的单模型验证

本轮调用 RLinf 官方 `rlinf.models.embodiment.gr00t.gr00t_n1d5.get_model`，构造真实 `GR00T_N1_5_ForRLActionPrediction`，并用 RLinf 的 `LiberoEnv` 和 `predict_action_batch` 完成固定 `libero_spatial` 任务的 8 步 episode。模型严格加载 Spatial-73f710e 权重，设备为 `musa:0`。

| 检查项 | 结果 |
|---|---|
| 模型类 | `GR00T_N1_5_ForRLActionPrediction` |
| 参数量 | 2,728,490,945 |
| 可训练参数 | 1,073,133,569 |
| reset 双视角 | `[1,256,256,3]`、`[1,256,256,3]` |
| reset state | `[1,8]`，全部有限 |
| 每步动作 | `[1,7]`，全部有限 |
| episode 长度 | 8 步 |
| reward | 8 步均为 `0.0`，累计 `0.0` |
| 终止状态 | 未 terminated，未 truncated |
| 设备显存 | allocated `5,476,926,464` bytes；reserved `6,945,767,424` bytes |

每一步的 `details` 都保留了 `prev_logprobs`、`prev_values` 和 `forward_inputs`，说明 RLinf PPO 所需的 rollout 元数据已经从模型接口返回；这仍然只是推理 rollout，没有执行 actor loss、value loss、反向传播或参数更新。

完整证据：[libero-rlinf-model-episode-v9.json](../routes/route2/evidence/libero/libero-rlinf-model-episode-v9.json)。本次使用真实文件入口并恢复 RLinf worker 默认 `spawn`；证据 SHA256：`0df0619fbf773e8568eb9725e0422ecc0d70b7a2faa8431d826b4cdf7a12883e`。v8 的 `fork` 结果保留作兼容诊断，不作为最终复现入口。

随后用同一真实 reset 的训练模式 rollout 元数据执行了一次 RLinf 官方 `compute_ppo_actor_critic_loss`、反向和 `AdamW` 更新。输入是一个真实 LIBERO 观测；为了只验证计算链，`advantages=1`、`returns=prev_values+1` 是显式合成 target，不是环境 GAE。loss 为 `-2.0622501`，322 个梯度张量全部 finite，gradient L2 为 `89.27794`，参数样本发生变化；峰值之后的 allocated/reserved 为 `20,496,821,248` / `22,542,286,848` bytes。完整证据：[libero-rlinf-ppo-one-step-v5.json](../routes/route2/evidence/libero/libero-rlinf-ppo-one-step-v5.json)，SHA256：`193f43ed5fecbc9ac25bbd6e18005d0a39e58b31ffb7209567de1040efffda96`。

在同一次开机周期内，已将这条计算链推进到真实多步 GAE。使用同一 `LiberoEnv` 的 8 步真实 rollout，按环境 reward、`terminated/truncated` 和最后一步 bootstrap value 调用 RLinf `compute_gae_advantages_and_returns`，再逐步累积官方 `compute_ppo_actor_critic_loss` 的梯度并执行一次 `AdamW`。8 个 reward 都是 `0.0`，但 GAE 的 advantages、returns 全部有限；322 个梯度张量 finite 且非零，gradient L2 为 `8.6351814`，参数样本在 optimizer step 后变化。loss sum 为 `-2.7540333`，峰值后的 allocated/reserved 为 `20,496,974,848` / `22,535,995,392` bytes。该结果证明真实观测到 PPO 参数更新的适配闭环，不代表 LIBERO 成功率或学习收益。

完整证据：[libero-rlinf-ppo-multi-step-v5.json](../routes/route2/evidence/libero/libero-rlinf-ppo-multi-step-v5.json)，精确命令见同目录的 `libero-rlinf-ppo-multi-step-v5.command.txt`。

## 适配措施与边界

本轮没有安装新的通用 Torch。由于镜像没有 vendor `flash_attn`，只对当前进程禁用了可选的 vendor FlashAttention 绑定；准备好的 Eagle 配置走 Transformers eager attention。`pytorch3d` 只在 GR00T transform 模块导入时被声明为可选空模块；LIBERO-Franka 的实际状态变换没有调用它。v9 已使用真实文件入口恢复 `spawn` 并通过，因此 `fork` 只属于 v8 的 stdin 兼容诊断，不是长期适配要求。

另外发现 GR00T N1.5 的自定义 `eval()` 不返回 `self`，探针已改为分开执行 `.to()` 和 `.eval()`。这属于上游包装器调用约定，已记录为适配点，没有修改上游源码。

PPO 单步的默认 action-head BF16 + Diffusers SDPA 路径先后暴露两个独立问题：默认路径在 backward 报 `MuDNNFlashSDPABwd MUDNN failed`；换成 FP32 attention processor 后，BF16 action-head 梯度仍出现非有限值。最终探针仅将 20 个 action-head attention 实例绑定到已验证的 `FP32AttentionProcessor`，并将 action head（含 value head）切换为 FP32；Eagle backbone 保持 BF16，通过 wrapper 的 `prepare_input` 显式分流 backbone/action dtype。该局部 fallback 使单步 PPO 计算通过，未修改上游源码或全局 attention 注册。

基础 RLinf 源树在没有显式开关时仍按设计拒绝 Torch 2.2：导入 `EmbodiedFSDPActor` 返回 `ImportError: Unsupported torch version: 2.2.0`。在独立 `RLinf-fsdp1` 源树应用 `fsdp1-experimental.patch` 与 `fsdp1-actor-dtensor.patch`，并设置 `RLINF_EXPERIMENTAL_FSDP1_TORCH22=1` 后，真实 `EmbodiedFSDPActor` 模块导入通过；证据见 [Actor 审计](../routes/route2/evidence/fsdp1/embodied-actor-torch22-optin-fsdp1-v3.json)。这份历史证据只证明导入边界；本次新增构造、训练、单rank同步与恢复证据另列于上方，多卡、Actor offload或完整Runner仍待验收。

动作是模型经过 RLinf 的 observation transform、denormalization 和 LIBERO 7 维转换后的结果；本轮没有把 8 步零 reward 解读为模型失效。预训练 Spatial 模型未经本任务 PPO 微调，短 episode 只验接口、有限性和环境连续运行。该历史节点尚未验证官方Actor；本次新增连续PPO/独立Rollout同步已补齐，长期success rate、多环境并行与完整Runner仍待验收。

## 下一步

官方连续三轮PPO与fresh-process同批恢复已经通过。两局完整episode已完成。下一步进入两卡MCCL/分片/同步验证，补齐完整Runner；学习收益需更大的固定预算与多seed完整任务评估。

复现实验使用 `/root/autodl-tmp/s4000-research/envs/route2/bin/python`，`PYTHONPATH` 依次包含 `route2-integration`、`route2`、`/root/autodl-tmp/rlinf-s4000`、系统 site-packages、RLinf 源树和 `Isaac-GR00T-spatial-eager` 源树；正式多步入口为 `libero_rlinf_model_episode_probe.py --worker-start-method spawn --rollout-mode train --ppo-multi-step --action-head-fp32`，环境配置使用 RLinf 源树内的 `examples/embodiment/config/env/libero_spatial.yaml`。

多卡进入条件、实测显存与磁盘建议见 [多卡计划](../routes/route2/planning/multi-gpu-entry.md)。
